"""
Layer 1 Analysis: Frequency-Domain Frame Spectral Change
=================================================

Motivation experiment for the frame-selection layer:

1. Show that state token oscillation correlates with frame redundancy
   (redundant frames = low structural spectral_change)
2. Show that skipping low-spectral_change frames reduces state oscillation
3. Measure accuracy (depth error) with and without frame filtering
4. Visualize spectral_change scores, kept vs skipped frames, and state oscillation

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/frame_spectral_change_analysis.py \
    --model_path /path/to/model.pth \
    --seq_path /path/to/scene/color \
    --output_dir analysis_results/frame_spectral_change \
    --depth_dir /path/to/scene/depth \
    --model_update_type cut3r --size 512 --max_frames 300
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r


# ─── data helpers (reuse from spectral_analysis) ────────────────────────────

def load_img_paths(rgb_dir, frame_interval=1, max_frames=300):
    exts = ('.jpg', '.jpeg', '.png', '.JPG')
    paths = sorted([os.path.join(rgb_dir, f)
                    for f in os.listdir(rgb_dir)
                    if f.endswith(exts)])
    paths = paths[::frame_interval]
    return paths[:max_frames]


def build_views(img_paths, size=512):
    from dust3r.utils.image import load_images
    from dust3r.utils.device import to_cpu
    imgs = load_images(img_paths, size=size, verbose=False)
    views = []
    for img_dict in imgs:
        view = {k: v for k, v in img_dict.items()}
        view['img_mask'] = torch.tensor(True).unsqueeze(0)
        view['ray_mask'] = torch.tensor(False).unsqueeze(0)
        view['ray_map']  = torch.zeros(1, *view['img'].shape[1:3], 3)
        view['update']   = torch.tensor(True).unsqueeze(0)
        view['reset']    = torch.tensor(False).unsqueeze(0)
        views.append(view)
    return views


def load_gt_depth(depth_path, depth_scale=1000.0):
    raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if raw is None:
        return None
    depth = raw.astype(np.float32) / depth_scale
    depth[depth <= 0] = np.nan
    return depth


def match_depth_path(img_path, depth_dir):
    """Try to find a matching depth file for an rgb image path."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    for ext in ('.png', '.PNG'):
        cand = os.path.join(depth_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def compute_depth_error(pred_depth, gt_depth, max_depth=10.0):
    mask = np.isfinite(gt_depth) & (gt_depth > 0) & (gt_depth < max_depth)
    if mask.sum() < 100:
        return np.nan
    h, w = gt_depth.shape
    ph, pw = pred_depth.shape
    if (ph, pw) != (h, w):
        pred_depth = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    valid_pred = pred_depth[mask]
    valid_gt   = gt_depth[mask]
    valid_pred = np.clip(valid_pred, 1e-3, max_depth)
    valid_gt   = np.clip(valid_gt, 1e-3, max_depth)
    # scale-invariant: align median
    scale = np.median(valid_gt) / (np.median(valid_pred) + 1e-6)
    valid_pred = valid_pred * scale
    return float(np.mean(np.abs(valid_pred - valid_gt) / (valid_gt + 1e-6)))


# ─── state oscillation metric ────────────────────────────────────────────────

def compute_state_oscillation(state_history):
    """
    Per-frame state change magnitude (L2 norm of token delta).
    Returns array of shape [T-1].
    """
    oscs = []
    for t in range(1, len(state_history)):
        delta = state_history[t].float() - state_history[t-1].float()
        osc = delta.norm(dim=-1).mean().item()
        oscs.append(osc)
    return np.array(oscs)


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--seq_path',   type=str, required=True,
                   help='RGB directory')
    p.add_argument('--depth_dir',  type=str, default='',
                   help='GT depth directory (optional)')
    p.add_argument('--output_dir', type=str, default='analysis_results/frame_spectral_change')
    p.add_argument('--model_update_type', type=str, default='cut3r')
    p.add_argument('--depth_scale', type=float, default=1000.0)
    p.add_argument('--size',        type=int, default=512)
    p.add_argument('--max_frames',  type=int, default=300)
    p.add_argument('--frame_interval', type=int, default=1)
    p.add_argument('--skip_ratio',  type=float, default=0.3,
                   help='Skip frames whose energy < skip_ratio * running_mean')
    p.add_argument('--warmup',      type=int, default=10,
                   help='Always keep first N frames for running mean warmup')
    p.add_argument('--max_depth',   type=float, default=10.0)
    p.add_argument('--device',      type=str, default='cuda')
    return p.parse_args()


def run_inference_with_state(model, views, device):
    """Run inference, capture state history via forward_recurrent_analysis."""
    with torch.no_grad():
        ress, analysis = model.forward_recurrent_analysis(views, device=device)
    return ress, analysis['state_history']


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    print(f"[model] Loading {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    img_paths = load_img_paths(args.seq_path, args.frame_interval, args.max_frames)
    print(f"[data] {len(img_paths)} frames from {args.seq_path}")

    all_views = build_views(img_paths, args.size)

    # ── Compute per-frame spectral_change (before inference) ──────────────────────────
    print("[spectral_change] Computing frame spectral_change scores ...")
    spectral_change_scores = [None]  # frame 0 has no prev
    imgs_tensor = [v['img'].float() for v in all_views]
    for i in range(1, len(imgs_tensor)):
        sc = ARCroco3DStereo.compute_frame_spectral_change(imgs_tensor[i-1], imgs_tensor[i])
        spectral_change_scores.append(sc)
    novelties_arr = np.array([n if n is not None else np.nan
                              for n in spectral_change_scores])

    # ── Run full sequence (baseline) ─────────────────────────────────────────
    print("[run] Full sequence (baseline) ...")
    ress_full, state_hist_full = run_inference_with_state(model, all_views, device)
    osc_full = compute_state_oscillation(state_hist_full)

    # ── Run filtered sequence ─────────────────────────────────────────────────
    print(f"[filter] skip_ratio={args.skip_ratio}, warmup={args.warmup}")
    kept_views, kept_indices, _ = ARCroco3DStereo.filter_views_by_spectral_change(
        all_views, skip_ratio=args.skip_ratio, warmup=args.warmup, device='cpu')
    skip_rate = 1.0 - len(kept_views) / len(all_views)
    print(f"[filter] Kept {len(kept_views)}/{len(all_views)} frames "
          f"(skip rate = {skip_rate:.1%})")

    ress_filt, state_hist_filt = run_inference_with_state(model, kept_views, device)
    osc_filt = compute_state_oscillation(state_hist_filt)

    # ── Depth error (if GT available) ─────────────────────────────────────────
    has_depth = args.depth_dir and os.path.isdir(args.depth_dir)
    errors_full = []
    errors_filt = []
    if has_depth:
        print("[depth] Computing depth errors ...")
        for t, (res, img_path) in enumerate(zip(ress_full, img_paths)):
            dpath = match_depth_path(img_path, args.depth_dir)
            if dpath is None:
                continue
            gt = load_gt_depth(dpath, args.depth_scale)
            if gt is None:
                continue
            pts = res['pts3d_in_self_view']
            pred_d = pts[0, :, :, 2].numpy()
            err = compute_depth_error(pred_d, gt, args.max_depth)
            errors_full.append(err)

        for t, (res, orig_idx) in enumerate(zip(ress_filt, kept_indices)):
            dpath = match_depth_path(img_paths[orig_idx], args.depth_dir)
            if dpath is None:
                continue
            gt = load_gt_depth(dpath, args.depth_scale)
            if gt is None:
                continue
            pts = res['pts3d_in_self_view']
            pred_d = pts[0, :, :, 2].numpy()
            err = compute_depth_error(pred_d, gt, args.max_depth)
            errors_filt.append(err)

    # ── Correlate oscillation vs spectral_change ─────────────────────────────────────
    # osc_full[t] corresponds to frames 1..T-1 (delta between t and t-1)
    nov_for_osc = novelties_arr[1:len(osc_full)+1]
    valid = np.isfinite(nov_for_osc) & np.isfinite(osc_full)
    if valid.sum() > 10:
        r_nov_osc, p_nov_osc = pearsonr(nov_for_osc[valid], osc_full[valid])
    else:
        r_nov_osc, p_nov_osc = np.nan, np.nan

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LAYER 1 ANALYSIS SUMMARY")
    print("="*60)
    print(f"Frames: {len(all_views)} total → {len(kept_views)} kept "
          f"({skip_rate:.1%} skipped)")
    print(f"\nState oscillation (mean ± std):")
    print(f"  Full sequence : {osc_full.mean():.4f} ± {osc_full.std():.4f}")
    print(f"  Filtered seq  : {osc_filt.mean():.4f} ± {osc_filt.std():.4f}")
    print(f"  Reduction     : {(osc_full.mean()-osc_filt.mean())/osc_full.mean()*100:.1f}%")
    print(f"\nSpectral Change ↔ Oscillation correlation: r={r_nov_osc:+.3f} (p={p_nov_osc:.3f})")
    if has_depth:
        ef = np.nanmean(errors_full)
        efilt = np.nanmean(errors_filt)
        print(f"\nDepth error (abs_rel):")
        print(f"  Full sequence : {ef:.4f}")
        print(f"  Filtered seq  : {efilt:.4f}")
        print(f"  Change        : {(efilt-ef)/ef*100:+.1f}%")
    print("="*60)

    with open(os.path.join(args.output_dir, 'spectral_change_summary.txt'), 'w') as f:
        f.write(f"seq_path: {args.seq_path}\n")
        f.write(f"model_update_type: {args.model_update_type}\n")
        f.write(f"skip_ratio={args.skip_ratio} warmup={args.warmup}\n")
        f.write(f"total_frames={len(all_views)} kept={len(kept_views)} "
                f"skip_rate={skip_rate:.3f}\n")
        f.write(f"osc_full={osc_full.mean():.4f}±{osc_full.std():.4f}\n")
        f.write(f"osc_filt={osc_filt.mean():.4f}±{osc_filt.std():.4f}\n")
        f.write(f"osc_reduction={100*(osc_full.mean()-osc_filt.mean())/osc_full.mean():.1f}%\n")
        f.write(f"r_spectral_change_oscillation={r_nov_osc:.4f} p={p_nov_osc:.4f}\n")
        if has_depth:
            f.write(f"depth_err_full={np.nanmean(errors_full):.4f}\n")
            f.write(f"depth_err_filt={np.nanmean(errors_filt):.4f}\n")

    # ── Visualizations ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    t_full = np.arange(len(novelties_arr))
    t_osc  = np.arange(1, len(osc_full)+1)

    # Panel 1: Spectral Change scores + kept/skipped
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_full, novelties_arr, color='C0', linewidth=0.8, label='Spectral Change score')
    ax1.axhline(np.nanmean(novelties_arr) * args.skip_ratio,
                color='red', linestyle='--', linewidth=1.0,
                label=f'skip threshold (≈{args.skip_ratio}×mean)')
    skipped = set(range(len(all_views))) - set(kept_indices)
    for si in skipped:
        ax1.axvspan(si - 0.5, si + 0.5, alpha=0.15, color='red')
    ax1.set_xlabel('Frame index')
    ax1.set_ylabel('Low-freq energy ratio')
    ax1.set_title(f'Frame Spectral Change Scores — {skip_rate:.1%} frames skipped (red)')
    ax1.legend(loc='upper right', fontsize=8)

    # Panel 2: State oscillation — full vs filtered
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_osc, osc_full, color='C0', linewidth=0.7, alpha=0.7,
             label=f'Full  μ={osc_full.mean():.3f}')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('State token Δ (L2)')
    ax2.set_title('State Oscillation: Full Sequence')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    t_filt = np.arange(1, len(osc_filt)+1)
    ax3.plot(t_filt, osc_filt, color='C2', linewidth=0.7, alpha=0.7,
             label=f'Filtered μ={osc_filt.mean():.3f}')
    ax3.set_xlabel('Frame (kept only)')
    ax3.set_ylabel('State token Δ (L2)')
    ax3.set_title('State Oscillation: Filtered Sequence')
    ax3.legend(fontsize=8)

    # Panel 3: Scatter spectral_change vs oscillation
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(nov_for_osc[valid], osc_full[valid],
                alpha=0.4, s=8, color='C0')
    ax4.set_xlabel('Frame spectral_change score')
    ax4.set_ylabel('State oscillation')
    ax4.set_title(f'Spectral Change ↔ Oscillation (r={r_nov_osc:+.3f})')

    # Panel 4: Depth error comparison (if available)
    ax5 = fig.add_subplot(gs[2, 1])
    if has_depth and errors_full and errors_filt:
        labels = ['Full sequence', 'Filtered sequence']
        means  = [np.nanmean(errors_full), np.nanmean(errors_filt)]
        stds   = [np.nanstd(errors_full),  np.nanstd(errors_filt)]
        bars = ax5.bar(labels, means, yerr=stds, capsize=5,
                       color=['C0', 'C2'], alpha=0.8)
        ax5.set_ylabel('Abs Rel Depth Error')
        ax5.set_title('Depth Error: Full vs Filtered')
        for bar, m in zip(bars, means):
            ax5.text(bar.get_x() + bar.get_width()/2, m + 0.001,
                     f'{m:.4f}', ha='center', va='bottom', fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'No GT depth available',
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Depth Error')

    plt.suptitle(f'Layer 1: Frequency-Domain Frame Spectral Change Analysis\n'
                 f'{os.path.basename(args.seq_path)} | {args.model_update_type}',
                 fontsize=11)
    out_fig = os.path.join(args.output_dir, 'frame_spectral_change_analysis.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[done] → {args.output_dir}")


if __name__ == '__main__':
    main()
