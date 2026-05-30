"""
Dynamic Token Analysis
======================
Key experiment for Direction C: validate that state tokens attending to
dynamic image regions develop high-frequency trajectories.

Pipeline:
  1. Run forward_recurrent_analysis on a TUM dynamic scene
  2. For each token, compute online EMA → high-freq residual energy (dynamic score)
  3. Project per-token dynamic scores to image space via cross-attention weights
  4. Visualize the resulting "dynamic heatmap" overlaid on RGB frames
  5. Quantitatively compare dynamic heatmap vs optical-flow magnitude map

If the hypothesis holds: high dynamic-score tokens should spatially correspond
to moving objects (people walking), NOT to the static background.

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/dynamic_token_analysis.py \
    --model_path /path/to/model.pth \
    --seq_path /path/to/tum/rgbd_dataset_freiburg3_walking_xyz/rgb \
    --output_dir analysis_results/dynamic_token \
    --model_update_type cut3r --size 512 --max_frames 150
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r


# ── helpers ───────────────────────────────────────────────────────────────────

def load_img_paths(rgb_dir, frame_interval=1, max_frames=150):
    exts = ('.jpg', '.jpeg', '.png', '.JPG')
    paths = sorted([os.path.join(rgb_dir, f)
                    for f in os.listdir(rgb_dir) if f.endswith(exts)])
    return paths[::frame_interval][:max_frames]


def build_views(img_paths, size=512):
    from dust3r.utils.image import load_images
    imgs = load_images(img_paths, size=size, verbose=False)
    views = []
    for img_dict in imgs:
        view = dict(img_dict)
        view['img_mask'] = torch.tensor(True).unsqueeze(0)
        view['ray_mask'] = torch.tensor(False).unsqueeze(0)
        view['ray_map']  = torch.zeros(1, *view['img'].shape[1:3], 3)
        view['update']   = torch.tensor(True).unsqueeze(0)
        view['reset']    = torch.tensor(False).unsqueeze(0)
        views.append(view)
    return views


def compute_optical_flow_magnitude(img_paths, size=512):
    """
    Compute dense optical flow magnitude between consecutive frames.
    Returns list of [H, W] arrays (one per frame, first is zeros).
    """
    flows = [None]
    prev_gray = None
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            flows.append(None)
            continue
        img_r = cv2.resize(img, (size, size))
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flows.append(mag)
        prev_gray = gray
    return flows  # len = len(img_paths), flows[0] = None


def project_token_scores_to_image(token_scores, cross_attn, img_shape):
    """
    Project per-token scores to image space using cross-attention weights.

    Args:
        token_scores: [n_state] tensor, per-token dynamic score
        cross_attn:   [n_state, n_patches] tensor, cross-attn weights
        img_shape:    (H_patches, W_patches) patch grid shape
    Returns:
        heatmap: [H_patches, W_patches] numpy array
    """
    H_p, W_p = img_shape
    # Weight patches by token score: sum over tokens, weighted by their attn
    # cross_attn: [n_state, n_patches], token_scores: [n_state]
    attn_norm = F.softmax(cross_attn.float(), dim=-1)  # [n_state, n_patches]
    weighted = (token_scores.float().unsqueeze(-1) * attn_norm).sum(0)  # [n_patches]
    n_expected = H_p * W_p
    if weighted.shape[0] > n_expected:
        weighted = weighted[1:]   # skip pose token if present
    if weighted.shape[0] == n_expected:
        heatmap = weighted.reshape(H_p, W_p).numpy()
    else:
        # Fallback: reshape to nearest square
        side = int(weighted.shape[0] ** 0.5)
        heatmap = weighted[:side*side].reshape(side, side).numpy()
    return heatmap


# ── online dynamic score tracker ─────────────────────────────────────────────

class TokenDynamicTracker:
    """
    Maintains online EMA of state tokens and running high-freq energy.
    Computes per-token 'dynamic score' = current high-freq energy / running mean.
    """
    def __init__(self, n_state, dim, device,
                 ema_momentum=0.9, running_momentum=0.9):
        self.mu    = ema_momentum
        self.gamma = running_momentum
        self.ema   = None
        self.running_energy = None
        self.warmed_up = False

    def update(self, state_feat):
        """
        Args:
            state_feat: [1, n_state, D] tensor
        Returns:
            dynamic_score: [n_state] tensor, higher = more dynamic
        """
        s = state_feat.squeeze(0)  # [n_state, D]

        if self.ema is None:
            self.ema = s.clone()
            self.running_energy = torch.zeros(s.shape[0], device=s.device)
            return torch.zeros(s.shape[0], device=s.device)

        self.ema = self.mu * self.ema + (1 - self.mu) * s
        high_freq = s - self.ema
        energy = high_freq.norm(dim=-1)  # [n_state]

        if not self.warmed_up:
            self.running_energy = energy.clone()
            self.warmed_up = True
        else:
            self.running_energy = (self.gamma * self.running_energy +
                                   (1 - self.gamma) * energy)

        dynamic_score = energy / (self.running_energy + 1e-6)
        return dynamic_score  # [n_state]


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',  type=str, required=True)
    p.add_argument('--seq_path',    type=str, required=True)
    p.add_argument('--output_dir',  type=str,
                   default='analysis_results/dynamic_token')
    p.add_argument('--model_update_type', type=str, default='cut3r')
    p.add_argument('--size',        type=int, default=512)
    p.add_argument('--max_frames',  type=int, default=150)
    p.add_argument('--frame_interval', type=int, default=1)
    p.add_argument('--ema_momentum',     type=float, default=0.9)
    p.add_argument('--running_momentum', type=float, default=0.9)
    p.add_argument('--device',      type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    print(f"[model] {args.model_update_type}")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    img_paths = load_img_paths(args.seq_path, args.frame_interval, args.max_frames)
    print(f"[data] {len(img_paths)} frames")

    all_views = build_views(img_paths, args.size)

    # Run inference and collect state + cross-attention history
    print("[run] forward_recurrent_analysis ...")
    with torch.no_grad():
        ress, analysis = model.forward_recurrent_analysis(all_views, device=device)

    state_history   = analysis['state_history']    # list[T] of [n_state, D]
    attn_history    = analysis['cross_attn_history']  # list[T] of [n_state, n_patches]
    img_shapes      = analysis['img_shapes']        # list[T] of (H_p, W_p)
    T = len(state_history)
    print(f"[data] T={T}  n_state={state_history[0].shape[0]}")

    # Compute optical flow magnitude for ground-truth dynamic regions
    print("[flow] Computing optical flow ...")
    flow_mags = compute_optical_flow_magnitude(img_paths, size=args.size)

    # Compute dynamic scores over time
    print("[score] Computing dynamic token scores ...")
    tracker = TokenDynamicTracker(
        n_state=state_history[0].shape[0],
        dim=state_history[0].shape[1],
        device='cpu',
        ema_momentum=args.ema_momentum,
        running_momentum=args.running_momentum,
    )

    dynamic_scores_all = []   # list[T] of [n_state]
    heatmaps_all       = []   # list[T] of [H_p, W_p]
    flow_mags_valid    = []   # parallel to dynamic_scores_all
    corr_scores        = []   # frame-level: mean(dynamic_score) vs mean(flow_mag)

    for t in range(T):
        state_t = state_history[t].unsqueeze(0)  # [1, n_state, D]
        ds = tracker.update(state_t)              # [n_state]
        dynamic_scores_all.append(ds)

        attn_t = attn_history[t]   # [n_state, n_patches]
        hmap = project_token_scores_to_image(ds, attn_t, img_shapes[t])
        heatmaps_all.append(hmap)

        if flow_mags[t] is not None:
            fm = flow_mags[t]
            fm_resized = cv2.resize(
                fm, (hmap.shape[1], hmap.shape[0]),
                interpolation=cv2.INTER_AREA)
            flow_mags_valid.append((t, fm_resized))
            corr_scores.append((ds.mean().item(), fm.mean()))

    # Correlation: mean dynamic score vs mean optical flow
    if len(corr_scores) > 10:
        ds_means  = [c[0] for c in corr_scores]
        flow_means = [c[1] for c in corr_scores]
        r_pearson, p_pearson = pearsonr(ds_means, flow_means)
        r_spearman, _        = spearmanr(ds_means, flow_means)
        print(f"\n[result] Dynamic score ↔ Optical flow correlation:")
        print(f"  Pearson  r = {r_pearson:+.3f}  (p={p_pearson:.4f})")
        print(f"  Spearman ρ = {r_spearman:+.3f}")
    else:
        r_pearson, r_spearman = np.nan, np.nan

    # Spatial correlation: per-frame heatmap vs flow_mag
    spatial_corrs = []
    for t, fm_r in flow_mags_valid:
        hmap = heatmaps_all[t]
        h_flat = hmap.flatten()
        f_flat = fm_r.flatten()
        valid = np.isfinite(h_flat) & np.isfinite(f_flat)
        if valid.sum() > 20:
            r, _ = pearsonr(h_flat[valid], f_flat[valid])
            spatial_corrs.append(r)
    mean_spatial_r = np.mean(spatial_corrs) if spatial_corrs else np.nan
    print(f"  Mean spatial r (heatmap vs flow) = {mean_spatial_r:+.3f}")

    # Save summary
    with open(os.path.join(args.output_dir, 'dynamic_summary.txt'), 'w') as f:
        f.write(f"seq_path: {args.seq_path}\n")
        f.write(f"model_update_type: {args.model_update_type}\n")
        f.write(f"T={T}  n_state={state_history[0].shape[0]}\n")
        f.write(f"temporal_r_pearson={r_pearson:.4f}  p={p_pearson:.4f}\n"
                if not np.isnan(r_pearson) else "temporal_r=nan\n")
        f.write(f"temporal_r_spearman={r_spearman:.4f}\n"
                if not np.isnan(r_spearman) else "")
        f.write(f"mean_spatial_r={mean_spatial_r:.4f}\n")

    # ── Visualise selected frames ─────────────────────────────────────────────
    print("[vis] Saving frame visualisations ...")
    vis_indices = list(range(0, T, max(1, T // 20)))[:20]

    for t in tqdm(vis_indices, desc='vis'):
        img_bgr = cv2.imread(img_paths[t])
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (args.size, args.size))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        hmap = heatmaps_all[t]
        hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        hmap_up = cv2.resize(hmap_norm, (args.size, args.size),
                             interpolation=cv2.INTER_LINEAR)

        fig = plt.figure(figsize=(12, 4))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(img_rgb)
        ax1.set_title(f'Frame {t}', fontsize=9)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(img_rgb)
        ax2.imshow(hmap_up, alpha=0.55, cmap='hot', vmin=0, vmax=1)
        ax2.set_title('Dynamic Token Score', fontsize=9)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[2])
        if flow_mags[t] is not None:
            fm = flow_mags[t]
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            ax3.imshow(fm_norm, cmap='hot', vmin=0, vmax=1)
            ax3.set_title('Optical Flow Magnitude', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'N/A', ha='center', va='center',
                     transform=ax3.transAxes)
            ax3.set_title('Optical Flow', fontsize=9)
        ax3.axis('off')

        plt.suptitle(
            f'{os.path.basename(args.seq_path)} | {args.model_update_type} | '
            f't={t}  spatial_r={spatial_corrs[min(len(spatial_corrs)-1, t//max(1,T//len(spatial_corrs)))]:+.3f}'
            if spatial_corrs else
            f'{os.path.basename(args.seq_path)} | t={t}',
            fontsize=8)
        plt.savefig(os.path.join(frames_dir, f'frame_{t:04d}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── Summary figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    t_arr = np.arange(T)
    ds_mean_arr = np.array([ds.mean().item() for ds in dynamic_scores_all])
    axes[0].plot(t_arr, ds_mean_arr, color='C1', linewidth=0.8)
    axes[0].set_ylabel('Mean dynamic score')
    axes[0].set_title('Token Dynamic Score over Time')

    flow_t   = [t for t, _ in flow_mags_valid]
    flow_val = [fm.mean() for _, fm in flow_mags_valid]
    axes[1].plot(flow_t, flow_val, color='C0', linewidth=0.8)
    axes[1].set_ylabel('Mean optical flow magnitude')
    axes[1].set_xlabel('Frame')
    axes[1].set_title(
        f'Optical Flow Magnitude  |  '
        f'temporal r={r_pearson:+.3f}  spatial r={mean_spatial_r:+.3f}')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'dynamic_overview.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[done] → {args.output_dir}")
    print(f"  Key metrics:")
    print(f"    temporal r (score↔flow) = {r_pearson:+.3f}")
    print(f"    spatial  r (heatmap↔flow) = {mean_spatial_r:+.3f}")


if __name__ == '__main__':
    main()
