"""
Batch Layer 1 Validation: Frequency-Domain Frame Spectral Change Filtering
===========================================================================

Runs frame spectral change filtering across multiple ScanNet + TUM scenes,
aggregating depth error and state oscillation statistics.

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/batch_frame_novelty.py \
    --model_path /path/to/model.pth \
    --scannet_root /path/to/scannetv2 \
    --tum_root /path/to/tum \
    --output_dir analysis_results/batch_frame_novelty \
    --num_scannet 10 --seed 42
"""

import os
import sys
import argparse
import glob

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r


# ── helpers ──────────────────────────────────────────────────────────────────

def load_img_paths(rgb_dir, frame_interval=1, max_frames=300):
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


def load_gt_depth(depth_path, depth_scale=1000.0):
    raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if raw is None:
        return None
    depth = raw.astype(np.float32) / depth_scale
    depth[depth <= 0] = np.nan
    return depth


def load_tum_associations(scene_dir):
    """Load TUM rgb→depth associations from depth.txt + rgb.txt by nearest timestamp."""
    rgb_txt   = os.path.join(scene_dir, 'rgb.txt')
    depth_txt = os.path.join(scene_dir, 'depth.txt')
    if not (os.path.exists(rgb_txt) and os.path.exists(depth_txt)):
        return {}

    def parse_txt(path):
        entries = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    entries[float(parts[0])] = parts[1]
        return entries

    rgb_entries   = parse_txt(rgb_txt)    # ts → rgb/xxxxx.png
    depth_entries = parse_txt(depth_txt)  # ts → depth/xxxxx.png
    depth_ts = sorted(depth_entries.keys())

    assoc = {}  # rgb filename → depth filepath
    for rts, rpath in rgb_entries.items():
        # nearest depth timestamp
        idx = np.searchsorted(depth_ts, rts)
        idx = min(max(idx, 0), len(depth_ts) - 1)
        if abs(depth_ts[idx] - rts) < 0.02:   # 20ms tolerance
            dpath = os.path.join(scene_dir, depth_entries[depth_ts[idx]])
            assoc[os.path.basename(rpath)] = dpath
    return assoc


def match_depth_path(img_path, depth_dir, dataset='scannet', tum_assoc=None):
    if dataset == 'tum' and tum_assoc:
        key = os.path.basename(img_path)
        return tum_assoc.get(key, None)
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
    vp = np.clip(pred_depth[mask], 1e-3, max_depth)
    vg = np.clip(gt_depth[mask],   1e-3, max_depth)
    scale = np.median(vg) / (np.median(vp) + 1e-6)
    return float(np.mean(np.abs(vp * scale - vg) / (vg + 1e-6)))


def compute_state_oscillation(state_history):
    oscs = []
    for t in range(1, len(state_history)):
        delta = state_history[t].float() - state_history[t-1].float()
        oscs.append(delta.norm(dim=-1).mean().item())
    return np.array(oscs)


# ── per-scene evaluation ──────────────────────────────────────────────────────

def eval_scene(model, img_paths, depth_dir, depth_scale, size,
               max_depth, skip_ratio, warmup, device, dataset, scene_dir=''):
    from dust3r.model import ARCroco3DStereo

    if len(img_paths) < 30:
        return None

    all_views = build_views(img_paths, size)

    # TUM depth associations
    tum_assoc = load_tum_associations(scene_dir) if dataset == 'tum' else {}

    # Compute spectral change scores
    imgs_tensor = [v['img'].float() for v in all_views]
    sc_scores = [0.0]
    for i in range(1, len(imgs_tensor)):
        sc = ARCroco3DStereo.compute_frame_spectral_change(
            imgs_tensor[i-1], imgs_tensor[i])
        sc_scores.append(sc)
    sc_arr = np.array(sc_scores)

    # Filter views
    kept_views, kept_indices, _ = ARCroco3DStereo.filter_views_by_spectral_change(
        all_views, skip_ratio=skip_ratio, warmup=warmup, device='cpu')
    skip_rate = 1.0 - len(kept_views) / len(all_views)

    # Inference: full sequence
    with torch.no_grad():
        ress_full, ana_full = model.forward_recurrent_analysis(all_views, device=device)
    osc_full = compute_state_oscillation(ana_full['state_history'])

    # Inference: filtered sequence
    with torch.no_grad():
        ress_filt, ana_filt = model.forward_recurrent_analysis(kept_views, device=device)
    osc_filt = compute_state_oscillation(ana_filt['state_history'])

    # Spectral change ↔ oscillation correlation
    sc_for_osc = sc_arr[1:len(osc_full)+1]
    valid = np.isfinite(sc_for_osc) & np.isfinite(osc_full) & (sc_for_osc > 0)
    r_sc_osc = pearsonr(sc_for_osc[valid], osc_full[valid])[0] if valid.sum() > 10 else np.nan

    # Depth errors — evaluated on the SAME set of frames (kept_indices) for fair comparison.
    # full_at_kept:  full-sequence predictions at kept frame positions
    # filt_at_kept:  filtered-sequence predictions at kept frame positions
    # This isolates the effect of skipping on reconstruction quality,
    # independent of which frames are selected.
    errors_full_at_kept, errors_filt = [], []

    # Build a map from original index → position in ress_filt
    filt_pos = {orig_idx: t for t, orig_idx in enumerate(kept_indices)}

    if depth_dir and os.path.isdir(depth_dir):
        for orig_idx, res_filt in zip(kept_indices, ress_filt):
            ipath = img_paths[orig_idx]
            dpath = match_depth_path(ipath, depth_dir, dataset, tum_assoc)
            if dpath is None:
                continue
            gt = load_gt_depth(dpath, depth_scale)
            if gt is None:
                continue

            # Full sequence prediction at this same frame
            pred_d_full = ress_full[orig_idx]['pts3d_in_self_view'][0, :, :, 2].numpy()
            err_full = compute_depth_error(pred_d_full, gt, max_depth)

            # Filtered sequence prediction at this frame
            pred_d_filt = res_filt['pts3d_in_self_view'][0, :, :, 2].numpy()
            err_filt = compute_depth_error(pred_d_filt, gt, max_depth)

            if not np.isnan(err_full):
                errors_full_at_kept.append(err_full)
            if not np.isnan(err_filt):
                errors_filt.append(err_filt)

    return {
        'skip_rate':     skip_rate,
        'osc_full':      osc_full.mean(),
        'osc_filt':      osc_filt.mean(),
        'r_sc_osc':      r_sc_osc,
        # Fair comparison: both evaluated on the same kept frames
        'err_full':      np.nanmean(errors_full_at_kept) if errors_full_at_kept else np.nan,
        'err_filt':      np.nanmean(errors_filt)         if errors_filt         else np.nan,
        'n_full':        len(all_views),
        'n_filt':        len(kept_views),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',    type=str, required=True)
    p.add_argument('--scannet_root',  type=str, default='')
    p.add_argument('--tum_root',      type=str, default='')
    p.add_argument('--output_dir',    type=str, default='analysis_results/batch_frame_novelty')
    p.add_argument('--model_update_type', type=str, default='cut3r')
    p.add_argument('--num_scannet',   type=int, default=10)
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--size',          type=int, default=512)
    p.add_argument('--max_frames',    type=int, default=300)
    p.add_argument('--frame_interval',type=int, default=1)
    p.add_argument('--skip_ratio',    type=float, default=0.3)
    p.add_argument('--warmup',        type=int, default=10)
    p.add_argument('--max_depth',     type=float, default=10.0)
    p.add_argument('--device',        type=str, default='cuda')
    return p.parse_args()


def discover_scenes(args):
    scenes = []
    rng = np.random.RandomState(args.seed)

    if args.scannet_root and os.path.isdir(args.scannet_root):
        all_sd = sorted(glob.glob(os.path.join(args.scannet_root, 'scene*')))
        valid = [sd for sd in all_sd
                 if os.path.isdir(os.path.join(sd, 'color'))
                 and os.path.isdir(os.path.join(sd, 'depth'))]
        if len(valid) > args.num_scannet:
            valid = [valid[i] for i in sorted(
                rng.choice(len(valid), args.num_scannet, replace=False))]
        for sd in valid:
            scenes.append({
                'name': os.path.basename(sd),
                'rgb_dir': os.path.join(sd, 'color'),
                'depth_dir': os.path.join(sd, 'depth'),
                'scene_dir': sd,
                'depth_scale': 1000.0,
                'frame_interval': args.frame_interval,
                'dataset': 'scannet',
            })

    if args.tum_root and os.path.isdir(args.tum_root):
        for td in sorted(glob.glob(os.path.join(args.tum_root, 'rgbd_dataset_*'))):
            rgb_dir   = os.path.join(td, 'rgb')
            depth_dir = os.path.join(td, 'depth')
            if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir):
                scenes.append({
                    'name': os.path.basename(td),
                    'rgb_dir': rgb_dir,
                    'depth_dir': depth_dir,
                    'scene_dir': td,          # needed for TUM associations
                    'depth_scale': 5000.0,
                    'frame_interval': args.frame_interval,
                    'dataset': 'tum',
                })

    return scenes


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    scenes = discover_scenes(args)
    n_sn  = sum(1 for s in scenes if s['dataset'] == 'scannet')
    n_tum = sum(1 for s in scenes if s['dataset'] == 'tum')
    print(f"[data] {len(scenes)} scenes ({n_sn} ScanNet, {n_tum} TUM)")

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    all_results = {}
    for scene in tqdm(scenes, desc='scenes'):
        img_paths = load_img_paths(
            scene['rgb_dir'], scene['frame_interval'], args.max_frames)
        try:
            r = eval_scene(model, img_paths,
                           scene['depth_dir'], scene['depth_scale'],
                           args.size, args.max_depth,
                           args.skip_ratio, args.warmup, device, scene['dataset'],
                           scene_dir=scene.get('scene_dir', ''))
        except Exception as e:
            print(f"  [warn] {scene['name']}: {e}")
            continue
        if r is not None:
            r['dataset'] = scene['dataset']
            all_results[scene['name']] = r
            print(f"  {scene['name']}: skip={r['skip_rate']:.1%}  "
                  f"r_sc_osc={r['r_sc_osc']:+.3f}  "
                  f"err_full={r['err_full']:.4f}  err_filt={r['err_filt']:.4f}  "
                  f"err_chg={100*(r['err_filt']-r['err_full'])/(r['err_full']+1e-8):+.1f}%")

    if not all_results:
        print("[error] No valid results.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def agg(key, ds=None):
        vals = [r[key] for r in all_results.values()
                if (ds is None or r['dataset'] == ds) and not np.isnan(r[key])]
        return (np.mean(vals), np.std(vals), len(vals)) if vals else (np.nan, np.nan, 0)

    lines = ['=' * 70, 'BATCH LAYER 1 VALIDATION SUMMARY', '=' * 70, '']
    lines.append(f"Model: {args.model_update_type}  |  skip_ratio={args.skip_ratio}")
    lines.append(f"Scenes: {len(all_results)} ({n_sn} ScanNet, {n_tum} TUM)\n")

    for ds_label, ds_key in [('ALL', None), ('ScanNet', 'scannet'), ('TUM', 'tum')]:
        sk  = agg('skip_rate', ds_key)
        r_  = agg('r_sc_osc', ds_key)
        ef  = agg('err_full', ds_key)
        efi = agg('err_filt', ds_key)
        if sk[2] == 0:
            continue
        err_chg = 100 * (efi[0] - ef[0]) / (ef[0] + 1e-8)
        lines.append(f"[{ds_label}]  n={sk[2]}")
        lines.append(f"  Skip rate          : {sk[0]:.1%} ± {sk[1]:.1%}")
        lines.append(f"  r(sc, oscillation) : {r_[0]:+.3f} ± {r_[1]:.3f}")
        lines.append(f"  Depth err (full)   : {ef[0]:.4f} ± {ef[1]:.4f}")
        lines.append(f"  Depth err (filt)   : {efi[0]:.4f} ± {efi[1]:.4f}")
        lines.append(f"  Depth err change   : {err_chg:+.1f}%\n")

    lines.append('--- Per-scene ---')
    hdr = f"{'Scene':<45s} {'dataset':>8s} {'skip':>6s} {'r_sc_osc':>9s} {'err_full':>9s} {'err_filt':>9s} {'chg%':>7s}"
    lines.append(hdr)
    lines.append('-' * 100)
    for name, r in sorted(all_results.items()):
        chg = 100 * (r['err_filt'] - r['err_full']) / (r['err_full'] + 1e-8)
        lines.append(
            f"{name:<45s} {r['dataset']:>8s} {r['skip_rate']:>5.1%} "
            f"{r['r_sc_osc']:>+9.3f} {r['err_full']:>9.4f} "
            f"{r['err_filt']:>9.4f} {chg:>+7.1f}%")
    lines.append('=' * 70)

    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(os.path.join(args.output_dir, 'batch_summary.txt'), 'w') as f:
        f.write(summary + '\n')

    # ── Plot ─────────────────────────────────────────────────────────────────
    names = list(all_results.keys())
    datasets = [all_results[n]['dataset'] for n in names]
    colors = ['C0' if d == 'scannet' else 'C1' for d in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: r(sc, oscillation) per scene
    ax = axes[0]
    bars = ax.bar(range(len(names)), [all_results[n]['r_sc_osc'] for n in names],
                  color=colors, alpha=0.8)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[-10:] for n in names], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Pearson r')
    ax.set_title('Spectral Change ↔ State Oscillation')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='C0', label='ScanNet'),
                        Patch(color='C1', label='TUM')], fontsize=8)

    # Panel 2: depth error full vs filt
    ax = axes[1]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, [all_results[n]['err_full'] for n in names],
           w, color=colors, alpha=0.5, label='Full')
    ax.bar(x + w/2, [all_results[n]['err_filt'] for n in names],
           w, color=colors, alpha=0.9, label='Filtered')
    ax.set_xticks(x)
    ax.set_xticklabels([n[-10:] for n in names], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Abs Rel Depth Error')
    ax.set_title('Depth Error: Full vs Filtered')
    ax.legend(fontsize=8)

    # Panel 3: skip rate per scene
    ax = axes[2]
    ax.bar(range(len(names)), [all_results[n]['skip_rate'] * 100 for n in names],
           color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[-10:] for n in names], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Skip rate (%)')
    ax.set_title(f'Frames Skipped (skip_ratio={args.skip_ratio})')

    plt.suptitle(f'Layer 1 Batch Validation — {args.model_update_type}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'batch_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[done] → {args.output_dir}')


if __name__ == '__main__':
    main()
