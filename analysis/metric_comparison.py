"""
Frame Change Metric Comparison
================================
Compares four inter-frame change metrics against state oscillation correlation
and depth error when used as frame filters.

Metrics:
  spectral_change  : low-frequency energy of frame difference (ours)
  l2_change        : L2 norm of full frame difference (naive baseline)
  high_freq_change : high-frequency energy of frame difference
  mid_freq_change  : mid-frequency energy of frame difference

For each metric we report:
  1. Pearson r with state oscillation (motivation evidence)
  2. Depth error (full) vs depth error (filtered) on the same kept frames

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/metric_comparison.py \
    --model_path /path/to/model.pth \
    --scannet_root /path/to/scannetv2 \
    --tum_root /path/to/tum \
    --output_dir analysis_results/metric_comparison \
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


# ── frame change metrics ──────────────────────────────────────────────────────

def compute_all_metrics(img_prev, img_curr):
    """
    Compute four frame-change metrics for img_curr relative to img_prev.

    Args:
        img_prev, img_curr: [B, C, H, W] float tensors
    Returns:
        dict with keys: spectral_change, l2_change, high_freq_change, mid_freq_change
    """
    diff = img_curr - img_prev                    # [B, C, H, W]
    diff_mean = diff.mean(dim=(0, 1))             # [H, W]

    # L2 baseline
    l2 = diff_mean.pow(2).mean().item()

    # FFT
    F = torch.fft.fft2(diff_mean)
    power = F.abs() ** 2                          # [H, W]
    H, W = power.shape

    # Frequency band boundaries
    h_lo = max(1, H // 8)   # low:  0 .. H/8
    h_mi = max(1, H // 4)   # mid:  H/8 .. H/4
    w_lo = max(1, W // 8)
    w_mi = max(1, W // 4)

    def corner_sum(p, hr, wr):
        return (p[:hr, :wr].sum() + p[:hr, -wr:].sum() +
                p[-hr:, :wr].sum() + p[-hr:, -wr:].sum()).item()

    low_e  = corner_sum(power, h_lo, w_lo)
    mid_e  = corner_sum(power, h_mi, w_mi) - low_e
    high_e = (power.sum() - corner_sum(power, h_mi, w_mi)).item()

    return {
        'spectral_change': low_e,
        'l2_change':       l2,
        'high_freq_change': high_e,
        'mid_freq_change':  mid_e,
    }


def adaptive_filter(scores, skip_ratio=0.3, warmup=10):
    """Return kept indices using EMA-based adaptive threshold."""
    running_mean = None
    gamma = 0.95
    kept = [0]   # always keep first
    img_prev_idx = 0

    for i in range(1, len(scores)):
        e = scores[i]
        if running_mean is None:
            running_mean = e
        else:
            running_mean = gamma * running_mean + (1 - gamma) * e

        is_informative = (i < warmup) or (e >= skip_ratio * running_mean)
        if is_informative:
            kept.append(i)
    return kept


# ── data helpers ─────────────────────────────────────────────────────────────

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
    rgb_entries   = parse_txt(rgb_txt)
    depth_entries = parse_txt(depth_txt)
    depth_ts = sorted(depth_entries.keys())
    assoc = {}
    for rts, rpath in rgb_entries.items():
        idx = np.searchsorted(depth_ts, rts)
        idx = min(max(idx, 0), len(depth_ts) - 1)
        if abs(depth_ts[idx] - rts) < 0.02:
            dpath = os.path.join(scene_dir, depth_entries[depth_ts[idx]])
            assoc[os.path.basename(rpath)] = dpath
    return assoc


def match_depth_path(img_path, depth_dir, dataset='scannet', tum_assoc=None):
    if dataset == 'tum' and tum_assoc:
        return tum_assoc.get(os.path.basename(img_path), None)
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


# ── per-scene eval ────────────────────────────────────────────────────────────

METRIC_NAMES = ['spectral_change', 'l2_change', 'high_freq_change', 'mid_freq_change']


def eval_scene(model, img_paths, depth_dir, depth_scale, size,
               max_depth, skip_ratio, warmup, device, dataset, scene_dir=''):

    if len(img_paths) < 30:
        return None

    all_views = build_views(img_paths, size)
    tum_assoc = load_tum_associations(scene_dir) if dataset == 'tum' else {}

    # Compute all metrics per frame
    imgs_tensor = [v['img'].float() for v in all_views]
    metric_scores = {m: [0.0] for m in METRIC_NAMES}
    for i in range(1, len(imgs_tensor)):
        m = compute_all_metrics(imgs_tensor[i-1], imgs_tensor[i])
        for k in METRIC_NAMES:
            metric_scores[k].append(m[k])
    metric_arrs = {k: np.array(v) for k, v in metric_scores.items()}

    # Full sequence inference (run once)
    with torch.no_grad():
        ress_full, ana_full = model.forward_recurrent_analysis(all_views, device=device)
    osc_full = compute_state_oscillation(ana_full['state_history'])

    # Correlations: each metric vs state oscillation
    osc_for_corr = osc_full  # [T-1]
    r_values = {}
    for metric_name in METRIC_NAMES:
        sc = metric_arrs[metric_name][1:len(osc_full)+1]
        valid = np.isfinite(sc) & np.isfinite(osc_for_corr) & (sc > 0)
        if valid.sum() > 10:
            r_values[metric_name] = pearsonr(sc[valid], osc_for_corr[valid])[0]
        else:
            r_values[metric_name] = np.nan

    # For each metric: filter → infer → depth error on kept frames
    depth_results = {}
    for metric_name in METRIC_NAMES:
        kept_indices = adaptive_filter(
            metric_arrs[metric_name].tolist(), skip_ratio=skip_ratio, warmup=warmup)
        kept_views = [all_views[i] for i in kept_indices]
        skip_rate  = 1.0 - len(kept_views) / len(all_views)

        with torch.no_grad():
            ress_filt, _ = model.forward_recurrent_analysis(kept_views, device=device)

        errors_full, errors_filt = [], []
        if depth_dir and os.path.isdir(depth_dir):
            for t, orig_idx in enumerate(kept_indices):
                ipath = img_paths[orig_idx]
                dpath = match_depth_path(ipath, depth_dir, dataset, tum_assoc)
                if dpath is None:
                    continue
                gt = load_gt_depth(dpath, depth_scale)
                if gt is None:
                    continue
                pred_full = ress_full[orig_idx]['pts3d_in_self_view'][0, :, :, 2].numpy()
                pred_filt = ress_filt[t]['pts3d_in_self_view'][0, :, :, 2].numpy()
                ef = compute_depth_error(pred_full, gt, max_depth)
                ef2 = compute_depth_error(pred_filt, gt, max_depth)
                if not np.isnan(ef):
                    errors_full.append(ef)
                if not np.isnan(ef2):
                    errors_filt.append(ef2)

        depth_results[metric_name] = {
            'skip_rate': skip_rate,
            'err_full':  np.nanmean(errors_full) if errors_full else np.nan,
            'err_filt':  np.nanmean(errors_filt) if errors_filt else np.nan,
        }

    return {'r_values': r_values, 'depth_results': depth_results}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',    type=str, required=True)
    p.add_argument('--scannet_root',  type=str, default='')
    p.add_argument('--tum_root',      type=str, default='')
    p.add_argument('--output_dir',    type=str, default='analysis_results/metric_comparison')
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
            scenes.append({'name': os.path.basename(sd),
                           'rgb_dir': os.path.join(sd, 'color'),
                           'depth_dir': os.path.join(sd, 'depth'),
                           'scene_dir': sd, 'depth_scale': 1000.0,
                           'frame_interval': args.frame_interval,
                           'dataset': 'scannet'})
    if args.tum_root and os.path.isdir(args.tum_root):
        for td in sorted(glob.glob(os.path.join(args.tum_root, 'rgbd_dataset_*'))):
            if os.path.isdir(os.path.join(td, 'rgb')):
                scenes.append({'name': os.path.basename(td),
                               'rgb_dir': os.path.join(td, 'rgb'),
                               'depth_dir': os.path.join(td, 'depth'),
                               'scene_dir': td, 'depth_scale': 5000.0,
                               'frame_interval': args.frame_interval,
                               'dataset': 'tum'})
    return scenes


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    scenes = discover_scenes(args)
    print(f"[data] {len(scenes)} scenes")

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    all_r      = {m: [] for m in METRIC_NAMES}  # per scene
    all_r_sn   = {m: [] for m in METRIC_NAMES}
    all_r_tum  = {m: [] for m in METRIC_NAMES}
    all_err_chg     = {m: [] for m in METRIC_NAMES}
    all_err_chg_sn  = {m: [] for m in METRIC_NAMES}
    all_err_chg_tum = {m: [] for m in METRIC_NAMES}

    for scene in tqdm(scenes, desc='scenes'):
        img_paths = load_img_paths(
            scene['rgb_dir'], scene['frame_interval'], args.max_frames)
        try:
            res = eval_scene(model, img_paths,
                             scene['depth_dir'], scene['depth_scale'],
                             args.size, args.max_depth,
                             args.skip_ratio, args.warmup,
                             device, scene['dataset'],
                             scene_dir=scene.get('scene_dir', ''))
        except Exception as e:
            print(f"  [warn] {scene['name']}: {e}")
            continue
        if res is None:
            continue

        ds = scene['dataset']
        for m in METRIC_NAMES:
            r = res['r_values'][m]
            dr = res['depth_results'][m]
            chg = 100 * (dr['err_filt'] - dr['err_full']) / (dr['err_full'] + 1e-8)

            if not np.isnan(r):
                all_r[m].append(r)
                (all_r_sn if ds == 'scannet' else all_r_tum)[m].append(r)
            if not np.isnan(chg):
                all_err_chg[m].append(chg)
                (all_err_chg_sn if ds == 'scannet' else all_err_chg_tum)[m].append(chg)

        # per-scene print
        r_line = '  '.join(f"{m[:8]}={res['r_values'][m]:+.3f}" for m in METRIC_NAMES)
        print(f"  {scene['name'][:30]:<30s} {r_line}")

    # ── Summary ──────────────────────────────────────────────────────────────
    lines = ['=' * 72, 'METRIC COMPARISON SUMMARY', '=' * 72, '']
    lines.append(f"{'Metric':<20s} {'r (ALL)':>10s} {'r (SN)':>10s} {'r (TUM)':>10s} "
                 f"{'err_chg (SN)':>14s} {'err_chg (TUM)':>14s}")
    lines.append('-' * 80)

    for m in METRIC_NAMES:
        r_all = f"{np.mean(all_r[m]):+.3f}±{np.std(all_r[m]):.3f}" if all_r[m] else 'N/A'
        r_sn  = f"{np.mean(all_r_sn[m]):+.3f}±{np.std(all_r_sn[m]):.3f}" if all_r_sn[m] else 'N/A'
        r_tum = f"{np.mean(all_r_tum[m]):+.3f}±{np.std(all_r_tum[m]):.3f}" if all_r_tum[m] else 'N/A'
        ec_sn  = f"{np.mean(all_err_chg_sn[m]):+.1f}%" if all_err_chg_sn[m] else 'N/A'
        ec_tum = f"{np.mean(all_err_chg_tum[m]):+.1f}%" if all_err_chg_tum[m] else 'N/A'
        lines.append(f"{m:<20s} {r_all:>10s} {r_sn:>10s} {r_tum:>10s} {ec_sn:>14s} {ec_tum:>14s}")

    lines.append('=' * 72)
    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(os.path.join(args.output_dir, 'metric_comparison.txt'), 'w') as f:
        f.write(summary + '\n')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(METRIC_NAMES))
    labels = ['spectral\n(low-freq)', 'L2\n(baseline)', 'high-freq', 'mid-freq']

    # Panel 1: correlation comparison
    ax = axes[0]
    for ds_label, r_dict, color in [
        ('ScanNet', all_r_sn, 'C0'), ('TUM', all_r_tum, 'C1')]:
        means = [np.mean(r_dict[m]) if r_dict[m] else 0 for m in METRIC_NAMES]
        stds  = [np.std(r_dict[m])  if r_dict[m] else 0 for m in METRIC_NAMES]
        offset = -0.15 if ds_label == 'ScanNet' else 0.15
        ax.bar(x + offset, means, 0.3, yerr=stds, label=ds_label,
               color=color, alpha=0.8, capsize=4)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Pearson r with state oscillation')
    ax.set_title('Correlation: Frame Metric ↔ State Oscillation')
    ax.legend()

    # Panel 2: depth error change
    ax = axes[1]
    for ds_label, ec_dict, color in [
        ('ScanNet', all_err_chg_sn, 'C0'), ('TUM', all_err_chg_tum, 'C1')]:
        means = [np.mean(ec_dict[m]) if ec_dict[m] else 0 for m in METRIC_NAMES]
        stds  = [np.std(ec_dict[m])  if ec_dict[m] else 0 for m in METRIC_NAMES]
        offset = -0.15 if ds_label == 'ScanNet' else 0.15
        ax.bar(x + offset, means, 0.3, yerr=stds, label=ds_label,
               color=color, alpha=0.8, capsize=4)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Depth error change (%)')
    ax.set_title('Depth Error: Filtered vs Full (lower = better)')
    ax.legend()

    plt.suptitle(f'Frame Metric Comparison — {args.model_update_type}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metric_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[done] → {args.output_dir}')


if __name__ == '__main__':
    main()
