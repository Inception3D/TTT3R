"""
Three-Layer Joint Ablation (Parallel)
=====================================

Evaluates all layer combinations to measure individual and joint contributions.
Optimized for GPU utilization:
  - Pre-caches views (raw + filtered) to avoid redundant I/O
  - Runs multiple configs in parallel on the same GPU via multiprocessing
  - Each worker loads one model copy and processes its assigned configs

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/joint_ablation.py \
    --model_path model/cut3r_512_dpt_4_64.pth \
    --scannet_root /mnt/sda/szy/research/dataset/scannetv2 \
    --tum_root /mnt/sda/szy/research/dataset/tum \
    --output_dir analysis_results/joint_ablation \
    --num_scannet 10 --seed 42 --n_workers 4
"""

import os
import sys
import argparse
import glob
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r

from analysis.spectral_analysis import (
    load_img_paths, build_views, load_gt_depth, compute_frame_depth_error,
    load_tum_associations, build_tum_timestamp_index, find_gt_depth_path,
)


# Best hyperparams from individual layer experiments
BEST_SPECTRAL_TAU = 1.0       # Layer 2: τ insensitive, use 1
BEST_GEO_TAU = 2.0            # Layer 3: τ=2
BEST_GEO_CUTOFF = 4           # Layer 3: cutoff=4 (25% bandwidth)
BEST_SKIP_RATIO = 0.3         # Layer 1: skip_ratio=0.3

# (label, update_type, extra_config, use_frame_filter)
CONFIGS = [
    # Baselines
    ("cut3r",        "cut3r",         {}, False),
    ("ttt3r",        "ttt3r",         {}, False),
    # Layer 1 only
    ("L1+cut3r",     "cut3r",         {}, True),
    ("L1+ttt3r",     "ttt3r",         {}, True),
    # Layer 2 only
    ("L2+cut3r",     "cut3r_spectral",
     {"spectral_temperature": BEST_SPECTRAL_TAU}, False),
    ("L2+ttt3r",     "ttt3r_spectral",
     {"spectral_temperature": BEST_SPECTRAL_TAU}, False),
    # Layer 3 only
    ("L3+cut3r",     "cut3r_geogate",
     {"geo_gate_tau": BEST_GEO_TAU, "geo_gate_freq_cutoff": BEST_GEO_CUTOFF}, False),
    ("L3+ttt3r",     "ttt3r_geogate",
     {"geo_gate_tau": BEST_GEO_TAU, "geo_gate_freq_cutoff": BEST_GEO_CUTOFF}, False),
    # Layer 2+3
    ("L23+cut3r",    "cut3r_joint",
     {"spectral_temperature": BEST_SPECTRAL_TAU,
      "geo_gate_tau": BEST_GEO_TAU, "geo_gate_freq_cutoff": BEST_GEO_CUTOFF}, False),
    ("L23+ttt3r",    "ttt3r_joint",
     {"spectral_temperature": BEST_SPECTRAL_TAU,
      "geo_gate_tau": BEST_GEO_TAU, "geo_gate_freq_cutoff": BEST_GEO_CUTOFF}, False),
    # Layer 1+2+3 (full framework)
    ("L123+cut3r",   "cut3r_joint",
     {"spectral_temperature": BEST_SPECTRAL_TAU,
      "geo_gate_tau": BEST_GEO_TAU, "geo_gate_freq_cutoff": BEST_GEO_CUTOFF}, True),
    ("L123+ttt3r",   "ttt3r_joint",
     {"spectral_temperature": BEST_SPECTRAL_TAU,
      "geo_gate_tau": BEST_GEO_TAU, "geo_gate_freq_cutoff": BEST_GEO_CUTOFF}, True),
]


def parse_args():
    p = argparse.ArgumentParser(description="Three-Layer Joint Ablation (Parallel)")
    p.add_argument("--model_path", type=str, default="model/cut3r_512_dpt_4_64.pth")
    p.add_argument("--scannet_root", type=str, default="")
    p.add_argument("--tum_root", type=str, default="")
    p.add_argument("--output_dir", type=str, default="analysis_results/joint_ablation")
    p.add_argument("--num_scannet", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_workers", type=int, default=4,
                   help="Number of parallel workers (each loads one model copy, ~7.5G VRAM)")
    return p.parse_args()


def discover_scenes(args):
    scenes = []
    rng = np.random.RandomState(args.seed)

    if args.scannet_root and os.path.isdir(args.scannet_root):
        scene_dirs = sorted(glob.glob(os.path.join(args.scannet_root, "scene*")))
        valid = [(sd, os.path.join(sd, "color"), os.path.join(sd, "depth"))
                 for sd in scene_dirs
                 if os.path.isdir(os.path.join(sd, "color"))
                 and os.path.isdir(os.path.join(sd, "depth"))]
        if len(valid) > args.num_scannet:
            idx = rng.choice(len(valid), args.num_scannet, replace=False)
            valid = [valid[i] for i in sorted(idx)]
        for sd, cd, dd in valid:
            scenes.append((os.path.basename(sd), cd, dd, 1000.0, 5, "scannet"))

    if args.tum_root and os.path.isdir(args.tum_root):
        tum_dirs = sorted(glob.glob(os.path.join(args.tum_root, "rgbd_dataset_*")))
        valid = [(td, os.path.join(td, "rgb"), os.path.join(td, "depth"))
                 for td in tum_dirs
                 if os.path.isdir(os.path.join(td, "rgb"))
                 and os.path.isdir(os.path.join(td, "depth"))]
        for td, rd, dd in valid:
            scenes.append((os.path.basename(td), rd, dd, 5000.0, 3, "tum"))

    return scenes


def eval_scene_from_views(model, views, img_paths, depth_dir, depth_scale,
                          max_depth, device, dataset):
    """Run inference on pre-built views and compute mean abs_rel depth error."""
    scene_dir = os.path.dirname(os.path.normpath(os.path.dirname(img_paths[0])))
    tum_assoc = load_tum_associations(scene_dir) if dataset == "tum" else None
    tum_depth_index = build_tum_timestamp_index(depth_dir) if dataset == "tum" else None

    with torch.no_grad():
        ress, _ = model.forward_recurrent_lighter(views, device=device)

    errors = []
    for t in range(len(ress)):
        pts3d = ress[t]["pts3d_in_self_view"]
        pred_depth = pts3d[0, :, :, 2].numpy()
        depth_path = find_gt_depth_path(img_paths[t], depth_dir,
                                        tum_assoc, tum_depth_index)
        if depth_path is not None:
            gt = load_gt_depth(depth_path, depth_scale)
            if gt is not None:
                err = compute_frame_depth_error(pred_depth, gt, max_depth)
                if not np.isnan(err):
                    errors.append(err)

    if not errors:
        return None
    return {"mean_error": np.mean(errors), "n_valid": len(errors), "n_frames": len(img_paths)}


def worker_fn(worker_id, config_indices, scenes, scene_cache, args, result_dict):
    """Worker process: loads one model, runs assigned configs sequentially."""
    device = args.device
    from dust3r.model import ARCroco3DStereo

    # Load model once for this worker
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    for ci in config_indices:
        config_name, update_type, extra_params, use_filter = CONFIGS[ci]
        print(f"  [worker {worker_id}] {config_name} (update={update_type}, filter={use_filter})")

        # Reconfigure model (no reload needed — same weights)
        model.config.model_update_type = update_type
        # Reset extra params to defaults first
        model.config.spectral_temperature = 2.0
        model.config.geo_gate_tau = 3.0
        model.config.geo_gate_freq_cutoff = 4
        for k, v in extra_params.items():
            setattr(model.config, k, v)

        scene_results = {}
        for si, (scene_name, rgb_dir, depth_dir, ds, fi, dataset) in enumerate(scenes):
            try:
                # Use pre-cached views
                cache_key = f"{si}_filter" if use_filter else f"{si}_raw"
                views, img_paths = scene_cache[cache_key]

                r = eval_scene_from_views(
                    model, views, img_paths, depth_dir, ds,
                    args.max_depth, device, dataset)
            except Exception as e:
                print(f"  [worker {worker_id}] [warn] {scene_name}: {e}")
                continue
            if r is not None:
                r["dataset"] = dataset
                scene_results[scene_name] = r

        result_dict[config_name] = scene_results
        print(f"  [worker {worker_id}] {config_name} done "
              f"({len(scene_results)} scenes, "
              f"mean_err={np.mean([r['mean_error'] for r in scene_results.values()]):.4f})")

    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    scenes = discover_scenes(args)
    if not scenes:
        print("[error] No scenes found.")
        return
    n_sn = sum(1 for s in scenes if s[5] == 'scannet')
    n_tum = sum(1 for s in scenes if s[5] == 'tum')
    print(f"[data] {len(scenes)} scenes ({n_sn} ScanNet, {n_tum} TUM)")

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    # ── Phase 1: Pre-cache views ──
    print(f"\n[phase 1] Pre-caching views for {len(scenes)} scenes...")
    t0 = time.time()

    # Determine which scenes need filtered views
    needs_filter = any(c[3] for c in CONFIGS)

    scene_cache = {}  # key: "{scene_idx}_raw" or "{scene_idx}_filter"
    for si, (scene_name, rgb_dir, depth_dir, ds, fi, dataset) in enumerate(scenes):
        img_paths = load_img_paths(rgb_dir, fi, args.max_frames)
        if len(img_paths) < 30:
            continue

        views = build_views(img_paths, args.size)
        scene_cache[f"{si}_raw"] = (views, img_paths)

        if needs_filter:
            filt_views, kept_indices, _ = ARCroco3DStereo.filter_views_by_spectral_change(
                views, skip_ratio=BEST_SKIP_RATIO, warmup=10, device=device)
            filt_img_paths = [img_paths[i] for i in kept_indices]
            scene_cache[f"{si}_filter"] = (filt_views, filt_img_paths)

        print(f"  [{si+1}/{len(scenes)}] {scene_name}: "
              f"{len(img_paths)} frames"
              + (f", filtered → {len(filt_img_paths)}" if needs_filter else ""))

    print(f"[phase 1] Done in {time.time()-t0:.1f}s")

    # ── Phase 2: Parallel config evaluation ──
    n_workers = min(args.n_workers, len(CONFIGS))
    print(f"\n[phase 2] Running {len(CONFIGS)} configs with {n_workers} workers...")
    t0 = time.time()

    # Distribute configs across workers (round-robin)
    worker_configs = [[] for _ in range(n_workers)]
    for i, _ in enumerate(CONFIGS):
        worker_configs[i % n_workers].append(i)

    for wi, wc in enumerate(worker_configs):
        labels = [CONFIGS[ci][0] for ci in wc]
        print(f"  worker {wi}: {labels}")

    # Use mp.Manager for shared result dict
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for wi in range(n_workers):
        p = mp.Process(
            target=worker_fn,
            args=(wi, worker_configs[wi], scenes, scene_cache, args, result_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_results = dict(result_dict)
    print(f"\n[phase 2] Done in {time.time()-t0:.1f}s")

    # ── Summary ──
    lines = ["=" * 80, "Three-Layer Joint Ablation Summary", "=" * 80, ""]

    config_stats = {}
    for config_name, scene_results in all_results.items():
        if not scene_results:
            continue
        errors_all = [r["mean_error"] for r in scene_results.values()]
        errors_sn = [r["mean_error"] for r in scene_results.values() if r["dataset"] == "scannet"]
        errors_tum = [r["mean_error"] for r in scene_results.values() if r["dataset"] == "tum"]

        stats = {
            "all": (np.mean(errors_all), np.std(errors_all), len(errors_all)),
            "scannet": (np.mean(errors_sn), np.std(errors_sn), len(errors_sn)) if errors_sn else (np.nan, np.nan, 0),
            "tum": (np.mean(errors_tum), np.std(errors_tum), len(errors_tum)) if errors_tum else (np.nan, np.nan, 0),
        }
        config_stats[config_name] = stats

    cut3r_all = config_stats.get("cut3r", {}).get("all", (None,))[0]

    lines.append(f"{'Config':<20s} {'ALL err':>12s} {'ScanNet':>12s} {'TUM':>12s} {'vs cut3r':>10s}")
    lines.append("-" * 70)
    for config_name, _, _, _ in CONFIGS:
        if config_name not in config_stats:
            continue
        s = config_stats[config_name]
        pct = ""
        if cut3r_all is not None and s['all'][0] is not None:
            pct = f"{(s['all'][0] / cut3r_all - 1) * 100:+.1f}%"
        lines.append(
            f"{config_name:<20s} "
            f"{s['all'][0]:>6.4f}+{s['all'][1]:.4f} "
            f"{s['scannet'][0]:>6.4f}+{s['scannet'][1]:.4f} "
            f"{s['tum'][0]:>6.4f}+{s['tum'][1]:.4f} "
            f"{pct:>10s}"
        )
    lines.append("")

    # Per-scene detail
    lines.append("--- Per-scene errors ---")
    config_labels = [c[0] for c in CONFIGS]
    hdr = f"{'Scene':<45s} {'ds':>3s} " + " ".join(f"{c[:11]:>11s}" for c in config_labels)
    lines.append(hdr)
    lines.append("-" * (50 + 12 * len(config_labels)))
    for scene_name, _, _, _, _, dataset in scenes:
        row = f"{scene_name:<45s} {dataset[:3]:>3s} "
        for config_name, _, _, _ in CONFIGS:
            if config_name in all_results and scene_name in all_results[config_name]:
                err = all_results[config_name][scene_name]["mean_error"]
                row += f"{err:>11.4f} "
            else:
                row += f"{'N/A':>11s} "
        lines.append(row)

    lines.append("=" * 80)
    summary = "\n".join(lines)
    print("\n" + summary)

    with open(os.path.join(args.output_dir, "joint_ablation_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, base_label in zip(axes, ["cut3r", "ttt3r"]):
        labels = []
        vals = []
        for config_name, _, _, _ in CONFIGS:
            if config_name not in config_stats:
                continue
            if base_label in config_name.lower() or config_name == base_label:
                labels.append(config_name)
                vals.append(config_stats[config_name]['all'][0])

        x = np.arange(len(labels))
        colors = ['C0' if l in ('cut3r', 'ttt3r') else
                  'C2' if l.startswith('L123') else 'C1' for l in labels]
        bars = ax.bar(x, vals, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean Abs Rel Depth Error")
        ax.set_title(f"Layer Contributions ({base_label} base)")

        if vals:
            baseline = vals[0]
            for i, (bar, v) in enumerate(zip(bars, vals)):
                if i > 0:
                    pct_val = (v / baseline - 1) * 100
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{pct_val:+.1f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "joint_ablation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] All outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
