"""
Spectral Ablation: Compare baseline vs spectral-modulated state updates
=========================================================================

Evaluates depth error across update types:
  cut3r, ttt3r, cut3r_spectral, ttt3r_spectral

with hyperparameter sweep over spectral temperature τ.

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/spectral_ablation.py \
    --model_path /path/to/model.pth \
    --scannet_root /path/to/scannetv2 \
    --tum_root /path/to/tum \
    --output_dir analysis_results/spectral_ablation \
    --num_scannet 10 --seed 42
"""

import os
import sys
import argparse
import glob
import warnings

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

from analysis.spectral_analysis import (
    load_img_paths, build_views, load_gt_depth, compute_frame_depth_error,
    derive_depth_dir, load_tum_associations, build_tum_timestamp_index,
    find_gt_depth_path,
)


def parse_args():
    p = argparse.ArgumentParser(description="Spectral Ablation")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--scannet_root", type=str, default="")
    p.add_argument("--tum_root", type=str, default="")
    p.add_argument("--output_dir", type=str, default="analysis_results/spectral_ablation")
    p.add_argument("--num_scannet", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def discover_scenes(args):
    """Return list of (name, rgb_dir, depth_dir, depth_scale, frame_interval, dataset)."""
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


def eval_scene(model, rgb_dir, depth_dir, depth_scale, frame_interval,
               max_frames, size, max_depth, device, dataset):
    """Run inference and compute mean abs_rel depth error."""
    img_paths = load_img_paths(rgb_dir, frame_interval, max_frames)
    if len(img_paths) < 30:
        return None

    views = build_views(img_paths, size)

    # Depth matching setup
    scene_dir = os.path.dirname(os.path.normpath(rgb_dir))
    tum_assoc = load_tum_associations(scene_dir) if dataset == "tum" else None
    tum_depth_index = build_tum_timestamp_index(depth_dir) if dataset == "tum" else None

    with torch.no_grad():
        ress, _ = model.forward_recurrent_lighter(views, device=device)

    errors = []
    confs = []
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

        if "conf_self" in ress[t]:
            confs.append(ress[t]["conf_self"][0].numpy().mean())

    if not errors:
        return None

    result = {"mean_error": np.mean(errors), "n_valid": len(errors)}
    if confs:
        # Confidence-error correlation
        min_len = min(len(errors), len(confs))
        if min_len > 10:
            r, _ = pearsonr(confs[:min_len], errors[:min_len])
            result["conf_error_r"] = r
    return result


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
    print(f"[data] {len(scenes)} scenes ({sum(1 for s in scenes if s[5]=='scannet')} ScanNet, "
          f"{sum(1 for s in scenes if s[5]=='tum')} TUM)")

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    # Configurations to evaluate
    configs = [
        ("cut3r", {}),
        ("ttt3r", {}),
        ("cut3r_spectral_t1", {"spectral_temperature": 1.0}),
        ("cut3r_spectral_t2", {"spectral_temperature": 2.0}),
        ("cut3r_spectral_t4", {"spectral_temperature": 4.0}),
        ("ttt3r_spectral_t1", {"spectral_temperature": 1.0}),
        ("ttt3r_spectral_t2", {"spectral_temperature": 2.0}),
        ("ttt3r_spectral_t4", {"spectral_temperature": 4.0}),
    ]

    all_results = {}

    for config_name, extra_params in configs:
        # Determine base update type
        if config_name.startswith("cut3r_spectral"):
            update_type = "cut3r_spectral"
        elif config_name.startswith("ttt3r_spectral"):
            update_type = "ttt3r_spectral"
        else:
            update_type = config_name

        print(f"\n{'='*60}")
        print(f"[config] {config_name} (update_type={update_type})")
        print(f"{'='*60}")

        model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
        model.config.model_update_type = update_type
        for k, v in extra_params.items():
            setattr(model.config, k, v)
        model.eval()

        scene_results = {}
        for scene_name, rgb_dir, depth_dir, ds, fi, dataset in tqdm(scenes, desc=config_name):
            try:
                r = eval_scene(model, rgb_dir, depth_dir, ds, fi,
                               args.max_frames, args.size, args.max_depth,
                               device, dataset)
            except Exception as e:
                print(f"  [warn] {scene_name}: {e}")
                continue
            if r is not None:
                r["dataset"] = dataset
                scene_results[scene_name] = r

        all_results[config_name] = scene_results

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    lines = ["=" * 80, "Spectral Ablation Summary", "=" * 80, ""]

    # Per-config aggregate
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

    # Table
    lines.append(f"{'Config':<25s} {'ALL err':>10s} {'ScanNet err':>12s} {'TUM err':>12s}")
    lines.append("-" * 60)
    for config_name in [c[0] for c in configs]:
        if config_name not in config_stats:
            continue
        s = config_stats[config_name]
        lines.append(
            f"{config_name:<25s} "
            f"{s['all'][0]:>6.4f}±{s['all'][1]:.4f} "
            f"{s['scannet'][0]:>6.4f}±{s['scannet'][1]:.4f} "
            f"{s['tum'][0]:>6.4f}±{s['tum'][1]:.4f}"
        )
    lines.append("")

    # Per-scene detail
    lines.append("--- Per-scene errors ---")
    scene_names = list(scenes)
    hdr = f"{'Scene':<45s} " + " ".join(f"{c[0][:10]:>10s}" for c in configs)
    lines.append(hdr)
    lines.append("-" * 80)
    for scene_name, _, _, _, _, _ in scenes:
        row = f"{scene_name:<45s} "
        for config_name, _ in configs:
            if config_name in all_results and scene_name in all_results[config_name]:
                err = all_results[config_name][scene_name]["mean_error"]
                row += f"{err:>10.4f} "
            else:
                row += f"{'N/A':>10s} "
        lines.append(row)

    lines.append("=" * 80)
    summary = "\n".join(lines)
    print("\n" + summary)

    with open(os.path.join(args.output_dir, "ablation_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 5))
    config_names = [c[0] for c in configs if c[0] in config_stats]
    x = np.arange(len(config_names))
    width = 0.25

    for i, (ds_label, ds_key, color) in enumerate([
        ("ScanNet", "scannet", "C0"), ("TUM", "tum", "C1"), ("ALL", "all", "C2")
    ]):
        means = [config_stats[c][ds_key][0] for c in config_names]
        stds = [config_stats[c][ds_key][1] for c in config_names]
        ax.bar(x + i * width, means, width, yerr=stds, label=ds_label,
               color=color, alpha=0.7, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Abs Rel Depth Error")
    ax.set_title("Spectral Ablation: Depth Error by Update Type")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ablation_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] All outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
