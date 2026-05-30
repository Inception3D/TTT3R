"""
B2 Memory Gate Ablation: Spectral-Change-Gated Memory Update
=============================================================

Compares depth error for four update types:
  cut3r          — no state gate, no memory gate (baseline)
  ttt3r          — TTT3R attention gate on state, no memory gate
  cut3r_memgate  — no state gate, spectral_change memory gate
  ttt3r_memgate  — TTT3R state gate + spectral_change memory gate

Also sweeps memory gate hyper-parameters (tau, skip_ratio) for cut3r_memgate.

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/memgate_ablation.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --scannet_root /mnt/sda/szy/research/dataset/scannet_seq \
    --tum_root /mnt/sda/szy/research/dataset/tum_dynamics \
    --output_dir analysis_results/memgate_ablation \
    --num_scannet 10 --seed 42
"""

import os
import sys
import argparse
import glob
import warnings

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
    load_tum_associations, build_tum_timestamp_index, find_gt_depth_path,
)


def parse_args():
    p = argparse.ArgumentParser(description="B2 Memory Gate Ablation")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--scannet_root", type=str, default="")
    p.add_argument("--tum_root", type=str, default="")
    p.add_argument("--output_dir", type=str, default="analysis_results/memgate_ablation")
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
    """Run inference and return mean abs_rel depth error."""
    img_paths = load_img_paths(rgb_dir, frame_interval, max_frames)
    if len(img_paths) < 30:
        return None

    views = build_views(img_paths, size)

    scene_dir = os.path.dirname(os.path.normpath(rgb_dir))
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
    return {"mean_error": np.mean(errors), "n_valid": len(errors)}


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
    print(f"[data] {len(scenes)} scenes "
          f"({sum(1 for s in scenes if s[5]=='scannet')} ScanNet, "
          f"{sum(1 for s in scenes if s[5]=='tum')} TUM)")

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    # Configurations: (label, update_type, extra_config_params)
    configs = [
        # Baselines
        ("cut3r",              "cut3r",         {}),
        ("ttt3r",              "ttt3r",         {}),
        # B2: memory gate only (cut3r state update)
        ("cut3r_mg_t2_sr0.5",  "cut3r_memgate", {"mem_gate_tau": 2.0, "mem_gate_skip_ratio": 0.5}),
        ("cut3r_mg_t3_sr0.5",  "cut3r_memgate", {"mem_gate_tau": 3.0, "mem_gate_skip_ratio": 0.5}),
        ("cut3r_mg_t3_sr0.3",  "cut3r_memgate", {"mem_gate_tau": 3.0, "mem_gate_skip_ratio": 0.3}),
        ("cut3r_mg_t5_sr0.5",  "cut3r_memgate", {"mem_gate_tau": 5.0, "mem_gate_skip_ratio": 0.5}),
        # B2 + TTT3R state gate
        ("ttt3r_mg_t3_sr0.5",  "ttt3r_memgate", {"mem_gate_tau": 3.0, "mem_gate_skip_ratio": 0.5}),
    ]

    all_results = {}

    for config_name, update_type, extra_params in configs:
        print(f"\n{'='*60}")
        print(f"[config] {config_name}  (update_type={update_type}, {extra_params})")
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

        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    lines = ["=" * 80, "B2 Memory Gate Ablation Summary", "=" * 80, ""]

    config_stats = {}
    for config_name, scene_results in all_results.items():
        if not scene_results:
            continue
        errors_all = [r["mean_error"] for r in scene_results.values()]
        errors_sn  = [r["mean_error"] for r in scene_results.values() if r.get("dataset") == "scannet"]
        errors_tum = [r["mean_error"] for r in scene_results.values() if r.get("dataset") == "tum"]

        config_stats[config_name] = {
            "all":     (np.mean(errors_all), np.std(errors_all), len(errors_all)),
            "scannet": (np.mean(errors_sn),  np.std(errors_sn),  len(errors_sn))  if errors_sn  else (np.nan, np.nan, 0),
            "tum":     (np.mean(errors_tum), np.std(errors_tum), len(errors_tum)) if errors_tum else (np.nan, np.nan, 0),
        }

    # Compute relative change vs cut3r baseline
    baseline_all = config_stats.get("cut3r", {}).get("all", (np.nan,))[0]
    baseline_sn  = config_stats.get("cut3r", {}).get("scannet", (np.nan,))[0]
    baseline_tum = config_stats.get("cut3r", {}).get("tum", (np.nan,))[0]

    lines.append(f"{'Config':<25s} {'ALL err':>12s} {'ScanNet':>12s} {'TUM':>12s} {'Δ ALL%':>8s}")
    lines.append("-" * 72)
    for cfg_name, _, _ in configs:
        if cfg_name not in config_stats:
            continue
        s = config_stats[cfg_name]
        delta_pct = (s["all"][0] - baseline_all) / (baseline_all + 1e-9) * 100
        lines.append(
            f"{cfg_name:<25s} "
            f"{s['all'][0]:>6.4f}±{s['all'][1]:.4f} "
            f"{s['scannet'][0]:>6.4f}±{s['scannet'][1]:.4f} "
            f"{s['tum'][0]:>6.4f}±{s['tum'][1]:.4f} "
            f"{delta_pct:>+7.2f}%"
        )
    lines.append("")

    # Per-scene detail
    lines.append("--- Per-scene errors ---")
    hdr = f"{'Scene':<45s} " + " ".join(f"{c[0][:12]:>12s}" for c in configs)
    lines.append(hdr)
    lines.append("-" * 90)
    for scene_name, *_ in scenes:
        row = f"{scene_name:<45s} "
        for cfg_name, _, _ in configs:
            if cfg_name in all_results and scene_name in all_results[cfg_name]:
                err = all_results[cfg_name][scene_name]["mean_error"]
                row += f"{err:>12.4f} "
            else:
                row += f"{'N/A':>12s} "
        lines.append(row)

    lines.append("=" * 80)
    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = os.path.join(args.output_dir, "memgate_ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # ── Plot ──
    valid_configs = [c[0] for c in configs if c[0] in config_stats]
    x = np.arange(len(valid_configs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (ds_label, ds_key, color) in enumerate([
        ("ScanNet", "scannet", "C0"), ("TUM", "tum", "C1"), ("ALL", "all", "C2")
    ]):
        means = [config_stats[c][ds_key][0] for c in valid_configs]
        stds  = [config_stats[c][ds_key][1] for c in valid_configs]
        ax.bar(x + i * width, means, width, yerr=stds, label=ds_label,
               color=color, alpha=0.7, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(valid_configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Abs Rel Depth Error")
    ax.set_title("B2 Memory Gate Ablation: Depth Error")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "memgate_ablation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] Outputs → {args.output_dir}")
    print(f"       Summary  → {summary_path}")


if __name__ == "__main__":
    main()
