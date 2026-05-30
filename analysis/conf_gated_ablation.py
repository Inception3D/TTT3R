"""
Ablation: Confidence-Gated State Update
========================================

Compares 3 update strategies on multiple ScanNet scenes:
  1. cut3r   – binary update
  2. ttt3r   – soft update via sigmoid(mean(cross_attn))
  3. ttt3r_conf – soft update * confidence gate

Evaluates per-frame depth error and confidence calibration.

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/conf_gated_ablation.py \
    --model_path /path/to/model.pth \
    --dataset_root /path/to/scannetv2 \
    --output_dir analysis_results/ablation_conf \
    --num_scenes 10 --seed 42
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
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Ablation: Confidence-Gated Update")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="analysis_results/ablation_conf")
    p.add_argument("--num_scenes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--frame_interval", type=int, default=5)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--depth_scale", type=float, default=1000.0)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    # Confidence gate hyperparameters
    p.add_argument("--conf_scales", type=str, default="5.0,7.5,10.0,15.0",
                   help="Comma-separated conf_gate_scale values to test.")
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def load_img_paths(color_dir, frame_interval, max_frames):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    all_paths = sorted(glob.glob(os.path.join(color_dir, "*")))
    img_paths = [p for p in all_paths if os.path.splitext(p)[1].lower() in img_exts]
    return img_paths[::frame_interval][:max_frames]


def build_views(img_paths, size):
    from dust3r.utils.image import load_images
    images = load_images(img_paths, size=size)
    views = []
    for i, img_dict in enumerate(images):
        view = {
            "img": img_dict["img"],
            "ray_map": torch.full(
                (img_dict["img"].shape[0], 6,
                 img_dict["img"].shape[-2], img_dict["img"].shape[-1]),
                torch.nan),
            "true_shape": torch.from_numpy(img_dict["true_shape"]),
            "idx": i, "instance": str(i),
            "camera_pose": torch.from_numpy(
                np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def load_gt_depth(depth_path, depth_scale=1000.0):
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32) / depth_scale
    d[d <= 0] = 0
    return d


def compute_frame_depth_error(pred_depth, gt_depth, max_depth):
    H_pred, W_pred = pred_depth.shape
    gt_resized = cv2.resize(gt_depth, (W_pred, H_pred),
                            interpolation=cv2.INTER_NEAREST)
    valid = (gt_resized > 0) & (gt_resized < max_depth) & (pred_depth > 1e-3)
    if valid.sum() < 10:
        return np.nan
    scale = np.median(gt_resized[valid]) / np.median(pred_depth[valid])
    pred_aligned = pred_depth * scale
    abs_rel = np.abs(pred_aligned[valid] - gt_resized[valid]) / gt_resized[valid]
    return abs_rel.mean()


def evaluate_scene(model, color_dir, depth_dir, args):
    """
    Run inference and return per-frame depth errors and confidences.
    Returns None if scene has too few frames.
    """
    img_paths = load_img_paths(color_dir, args.frame_interval, args.max_frames)
    if len(img_paths) < 20:
        return None

    views = build_views(img_paths, args.size)

    with torch.no_grad():
        ress, _ = model.forward_recurrent_analysis(views, device=args.device)

    T = len(ress)
    depth_errors = np.zeros(T)
    confidences = np.zeros(T)

    for t in range(T):
        pts3d = ress[t]["pts3d_in_self_view"]
        pred_depth = pts3d[0, :, :, 2].numpy()

        basename = os.path.splitext(os.path.basename(img_paths[t]))[0]
        depth_path = os.path.join(depth_dir, basename + ".png")

        if os.path.exists(depth_path):
            gt = load_gt_depth(depth_path, args.depth_scale)
            depth_errors[t] = compute_frame_depth_error(
                pred_depth, gt, args.max_depth) if gt is not None else np.nan
        else:
            depth_errors[t] = np.nan

        if "conf_self" in ress[t]:
            confidences[t] = ress[t]["conf_self"][0].numpy().mean()
        elif "conf" in ress[t]:
            confidences[t] = ress[t]["conf"][0].numpy().mean()

    return {
        "depth_errors": depth_errors,
        "confidences": confidences,
        "n_frames": T,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    args.device = device

    conf_scales = [float(x) for x in args.conf_scales.split(",")]

    # ── Find scenes ────────────────────────────────────────────────────────────
    scene_dirs = sorted(glob.glob(os.path.join(args.dataset_root, "scene*")))
    valid_scenes = [sd for sd in scene_dirs
                    if os.path.isdir(os.path.join(sd, "color"))
                    and os.path.isdir(os.path.join(sd, "depth"))]

    rng = np.random.RandomState(args.seed)
    if len(valid_scenes) > args.num_scenes:
        indices = rng.choice(len(valid_scenes), args.num_scenes, replace=False)
        selected = [valid_scenes[i] for i in sorted(indices)]
    else:
        selected = valid_scenes

    scene_names = [os.path.basename(s) for s in selected]
    print(f"[data] {len(selected)} scenes selected")

    # ── Model ──────────────────────────────────────────────────────────────────
    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    print(f"[model] Loading from {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # ── Define strategies ──────────────────────────────────────────────────────
    strategies = [("cut3r", {}), ("ttt3r", {})]
    for cs in conf_scales:
        strategies.append((f"ttt3r_conf(s={cs})", {"conf_gate_scale": cs}))

    # ── Run all strategies ─────────────────────────────────────────────────────
    # results[strategy_name][scene_name] = {depth_errors, confidences, ...}
    all_results = {}

    for strategy_name, extra_config in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        # Set model update type
        if strategy_name == "cut3r":
            model.config.model_update_type = "cut3r"
        elif strategy_name == "ttt3r":
            model.config.model_update_type = "ttt3r"
        else:
            model.config.model_update_type = "ttt3r_conf"
            for k, v in extra_config.items():
                setattr(model.config, k, v)

        strategy_results = {}
        for scene_dir in tqdm(selected, desc=strategy_name):
            scene_name = os.path.basename(scene_dir)
            color_dir = os.path.join(scene_dir, "color")
            depth_dir = os.path.join(scene_dir, "depth")

            try:
                result = evaluate_scene(model, color_dir, depth_dir, args)
            except Exception as e:
                print(f"  [warn] {scene_name}: {e}")
                continue

            if result is not None:
                strategy_results[scene_name] = result

        all_results[strategy_name] = strategy_results

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")

    # Compute per-scene mean error and conf-error correlation for each strategy
    table_rows = []
    strategy_scene_errors = {}  # for paired tests

    for strategy_name, strategy_results in all_results.items():
        scene_errors = []
        scene_confs = []
        conf_corrs = []

        for sn in scene_names:
            if sn not in strategy_results:
                scene_errors.append(np.nan)
                scene_confs.append(np.nan)
                conf_corrs.append(np.nan)
                continue
            data = strategy_results[sn]
            errs = data["depth_errors"]
            confs = data["confidences"]

            scene_errors.append(np.nanmean(errs))
            scene_confs.append(np.nanmean(confs))

            mask = np.isfinite(errs) & np.isfinite(confs)
            if mask.sum() > 10:
                r, _ = pearsonr(confs[mask], errs[mask])
                conf_corrs.append(r)
            else:
                conf_corrs.append(np.nan)

        scene_errors = np.array(scene_errors)
        scene_confs = np.array(scene_confs)
        conf_corrs = np.array(conf_corrs)

        strategy_scene_errors[strategy_name] = scene_errors

        valid_e = scene_errors[np.isfinite(scene_errors)]
        valid_c = conf_corrs[np.isfinite(conf_corrs)]

        table_rows.append({
            "strategy": strategy_name,
            "mean_error": np.mean(valid_e) if len(valid_e) > 0 else np.nan,
            "std_error": np.std(valid_e) if len(valid_e) > 0 else np.nan,
            "mean_conf_corr": np.mean(valid_c) if len(valid_c) > 0 else np.nan,
            "std_conf_corr": np.std(valid_c) if len(valid_c) > 0 else np.nan,
            "scene_errors": scene_errors,
            "conf_corrs": conf_corrs,
        })

    # Print table
    lines = [
        f"{'Strategy':<25s} {'MeanError':>10s} {'±std':>8s} {'ConfCorr':>10s} {'±std':>8s}",
        "-" * 65,
    ]
    for r in table_rows:
        lines.append(
            f"{r['strategy']:<25s} {r['mean_error']:>10.4f} {r['std_error']:>8.4f} "
            f"{r['mean_conf_corr']:>+10.4f} {r['std_conf_corr']:>8.4f}")

    # Improvement over ttt3r baseline
    ttt3r_error = next(r["mean_error"] for r in table_rows if r["strategy"] == "ttt3r")
    lines.append("")
    lines.append("--- Relative to TTT3R baseline ---")
    for r in table_rows:
        if r["strategy"] == "ttt3r":
            continue
        delta = r["mean_error"] - ttt3r_error
        pct = delta / ttt3r_error * 100
        lines.append(f"  {r['strategy']:<25s}: Δerror = {delta:+.4f} ({pct:+.1f}%)")

    summary = "\n".join(lines)
    print(summary)

    # Per-scene table
    per_scene_lines = ["\n--- Per-scene depth errors ---",
                       f"{'Scene':<20s}" + "".join(
                           f" {r['strategy'][:15]:>15s}" for r in table_rows)]
    per_scene_lines.append("-" * (20 + 16 * len(table_rows)))
    for i, sn in enumerate(scene_names):
        row = f"{sn:<20s}"
        for r in table_rows:
            val = r["scene_errors"][i]
            row += f" {val:>15.4f}" if np.isfinite(val) else f" {'N/A':>15s}"
        per_scene_lines.append(row)
    per_scene = "\n".join(per_scene_lines)
    print(per_scene)

    # Save
    full_summary = summary + "\n" + per_scene
    with open(os.path.join(args.output_dir, "ablation_summary.txt"), "w") as f:
        f.write(full_summary + "\n")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Bar chart: mean error per strategy
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["strategy"] for r in table_rows]
    errors = [r["mean_error"] for r in table_rows]
    stds = [r["std_error"] for r in table_rows]
    x = np.arange(len(names))
    bars = ax.bar(x, errors, yerr=stds, capsize=4, color="steelblue", alpha=0.8)

    # Highlight best
    best_idx = np.argmin(errors)
    bars[best_idx].set_color("C2")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean depth error (abs_rel)")
    ax.set_title("Ablation: Update Strategy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "strategy_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Per-scene comparison: grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    n_strategies = len(table_rows)
    bar_width = 0.8 / n_strategies
    colors = plt.cm.Set2(np.linspace(0, 1, n_strategies))

    for j, r in enumerate(table_rows):
        offsets = np.arange(len(scene_names)) + j * bar_width
        vals = [r["scene_errors"][i] if np.isfinite(r["scene_errors"][i]) else 0
                for i in range(len(scene_names))]
        ax.bar(offsets, vals, bar_width, label=r["strategy"],
               color=colors[j], alpha=0.85)

    ax.set_xticks(np.arange(len(scene_names)) + bar_width * (n_strategies - 1) / 2)
    ax.set_xticklabels(scene_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean depth error (abs_rel)")
    ax.set_title("Per-scene depth error by strategy")
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "per_scene_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Confidence calibration: conf-error correlation per strategy
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["strategy"] for r in table_rows]
    corrs = [r["mean_conf_corr"] for r in table_rows]
    corr_stds = [r["std_conf_corr"] for r in table_rows]
    bars = ax.bar(x, corrs, yerr=corr_stds, capsize=4, color="coral", alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Pearson r (confidence vs error)")
    ax.set_title("Confidence Calibration by Strategy (more negative = better calibrated)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "conf_calibration.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save raw data
    np.savez_compressed(
        os.path.join(args.output_dir, "ablation_data.npz"),
        scene_names=np.array(scene_names),
        strategy_names=np.array([r["strategy"] for r in table_rows]),
        scene_errors=np.array([r["scene_errors"] for r in table_rows]),
        conf_corrs=np.array([r["conf_corrs"] for r in table_rows]),
    )

    print(f"\n[done] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
