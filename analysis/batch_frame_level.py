"""
Batch Frame-Level Analysis Across Multiple Scenes
===================================================

Runs frame_level_analysis logic on multiple ScanNet scenes, collects
per-scene confidence-error and state_change-error correlations, and
produces an aggregate summary comparing TTT3R vs CUT3R.

Usage
-----
# Run TTT3R on GPU 0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/batch_frame_level.py \
    --model_path /path/to/model.pth \
    --dataset_root /path/to/scannetv2 \
    --output_dir analysis_results/batch_ttt3r \
    --model_update_type ttt3r \
    --num_scenes 10 --seed 42

# Run CUT3R on GPU 1 (same seed for same scenes)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/batch_frame_level.py \
    --model_path /path/to/model.pth \
    --dataset_root /path/to/scannetv2 \
    --output_dir analysis_results/batch_cut3r \
    --model_update_type cut3r \
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
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Batch frame-level analysis")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--dataset_root", type=str, required=True,
                   help="Root of ScanNet dataset (contains scene*/ dirs).")
    p.add_argument("--output_dir", type=str, default="analysis_results/batch")
    p.add_argument("--model_update_type", type=str, default="ttt3r",
                   choices=["cut3r", "ttt3r"])
    p.add_argument("--num_scenes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--frame_interval", type=int, default=5)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--depth_scale", type=float, default=1000.0)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# =============================================================================
# Helpers (from frame_level_analysis.py, minimal version)
# =============================================================================

def load_img_paths(color_dir, frame_interval, max_frames):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    all_paths = sorted(glob.glob(os.path.join(color_dir, "*")))
    img_paths = [p for p in all_paths if os.path.splitext(p)[1].lower() in img_exts]
    img_paths = img_paths[::frame_interval][:max_frames]
    return img_paths


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


def analyze_one_scene(model, color_dir, depth_dir, args):
    """
    Run inference on one scene and return per-frame signals.

    Returns dict with keys:
        depth_errors, confidences, state_changes, state_drift
    Each is a numpy array of shape (T,).
    Returns None if scene has too few frames.
    """
    img_paths = load_img_paths(color_dir, args.frame_interval, args.max_frames)
    if len(img_paths) < 20:
        return None

    views = build_views(img_paths, args.size)

    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(
            views, device=args.device)

    state_history = analysis_data["state_history"]
    T = len(state_history)

    # State changes
    state_changes = np.zeros(T)
    for t in range(1, T):
        diff = state_history[t] - state_history[t - 1]
        state_changes[t] = diff.norm(dim=-1).mean().item()

    # State drift
    init_state = state_history[0]
    state_drift = np.zeros(T)
    for t in range(T):
        diff = state_history[t] - init_state
        state_drift[t] = diff.norm(dim=-1).mean().item()

    # Depth errors and confidence
    depth_errors = np.zeros(T)
    confidences = np.zeros(T)

    for t in range(T):
        pts3d = ress[t]["pts3d_in_self_view"]
        pred_depth = pts3d[0, :, :, 2].numpy()

        # Depth path
        basename = os.path.splitext(os.path.basename(img_paths[t]))[0]
        depth_path = os.path.join(depth_dir, basename + ".png")

        if os.path.exists(depth_path):
            gt_depth = load_gt_depth(depth_path, args.depth_scale)
            if gt_depth is not None:
                depth_errors[t] = compute_frame_depth_error(
                    pred_depth, gt_depth, args.max_depth)
            else:
                depth_errors[t] = np.nan
        else:
            depth_errors[t] = np.nan

        # Confidence
        if "conf_self" in ress[t]:
            confidences[t] = ress[t]["conf_self"][0].numpy().mean()
        elif "conf" in ress[t]:
            confidences[t] = ress[t]["conf"][0].numpy().mean()
        else:
            confidences[t] = np.nan

    return {
        "depth_errors": depth_errors,
        "confidences": confidences,
        "state_changes": state_changes,
        "state_drift": state_drift,
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

    # ── Find scenes ────────────────────────────────────────────────────────────
    scene_dirs = sorted(glob.glob(os.path.join(args.dataset_root, "scene*")))
    # Filter: must have color/ and depth/ subdirs
    valid_scenes = []
    for sd in scene_dirs:
        color_dir = os.path.join(sd, "color")
        depth_dir = os.path.join(sd, "depth")
        if os.path.isdir(color_dir) and os.path.isdir(depth_dir):
            valid_scenes.append(sd)

    print(f"[data] Found {len(valid_scenes)} valid scenes")

    # Sample
    rng = np.random.RandomState(args.seed)
    if len(valid_scenes) > args.num_scenes:
        indices = rng.choice(len(valid_scenes), args.num_scenes, replace=False)
        selected = [valid_scenes[i] for i in sorted(indices)]
    else:
        selected = valid_scenes

    print(f"[data] Selected {len(selected)} scenes:")
    for s in selected:
        print(f"  {os.path.basename(s)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    print(f"[model] Loading {args.model_update_type} from {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    # ── Run per scene ──────────────────────────────────────────────────────────
    scene_results = {}

    for scene_dir in tqdm(selected, desc="Scenes"):
        scene_name = os.path.basename(scene_dir)
        color_dir = os.path.join(scene_dir, "color")
        depth_dir = os.path.join(scene_dir, "depth")

        try:
            result = analyze_one_scene(model, color_dir, depth_dir, args)
        except Exception as e:
            print(f"[warn] {scene_name} failed: {e}")
            continue

        if result is None:
            print(f"[warn] {scene_name} skipped (too few frames)")
            continue

        scene_results[scene_name] = result

    if not scene_results:
        print("[error] No scenes produced results.")
        return

    # ── Aggregate per-scene correlations ───────────────────────────────────────
    print(f"\n[analysis] Aggregating {len(scene_results)} scenes …")

    records = []
    for scene_name, data in scene_results.items():
        errs = data["depth_errors"]
        confs = data["confidences"]
        changes = data["state_changes"]
        drift = data["state_drift"]

        row = {"scene": scene_name, "n_frames": data["n_frames"]}

        # Confidence vs error
        mask = np.isfinite(errs) & np.isfinite(confs)
        if mask.sum() > 10:
            r_p, _ = pearsonr(confs[mask], errs[mask])
            r_s, _ = spearmanr(confs[mask], errs[mask])
            row["conf_pearson"] = r_p
            row["conf_spearman"] = r_s
        else:
            row["conf_pearson"] = np.nan
            row["conf_spearman"] = np.nan

        # State change vs error
        mask = np.isfinite(errs) & np.isfinite(changes)
        if mask.sum() > 10:
            r_p, _ = pearsonr(changes[mask], errs[mask])
            r_s, _ = spearmanr(changes[mask], errs[mask])
            row["change_pearson"] = r_p
            row["change_spearman"] = r_s
        else:
            row["change_pearson"] = np.nan
            row["change_spearman"] = np.nan

        # State drift vs error
        mask = np.isfinite(errs) & np.isfinite(drift)
        if mask.sum() > 10:
            r_p, _ = pearsonr(drift[mask], errs[mask])
            r_s, _ = spearmanr(drift[mask], errs[mask])
            row["drift_pearson"] = r_p
            row["drift_spearman"] = r_s
        else:
            row["drift_pearson"] = np.nan
            row["drift_spearman"] = np.nan

        # Mean depth error for this scene
        row["mean_error"] = np.nanmean(errs)
        row["mean_conf"] = np.nanmean(confs)

        records.append(row)

    # ── Summary statistics ─────────────────────────────────────────────────────
    conf_pearsons = np.array([r["conf_pearson"] for r in records])
    conf_spearmans = np.array([r["conf_spearman"] for r in records])
    change_pearsons = np.array([r["change_pearson"] for r in records])
    change_spearmans = np.array([r["change_spearman"] for r in records])
    drift_pearsons = np.array([r["drift_pearson"] for r in records])
    mean_errors = np.array([r["mean_error"] for r in records])
    mean_confs = np.array([r["mean_conf"] for r in records])

    def safe_mean_std(arr):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            return np.nan, np.nan
        return v.mean(), v.std()

    lines = [
        "=" * 70,
        f"Batch Frame-Level Analysis: {args.model_update_type.upper()}",
        "=" * 70,
        f"Scenes analyzed:  {len(records)}",
        f"Model:            {args.model_update_type}",
        f"Frame interval:   {args.frame_interval}",
        f"Max frames/scene: {args.max_frames}",
        "",
        "--- Per-scene results ---",
        f"{'Scene':<25s} {'Frames':>6s} {'MeanErr':>8s} {'MeanConf':>8s} "
        f"{'Conf_r':>7s} {'Conf_ρ':>7s} {'Chg_r':>7s} {'Drft_r':>7s}",
        "-" * 70,
    ]
    for r in records:
        lines.append(
            f"{r['scene']:<25s} {r['n_frames']:>6d} {r['mean_error']:>8.4f} "
            f"{r['mean_conf']:>8.2f} {r['conf_pearson']:>+7.3f} "
            f"{r['conf_spearman']:>+7.3f} {r['change_pearson']:>+7.3f} "
            f"{r['drift_pearson']:>+7.3f}")

    m, s = safe_mean_std(conf_pearsons)
    ms, ss = safe_mean_std(conf_spearmans)
    mc, sc = safe_mean_std(change_pearsons)
    md, sd_ = safe_mean_std(drift_pearsons)

    lines += [
        "-" * 70,
        "",
        "--- Aggregate (mean ± std across scenes) ---",
        f"  Confidence-Error Pearson r:   {m:+.4f} ± {s:.4f}",
        f"  Confidence-Error Spearman ρ:  {ms:+.4f} ± {ss:.4f}",
        f"  StateChange-Error Pearson r:  {mc:+.4f} ± {sc:.4f}",
        f"  StateDrift-Error Pearson r:   {md:+.4f} ± {sd_:.4f}",
        f"  Mean depth error:             {np.nanmean(mean_errors):.4f} ± {np.nanstd(mean_errors):.4f}",
        f"  Mean confidence:              {np.nanmean(mean_confs):.2f} ± {np.nanstd(mean_confs):.2f}",
        "=" * 70,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = os.path.join(args.output_dir, "batch_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")

    # ── Save raw data ──────────────────────────────────────────────────────────
    npz_path = os.path.join(args.output_dir, "batch_data.npz")
    np.savez_compressed(
        npz_path,
        scene_names=np.array([r["scene"] for r in records]),
        conf_pearsons=conf_pearsons,
        conf_spearmans=conf_spearmans,
        change_pearsons=change_pearsons,
        change_spearmans=change_spearmans,
        drift_pearsons=drift_pearsons,
        mean_errors=mean_errors,
        mean_confs=mean_confs,
    )

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Bar chart: per-scene confidence-error correlation
    fig, ax = plt.subplots(figsize=(12, 4))
    scene_names = [r["scene"] for r in records]
    x = np.arange(len(scene_names))
    valid_cp = np.where(np.isfinite(conf_pearsons), conf_pearsons, 0)
    colors = ["C3" if v > 0 else "C0" for v in valid_cp]
    ax.bar(x, valid_cp, color=colors, alpha=0.7)
    ax.axhline(m, color="black", linestyle="--", linewidth=1,
               label=f"Mean = {m:+.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(scene_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r (confidence vs error)")
    ax.set_title(f"Per-scene Confidence–Error correlation [{args.model_update_type.upper()}]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "per_scene_conf_error.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Histogram: distribution of confidence-error correlations
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_vals = conf_pearsons[np.isfinite(conf_pearsons)]
    ax.hist(valid_vals, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(m, color="crimson", linestyle="--",
               label=f"Mean = {m:+.3f}")
    ax.set_xlabel("Pearson r (confidence vs depth error)")
    ax.set_ylabel("Number of scenes")
    ax.set_title(f"Distribution of confidence–error correlation [{args.model_update_type.upper()}]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "conf_error_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Scatter: mean confidence vs mean error across scenes
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean_confs, mean_errors, s=30, color="steelblue")
    for i, name in enumerate(scene_names):
        ax.annotate(name[-8:], (mean_confs[i], mean_errors[i]),
                    fontsize=6, alpha=0.7)
    mask = np.isfinite(mean_confs) & np.isfinite(mean_errors)
    if mask.sum() > 3:
        r_p, _ = pearsonr(mean_confs[mask], mean_errors[mask])
        z = np.polyfit(mean_confs[mask], mean_errors[mask], 1)
        x_line = np.linspace(mean_confs[mask].min(), mean_confs[mask].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1)
        ax.set_title(f"Scene-level: Mean confidence vs Mean error (r={r_p:.3f})")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Mean depth error (abs_rel)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scene_conf_vs_error.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
