"""
Batch Spectral Analysis Across Multiple Scenes (ScanNet + TUM)
===============================================================

Runs spectral analysis on multiple scenes, collects per-scene correlations
for all spectral features vs depth error, and produces aggregate summary.

Usage
-----
# ScanNet + TUM combined, TTT3R on GPU 0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/batch_spectral.py \
    --model_path /path/to/model.pth \
    --scannet_root /path/to/scannetv2 \
    --tum_root /path/to/tum \
    --output_dir analysis_results/batch_spectral_ttt3r \
    --model_update_type ttt3r --num_scannet 10 --seed 42

# CUT3R on GPU 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/batch_spectral.py \
    --model_path /path/to/model.pth \
    --scannet_root /path/to/scannetv2 \
    --tum_root /path/to/tum \
    --output_dir analysis_results/batch_spectral_cut3r \
    --model_update_type cut3r --num_scannet 10 --seed 42
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

# Reuse helpers from spectral_analysis
from analysis.spectral_analysis import (
    load_img_paths, build_views, load_gt_depth, compute_frame_depth_error,
    derive_depth_dir, load_tum_associations, build_tum_timestamp_index,
    find_gt_depth_path, compute_token_power_spectra, compute_band_energies,
    compute_running_spectral_features, compute_online_spectral_features,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Batch Spectral Analysis")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--scannet_root", type=str, default="",
                   help="Root of ScanNet dataset (contains scene*/ dirs).")
    p.add_argument("--tum_root", type=str, default="",
                   help="Root of TUM dataset (contains rgbd_dataset_*/ dirs).")
    p.add_argument("--output_dir", type=str, default="analysis_results/batch_spectral")
    p.add_argument("--model_update_type", type=str, default="ttt3r",
                   choices=["cut3r", "ttt3r"])
    p.add_argument("--num_scannet", type=int, default=10,
                   help="Number of ScanNet scenes to sample.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--frame_interval_scannet", type=int, default=5)
    p.add_argument("--frame_interval_tum", type=int, default=3)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--depth_scale_scannet", type=float, default=1000.0)
    p.add_argument("--depth_scale_tum", type=float, default=5000.0)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# =============================================================================
# Scene discovery
# =============================================================================

def discover_scenes(args):
    """Return list of (scene_name, rgb_dir, depth_dir, depth_scale, frame_interval, dataset)."""
    scenes = []
    rng = np.random.RandomState(args.seed)

    # ScanNet
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
            scenes.append((os.path.basename(sd), cd, dd,
                           args.depth_scale_scannet, args.frame_interval_scannet,
                           "scannet"))
        print(f"[data] ScanNet: {len(valid)} scenes")

    # TUM
    if args.tum_root and os.path.isdir(args.tum_root):
        tum_dirs = sorted(glob.glob(os.path.join(args.tum_root, "rgbd_dataset_*")))
        valid = [(td, os.path.join(td, "rgb"), os.path.join(td, "depth"))
                 for td in tum_dirs
                 if os.path.isdir(os.path.join(td, "rgb"))
                 and os.path.isdir(os.path.join(td, "depth"))]
        for td, rd, dd in valid:
            scenes.append((os.path.basename(td), rd, dd,
                           args.depth_scale_tum, args.frame_interval_tum,
                           "tum"))
        print(f"[data] TUM: {len(valid)} scenes")

    return scenes


# =============================================================================
# Per-scene analysis
# =============================================================================

def analyze_one_scene(model, rgb_dir, depth_dir, depth_scale,
                      frame_interval, max_frames, size, window_size,
                      device, dataset):
    """Run spectral analysis on one scene. Returns dict of per-scene metrics."""
    img_paths = load_img_paths(rgb_dir, frame_interval, max_frames)
    if len(img_paths) < 30:
        return None

    views = build_views(img_paths, size)

    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(
            views, device=device)

    state_history = analysis_data["state_history"]
    T = len(state_history)
    state_stack = torch.stack(state_history).numpy()  # [T, n_state, D]

    # --- Spectral features ---
    freqs, power_spectra = compute_token_power_spectra(state_stack)
    band_energies = compute_band_energies(freqs, power_spectra)
    frame_high, frame_low, frame_ratio = compute_running_spectral_features(
        state_stack, window_size)
    high_freq_energy, _ = compute_online_spectral_features(state_stack)
    frame_high_online = high_freq_energy.mean(axis=1)  # [T]

    # --- Depth errors, confidence, state_change ---
    # Depth matching setup
    scene_dir = os.path.dirname(os.path.normpath(rgb_dir))
    tum_assoc = load_tum_associations(scene_dir) if dataset == "tum" else None
    tum_depth_index = build_tum_timestamp_index(depth_dir) if dataset == "tum" else None

    depth_errors = np.full(T, np.nan)
    confidences = np.full(T, np.nan)
    state_changes = np.zeros(T)

    for t in range(T):
        # Depth error
        pts3d = ress[t]["pts3d_in_self_view"]
        pred_depth = pts3d[0, :, :, 2].numpy()
        depth_path = find_gt_depth_path(img_paths[t], depth_dir,
                                        tum_assoc, tum_depth_index)
        if depth_path is not None:
            gt = load_gt_depth(depth_path, depth_scale)
            if gt is not None:
                depth_errors[t] = compute_frame_depth_error(
                    pred_depth, gt, 10.0)

        # Confidence
        if "conf_self" in ress[t]:
            confidences[t] = ress[t]["conf_self"][0].numpy().mean()
        elif "conf" in ress[t]:
            confidences[t] = ress[t]["conf"][0].numpy().mean()

        # State change
        if t > 0:
            diff = state_history[t] - state_history[t - 1]
            state_changes[t] = diff.norm(dim=-1).mean().item()

    # --- Compute correlations ---
    signals = {
        "high_freq_win": frame_high,
        "high_freq_online": frame_high_online,
        "low_freq_win": frame_low,
        "high_total_ratio": frame_ratio,
        "state_change": state_changes,
        "confidence": confidences,
    }

    correlations = {}
    for name, signal in signals.items():
        mask = np.isfinite(depth_errors) & np.isfinite(signal)
        if mask.sum() > 10:
            r_p, p_p = pearsonr(signal[mask], depth_errors[mask])
            r_s, p_s = spearmanr(signal[mask], depth_errors[mask])
            correlations[name] = {"pearson": r_p, "spearman": r_s,
                                  "p_pearson": p_p, "p_spearman": p_s}
        else:
            correlations[name] = {"pearson": np.nan, "spearman": np.nan,
                                  "p_pearson": np.nan, "p_spearman": np.nan}

    return {
        "n_frames": T,
        "n_valid_depth": int(np.isfinite(depth_errors).sum()),
        "mean_error": float(np.nanmean(depth_errors)),
        "mean_conf": float(np.nanmean(confidences)),
        "mean_high_ratio": float(band_energies["high_ratio"].mean()),
        "mean_total_energy": float(band_energies["total"].mean()),
        "correlations": correlations,
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

    scenes = discover_scenes(args)
    if not scenes:
        print("[error] No scenes found.")
        return
    print(f"[data] Total: {len(scenes)} scenes")

    # ── Model ──
    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    print(f"[model] Loading {args.model_update_type} from {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    # ── Run per scene ──
    results = {}
    for scene_name, rgb_dir, depth_dir, depth_scale, fi, dataset in tqdm(scenes, desc="Scenes"):
        try:
            r = analyze_one_scene(
                model, rgb_dir, depth_dir, depth_scale,
                fi, args.max_frames, args.size, args.window_size,
                device, dataset)
        except Exception as e:
            print(f"[warn] {scene_name} failed: {e}")
            continue
        if r is None:
            print(f"[warn] {scene_name} skipped (too few frames)")
            continue
        r["dataset"] = dataset
        results[scene_name] = r

    if not results:
        print("[error] No scenes produced results.")
        return

    # ── Aggregate ──
    signal_names = ["high_freq_win", "high_freq_online", "low_freq_win",
                    "high_total_ratio", "state_change", "confidence"]

    # Collect per-scene correlations
    records = []
    for scene_name, data in results.items():
        row = {
            "scene": scene_name,
            "dataset": data["dataset"],
            "n_frames": data["n_frames"],
            "n_valid_depth": data["n_valid_depth"],
            "mean_error": data["mean_error"],
            "mean_conf": data["mean_conf"],
        }
        for sig in signal_names:
            row[f"{sig}_r"] = data["correlations"][sig]["pearson"]
            row[f"{sig}_rho"] = data["correlations"][sig]["spearman"]
        records.append(row)

    # ── Summary text ──
    def safe_stats(arr):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            return np.nan, np.nan
        return v.mean(), v.std()

    lines = [
        "=" * 80,
        f"Batch Spectral Analysis: {args.model_update_type.upper()}",
        "=" * 80,
        f"Scenes analyzed: {len(records)}",
        f"  ScanNet: {sum(1 for r in records if r['dataset']=='scannet')}",
        f"  TUM:     {sum(1 for r in records if r['dataset']=='tum')}",
        "",
    ]

    # Per-scene table
    hdr = f"{'Scene':<45s} {'DS':>3s} {'Frm':>4s} {'Err':>6s} "
    hdr += " ".join(f"{s[:8]:>8s}" for s in signal_names)
    lines.append(hdr)
    lines.append("-" * 80)

    for r in records:
        row_str = f"{r['scene']:<45s} {r['dataset'][:3]:>3s} {r['n_frames']:>4d} {r['mean_error']:>6.3f} "
        row_str += " ".join(f"{r[f'{s}_r']:>+8.3f}" for s in signal_names)
        lines.append(row_str)

    lines.append("-" * 80)
    lines.append("")

    # Aggregate by dataset
    for ds_label, ds_filter in [("ALL", None), ("ScanNet", "scannet"), ("TUM", "tum")]:
        subset = [r for r in records if ds_filter is None or r["dataset"] == ds_filter]
        if not subset:
            continue
        lines.append(f"--- Aggregate: {ds_label} ({len(subset)} scenes) ---")
        for sig in signal_names:
            vals_r = np.array([r[f"{sig}_r"] for r in subset])
            vals_rho = np.array([r[f"{sig}_rho"] for r in subset])
            m_r, s_r = safe_stats(vals_r)
            m_rho, s_rho = safe_stats(vals_rho)
            lines.append(f"  {sig:<22s}: r={m_r:+.4f}±{s_r:.4f}  ρ={m_rho:+.4f}±{s_rho:.4f}")
        m_err, s_err = safe_stats(np.array([r["mean_error"] for r in subset]))
        lines.append(f"  {'mean_depth_error':<22s}: {m_err:.4f}±{s_err:.4f}")
        lines.append("")

    lines.append("=" * 80)
    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = os.path.join(args.output_dir, "batch_spectral_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")

    # ── Save raw data ──
    np.savez_compressed(
        os.path.join(args.output_dir, "batch_spectral_data.npz"),
        scene_names=np.array([r["scene"] for r in records]),
        datasets=np.array([r["dataset"] for r in records]),
        **{f"{sig}_r": np.array([r[f"{sig}_r"] for r in records])
           for sig in signal_names},
        **{f"{sig}_rho": np.array([r[f"{sig}_rho"] for r in records])
           for sig in signal_names},
        mean_errors=np.array([r["mean_error"] for r in records]),
        mean_confs=np.array([r["mean_conf"] for r in records]),
    )

    # ── Plots ──
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Per-scene correlation bar chart for each signal
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for ax, sig in zip(axes.ravel(), signal_names):
        vals = np.array([r[f"{sig}_r"] for r in records])
        ds_labels = [r["dataset"] for r in records]
        colors = ["C0" if d == "scannet" else "C1" for d in ds_labels]
        x = np.arange(len(records))
        valid_vals = np.where(np.isfinite(vals), vals, 0)
        ax.bar(x, valid_vals, color=colors, alpha=0.7)
        m, _ = safe_stats(vals)
        ax.axhline(m, color="black", linestyle="--", linewidth=1,
                   label=f"mean={m:+.3f}")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"{sig} vs depth_error")
        ax.set_xticks(x)
        ax.set_xticklabels([r["scene"][-12:] for r in records],
                           rotation=60, ha="right", fontsize=6)
        ax.legend(fontsize=7)
    plt.suptitle(f"Per-scene signal–error correlations [{args.model_update_type.upper()}]"
                 f"\n(blue=ScanNet, orange=TUM)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "per_scene_correlations.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Aggregate comparison: mean correlation per signal (grouped bar by dataset)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(signal_names))
    width = 0.25
    for i, (ds_label, ds_filter, color) in enumerate([
        ("ScanNet", "scannet", "C0"), ("TUM", "tum", "C1"), ("ALL", None, "C2")
    ]):
        subset = [r for r in records if ds_filter is None or r["dataset"] == ds_filter]
        if not subset:
            continue
        means = []
        stds = []
        for sig in signal_names:
            vals = np.array([r[f"{sig}_r"] for r in subset])
            m, s = safe_stats(vals)
            means.append(m)
            stds.append(s)
        ax.bar(x + i * width, means, width, yerr=stds, label=ds_label,
               color=color, alpha=0.7, capsize=3)
    ax.set_xticks(x + width)
    ax.set_xticklabels(signal_names, rotation=30, ha="right")
    ax.set_ylabel("Mean Pearson r (vs depth error)")
    ax.set_title(f"Signal–Error Correlation by Dataset [{args.model_update_type.upper()}]")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "aggregate_by_dataset.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Confidence vs spectral: scatter of per-scene conf_r vs high_freq_r
    fig, ax = plt.subplots(figsize=(6, 5))
    conf_rs = np.array([r["confidence_r"] for r in records])
    hf_rs = np.array([r["high_freq_win_r"] for r in records])
    ds_colors = ["C0" if r["dataset"] == "scannet" else "C1" for r in records]
    ax.scatter(conf_rs, hf_rs, c=ds_colors, s=40, alpha=0.7)
    for i, r in enumerate(records):
        ax.annotate(r["scene"][-10:], (conf_rs[i], hf_rs[i]),
                    fontsize=5, alpha=0.6)
    ax.set_xlabel("Confidence–Error Pearson r")
    ax.set_ylabel("High-freq–Error Pearson r")
    ax.set_title("Per-scene: Confidence vs Spectral signal strength")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "conf_vs_spectral_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] All outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
