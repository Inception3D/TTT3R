"""
Experiment A2: Frame-Level State Dynamics vs Reconstruction Quality
====================================================================

Instead of patch-level spatial correlation (which fails due to sparse
frequency projection), this script analyzes FRAME-LEVEL relationships:

1. State change magnitude per frame  vs  mean depth error per frame
2. Model confidence per frame         vs  mean depth error per frame
3. TTT3R update mask magnitude        vs  mean depth error per frame

These frame-level signals are more natural proxies for "how much the
model is adapting" and "how well it's reconstructing".

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/frame_level_analysis.py \
    --model_path /path/to/model.pth \
    --seq_path /path/to/scene/color \
    --output_dir analysis_results/expA2 \
    --model_update_type ttt3r \
    --size 512 --frame_interval 5 --max_frames 200
"""

import os
import sys
import argparse
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp A2 – Frame-Level State Dynamics vs Reconstruction Quality")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--seq_path", type=str, required=True,
                   help="Path to ScanNet color/ directory.")
    p.add_argument("--output_dir", type=str, default="analysis_results/expA2")
    p.add_argument("--model_update_type", type=str, default="ttt3r",
                   choices=["cut3r", "ttt3r"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--frame_interval", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--depth_scale", type=float, default=1000.0)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# =============================================================================
# Input helpers
# =============================================================================

def load_img_paths(seq_path, frame_interval, max_frames):
    import glob
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    all_paths = sorted(glob.glob(os.path.join(seq_path, "*")))
    img_paths = [p for p in all_paths if os.path.splitext(p)[1].lower() in img_exts]
    img_paths = img_paths[::frame_interval][:max_frames]
    print(f"[load] Using {len(img_paths)} frames (interval={frame_interval})")
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
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(
                np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def derive_depth_dir(seq_path):
    color_dir = os.path.normpath(seq_path)
    parent = os.path.dirname(color_dir)
    base = os.path.basename(color_dir)
    depth_base = base.replace("color", "depth", 1)
    return os.path.join(parent, depth_base)


def color_to_depth_path(color_path, depth_dir):
    basename = os.path.splitext(os.path.basename(color_path))[0]
    return os.path.join(depth_dir, basename + ".png")


def load_gt_depth(depth_path, depth_scale=1000.0):
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32) / depth_scale
    d[d <= 0] = 0
    return d


# =============================================================================
# Per-frame metrics
# =============================================================================

def compute_state_change(state_history):
    """
    Per-frame state change: L2 norm of (state_t - state_{t-1}), averaged
    over all tokens and feature dimensions.

    Returns:
        state_changes: list of T floats (first frame = 0)
    """
    changes = [0.0]
    for t in range(1, len(state_history)):
        diff = state_history[t] - state_history[t - 1]  # [n_state, dec_dim]
        change = diff.norm(dim=-1).mean().item()          # mean over tokens
        changes.append(change)
    return changes


def compute_per_token_state_change(state_history):
    """
    Per-frame, per-token state change magnitude.

    Returns:
        token_changes: [T, n_state] numpy array (first frame = 0)
    """
    T = len(state_history)
    n_state = state_history[0].shape[0]
    token_changes = np.zeros((T, n_state))
    for t in range(1, T):
        diff = state_history[t] - state_history[t - 1]
        token_changes[t] = diff.norm(dim=-1).numpy()
    return token_changes


def compute_update_mask_stats(cross_attn_history, model_update_type):
    """
    For TTT3R: compute the actual update mask magnitude per frame.
    update_mask = sigmoid(mean(cross_attn_logits over patches and layers/heads))

    Note: cross_attn_history contains post-softmax values from the analysis method.
    For TTT3R update mask we need pre-softmax logits, which we don't have here.
    Instead, we compute the mean attention entropy as a proxy for update selectivity.

    Returns:
        mask_magnitudes: list of T floats (mean update weight per frame)
        mask_entropies:  list of T floats (attention entropy per frame)
    """
    magnitudes = []
    entropies = []
    for t in range(len(cross_attn_history)):
        attn = cross_attn_history[t]  # [n_state, n_patches], post-softmax
        # Mean attention value (proxy for how spread out attention is)
        magnitudes.append(attn.mean().item())

        # Attention entropy per token, averaged
        # Higher entropy = more uniform attention = less selective
        eps = 1e-8
        ent = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()
        entropies.append(ent)

    return magnitudes, entropies


def compute_frame_depth_error(pred_depth, gt_depth, max_depth):
    """
    Compute mean absolute relative depth error for one frame.

    Returns:
        mean_error: float (NaN if insufficient valid pixels)
        valid_ratio: float
    """
    H_pred, W_pred = pred_depth.shape
    gt_resized = cv2.resize(gt_depth, (W_pred, H_pred),
                            interpolation=cv2.INTER_NEAREST)

    valid = (gt_resized > 0) & (gt_resized < max_depth) & (pred_depth > 1e-3)
    valid_ratio = valid.sum() / valid.size

    if valid.sum() < 10:
        return np.nan, valid_ratio

    # Median-ratio scale alignment
    scale = np.median(gt_resized[valid]) / np.median(pred_depth[valid])
    pred_aligned = pred_depth * scale

    abs_rel = np.abs(pred_aligned[valid] - gt_resized[valid]) / gt_resized[valid]
    return abs_rel.mean(), valid_ratio


def compute_frame_confidence(res):
    """Extract mean model confidence for one frame."""
    if "conf_self" in res:
        conf = res["conf_self"][0].numpy()  # (H, W, 1)
        return conf.mean()
    elif "conf" in res:
        conf = res["conf"][0].numpy()
        return conf.mean()
    return np.nan


# =============================================================================
# Visualization
# =============================================================================

def plot_dual_axis(x, y1, y2, xlabel, y1label, y2label, title, out_path,
                   corr_label=None):
    """Scatter plot with two y variables against x, and correlation annotation."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "C0"
    color2 = "C3"

    ax1.scatter(x, y1, s=8, alpha=0.6, color=color1, label=y1label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    if corr_label:
        ax1.set_title(f"{title}\n{corr_label}", fontsize=11)
    else:
        ax1.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


def plot_time_series_comparison(frames, series_dict, title, ylabel, out_path):
    """Plot multiple time series on the same axes."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, (values, color) in series_dict.items():
        ax.plot(frames, values, linewidth=1, alpha=0.8, color=color, label=label)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


def plot_scatter_with_regression(x, y, xlabel, ylabel, title, out_path):
    """Simple scatter with regression line and correlation stats."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid, y_valid = x[mask], y[mask]

    if len(x_valid) < 5:
        print(f"[warn] Not enough data for {out_path}")
        return

    r_p, p_p = pearsonr(x_valid, y_valid)
    r_s, p_s = spearmanr(x_valid, y_valid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_valid, y_valid, s=15, alpha=0.6, color="steelblue")

    # Regression line
    z = np.polyfit(x_valid, y_valid, 1)
    x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nPearson r={r_p:.3f} (p={p_p:.3g})  "
                 f"Spearman ρ={r_s:.3f} (p={p_s:.3g})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


def plot_correlation_matrix(data_dict, out_path):
    """
    Correlation matrix heatmap of all frame-level signals.
    data_dict: {name: np.array of shape (T,)}
    """
    names = list(data_dict.keys())
    n = len(names)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mask = np.isfinite(data_dict[names[i]]) & np.isfinite(data_dict[names[j]])
            if mask.sum() > 5:
                corr_matrix[i, j], _ = pearsonr(
                    data_dict[names[i]][mask], data_dict[names[j]][mask])
            else:
                corr_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Frame-level signal correlation matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


def plot_top_bottom_tokens(token_changes, depth_errors, out_path, top_k=20):
    """
    Compare state change of top-K most-changing tokens vs bottom-K
    against depth error over time.
    """
    mean_change = np.nanmean(token_changes[1:], axis=0)  # skip frame 0
    ranked = np.argsort(mean_change)[::-1]
    top_idx = ranked[:top_k]
    bot_idx = ranked[-top_k:]

    T = token_changes.shape[0]
    frames = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Top-K token changes
    for idx in top_idx:
        axes[0].plot(frames, token_changes[:, idx], alpha=0.4, linewidth=0.6)
    axes[0].set_ylabel("State change (L2)")
    axes[0].set_title(f"Top-{top_k} most dynamic tokens")

    # Bottom-K token changes
    for idx in bot_idx:
        axes[1].plot(frames, token_changes[:, idx], alpha=0.4, linewidth=0.6)
    axes[1].set_ylabel("State change (L2)")
    axes[1].set_title(f"Bottom-{top_k} most stable tokens")

    # Depth error
    axes[2].plot(frames, depth_errors, "k-", linewidth=1, alpha=0.8)
    axes[2].set_ylabel("Mean depth error (abs_rel)")
    axes[2].set_xlabel("Frame index")
    axes[2].set_title("Frame depth error")

    plt.suptitle("Per-token dynamics vs reconstruction error", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU.")
        device = "cpu"

    # ── Depth dir ──────────────────────────────────────────────────────────────
    depth_dir = derive_depth_dir(args.seq_path)
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    print(f"[depth] GT depth dir: {depth_dir}")

    # ── Model ──────────────────────────────────────────────────────────────────
    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    print(f"[model] Loading from {args.model_path} …")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    # ── Data ───────────────────────────────────────────────────────────────────
    img_paths = load_img_paths(args.seq_path, args.frame_interval, args.max_frames)
    if not img_paths:
        raise RuntimeError(f"No images found at {args.seq_path}")

    depth_paths = [color_to_depth_path(p, depth_dir) for p in img_paths]
    views = build_views(img_paths, args.size)

    # ── Inference ──────────────────────────────────────────────────────────────
    print("[inference] Running forward_recurrent_analysis …")
    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(views, device=device)

    state_history = analysis_data["state_history"]
    cross_attn_history = analysis_data["cross_attn_history"]
    T = len(state_history)
    print(f"[data] T={T} frames")

    # ── Frame-level signals ────────────────────────────────────────────────────
    print("[analysis] Computing frame-level signals …")

    # 1. State change magnitude
    state_changes = compute_state_change(state_history)

    # 2. Per-token state change
    token_changes = compute_per_token_state_change(state_history)

    # 3. Attention stats
    attn_magnitudes, attn_entropies = compute_update_mask_stats(
        cross_attn_history, args.model_update_type)

    # 4. Depth errors and confidence
    depth_errors = []
    confidences = []
    for t in tqdm(range(T), desc="Depth errors"):
        # Predicted depth
        pts3d = ress[t]["pts3d_in_self_view"]
        pred_depth = pts3d[0, :, :, 2].numpy()

        # GT depth
        if t < len(depth_paths) and os.path.exists(depth_paths[t]):
            gt_depth = load_gt_depth(depth_paths[t], args.depth_scale)
        else:
            gt_depth = None

        if gt_depth is not None:
            err, _ = compute_frame_depth_error(pred_depth, gt_depth, args.max_depth)
        else:
            err = np.nan
        depth_errors.append(err)

        # Confidence
        conf = compute_frame_confidence(ress[t])
        confidences.append(conf)

    # Convert to arrays
    state_changes = np.array(state_changes)
    depth_errors = np.array(depth_errors)
    confidences = np.array(confidences)
    attn_magnitudes = np.array(attn_magnitudes)
    attn_entropies = np.array(attn_entropies)

    # 5. Cumulative state drift (distance from initial state)
    state_drift = np.zeros(T)
    init_state = state_history[0]
    for t in range(T):
        diff = state_history[t] - init_state
        state_drift[t] = diff.norm(dim=-1).mean().item()

    # ── Correlation analysis ───────────────────────────────────────────────────
    print("[analysis] Computing correlations …")

    signals = {
        "state_change": state_changes,
        "state_drift": state_drift,
        "attn_entropy": attn_entropies,
        "depth_error": depth_errors,
        "confidence": confidences,
    }

    # Print pairwise correlations with depth error
    print("\n--- Correlations with depth error ---")
    results = {}
    for name, values in signals.items():
        if name == "depth_error":
            continue
        mask = np.isfinite(values) & np.isfinite(depth_errors)
        if mask.sum() > 5:
            r_p, p_p = pearsonr(values[mask], depth_errors[mask])
            r_s, p_s = spearmanr(values[mask], depth_errors[mask])
            results[name] = (r_p, p_p, r_s, p_s)
            print(f"  {name:20s}: Pearson r={r_p:+.4f} (p={p_p:.3g}), "
                  f"Spearman ρ={r_s:+.4f} (p={p_s:.3g})")

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 60,
        "Experiment A2: Frame-Level State Dynamics vs Reconstruction",
        "=" * 60,
        f"Sequence:          {args.seq_path}",
        f"Model:             {args.model_update_type}",
        f"Frames processed:  {T}",
        "",
        "--- Correlations with depth error ---",
    ]
    for name, (r_p, p_p, r_s, p_s) in results.items():
        summary_lines.append(
            f"  {name:20s}: Pearson r={r_p:+.4f} (p={p_p:.3g}), "
            f"Spearman ρ={r_s:+.4f} (p={p_s:.3g})")
    summary_lines.append("=" * 60)
    summary = "\n".join(summary_lines)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\n{summary}")

    # ── Save raw data ──────────────────────────────────────────────────────────
    npz_path = os.path.join(args.output_dir, "frame_level_data.npz")
    np.savez_compressed(
        npz_path,
        state_changes=state_changes,
        state_drift=state_drift,
        token_changes=token_changes,
        attn_magnitudes=attn_magnitudes,
        attn_entropies=attn_entropies,
        depth_errors=depth_errors,
        confidences=confidences,
    )
    print(f"[save] Raw data → {npz_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Time series: all signals together
    frames = np.arange(T)

    # Normalize for overlay plot
    def norm01(x):
        x = x.copy()
        mask = np.isfinite(x)
        if mask.sum() == 0:
            return x
        mn, mx = x[mask].min(), x[mask].max()
        if mx - mn > 1e-8:
            x[mask] = (x[mask] - mn) / (mx - mn)
        return x

    plot_time_series_comparison(
        frames,
        {
            "State change (norm)": (norm01(state_changes), "C0"),
            "Depth error (norm)": (norm01(depth_errors), "C3"),
            "Confidence (norm)": (norm01(confidences), "C2"),
        },
        "Frame-level signals over time (normalized to [0,1])",
        "Normalized value",
        os.path.join(plots_dir, "time_series_overlay.png"),
    )

    # 2. Scatter: state change vs depth error
    plot_scatter_with_regression(
        state_changes, depth_errors,
        "State change magnitude (mean L2)",
        "Mean depth error (abs_rel)",
        "State change vs Depth error",
        os.path.join(plots_dir, "scatter_state_change_vs_error.png"),
    )

    # 3. Scatter: state drift vs depth error
    plot_scatter_with_regression(
        state_drift, depth_errors,
        "State drift from init (mean L2)",
        "Mean depth error (abs_rel)",
        "Cumulative state drift vs Depth error",
        os.path.join(plots_dir, "scatter_state_drift_vs_error.png"),
    )

    # 4. Scatter: confidence vs depth error
    plot_scatter_with_regression(
        confidences, depth_errors,
        "Mean model confidence",
        "Mean depth error (abs_rel)",
        "Confidence vs Depth error",
        os.path.join(plots_dir, "scatter_confidence_vs_error.png"),
    )

    # 5. Scatter: attention entropy vs depth error
    plot_scatter_with_regression(
        attn_entropies, depth_errors,
        "Mean attention entropy",
        "Mean depth error (abs_rel)",
        "Attention entropy vs Depth error",
        os.path.join(plots_dir, "scatter_attn_entropy_vs_error.png"),
    )

    # 6. Correlation matrix
    plot_correlation_matrix(signals, os.path.join(plots_dir, "correlation_matrix.png"))

    # 7. Top/bottom tokens vs error
    plot_top_bottom_tokens(
        token_changes, depth_errors,
        os.path.join(plots_dir, "top_bottom_tokens_vs_error.png"),
    )

    print(f"\n[done] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
