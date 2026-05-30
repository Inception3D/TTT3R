"""
Experiment A: Frequency vs Reconstruction Error Correlation
============================================================

Hypothesis:
    State tokens with high temporal variance ("high frequency") attend to
    scene regions where reconstruction is poor. If this correlation holds,
    frequency can serve as a self-supervised quality signal for guiding
    state updates, frame selection, and memory management.

What this script produces
--------------------------
output_dir/
  plots/
    scatter_freq_vs_error.png        – aggregate scatter: patch freq vs patch error
    correlation_over_time.png        – per-frame Pearson/Spearman r
    side_by_side/
      frame_000000.png               – 4-panel: RGB | freq map | error map | overlay
      ...
  freq_error_data.npz                – raw arrays
  correlation_summary.txt            – text summary

Usage
-----
python analysis/freq_error_analysis.py \
    --model_path /path/to/model.pth \
    --seq_path /path/to/scene/color \
    --output_dir analysis_results/expA \
    --model_update_type ttt3r \
    --size 512 --frame_interval 5 --max_frames 200

Notes
-----
- --seq_path must point to the color/ directory; depth/ is derived as a sibling.
- Requires ScanNet-style GT depth (uint16 PNG, mm → /1000 → meters).
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
import matplotlib.cm as cm
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
        description="Exp A – Frequency vs Reconstruction Error Correlation")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--seq_path", type=str, required=True,
                   help="Path to ScanNet color/ directory.")
    p.add_argument("--output_dir", type=str, default="analysis_results/expA")
    p.add_argument("--model_update_type", type=str, default="ttt3r",
                   choices=["cut3r", "ttt3r"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--frame_interval", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--depth_scale", type=float, default=1000.0,
                   help="Divisor to convert uint16 depth to meters.")
    p.add_argument("--max_depth", type=float, default=10.0,
                   help="Ignore GT depth beyond this (meters).")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# =============================================================================
# Input helpers (reused from state_freq_analysis.py)
# =============================================================================

def load_img_paths(seq_path, frame_interval, max_frames):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    import glob
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


# =============================================================================
# GT depth helpers
# =============================================================================

def derive_depth_dir(seq_path):
    """Given .../scene/color, return .../scene/depth."""
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
# Frequency computation (reused from state_freq_analysis.py)
# =============================================================================

def compute_token_variance(state_history):
    state_stack = torch.stack(state_history)
    token_mean = state_stack.mean(dim=0)
    token_var = ((state_stack - token_mean) ** 2).mean(dim=(0, 2))
    return token_var


def project_freq_to_image(token_var, cross_attn, img_shape_patches):
    weighted = (token_var.unsqueeze(1) * cross_attn).sum(dim=0)
    H_p, W_p = img_shape_patches
    return weighted.reshape(H_p, W_p).numpy().astype(np.float32)


# =============================================================================
# Depth error computation
# =============================================================================

def extract_pred_depth(res):
    """Extract depth (z-component) from model prediction."""
    pts3d = res["pts3d_in_self_view"]  # (B, H, W, 3)
    return pts3d[0, :, :, 2].numpy()   # (H, W)


def align_and_compute_error(pred_depth, gt_depth, max_depth, patch_size):
    """
    Scale-align predicted depth to GT (median ratio), compute abs_rel error,
    and aggregate to patch level.

    Returns:
        pixel_error: (H, W) absolute relative error (NaN for invalid)
        patch_error: (H_p, W_p) patch-level mean error (NaN for invalid patches)
        valid_ratio: fraction of valid pixels
    """
    H_pred, W_pred = pred_depth.shape

    # Resize GT to prediction resolution
    gt_resized = cv2.resize(gt_depth, (W_pred, H_pred),
                            interpolation=cv2.INTER_NEAREST)

    # Valid mask
    valid = (gt_resized > 0) & (gt_resized < max_depth) & (pred_depth > 1e-3)
    valid_ratio = valid.sum() / valid.size

    if valid.sum() < 10:
        pixel_error = np.full_like(pred_depth, np.nan)
        H_p, W_p = H_pred // patch_size, W_pred // patch_size
        patch_error = np.full((H_p, W_p), np.nan)
        return pixel_error, patch_error, valid_ratio

    # Median-ratio scale alignment
    scale = np.median(gt_resized[valid]) / np.median(pred_depth[valid])
    pred_aligned = pred_depth * scale

    # Absolute relative error
    pixel_error = np.full_like(pred_depth, np.nan)
    pixel_error[valid] = np.abs(pred_aligned[valid] - gt_resized[valid]) / gt_resized[valid]

    # Aggregate to patch level
    H_p = H_pred // patch_size
    W_p = W_pred // patch_size
    H_crop = H_p * patch_size
    W_crop = W_p * patch_size
    err_cropped = pixel_error[:H_crop, :W_crop]
    err_patches = err_cropped.reshape(H_p, patch_size, W_p, patch_size)
    err_patches = err_patches.transpose(0, 2, 1, 3).reshape(H_p, W_p, -1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        patch_error = np.nanmean(err_patches, axis=-1)

    return pixel_error, patch_error, valid_ratio


# =============================================================================
# Visualization
# =============================================================================

def upsample_heatmap(hm, target_hw):
    t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    t_up = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t_up[0, 0].numpy()


def normalize(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def plot_scatter(all_freq, all_error, out_path):
    """Hexbin scatter of frequency vs error across all frames."""
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(all_freq, all_error, gridsize=60, cmap="YlOrRd",
                   mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, label="Count")

    # Regression line
    mask = np.isfinite(all_freq) & np.isfinite(all_error)
    if mask.sum() > 2:
        z = np.polyfit(all_freq[mask], all_error[mask], 1)
        x_line = np.linspace(np.nanmin(all_freq), np.nanmax(all_freq), 100)
        ax.plot(x_line, np.polyval(z, x_line), "b--", linewidth=1.5, label="Linear fit")
        r_p, _ = pearsonr(all_freq[mask], all_error[mask])
        r_s, _ = spearmanr(all_freq[mask], all_error[mask])
        ax.set_title(f"Frequency vs Depth Error  (Pearson r={r_p:.3f}, Spearman ρ={r_s:.3f})",
                     fontsize=12)
        ax.legend()

    ax.set_xlabel("Patch frequency (variance-weighted attention)")
    ax.set_ylabel("Patch depth error (abs_rel)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


def plot_correlation_over_time(pearson_rs, spearman_rs, out_path):
    """Per-frame correlation coefficients over time."""
    T = len(pearson_rs)
    frames = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, pearson_rs, "o-", markersize=2, linewidth=1, label="Pearson r", alpha=0.8)
    ax.plot(frames, spearman_rs, "s-", markersize=2, linewidth=1, label="Spearman ρ", alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    valid_p = [x for x in pearson_rs if np.isfinite(x)]
    valid_s = [x for x in spearman_rs if np.isfinite(x)]
    if valid_p:
        ax.axhline(np.mean(valid_p), color="C0", linestyle=":", linewidth=1,
                   label=f"Mean Pearson = {np.mean(valid_p):.3f}")
    if valid_s:
        ax.axhline(np.mean(valid_s), color="C1", linestyle=":", linewidth=1,
                   label=f"Mean Spearman = {np.mean(valid_s):.3f}")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("Frequency–Error correlation over time")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


def save_side_by_side(img_paths, freq_maps, error_maps, img_shapes_list,
                      out_dir, size, patch_size, max_save=50):
    """Save 4-panel figures: RGB | freq map | error map | overlay."""
    os.makedirs(out_dir, exist_ok=True)
    from dust3r.utils.image import load_images

    n_frames = min(len(freq_maps), max_save)
    step = max(1, len(freq_maps) // n_frames)
    indices = list(range(0, len(freq_maps), step))[:n_frames]

    images = load_images([img_paths[i] for i in indices], size=size)

    for save_idx, t in enumerate(tqdm(indices, desc="Saving side-by-side")):
        img_np = images[save_idx]["img"][0].permute(1, 2, 0).numpy()
        img_np = ((img_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        H, W = img_np.shape[:2]

        freq_up = normalize(upsample_heatmap(freq_maps[t], (H, W)))
        err_up = upsample_heatmap(error_maps[t], (H, W))
        err_up_n = normalize(err_up)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        im1 = axes[1].imshow(freq_up, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("Frequency map")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(err_up_n, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Depth error (abs_rel)")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Overlay: freq (red) + error (green)
        overlay = np.stack([freq_up, err_up_n, np.zeros_like(freq_up)], axis=-1)
        overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
        blended = ((0.5 * img_np.astype(np.float32)
                     + 0.5 * overlay.astype(np.float32))
                    .clip(0, 255).astype(np.uint8))
        axes[3].imshow(blended)
        axes[3].set_title("Overlay (R=freq, G=error)")
        axes[3].axis("off")

        fig.suptitle(f"Frame {t:04d}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"frame_{t:06d}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


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

    # ── Derive depth directory ─────────────────────────────────────────────────
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

    # Verify depth files exist
    depth_paths = [color_to_depth_path(p, depth_dir) for p in img_paths]
    missing = [p for p in depth_paths if not os.path.exists(p)]
    if missing:
        print(f"[warn] {len(missing)} depth files missing; first: {missing[0]}")

    views = build_views(img_paths, args.size)

    # ── Inference ──────────────────────────────────────────────────────────────
    print("[inference] Running forward_recurrent_analysis …")
    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(views, device=device)

    state_history = analysis_data["state_history"]
    cross_attn_history = analysis_data["cross_attn_history"]
    img_shapes_list = analysis_data["img_shapes"]

    T = len(state_history)
    print(f"[data] T={T} frames, cross_attn captured: {len(cross_attn_history)}")

    # ── Token variance ─────────────────────────────────────────────────────────
    token_var = compute_token_variance(state_history)

    # ── Per-frame analysis ─────────────────────────────────────────────────────
    print("[analysis] Computing per-frame frequency-error correlation …")

    pearson_rs = []
    spearman_rs = []
    all_freq_patches = []
    all_error_patches = []
    freq_maps = []
    error_maps = []
    valid_ratios = []

    for t in tqdm(range(min(T, len(cross_attn_history))), desc="Frames"):
        # Frequency map
        freq_map = project_freq_to_image(
            token_var, cross_attn_history[t], img_shapes_list[t])
        freq_maps.append(freq_map)

        # Predicted depth
        pred_depth = extract_pred_depth(ress[t])

        # GT depth
        if t < len(depth_paths) and os.path.exists(depth_paths[t]):
            gt_depth = load_gt_depth(depth_paths[t], args.depth_scale)
        else:
            gt_depth = None

        if gt_depth is None:
            error_maps.append(np.full_like(freq_map, np.nan))
            pearson_rs.append(np.nan)
            spearman_rs.append(np.nan)
            valid_ratios.append(0.0)
            continue

        # Compute error
        pixel_error, patch_error, valid_ratio = align_and_compute_error(
            pred_depth, gt_depth, args.max_depth, args.patch_size)

        error_maps.append(patch_error)
        valid_ratios.append(valid_ratio)

        # Correlation (flatten, mask NaN)
        f_flat = freq_map.ravel()
        e_flat = patch_error.ravel()
        mask = np.isfinite(f_flat) & np.isfinite(e_flat)

        if mask.sum() > 10:
            r_p, _ = pearsonr(f_flat[mask], e_flat[mask])
            r_s, _ = spearmanr(f_flat[mask], e_flat[mask])
            pearson_rs.append(r_p)
            spearman_rs.append(r_s)
            all_freq_patches.append(f_flat[mask])
            all_error_patches.append(e_flat[mask])
        else:
            pearson_rs.append(np.nan)
            spearman_rs.append(np.nan)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    if all_freq_patches:
        all_freq = np.concatenate(all_freq_patches)
        all_error = np.concatenate(all_error_patches)
        agg_pearson, _ = pearsonr(all_freq, all_error)
        agg_spearman, _ = spearmanr(all_freq, all_error)
    else:
        all_freq, all_error = np.array([]), np.array([])
        agg_pearson, agg_spearman = np.nan, np.nan

    valid_pearson = [x for x in pearson_rs if np.isfinite(x)]
    valid_spearman = [x for x in spearman_rs if np.isfinite(x)]

    # ── Summary ────────────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 60,
        "Experiment A: Frequency vs Reconstruction Error Correlation",
        "=" * 60,
        f"Sequence:          {args.seq_path}",
        f"Model:             {args.model_update_type}",
        f"Frames processed:  {T}",
        f"Valid frames:      {len(valid_pearson)}/{T}",
        f"Mean valid ratio:  {np.mean(valid_ratios):.3f}",
        "",
        "--- Aggregate correlation (all patches pooled) ---",
        f"  Pearson r  = {agg_pearson:.4f}",
        f"  Spearman ρ = {agg_spearman:.4f}",
        f"  N patches  = {len(all_freq)}",
        "",
        "--- Per-frame correlation (mean ± std) ---",
        f"  Pearson r  = {np.mean(valid_pearson):.4f} ± {np.std(valid_pearson):.4f}"
        if valid_pearson else "  Pearson r  = N/A",
        f"  Spearman ρ = {np.mean(valid_spearman):.4f} ± {np.std(valid_spearman):.4f}"
        if valid_spearman else "  Spearman ρ = N/A",
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    summary_path = os.path.join(args.output_dir, "correlation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")

    # ── Save raw data ──────────────────────────────────────────────────────────
    npz_path = os.path.join(args.output_dir, "freq_error_data.npz")
    np.savez_compressed(
        npz_path,
        token_var=token_var.numpy(),
        pearson_rs=np.array(pearson_rs),
        spearman_rs=np.array(spearman_rs),
        all_freq=all_freq,
        all_error=all_error,
        valid_ratios=np.array(valid_ratios),
    )
    print(f"[save] Raw data → {npz_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if len(all_freq) > 0:
        plot_scatter(all_freq, all_error,
                     os.path.join(plots_dir, "scatter_freq_vs_error.png"))

    plot_correlation_over_time(
        pearson_rs, spearman_rs,
        os.path.join(plots_dir, "correlation_over_time.png"))

    if cross_attn_history and error_maps:
        save_side_by_side(
            img_paths, freq_maps, error_maps, img_shapes_list,
            os.path.join(plots_dir, "side_by_side"),
            args.size, args.patch_size)

    print(f"\n[done] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
