"""
Experiment 1: State Token Frequency Visualization
==================================================

Hypothesis:
    State tokens with high temporal variance (high "temporal frequency") correspond
    to unstable or challenging scene regions — dynamic objects, occlusion boundaries,
    and scene transitions. Low-variance tokens represent stable, well-converged scene
    elements.

What this script produces
--------------------------
output_dir/
  freq_heatmaps/
    frame_000000.png   – per-frame: original image + variance-weighted attention overlay
    ...
  plots/
    token_variance_hist.png    – histogram of per-token temporal variance
    token_variance_evolution.png – variance evolution for top-K / bottom-K tokens
    high_low_token_attention.png – where high-freq vs low-freq tokens attend (time-avg)
  state_freq_data.npz          – raw arrays for further analysis

Usage
-----
python analysis/state_freq_analysis.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/taylor.mp4 \
    --output_dir analysis_results/exp1 \
    --model_update_type ttt3r \
    --size 512 \
    --frame_interval 1 \
    --max_frames 200

Notes
-----
- Run from the project root (TTT3R/).
- Requires the model checkpoint at --model_path.
- Both video files and image directories are accepted for --seq_path.
"""

import os
import sys
import argparse
import glob
import tempfile
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    p = argparse.ArgumentParser(description="Exp 1 – State Token Frequency Analysis")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--seq_path",   type=str, required=True,
                   help="Path to a video file or a directory of images.")
    p.add_argument("--output_dir", type=str, default="analysis_results/exp1")
    p.add_argument("--model_update_type", type=str, default="ttt3r",
                   choices=["cut3r", "ttt3r"])
    p.add_argument("--size",           type=int, default=512)
    p.add_argument("--frame_interval", type=int, default=1,
                   help="Use every Nth frame (1 = every frame).")
    p.add_argument("--max_frames",     type=int, default=200,
                   help="Maximum number of frames to process.")
    p.add_argument("--top_k_tokens",   type=int, default=20,
                   help="Number of high / low variance tokens to visualise.")
    p.add_argument("--window_size",    type=int, default=10,
                   help="Sliding window (frames) for running-variance evolution plot.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# =============================================================================
# Input helpers
# =============================================================================

def load_img_paths(seq_path: str, frame_interval: int, max_frames: int):
    """Return (list_of_img_paths, tmp_dir_or_None)."""
    tmpdirname = None
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    if os.path.isdir(seq_path):
        all_paths = sorted(glob.glob(os.path.join(seq_path, "*")))
        img_paths = [p for p in all_paths if os.path.splitext(p)[1].lower() in img_exts]
        img_paths = img_paths[::frame_interval]
    else:
        cap = cv2.VideoCapture(seq_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {seq_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, frame_interval))
        tmpdirname = tempfile.mkdtemp()
        img_paths = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            path = os.path.join(tmpdirname, f"frame_{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            img_paths.append(path)
        cap.release()

    img_paths = img_paths[:max_frames]
    print(f"[load] Using {len(img_paths)} frames (interval={frame_interval})")
    return img_paths, tmpdirname


def build_views(img_paths, size, reset_interval=10_000_000):
    """Replicate demo.py's prepare_input for a plain image list."""
    from dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []
    for i, img_dict in enumerate(images):
        view = {
            "img": img_dict["img"],
            "ray_map": torch.full(
                (img_dict["img"].shape[0], 6,
                 img_dict["img"].shape[-2], img_dict["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(img_dict["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask":  torch.tensor(True).unsqueeze(0),
            "ray_mask":  torch.tensor(False).unsqueeze(0),
            "update":    torch.tensor(True).unsqueeze(0),
            "reset":     torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
        }
        views.append(view)
    return views


# =============================================================================
# Core analysis computations
# =============================================================================

def compute_token_variance(state_history):
    """
    Args:
        state_history: list of T tensors, each [n_state, dec_dim]
    Returns:
        token_var: [n_state]  – mean squared deviation averaged over time and feature dim
    """
    state_stack = torch.stack(state_history)     # [T, n_state, dec_dim]
    token_mean  = state_stack.mean(dim=0)        # [n_state, dec_dim]
    token_var   = ((state_stack - token_mean) ** 2).mean(dim=(0, 2))  # [n_state]
    return token_var


def compute_running_variance(state_history, window_size):
    """
    Sliding-window temporal variance per token.

    Args:
        state_history: list of T tensors [n_state, dec_dim]
        window_size:   int
    Returns:
        running_var: [T, n_state]  (NaN for frames before the first full window)
    """
    T = len(state_history)
    n_state = state_history[0].shape[0]
    running_var = torch.full((T, n_state), float("nan"))

    state_stack = torch.stack(state_history)     # [T, n_state, dec_dim]
    for t in range(window_size - 1, T):
        window = state_stack[t - window_size + 1 : t + 1]   # [W, n_state, dec_dim]
        mu = window.mean(dim=0)
        var = ((window - mu) ** 2).mean(dim=(0, 2))          # [n_state]
        running_var[t] = var

    return running_var


def project_freq_to_image(token_var, cross_attn, img_shape_patches):
    """
    Compute a spatial frequency heatmap for one frame.

    freq_map[patch] = sum_k ( token_var[k] * cross_attn[k, patch] )

    Args:
        token_var:        [n_state]               – global token variance
        cross_attn:       [n_state, n_img_patches] – per-frame attention
        img_shape_patches: (H_p, W_p)
    Returns:
        freq_map_2d: [H_p, W_p]  (numpy float32)
    """
    # Weight attention maps by token variance and sum
    weighted = (token_var.unsqueeze(1) * cross_attn).sum(dim=0)   # [n_patches]
    H_p, W_p = img_shape_patches
    freq_map_2d = weighted.reshape(H_p, W_p).numpy().astype(np.float32)
    return freq_map_2d


def upsample_heatmap(heatmap_2d, target_hw):
    """Bilinearly upsample a (H_p, W_p) heatmap to target_hw = (H, W)."""
    t = torch.from_numpy(heatmap_2d).unsqueeze(0).unsqueeze(0)   # [1,1,H_p,W_p]
    t_up = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t_up[0, 0].numpy()


def normalize_heatmap(heatmap):
    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - mn) / (mx - mn)


def overlay_heatmap_on_image(img_rgb, heatmap_norm, alpha=0.55, colormap="inferno"):
    """
    Args:
        img_rgb:      [H, W, 3] uint8 numpy array
        heatmap_norm: [H, W]    float in [0, 1]
    Returns:
        blended: [H, W, 3] uint8
    """
    cmap = cm.get_cmap(colormap)
    heat_rgba = (cmap(heatmap_norm) * 255).astype(np.uint8)   # [H, W, 4]
    heat_rgb  = heat_rgba[..., :3]
    blended   = ((1 - alpha) * img_rgb.astype(np.float32)
                 + alpha * heat_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return blended


# =============================================================================
# Visualisation helpers
# =============================================================================

def save_freq_heatmaps(img_paths, cross_attn_history, img_shapes_list, token_var,
                       out_dir, size):
    """Save per-frame heatmap PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    from dust3r.utils.image import load_images

    # Load original images at inference size for display
    images = load_images(img_paths, size=size)

    for t, (cross_attn, img_shape_patches) in enumerate(
            tqdm(zip(cross_attn_history, img_shapes_list),
                 total=len(cross_attn_history), desc="Saving heatmaps")):

        # Original image  [H, W, 3] in [0,1]
        img_np = images[t]["img"][0].permute(1, 2, 0).numpy()  # [-1,1] → need to rescale
        img_np = ((img_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        H, W   = img_np.shape[:2]

        freq_map = project_freq_to_image(token_var, cross_attn, img_shape_patches)
        freq_up  = upsample_heatmap(freq_map, (H, W))
        freq_n   = normalize_heatmap(freq_up)
        blended  = overlay_heatmap_on_image(img_np, freq_n)

        # Side-by-side: original | heatmap overlay
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original frame")
        axes[0].axis("off")

        axes[1].imshow(blended)
        axes[1].set_title("State-token frequency map")
        axes[1].axis("off")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="inferno",
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04,
                     label="Variance-weighted attention (normalised)")

        fig.suptitle(f"Frame {t:04d}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"frame_{t:06d}.png"), dpi=120,
                    bbox_inches="tight")
        plt.close(fig)


def plot_token_variance_histogram(token_var, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_var.numpy(), bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Temporal variance (mean over feature dim)")
    ax.set_ylabel("Number of state tokens")
    ax.set_title("Distribution of state-token temporal variance")
    ax.axvline(token_var.mean().item(), color="crimson", linestyle="--",
               label=f"Mean = {token_var.mean():.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


def plot_token_variance_evolution(running_var, top_k, out_path):
    """
    Plot running-window variance over time for the top-K and bottom-K tokens
    (ranked by overall temporal variance = mean of running_var, ignoring NaN).
    """
    # mean over valid (non-NaN) frames
    mean_var = torch.nanmean(running_var, dim=0)   # [n_state]
    ranked   = mean_var.argsort(descending=True)

    top_idx = ranked[:top_k].tolist()
    bot_idx = ranked[-top_k:].tolist()

    T = running_var.shape[0]
    frames = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for idx in top_idx:
        axes[0].plot(frames, running_var[:, idx].numpy(), alpha=0.6, linewidth=0.8)
    axes[0].set_title(f"Top-{top_k} highest-variance tokens (unstable)")
    axes[0].set_ylabel("Running variance")

    for idx in bot_idx:
        axes[1].plot(frames, running_var[:, idx].numpy(), alpha=0.6, linewidth=0.8)
    axes[1].set_title(f"Bottom-{top_k} lowest-variance tokens (stable)")
    axes[1].set_ylabel("Running variance")
    axes[1].set_xlabel("Frame index")

    plt.suptitle("State-token temporal variance evolution", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


def plot_high_low_token_attention(token_var, cross_attn_history, img_shapes_list,
                                  top_k, out_path):
    """
    Time-averaged attention maps for the top-K (high-freq) and bottom-K (low-freq) tokens.
    Shows WHERE in the image unstable vs stable tokens tend to look.
    """
    ranked  = token_var.argsort(descending=True)
    top_idx = ranked[:top_k]
    bot_idx = ranked[-top_k:]

    # Accumulate time-averaged attention for each group
    # Use the first frame's shape as reference (resize all to it)
    H_ref, W_ref = img_shapes_list[0]

    top_avg = torch.zeros(H_ref, W_ref)
    bot_avg = torch.zeros(H_ref, W_ref)
    count   = 0

    for cross_attn, (H_p, W_p) in zip(cross_attn_history, img_shapes_list):
        # [top_k, n_patches] → [top_k, H_p, W_p]
        n_p = cross_attn.shape[1]

        top_map = cross_attn[top_idx].mean(dim=0).reshape(H_p, W_p)  # [H_p, W_p]
        bot_map = cross_attn[bot_idx].mean(dim=0).reshape(H_p, W_p)

        # Resize to reference patch resolution if needed
        if (H_p, W_p) != (H_ref, W_ref):
            top_map = F.interpolate(top_map.unsqueeze(0).unsqueeze(0),
                                    size=(H_ref, W_ref), mode="bilinear",
                                    align_corners=False)[0, 0]
            bot_map = F.interpolate(bot_map.unsqueeze(0).unsqueeze(0),
                                    size=(H_ref, W_ref), mode="bilinear",
                                    align_corners=False)[0, 0]

        top_avg += top_map
        bot_avg += bot_map
        count   += 1

    top_avg = (top_avg / count).numpy()
    bot_avg = (bot_avg / count).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(top_avg, cmap="hot")
    axes[0].set_title(f"High-freq tokens (top {top_k})\nWhere they attend")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(bot_avg, cmap="cool")
    axes[1].set_title(f"Low-freq tokens (bottom {top_k})\nWhere they attend")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Time-averaged attention: high-freq vs low-freq state tokens", fontsize=13)
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

    # ── Model + imports ───────────────────────────────────────────────────────
    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    print(f"[model] Loading from {args.model_path} …")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    # ── Data ─────────────────────────────────────────────────────────────────
    img_paths, tmpdirname = load_img_paths(
        args.seq_path, args.frame_interval, args.max_frames
    )
    if not img_paths:
        raise RuntimeError(f"No images found at {args.seq_path}")

    views = build_views(img_paths, args.size)

    # ── Inference with analysis ───────────────────────────────────────────────
    print("[inference] Running forward_recurrent_analysis …")
    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(views, device=device)

    if tmpdirname:
        shutil.rmtree(tmpdirname)

    state_history      = analysis_data["state_history"]       # list[T] of [n_state, dec_dim]
    cross_attn_history = analysis_data["cross_attn_history"]  # list[T] of [n_state, n_patches]
    img_shapes_list    = analysis_data["img_shapes"]          # list[T] of (H_p, W_p)

    T       = len(state_history)
    n_state = state_history[0].shape[0]
    print(f"[data] T={T} frames, n_state={n_state} tokens")
    print(f"[data] cross_attn frames captured: {len(cross_attn_history)}")

    # ── Compute statistics ────────────────────────────────────────────────────
    print("[analysis] Computing token temporal variance …")
    token_var    = compute_token_variance(state_history)           # [n_state]
    running_var  = compute_running_variance(state_history, args.window_size)  # [T, n_state]

    print(f"[analysis] Token variance – min={token_var.min():.5f}  "
          f"mean={token_var.mean():.5f}  max={token_var.max():.5f}")

    # ── Save raw data ─────────────────────────────────────────────────────────
    npz_path = os.path.join(args.output_dir, "state_freq_data.npz")
    np.savez_compressed(
        npz_path,
        token_var   = token_var.numpy(),                                      # [n_state]
        running_var = running_var.numpy(),                                     # [T, n_state]
        state_stack = torch.stack(state_history).numpy(),                     # [T, n_state, dec_dim]
        cross_attn  = torch.stack(cross_attn_history).numpy()                 # [T, n_state, n_patches]
            if cross_attn_history else np.array([]),
        img_shapes  = np.array(img_shapes_list),                              # [T, 2]
    )
    print(f"[save] Raw data → {npz_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_token_variance_histogram(
        token_var,
        os.path.join(plots_dir, "token_variance_hist.png")
    )
    plot_token_variance_evolution(
        running_var, args.top_k_tokens,
        os.path.join(plots_dir, "token_variance_evolution.png")
    )
    if cross_attn_history:
        plot_high_low_token_attention(
            token_var, cross_attn_history, img_shapes_list,
            args.top_k_tokens,
            os.path.join(plots_dir, "high_low_token_attention.png")
        )

    # ── Per-frame heatmaps ────────────────────────────────────────────────────
    if cross_attn_history:
        save_freq_heatmaps(
            img_paths[:len(cross_attn_history)],
            cross_attn_history,
            img_shapes_list,
            token_var,
            os.path.join(args.output_dir, "freq_heatmaps"),
            args.size,
        )

    print(f"\n[done] All outputs saved to: {args.output_dir}")
    print("Summary of output files:")
    print(f"  {args.output_dir}/state_freq_data.npz")
    print(f"  {args.output_dir}/plots/token_variance_hist.png")
    print(f"  {args.output_dir}/plots/token_variance_evolution.png")
    print(f"  {args.output_dir}/plots/high_low_token_attention.png")
    print(f"  {args.output_dir}/freq_heatmaps/frame_XXXXXX.png  (x{len(cross_attn_history)})")


if __name__ == "__main__":
    main()
