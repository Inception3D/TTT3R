"""
Spectral Analysis of State Token Dynamics
==========================================

Core analysis for understanding state token behavior in frequency domain.
Goes beyond simple variance (Exp1) by decomposing token trajectories into
frequency bands and correlating spectral features with reconstruction quality.

Key questions:
  1. What do state token power spectra look like? (stable vs unstable tokens)
  2. Does high-frequency energy (oscillation) predict reconstruction error?
  3. Does low-frequency change (drift) predict scene transitions?
  4. Do spectral features carry information beyond simple variance?

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/spectral_analysis.py \
    --model_path /path/to/model.pth \
    --seq_path /path/to/scene/color \
    --output_dir analysis_results/spectral \
    --model_update_type ttt3r \
    --size 512 --frame_interval 5 --max_frames 200
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
from scipy.signal import stft
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Spectral Analysis of State Tokens")
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--seq_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="analysis_results/spectral")
    p.add_argument("--model_update_type", type=str, default="ttt3r",
                   choices=["cut3r", "ttt3r"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--frame_interval", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--depth_scale", type=float, default=1000.0)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--window_size", type=int, default=32,
                   help="STFT window size (frames).")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# =============================================================================
# Input helpers
# =============================================================================

def load_img_paths(seq_path, frame_interval, max_frames):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    all_paths = sorted(glob.glob(os.path.join(seq_path, "*")))
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


def derive_depth_dir(seq_path):
    color_dir = os.path.normpath(seq_path)
    parent = os.path.dirname(color_dir)
    base = os.path.basename(color_dir)
    # ScanNet: color -> depth; TUM: rgb -> depth
    for src in ("color", "rgb"):
        if src in base:
            return os.path.join(parent, base.replace(src, "depth", 1))
    return os.path.join(parent, "depth")


def load_tum_associations(scene_dir):
    """Load TUM associations.txt to build rgb->depth filename mapping."""
    assoc_path = os.path.join(scene_dir, "associations.txt")
    if not os.path.exists(assoc_path):
        return None
    mapping = {}
    with open(assoc_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                # format: ts_rgb rgb/file ts_depth depth/file
                rgb_file = os.path.basename(parts[1])
                depth_file = os.path.basename(parts[3])
                mapping[rgb_file] = depth_file
    return mapping if mapping else None


def build_tum_timestamp_index(depth_dir):
    """Build sorted list of (timestamp, filename) for TUM depth images."""
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    index = []
    for f in depth_files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            ts = float(name)
            index.append((ts, f))
        except ValueError:
            continue
    return index


def find_gt_depth_path(img_path, depth_dir, tum_assoc=None,
                       tum_depth_index=None, max_dt=0.05):
    """Find corresponding GT depth path for an image."""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    # Direct match (ScanNet style)
    direct = os.path.join(depth_dir, basename + ".png")
    if os.path.exists(direct):
        return direct
    # TUM association match
    if tum_assoc is not None:
        rgb_file = os.path.basename(img_path)
        if rgb_file in tum_assoc:
            depth_path = os.path.join(depth_dir, tum_assoc[rgb_file])
            if os.path.exists(depth_path):
                return depth_path
    # TUM timestamp nearest-neighbor match
    if tum_depth_index is not None:
        try:
            rgb_ts = float(basename)
        except ValueError:
            return None
        # Binary search for closest timestamp
        timestamps = [t for t, _ in tum_depth_index]
        idx = np.searchsorted(timestamps, rgb_ts)
        best_path, best_dt = None, max_dt
        for i in [idx - 1, idx]:
            if 0 <= i < len(tum_depth_index):
                dt = abs(tum_depth_index[i][0] - rgb_ts)
                if dt < best_dt:
                    best_dt = dt
                    best_path = tum_depth_index[i][1]
        return best_path
    return None


# =============================================================================
# Spectral Analysis
# =============================================================================

def compute_token_power_spectra(state_stack):
    """
    Compute power spectrum for each state token.

    Args:
        state_stack: [T, n_state, D] numpy array
    Returns:
        freqs: [T//2+1] normalized frequencies
        power_spectra: [n_state, T//2+1] power averaged over feature dims
    """
    T, n_state, D = state_stack.shape

    # FFT per token per feature dim, then average power over dims
    power_spectra = np.zeros((n_state, T // 2 + 1))

    for k in range(n_state):
        # [T, D] trajectory for token k
        traj = state_stack[:, k, :]
        # Remove DC (mean) per feature dim
        traj = traj - traj.mean(axis=0, keepdims=True)
        # FFT along time axis
        fft_vals = np.fft.rfft(traj, axis=0)  # [T//2+1, D]
        power = np.abs(fft_vals) ** 2          # [T//2+1, D]
        power_spectra[k] = power.mean(axis=1)  # average over feature dims

    freqs = np.fft.rfftfreq(T)  # normalized frequencies [0, 0.5]
    return freqs, power_spectra


def compute_band_energies(freqs, power_spectra):
    """
    Decompose power spectrum into frequency bands.

    Returns:
        band_energies: dict of {band_name: [n_state] array}
    """
    n_state = power_spectra.shape[0]

    # Define bands (fraction of Nyquist = 0.5)
    bands = {
        "dc": (0, 0.01),
        "low": (0.01, 0.1),
        "mid": (0.1, 0.25),
        "high": (0.25, 0.5),
    }

    band_energies = {}
    for name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if mask.sum() == 0:
            band_energies[name] = np.zeros(n_state)
        else:
            band_energies[name] = power_spectra[:, mask].sum(axis=1)

    # Derived features
    total = power_spectra[:, 1:].sum(axis=1) + 1e-10  # exclude DC
    band_energies["high_ratio"] = band_energies["high"] / total
    band_energies["low_ratio"] = band_energies["low"] / total
    band_energies["total"] = total

    return band_energies


def compute_running_spectral_features(state_stack, window_size):
    """
    Sliding-window spectral analysis for frame-level features.

    Args:
        state_stack: [T, n_state, D]
        window_size: int
    Returns:
        frame_high_energy: [T] mean high-freq energy across tokens at each frame
        frame_low_energy:  [T] mean low-freq energy
        frame_spectral_ratio: [T] high/total ratio
    """
    T, n_state, D = state_stack.shape
    frame_high = np.full(T, np.nan)
    frame_low = np.full(T, np.nan)
    frame_ratio = np.full(T, np.nan)

    for t in range(window_size - 1, T):
        window = state_stack[t - window_size + 1:t + 1]  # [W, n_state, D]
        window = window - window.mean(axis=0, keepdims=True)

        # FFT for each token, average over D
        fft_vals = np.fft.rfft(window, axis=0)  # [W//2+1, n_state, D]
        power = np.abs(fft_vals) ** 2             # [W//2+1, n_state, D]
        power = power.mean(axis=2)                # [W//2+1, n_state]

        freqs = np.fft.rfftfreq(window_size)
        low_mask = (freqs >= 0.01) & (freqs < 0.1)
        high_mask = (freqs >= 0.25) & (freqs <= 0.5)
        total_mask = freqs > 0

        low_e = power[low_mask].sum(axis=0) if low_mask.sum() > 0 else np.zeros(n_state)
        high_e = power[high_mask].sum(axis=0) if high_mask.sum() > 0 else np.zeros(n_state)
        total_e = power[total_mask].sum(axis=0) + 1e-10

        frame_high[t] = high_e.mean()
        frame_low[t] = low_e.mean()
        frame_ratio[t] = (high_e / total_e).mean()

    return frame_high, frame_low, frame_ratio


def compute_online_spectral_features(state_stack, momentum=0.95):
    """
    Online (causal) spectral decomposition using exponential moving average.
    This simulates what we'd actually do at inference time.

    Returns:
        high_freq_energy: [T, n_state] per-token high-freq energy
        low_freq_signal:  [T, n_state, D] low-pass filtered state
    """
    T, n_state, D = state_stack.shape
    ema = state_stack[0].copy()  # [n_state, D]
    high_freq_energy = np.zeros((T, n_state))
    low_freq_signal = np.zeros_like(state_stack)

    low_freq_signal[0] = ema

    for t in range(1, T):
        ema = momentum * ema + (1 - momentum) * state_stack[t]
        low_freq_signal[t] = ema
        high_freq = state_stack[t] - ema
        high_freq_energy[t] = np.linalg.norm(high_freq, axis=-1)  # [n_state]

    return high_freq_energy, low_freq_signal


# =============================================================================
# Visualization
# =============================================================================

def plot_power_spectra(freqs, power_spectra, band_energies, out_dir):
    """Plot average power spectrum and stable vs unstable token comparison."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Average power spectrum (log scale)
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_power = power_spectra.mean(axis=0)
    ax.semilogy(freqs[1:], mean_power[1:], color="steelblue", linewidth=1.2)
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Average power spectrum of state token trajectories")
    # Mark band boundaries
    for f in [0.01, 0.1, 0.25]:
        ax.axvline(f, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.text(0.05, ax.get_ylim()[1] * 0.5, "Low", fontsize=8, ha="center")
    ax.text(0.175, ax.get_ylim()[1] * 0.5, "Mid", fontsize=8, ha="center")
    ax.text(0.375, ax.get_ylim()[1] * 0.5, "High", fontsize=8, ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avg_power_spectrum.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Stable vs unstable tokens
    total = band_energies["total"]
    ranked = np.argsort(total)
    top_20 = ranked[-20:]   # most energetic (unstable)
    bot_20 = ranked[:20]    # least energetic (stable)

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx in top_20:
        ax.semilogy(freqs[1:], power_spectra[idx, 1:], color="C3",
                    alpha=0.15, linewidth=0.5)
    for idx in bot_20:
        ax.semilogy(freqs[1:], power_spectra[idx, 1:], color="C0",
                    alpha=0.15, linewidth=0.5)
    ax.semilogy(freqs[1:], power_spectra[top_20].mean(axis=0)[1:],
                color="C3", linewidth=2, label="Top-20 unstable (mean)")
    ax.semilogy(freqs[1:], power_spectra[bot_20].mean(axis=0)[1:],
                color="C0", linewidth=2, label="Top-20 stable (mean)")
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Power spectra: stable vs unstable tokens")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stable_vs_unstable_spectra.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Band energy distribution
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, band in zip(axes, ["dc", "low", "mid", "high"]):
        ax.hist(np.log10(band_energies[band] + 1e-10), bins=40,
                color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{band} band (log10)")
        ax.set_xlabel("log10(energy)")
        ax.set_ylabel("# tokens")
    plt.suptitle("Distribution of per-token band energies", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_energy_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. High-freq ratio vs variance scatter
    token_var = band_energies["total"]  # proxy for variance
    high_ratio = band_energies["high_ratio"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(token_var, high_ratio, s=5, alpha=0.5, color="steelblue")
    ax.set_xlabel("Total spectral energy (≈ variance)")
    ax.set_ylabel("High-frequency ratio")
    ax.set_title("Total energy vs High-freq ratio per token")
    mask = np.isfinite(token_var) & np.isfinite(high_ratio)
    if mask.sum() > 5:
        r, _ = pearsonr(token_var[mask], high_ratio[mask])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=10, verticalalignment="top")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_vs_highfreq_ratio.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frame_level_spectral(frames, depth_errors, frame_high, frame_low,
                               frame_ratio, online_high_mean, out_dir):
    """Frame-level spectral features vs depth error."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Time series overlay
    def norm01(x):
        x = x.copy()
        m = np.isfinite(x)
        if m.sum() == 0: return x
        mn, mx = x[m].min(), x[m].max()
        if mx - mn > 1e-8: x[m] = (x[m] - mn) / (mx - mn)
        return x

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(frames, norm01(depth_errors), "C3-", linewidth=1, alpha=0.8,
            label="Depth error")
    ax.plot(frames, norm01(frame_high), "C0-", linewidth=1, alpha=0.8,
            label="High-freq energy (windowed)")
    ax.plot(frames, norm01(online_high_mean), "C2--", linewidth=1, alpha=0.8,
            label="High-freq energy (online EMA)")
    ax.plot(frames, norm01(frame_ratio), "C4:", linewidth=1, alpha=0.6,
            label="High/total ratio")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Normalized value")
    ax.set_title("Spectral features vs depth error over time")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectral_vs_error_timeseries.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Scatter plots: each spectral feature vs depth error
    spectral_signals = {
        "High-freq energy (window)": frame_high,
        "High-freq energy (online)": online_high_mean,
        "Low-freq energy (window)": frame_low,
        "High/total ratio": frame_ratio,
    }

    fig, axes = plt.subplots(1, len(spectral_signals), figsize=(5 * len(spectral_signals), 4))
    for ax, (name, signal) in zip(axes, spectral_signals.items()):
        mask = np.isfinite(signal) & np.isfinite(depth_errors)
        if mask.sum() > 5:
            ax.scatter(signal[mask], depth_errors[mask], s=10, alpha=0.5,
                       color="steelblue")
            r_p, p_p = pearsonr(signal[mask], depth_errors[mask])
            r_s, p_s = spearmanr(signal[mask], depth_errors[mask])
            ax.set_title(f"{name}\nr={r_p:.3f} (p={p_p:.2g}), ρ={r_s:.3f}",
                         fontsize=9)
            # Regression line
            z = np.polyfit(signal[mask], depth_errors[mask], 1)
            x_line = np.linspace(signal[mask].min(), signal[mask].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("Depth error (abs_rel)", fontsize=8)
    plt.suptitle("Spectral features vs depth error (frame-level)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectral_scatter_vs_error.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_correlation_matrix(depth_errors, frame_high, frame_low,
                                      frame_ratio, online_high_mean,
                                      state_changes, confidences, out_dir):
    """Correlation matrix: all signals including spectral features."""
    signals = {
        "depth_error": depth_errors,
        "high_freq (win)": frame_high,
        "high_freq (ema)": online_high_mean,
        "low_freq (win)": frame_low,
        "high/total ratio": frame_ratio,
        "state_change": state_changes,
        "confidence": confidences,
    }

    names = list(signals.keys())
    n = len(names)
    corr = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mask = np.isfinite(signals[names[i]]) & np.isfinite(signals[names[j]])
            if mask.sum() > 5:
                corr[i, j], _ = pearsonr(
                    signals[names[i]][mask], signals[names[j]][mask])
            else:
                corr[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation matrix: spectral + baseline signals")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectral_correlation_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ── Depth dir ──────────────────────────────────────────────────────────────
    depth_dir = derive_depth_dir(args.seq_path)
    has_depth = os.path.isdir(depth_dir)
    # TUM association file: lives in parent of rgb/
    scene_dir = os.path.dirname(os.path.normpath(args.seq_path))
    tum_assoc = load_tum_associations(scene_dir) if has_depth else None
    tum_depth_index = build_tum_timestamp_index(depth_dir) if has_depth else None
    if has_depth:
        print(f"[depth] GT depth dir: {depth_dir}")
        if tum_assoc:
            print(f"[depth] TUM associations loaded: {len(tum_assoc)} pairs")
        elif tum_depth_index:
            print(f"[depth] TUM timestamp index built: {len(tum_depth_index)} depth frames")
    else:
        print(f"[warn] No depth dir found, skipping error correlation")

    # ── Model ──────────────────────────────────────────────────────────────────
    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    print(f"[model] Loading {args.model_update_type} from {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    # ── Data ───────────────────────────────────────────────────────────────────
    img_paths = load_img_paths(args.seq_path, args.frame_interval, args.max_frames)
    if not img_paths:
        raise RuntimeError(f"No images found at {args.seq_path}")
    print(f"[load] {len(img_paths)} frames")

    views = build_views(img_paths, args.size)

    # ── Inference ──────────────────────────────────────────────────────────────
    print("[inference] Running forward_recurrent_analysis …")
    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(
            views, device=device)

    state_history = analysis_data["state_history"]
    T = len(state_history)
    state_stack = torch.stack(state_history).numpy()  # [T, n_state, D]
    n_state, D = state_stack.shape[1], state_stack.shape[2]
    print(f"[data] T={T}, n_state={n_state}, D={D}")

    # ── Depth errors and confidence ────────────────────────────────────────────
    depth_errors = np.full(T, np.nan)
    confidences = np.full(T, np.nan)
    state_changes = np.zeros(T)

    for t in range(T):
        # Depth error
        if has_depth:
            pts3d = ress[t]["pts3d_in_self_view"]
            pred_depth = pts3d[0, :, :, 2].numpy()
            depth_path = find_gt_depth_path(img_paths[t], depth_dir,
                                              tum_assoc, tum_depth_index)
            if depth_path is not None:
                gt = load_gt_depth(depth_path, args.depth_scale)
                if gt is not None:
                    depth_errors[t] = compute_frame_depth_error(
                        pred_depth, gt, args.max_depth)

        # Confidence
        if "conf_self" in ress[t]:
            confidences[t] = ress[t]["conf_self"][0].numpy().mean()

        # State change
        if t > 0:
            diff = state_history[t] - state_history[t - 1]
            state_changes[t] = diff.norm(dim=-1).mean().item()

    # ── Spectral Analysis ──────────────────────────────────────────────────────
    print("[spectral] Computing power spectra …")
    freqs, power_spectra = compute_token_power_spectra(state_stack)
    band_energies = compute_band_energies(freqs, power_spectra)

    print("[spectral] Computing windowed spectral features …")
    frame_high, frame_low, frame_ratio = compute_running_spectral_features(
        state_stack, args.window_size)

    print("[spectral] Computing online (EMA) spectral features …")
    online_high_energy, low_freq_signal = compute_online_spectral_features(
        state_stack, momentum=0.95)
    online_high_mean = online_high_energy.mean(axis=1)  # [T], mean over tokens

    # ── Correlations ───────────────────────────────────────────────────────────
    print("\n--- Spectral features vs depth error ---")
    spectral_signals = {
        "high_freq_windowed": frame_high,
        "high_freq_online": online_high_mean,
        "low_freq_windowed": frame_low,
        "high_total_ratio": frame_ratio,
        "state_change": state_changes,
        "confidence": confidences,
    }

    results = {}
    for name, signal in spectral_signals.items():
        mask = np.isfinite(signal) & np.isfinite(depth_errors)
        if mask.sum() > 10:
            r_p, p_p = pearsonr(signal[mask], depth_errors[mask])
            r_s, p_s = spearmanr(signal[mask], depth_errors[mask])
            results[name] = (r_p, p_p, r_s, p_s)
            print(f"  {name:25s}: Pearson r={r_p:+.4f} (p={p_p:.3g}), "
                  f"Spearman ρ={r_s:+.4f} (p={p_s:.3g})")

    # ── Summary ────────────────────────────────────────────────────────────────
    lines = [
        "=" * 70,
        "Spectral Analysis of State Token Dynamics",
        "=" * 70,
        f"Sequence:  {args.seq_path}",
        f"Model:     {args.model_update_type}",
        f"Frames:    {T}",
        f"Tokens:    {n_state}, Dim: {D}",
        f"Window:    {args.window_size} frames",
        "",
        "--- Token spectral statistics ---",
        f"  Mean total energy:     {band_energies['total'].mean():.4f}",
        f"  Mean high-freq ratio:  {band_energies['high_ratio'].mean():.4f}",
        f"  Mean low-freq ratio:   {band_energies['low_ratio'].mean():.4f}",
        "",
        "--- Correlations with depth error ---",
    ]
    for name, (r_p, p_p, r_s, p_s) in results.items():
        lines.append(f"  {name:25s}: r={r_p:+.4f} (p={p_p:.3g}), ρ={r_s:+.4f}")
    lines.append("=" * 70)
    summary = "\n".join(lines)
    print(f"\n{summary}")

    with open(os.path.join(args.output_dir, "spectral_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ── Save data ──────────────────────────────────────────────────────────────
    np.savez_compressed(
        os.path.join(args.output_dir, "spectral_data.npz"),
        freqs=freqs,
        power_spectra=power_spectra,
        band_energies_high=band_energies["high"],
        band_energies_low=band_energies["low"],
        band_energies_mid=band_energies["mid"],
        band_energies_total=band_energies["total"],
        high_ratio=band_energies["high_ratio"],
        frame_high=frame_high,
        frame_low=frame_low,
        frame_ratio=frame_ratio,
        online_high_energy=online_high_energy,
        online_high_mean=online_high_mean,
        depth_errors=depth_errors,
        confidences=confidences,
        state_changes=state_changes,
    )

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, "plots")

    plot_power_spectra(freqs, power_spectra, band_energies, plots_dir)

    frames = np.arange(T)
    plot_frame_level_spectral(frames, depth_errors, frame_high, frame_low,
                               frame_ratio, online_high_mean, plots_dir)

    plot_spectral_correlation_matrix(
        depth_errors, frame_high, frame_low, frame_ratio,
        online_high_mean, state_changes, confidences, plots_dir)

    print(f"\n[done] All outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
