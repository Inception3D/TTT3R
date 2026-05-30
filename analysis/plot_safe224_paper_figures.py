import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def plot_alpha_curve(per_seq_df: pd.DataFrame, out_path: Path) -> None:
    df = per_seq_df.copy()
    # map method names to alpha_drift values for curve plotting
    alpha_map = {
        "ttt3r_momentum_inv_t1_drift0": 0.0,
        "ttt3r_momentum_inv_t1": 0.15,
    }
    df["alpha_drift_plot"] = df["method"].map(alpha_map)
    df = df.dropna(subset=["alpha_drift_plot"])

    # aggregate over objects for each sequence length
    agg = (
        df.groupby(["seq_length", "alpha_drift_plot"], as_index=False)["basic_consistency_score"]
        .mean()
        .sort_values(["seq_length", "alpha_drift_plot"])
    )

    plt.figure(figsize=(8, 5), dpi=180)
    for seq_len, g in agg.groupby("seq_length"):
        plt.plot(
            g["alpha_drift_plot"],
            g["basic_consistency_score"],
            marker="o",
            linewidth=2.2,
            label=f"Sequence length={int(seq_len)}",
        )
    plt.xlabel("alpha_drift")
    plt.ylabel("Basic Consistency Score (lower is better)")
    plt.title("SAFE224: Metric vs alpha_drift")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_sequence_improvement(per_seq_df: pd.DataFrame, out_path: Path) -> None:
    df = per_seq_df.copy()
    metric = "basic_consistency_score"

    pivot = (
        df.pivot_table(
            index=["object_id", "sequence_id", "seq_length"],
            columns="method",
            values=metric,
            aggfunc="mean",
        )
        .reset_index()
    )
    if (
        "ttt3r_momentum_inv_t1" not in pivot.columns
        or "ttt3r_momentum_inv_t1_drift0" not in pivot.columns
    ):
        raise ValueError("Required methods not found in per-sequence metrics.")

    # improvement > 0 means inv_t1 is better (smaller metric)
    pivot["improvement"] = (
        pivot["ttt3r_momentum_inv_t1_drift0"] - pivot["ttt3r_momentum_inv_t1"]
    )
    pivot["label"] = pivot["object_id"] + "_L" + pivot["seq_length"].astype(int).astype(str)
    pivot = pivot.sort_values("improvement", ascending=False).reset_index(drop=True)

    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in pivot["improvement"]]
    plt.figure(figsize=(9, 5), dpi=180)
    plt.bar(np.arange(len(pivot)), pivot["improvement"], color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(np.arange(len(pivot)), pivot["label"], rotation=25, ha="right")
    plt.ylabel("Improvement over drift0 (positive = better)")
    plt.title("Per-sequence Improvement Distribution (inv_t1 vs drift0)")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _load_depth(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _norm_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    valid = d > 1e-8
    if not np.any(valid):
        return np.zeros_like(d)
    lo = np.percentile(d[valid], 2)
    hi = np.percentile(d[valid], 98)
    if hi <= lo:
        hi = lo + 1e-6
    d = np.clip((d - lo) / (hi - lo), 0, 1)
    return d


def plot_typical_before_after(pred_root: Path, out_path: Path, frame_id: str = "000008") -> None:
    seq = "apple/540_79043_153212_len024"

    color_path = pred_root / "ttt3r_momentum_inv_t1" / seq / "color" / f"{frame_id}.png"
    # fallback to raw sequence image if exported color frame is all black
    raw_seq_path = (
        pred_root.parent / "sequences" / "apple" / "540_79043_153212_len024" / f"{frame_id}.jpg"
    )
    d_inv = pred_root / "ttt3r_momentum_inv_t1" / seq / "depth" / f"{frame_id}.npy"
    d_zero = pred_root / "ttt3r_momentum_inv_t1_drift0" / seq / "depth" / f"{frame_id}.npy"

    if not (color_path.exists() and d_inv.exists() and d_zero.exists()):
        raise FileNotFoundError("Typical sequence files not found for visualization.")

    img = plt.imread(color_path)
    # if color export is fully black, use original sequence RGB for better visualization
    if np.max(img) <= 0 and raw_seq_path.exists():
        img = plt.imread(raw_seq_path)
    depth_inv = _load_depth(d_inv)
    depth_zero = _load_depth(d_zero)
    diff = np.abs(depth_inv - depth_zero)

    n_inv = _norm_depth(depth_inv)
    n_zero = _norm_depth(depth_zero)
    n_diff = _norm_depth(diff)

    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8), dpi=180)
    axs[0].imshow(img)
    axs[0].set_title("Input RGB")
    axs[1].imshow(n_zero, cmap="viridis")
    axs[1].set_title("Depth (drift0)")
    axs[2].imshow(n_inv, cmap="viridis")
    axs[2].set_title("Depth (inv_t1)")
    axs[3].imshow(n_diff, cmap="magma")
    axs[3].set_title("|Depth diff|")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"Typical Sequence Visual Comparison (apple_len024, frame {int(frame_id)})", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAFE224 paper-friendly figures.")
    parser.add_argument(
        "--safe_root",
        type=Path,
        default=Path("/home/chen/TTT3R/benchmark_single_object/outputs_ablation_safe"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/mnt/c/Users/Chen/Desktop/codes/TTT3R/docs/figures/safe224"),
    )
    args = parser.parse_args()

    metrics_root = args.safe_root / "metrics"
    pred_root = args.safe_root / "predictions"

    _ensure_dir(args.output_dir)

    per_seq = _read_csv(metrics_root / "per_sequence_results.csv")

    plot_alpha_curve(per_seq, args.output_dir / "fig_alpha_drift_curve.png")
    plot_sequence_improvement(per_seq, args.output_dir / "fig_sequence_improvement_distribution.png")
    plot_typical_before_after(
        pred_root, args.output_dir / "fig_typical_before_after_depth.png", frame_id="000008"
    )
    # cache-busting copy with explicit frame suffix for web preview
    plot_typical_before_after(
        pred_root,
        args.output_dir / "fig_typical_before_after_depth_frame008.png",
        frame_id="000008",
    )

    print("[DONE] Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
