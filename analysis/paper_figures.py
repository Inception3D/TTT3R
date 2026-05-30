"""
Generate paper-style summary figures from existing analysis outputs.

Outputs:
- fig_relpose_per_scene.png
- fig_dynamics_summary.png
- fig_tau_sensitivity.png
- README.md
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_results" / "paper_figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        value = row[key]
        vals.append(float(value) if value not in ("", "nan", "NaN") else np.nan)
    return np.asarray(vals, dtype=np.float64)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
        }
    )


def make_relpose_figure() -> None:
    scannet = read_csv(ROOT / "analysis_results" / "a3_scannet_momentum_inv_t1" / "per_scene_comparison.csv")
    tum = read_csv(ROOT / "analysis_results" / "a3_tum_momentum_inv_t1" / "per_scene_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    datasets = [
        ("ScanNet", scannet, axes[0]),
        ("TUM", tum, axes[1]),
    ]

    for title, rows, ax in datasets:
        x = to_float_array(rows, "random_x0.5_ate")
        y = to_float_array(rows, "stability_brake_t1_ate")
        improved = np.asarray([r["is_improved"] == "True" for r in rows], dtype=bool)
        colors = np.where(improved, "#1f77b4", "#d95f02")

        ax.scatter(x, y, c=colors, s=28, alpha=0.9, edgecolors="white", linewidths=0.4)
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        pad = (hi - lo) * 0.06
        lo = max(0.0, lo - pad)
        hi = hi + pad
        ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Constant dampening ATE")
        ax.set_ylabel("Stability brake ATE")

        n_improved = int(improved.sum())
        n_total = len(rows)
        median_base = float(np.nanmedian(x))
        median_method = float(np.nanmedian(y))
        ax.text(
            0.04,
            0.96,
            f"Improved: {n_improved}/{n_total}\nMedian: {median_base:.3f} -> {median_method:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7e6", edgecolor="#d9c59a", alpha=0.95),
        )

    fig.suptitle("Per-scene relpose comparison", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_relpose_per_scene.png", bbox_inches="tight")
    plt.close(fig)


def make_tau_figure() -> None:
    scannet = read_csv(ROOT / "analysis_results" / "s3_scannet_momentum_inv" / "tau_sensitivity_summary.csv")
    tum = read_csv(ROOT / "analysis_results" / "s3_tum_momentum_inv" / "tau_sensitivity_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    datasets = [
        ("ScanNet", scannet, axes[0]),
        ("TUM", tum, axes[1]),
    ]

    for title, rows, ax in datasets:
        tau = to_float_array(rows, "tau")
        median_ate = to_float_array(rows, "median_ate")
        mean_ate = to_float_array(rows, "mean_ate")

        order = np.argsort(tau)
        tau = tau[order]
        median_ate = median_ate[order]
        mean_ate = mean_ate[order]

        ax.plot(tau, median_ate, marker="o", linewidth=2.2, color="#1f77b4", label="Median ATE")
        ax.plot(tau, mean_ate, marker="s", linewidth=2.0, color="#d95f02", label="Mean ATE")
        ax.set_title(title)
        ax.set_xlabel("Tau")
        ax.set_ylabel("ATE")
        ax.legend(frameon=False)

    fig.suptitle("Tau sensitivity of stability brake", y=1.03, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tau_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_a2_panel(ax: plt.Axes) -> None:
    rows = read_csv(ROOT / "analysis_results" / "a2_proxy_co3d" / "a2_proxy_points.csv")
    x = to_float_array(rows, "baseline_var_delta_cosine")
    y = to_float_array(rows, "convergence_improve_pct")

    seq_names = [row["sequence"] for row in rows]
    colors = ["#1f77b4" if "apple" in name else "#2ca02c" for name in seq_names]
    ax.scatter(x, y, c=colors, s=34, alpha=0.9, edgecolors="white", linewidths=0.4)
    z = np.polyfit(x, y, 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    yy = z[0] * xx + z[1]
    ax.plot(xx, yy, color="#444444", linestyle="--", linewidth=1.6)

    summary = read_csv(ROOT / "analysis_results" / "a2_proxy_co3d" / "summary.csv")[0]
    ax.set_title("A2 proxy")
    ax.set_xlabel("Var(cos(delta_t, delta_{t-1}))")
    ax.set_ylabel("Convergence improvement (%)")
    ax.text(
        0.04,
        0.96,
        f"Pearson = {float(summary['pearson_corr']):.3f}\nSpearman = {float(summary['spearman_corr']):.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef6ff", edgecolor="#bfd8f5", alpha=0.95),
    )


def plot_a4_panel(ax: plt.Axes, seq_name: str, csv_path: Path) -> None:
    rows = read_csv(csv_path)
    cut_rows = [r for r in rows if r["update_type"] == "cut3r"]
    ttt_rows = [r for r in rows if r["update_type"] == "ttt3r"]

    cut_frame = to_float_array(cut_rows, "frame_idx")
    cut_delta = to_float_array(cut_rows, "delta_norm")
    ttt_frame = to_float_array(ttt_rows, "frame_idx")
    ttt_delta = to_float_array(ttt_rows, "delta_norm")

    if np.nanmax(cut_delta) > 0:
        cut_delta = cut_delta / np.nanmax(cut_delta)
    if np.nanmax(ttt_delta) > 0:
        ttt_delta = ttt_delta / np.nanmax(ttt_delta)

    ax.plot(cut_frame, cut_delta, color="#d95f02", linewidth=2.0, label="CUT3R")
    ax.plot(ttt_frame, ttt_delta, color="#1f77b4", linewidth=2.2, label="TTT3R")
    ax.set_title(seq_name)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Normalized delta norm")


def make_dynamics_figure() -> None:
    fig = plt.figure(figsize=(13.2, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    plot_a2_panel(ax0)
    plot_a4_panel(ax1, "A4 Apple", ROOT / "analysis_results" / "a4_co3d_apple" / "state_convergence.csv")
    plot_a4_panel(ax2, "A4 Bottle", ROOT / "analysis_results" / "a4_co3d_bottle" / "state_convergence.csv")
    ax2.legend(frameon=False, loc="upper right")

    fig.suptitle("State-dynamics analysis", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_dynamics_summary.png", bbox_inches="tight")
    plt.close(fig)


def write_readme() -> None:
    text = """# Paper Figures

This directory contains paper-style summary figures generated from the current local analysis outputs.

## Files

- `fig_relpose_per_scene.png`: ScanNet and TUM per-scene scatter, constant dampening vs stability brake
- `fig_tau_sensitivity.png`: available tau sensitivity curves on ScanNet and TUM
- `fig_dynamics_summary.png`: A2 proxy correlation and A4 local state-convergence plots

These figures are intended for slides, reports, supplementary material, or later paper polishing.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    make_relpose_figure()
    make_tau_figure()
    make_dynamics_figure()
    write_readme()
    print(f"[OK] Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
