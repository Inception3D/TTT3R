"""
A3: Per-Scene Improvement Distribution Analysis

Compare ttt3r_momentum_inv_t1 (stability brake) vs ttt3r_random (constant x0.5)
and other configs on per-scene ATE.

Outputs to analysis_results/a3_per_scene/:
  - scatter_scannet.png: x=random ATE, y=inv_t1 ATE, y=x reference
  - scatter_tum.png: same for TUM
  - boxplot_scannet.png: ATE distribution per config
  - boxplot_tum.png: same for TUM
  - summary.txt: statistics
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/szy/research/TTT3R")
SCANNET_DIR = BASE / "eval_results/relpose/scannet_s3_1000"
TUM_DIR = BASE / "eval_results/relpose/tum_s1_1000"
OUT_DIR = BASE / "analysis_results/a3_per_scene"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["cut3r", "ttt3r", "ttt3r_random", "ttt3r_momentum_inv_t1"]
CONFIG_LABELS = {
    "cut3r": "CUT3R (baseline)",
    "ttt3r": "TTT3R (no dampening)",
    "ttt3r_random": "TTT3R + constant 0.5",
    "ttt3r_momentum_inv_t1": "TTT3R + stability brake",
}
# Colors for each config
CONFIG_COLORS = {
    "cut3r": "#888888",
    "ttt3r": "#4477AA",
    "ttt3r_random": "#EE7733",
    "ttt3r_momentum_inv_t1": "#228833",
}


def parse_error_log(path):
    """Parse _error_log.txt, return dict {scene_name: ATE}."""
    results = {}
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return results

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip summary lines
            if line.startswith("Average"):
                continue
            # Skip exception lines
            if line.startswith("Exception"):
                continue
            # Skip bare numeric lines (redundant echo of previous values)
            try:
                float(line)
                continue
            except ValueError:
                pass
            # Parse data lines: "dataset-scene_name     | ATE: 0.123, RPE trans: 0.456, RPE rot: 0.789"
            m = re.match(r"^[\w_]+-(.+?)\s*\|\s*ATE:\s*([\d.]+)", line)
            if m:
                scene = m.group(1).strip()
                ate = float(m.group(2))
                results[scene] = ate
    return results


def load_dataset(base_dir, configs):
    """Load per-scene ATE for all configs. Return {config: {scene: ATE}}."""
    data = {}
    for cfg in configs:
        log_path = base_dir / cfg / "_error_log.txt"
        data[cfg] = parse_error_log(log_path)
        print(f"  {cfg}: {len(data[cfg])} scenes")
    return data


def get_common_scenes(data, configs):
    """Return sorted list of scenes present in ALL configs."""
    scene_sets = [set(data[cfg].keys()) for cfg in configs]
    common = scene_sets[0]
    for s in scene_sets[1:]:
        common = common & s
    return sorted(common)


def compute_stats(random_ates, brake_ates, scene_names):
    """Compute comparison statistics between random and brake."""
    improvements = []  # positive = brake is better (lower ATE)
    pct_improvements = []

    for i, scene in enumerate(scene_names):
        r = random_ates[i]
        b = brake_ates[i]
        improvements.append(r - b)
        if r > 0:
            pct_improvements.append((r - b) / r * 100)
        else:
            pct_improvements.append(0.0)

    improvements = np.array(improvements)
    pct_improvements = np.array(pct_improvements)

    n_improved = np.sum(improvements > 0)
    n_total = len(improvements)

    # Best and worst scenes
    best_idx = np.argmax(pct_improvements)
    worst_idx = np.argmin(pct_improvements)

    stats = {
        "n_improved": int(n_improved),
        "n_total": n_total,
        "pct_improved": n_improved / n_total * 100,
        "mean_pct": float(np.mean(pct_improvements)),
        "median_pct": float(np.median(pct_improvements)),
        "std_pct": float(np.std(pct_improvements)),
        "best_scene": scene_names[best_idx],
        "best_pct": float(pct_improvements[best_idx]),
        "best_random_ate": float(random_ates[best_idx]),
        "best_brake_ate": float(brake_ates[best_idx]),
        "worst_scene": scene_names[worst_idx],
        "worst_pct": float(pct_improvements[worst_idx]),
        "worst_random_ate": float(random_ates[worst_idx]),
        "worst_brake_ate": float(brake_ates[worst_idx]),
    }
    return stats


def plot_scatter(random_ates, brake_ates, scene_names, dataset_name, out_path):
    """Scatter plot: x=random ATE, y=brake ATE, with y=x line."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(random_ates, brake_ates, s=40, alpha=0.7, edgecolors="k",
               linewidths=0.5, color=CONFIG_COLORS["ttt3r_momentum_inv_t1"],
               zorder=3)

    # y=x reference line
    all_vals = np.concatenate([random_ates, brake_ates])
    lo, hi = 0, max(all_vals) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, label="y = x")

    ax.set_xlabel("TTT3R + constant 0.5 (ATE)", fontsize=12)
    ax.set_ylabel("TTT3R + stability brake (ATE)", fontsize=12)
    ax.set_title(f"{dataset_name}: Per-Scene ATE Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Count improved / degraded
    n_improved = np.sum(brake_ates < random_ates)
    n_total = len(brake_ates)
    ax.text(0.05, 0.95,
            f"Improved: {n_improved}/{n_total} scenes\n"
            f"Below line = brake better",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_boxplot(data, common_scenes, configs, dataset_name, out_path):
    """Box plot of ATE distribution for each config."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ate_lists = []
    labels = []
    colors = []
    for cfg in configs:
        ates = [data[cfg][s] for s in common_scenes]
        ate_lists.append(ates)
        labels.append(CONFIG_LABELS[cfg])
        colors.append(CONFIG_COLORS[cfg])

    bp = ax.boxplot(ate_lists, labels=labels, patch_artist=True,
                    widths=0.5, showfliers=True,
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("ATE", fontsize=12)
    ax.set_title(f"{dataset_name}: ATE Distribution by Config ({len(common_scenes)} scenes)",
                 fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    # Rotate labels if needed
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def format_stats(stats, dataset_name):
    """Format stats dict as text."""
    lines = [
        f"=== {dataset_name}: Stability Brake vs Constant 0.5 ===",
        f"",
        f"Scenes improved: {stats['n_improved']} / {stats['n_total']} "
        f"({stats['pct_improved']:.1f}%)",
        f"",
        f"ATE reduction (%):",
        f"  Mean:   {stats['mean_pct']:+.1f}%",
        f"  Median: {stats['median_pct']:+.1f}%",
        f"  Std:    {stats['std_pct']:.1f}%",
        f"",
        f"Best scene:  {stats['best_scene']}",
        f"  random ATE: {stats['best_random_ate']:.5f}  ->  brake ATE: {stats['best_brake_ate']:.5f}  "
        f"({stats['best_pct']:+.1f}%)",
        f"",
        f"Worst scene: {stats['worst_scene']}",
        f"  random ATE: {stats['worst_random_ate']:.5f}  ->  brake ATE: {stats['worst_brake_ate']:.5f}  "
        f"({stats['worst_pct']:+.1f}%)",
        f"",
    ]
    return "\n".join(lines)


def process_dataset(base_dir, dataset_name):
    """Process one dataset: load data, make plots, return stats text."""
    print(f"\n--- {dataset_name} ---")
    data = load_dataset(base_dir, CONFIGS)
    common_scenes = get_common_scenes(data, CONFIGS)
    print(f"  Common scenes: {len(common_scenes)}")

    if len(common_scenes) == 0:
        return f"=== {dataset_name}: No common scenes found ===\n"

    # Extract arrays for scatter
    random_ates = np.array([data["ttt3r_random"][s] for s in common_scenes])
    brake_ates = np.array([data["ttt3r_momentum_inv_t1"][s] for s in common_scenes])

    # Scatter plot
    plot_scatter(random_ates, brake_ates, common_scenes, dataset_name,
                 OUT_DIR / f"scatter_{dataset_name.lower()}.png")

    # Box plot
    plot_boxplot(data, common_scenes, CONFIGS, dataset_name,
                 OUT_DIR / f"boxplot_{dataset_name.lower()}.png")

    # Statistics
    stats = compute_stats(random_ates, brake_ates, common_scenes)
    stats_text = format_stats(stats, dataset_name)

    # Also add per-config summary (mean/median ATE)
    config_lines = [f"--- {dataset_name}: Per-Config ATE Summary ---", ""]
    config_lines.append(f"{'Config':<35s}  {'Mean ATE':>10s}  {'Median ATE':>10s}")
    config_lines.append("-" * 60)
    for cfg in CONFIGS:
        ates = np.array([data[cfg][s] for s in common_scenes])
        config_lines.append(
            f"{CONFIG_LABELS[cfg]:<35s}  {np.mean(ates):10.5f}  {np.median(ates):10.5f}"
        )
    config_lines.append("")

    return stats_text + "\n".join(config_lines) + "\n"


def main():
    summary_parts = []
    summary_parts.append("A3: Per-Scene Improvement Distribution Analysis")
    summary_parts.append("=" * 55)
    summary_parts.append("")

    # ScanNet
    scannet_text = process_dataset(SCANNET_DIR, "ScanNet")
    summary_parts.append(scannet_text)

    # TUM
    tum_text = process_dataset(TUM_DIR, "TUM")
    summary_parts.append(tum_text)

    # Write summary
    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_parts))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
