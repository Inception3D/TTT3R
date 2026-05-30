"""
Experiment A3: Per-scene improvement distribution for relpose results.

This script compares two relpose evaluation result directories, parses their
`_error_log*.txt` files, and produces:
1. A merged per-scene CSV
2. Scatter plots (baseline ATE vs method ATE)
3. Improvement summary statistics

Typical use:
python analysis/per_scene_improvement_analysis.py ^
    --baseline_root eval_results/relpose/scannet_s3_1000/cut3r ^
    --method_root eval_results/relpose/scannet_s3_1000/ttt3r_momentum ^
    --output_dir analysis_results/a3_scannet_brake
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LINE_RE = re.compile(
    r"^(?P<dataset>[^-]+)-(?P<scene>[^|]+)\|\s*ATE:\s*(?P<ate>[-+0-9.eE]+),\s*"
    r"RPE trans:\s*(?P<rpe_t>[-+0-9.eE]+),\s*RPE rot:\s*(?P<rpe_r>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-scene relpose improvement distribution."
    )
    parser.add_argument("--baseline_root", type=str, required=True)
    parser.add_argument("--method_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--method_name", type=str, default="method")
    parser.add_argument("--title", type=str, default="Per-scene ATE comparison")
    return parser.parse_args()


def collect_error_logs(root: Path) -> list[Path]:
    return sorted(root.rglob("_error_log*.txt"))


def parse_error_log(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = LINE_RE.match(line)
            if not match:
                continue
            rows.append(
                {
                    "dataset": match.group("dataset").strip(),
                    "scene": match.group("scene").strip(),
                    "ate": float(match.group("ate")),
                    "rpe_trans": float(match.group("rpe_t")),
                    "rpe_rot": float(match.group("rpe_r")),
                    "source_file": str(path),
                }
            )
    return rows


def load_result_table(root: Path, value_prefix: str) -> list[dict[str, object]]:
    logs = collect_error_logs(root)
    if not logs:
        raise FileNotFoundError(f"No _error_log*.txt found under {root}")

    rows: list[dict[str, object]] = []
    for p in logs:
        rows.extend(parse_error_log(p))
    if not rows:
        raise RuntimeError(f"Parsed zero scene rows from {root}")

    dedup: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        dedup[(str(row["dataset"]), str(row["scene"]))] = {
            "dataset": row["dataset"],
            "scene": row["scene"],
            f"{value_prefix}_ate": row["ate"],
            f"{value_prefix}_rpe_trans": row["rpe_trans"],
            f"{value_prefix}_rpe_rot": row["rpe_rot"],
        }
    return list(dedup.values())


def merge_tables(
    baseline_rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
    baseline_name: str,
    method_name: str,
) -> list[dict[str, object]]:
    baseline_map = {(str(r["dataset"]), str(r["scene"])): r for r in baseline_rows}
    method_map = {(str(r["dataset"]), str(r["scene"])): r for r in method_rows}
    keys = sorted(set(baseline_map) & set(method_map))
    merged = []
    for key in keys:
        row = {}
        row.update(baseline_map[key])
        row.update(method_map[key])
        row["ate_abs_improve"] = row[f"{baseline_name}_ate"] - row[f"{method_name}_ate"]
        row["ate_rel_improve_pct"] = (
            row["ate_abs_improve"] / (row[f"{baseline_name}_ate"] + 1e-12) * 100.0
        )
        row["is_improved"] = row["ate_abs_improve"] > 0
        merged.append(row)
    return merged


def plot_scatter(rows: list[dict[str, object]], baseline_name: str, method_name: str, out_path: Path, title: str) -> None:
    x = np.asarray([float(r[f"{baseline_name}_ate"]) for r in rows])
    y = np.asarray([float(r[f"{method_name}_ate"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    colors = np.where(np.asarray([bool(r["is_improved"]) for r in rows]), "tab:blue", "tab:red")
    ax.scatter(x, y, c=colors, alpha=0.8, s=26)

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")

    improved = int(sum(bool(r["is_improved"]) for r in rows))
    total = int(len(rows))
    ax.set_xlabel(f"{baseline_name} ATE")
    ax.set_ylabel(f"{method_name} ATE")
    ax.set_title(f"{title}\nImproved scenes: {improved}/{total}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(rows: list[dict[str, object]], out_path: Path, method_name: str) -> None:
    vals = np.asarray([float(r["ate_rel_improve_pct"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(vals, bins=20, color="tab:green", alpha=0.85)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(f"ATE relative improvement of {method_name} over baseline (%)")
    ax.set_ylabel("Scene count")
    ax.set_title("Per-scene improvement distribution")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = Path(args.baseline_root)
    method_root = Path(args.method_root)

    baseline_rows = load_result_table(baseline_root, args.baseline_name)
    method_rows = load_result_table(method_root, args.method_name)

    merged = merge_tables(baseline_rows, method_rows, args.baseline_name, args.method_name)
    if not merged:
        raise RuntimeError("No overlapping scenes found between baseline and method logs.")

    merged = sorted(merged, key=lambda r: (str(r["dataset"]), -float(r["ate_rel_improve_pct"])))

    merged_csv = output_dir / "per_scene_comparison.csv"
    with merged_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

    baseline_ates = np.asarray([float(r[f"{args.baseline_name}_ate"]) for r in merged])
    method_ates = np.asarray([float(r[f"{args.method_name}_ate"]) for r in merged])
    rel_improve = np.asarray([float(r["ate_rel_improve_pct"]) for r in merged])
    summary_row = {
        "num_scenes": int(len(merged)),
        "num_improved": int(sum(bool(r["is_improved"]) for r in merged)),
        "num_degraded": int(sum(not bool(r["is_improved"]) for r in merged)),
        "median_baseline_ate": float(np.median(baseline_ates)),
        "median_method_ate": float(np.median(method_ates)),
        "mean_rel_improve_pct": float(np.mean(rel_improve)),
        "median_rel_improve_pct": float(np.median(rel_improve)),
        "best_scene": str(merged[0]["scene"]),
        "worst_scene": str(merged[-1]["scene"]),
    }
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    plot_scatter(
        merged,
        args.baseline_name,
        args.method_name,
        output_dir / "ate_scatter.png",
        args.title,
    )
    plot_histogram(merged, output_dir / "improvement_hist.png", args.method_name)

    with (output_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write(
            "Outputs:\n"
            "- per_scene_comparison.csv: merged per-scene metrics\n"
            "- summary.csv: aggregate summary\n"
            "- ate_scatter.png: baseline vs method scatter\n"
            "- improvement_hist.png: relative improvement histogram\n"
        )

    print(f"[OK] Wrote {merged_csv}")
    print(f"[OK] Wrote {output_dir / 'summary.csv'}")
    print(f"[OK] Wrote {output_dir / 'ate_scatter.png'}")
    print(f"[OK] Wrote {output_dir / 'improvement_hist.png'}")


if __name__ == "__main__":
    main()
