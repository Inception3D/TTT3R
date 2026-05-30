"""
S3: Stability brake tau sensitivity summarizer.

Parses relpose `_error_log*.txt` files from multiple experiment directories
whose names contain tau tags such as:
    ttt3r_momentum_inv_t0.5
    ttt3r_momentum_inv_t1
    ttt3r_momentum_inv_t1.5

It summarizes per-run median ATE / RPE and produces a simple tau curve.
"""

from __future__ import annotations

import argparse
import csv
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
TAU_RE = re.compile(r"_t(?P<tau>[0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stability brake tau sensitivity.")
    parser.add_argument("--root", type=str, required=True, help="Root dir containing multiple config subdirs.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="ttt3r_momentum_inv_t*")
    return parser.parse_args()


def parse_log(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            rows.append(
                {
                    "ate": float(m.group("ate")),
                    "rpe_trans": float(m.group("rpe_t")),
                    "rpe_rot": float(m.group("rpe_r")),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for config_dir in sorted(root.glob(args.pattern)):
        if not config_dir.is_dir():
            continue
        tau_match = TAU_RE.search(config_dir.name)
        if not tau_match:
            continue
        tau = float(tau_match.group("tau"))
        logs = sorted(config_dir.rglob("_error_log*.txt"))
        metrics = []
        for log in logs:
            metrics.extend(parse_log(log))
        if not metrics:
            continue

        ate = np.asarray([m["ate"] for m in metrics], dtype=float)
        rpe_t = np.asarray([m["rpe_trans"] for m in metrics], dtype=float)
        rpe_r = np.asarray([m["rpe_rot"] for m in metrics], dtype=float)
        summary_rows.append(
            {
                "config": config_dir.name,
                "tau": tau,
                "num_scenes": int(len(metrics)),
                "median_ate": float(np.median(ate)),
                "mean_ate": float(np.mean(ate)),
                "median_rpe_trans": float(np.median(rpe_t)),
                "median_rpe_rot": float(np.median(rpe_r)),
            }
        )

    if not summary_rows:
        raise RuntimeError(f"No matching sensitivity result dirs found under {root}")

    summary_rows.sort(key=lambda r: float(r["tau"]))

    csv_path = output_dir / "tau_sensitivity_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    taus = np.asarray([float(r["tau"]) for r in summary_rows], dtype=float)
    med_ate = np.asarray([float(r["median_ate"]) for r in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(taus, med_ate, marker="o", linewidth=1.8)
    ax.set_xlabel("tau")
    ax.set_ylabel("Median ATE")
    ax.set_title("Stability brake tau sensitivity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "tau_sensitivity_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {output_dir / 'tau_sensitivity_curve.png'}")


if __name__ == "__main__":
    main()
