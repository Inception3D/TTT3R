from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SAFE224 reset-interval sensitivity.")
    parser.add_argument(
        "--raw_csv",
        type=Path,
        default=Path(
            "benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/reset_raw_results.csv"
        ),
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path(
            "benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/summary_by_reset_method.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_results/s5_reset_interval_sensitivity"),
    )
    return parser.parse_args()


def save_consistency_plot(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, label in [
        ("ttt3r_momentum_inv_t1", "brake (alpha=0.15)"),
        ("ttt3r_momentum_inv_t1_drift0", "drift0 (alpha=0.0)"),
    ]:
        part = summary[summary["method"] == method].sort_values("reset_interval")
        ax.errorbar(
            part["reset_interval"],
            part["basic_consistency_score_mean"],
            yerr=part["basic_consistency_score_std"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    ax.set_xlabel("reset_interval")
    ax.set_ylabel("basic_consistency_score (lower is better)")
    ax.set_title("Reset sensitivity under SAFE224 (mean ± std)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_runtime_plot(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, label in [
        ("ttt3r_momentum_inv_t1", "brake (alpha=0.15)"),
        ("ttt3r_momentum_inv_t1_drift0", "drift0 (alpha=0.0)"),
    ]:
        part = summary[summary["method"] == method].sort_values("reset_interval")
        ax.errorbar(
            part["reset_interval"],
            part["runtime_sec_mean"],
            yerr=part["runtime_sec_std"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    ax.set_xlabel("reset_interval")
    ax.set_ylabel("runtime_sec")
    ax.set_title("Runtime overhead vs reset_interval (mean ± std)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_delta_distribution(raw: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    paired = (
        raw.pivot_table(
            index=["reset_interval", "seed_eval", "sequence_id"],
            columns="method",
            values="basic_consistency_score",
            aggfunc="mean",
        )
        .reset_index()
        .dropna()
    )
    paired["delta_drift0_minus_brake"] = (
        paired["ttt3r_momentum_inv_t1_drift0"] - paired["ttt3r_momentum_inv_t1"]
    )

    resets = sorted(paired["reset_interval"].unique().tolist())
    data = [paired.loc[paired["reset_interval"] == r, "delta_drift0_minus_brake"] for r in resets]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.boxplot(data, tick_labels=[str(r) for r in resets], showfliers=False)
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="gray")
    ax.set_xlabel("reset_interval")
    ax.set_ylabel("delta consistency (drift0 - brake)")
    ax.set_title("Per-sequence consistency delta distribution")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paired


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw_csv)
    summary = pd.read_csv(args.summary_csv)

    save_consistency_plot(summary, args.out_dir / "fig_consistency_vs_reset.png")
    save_runtime_plot(summary, args.out_dir / "fig_runtime_vs_reset.png")
    paired = save_delta_distribution(raw, args.out_dir / "fig_delta_distribution.png")

    # Key numbers table for paper text.
    key = []
    for r in sorted(summary["reset_interval"].unique().tolist()):
        a = summary[
            (summary["method"] == "ttt3r_momentum_inv_t1")
            & (summary["reset_interval"] == r)
        ].iloc[0]
        b = summary[
            (summary["method"] == "ttt3r_momentum_inv_t1_drift0")
            & (summary["reset_interval"] == r)
        ].iloc[0]
        key.append(
            {
                "reset_interval": int(r),
                "consistency_brake_mean": float(a["basic_consistency_score_mean"]),
                "consistency_drift0_mean": float(b["basic_consistency_score_mean"]),
                "delta_consistency_drift0_minus_brake": float(
                    b["basic_consistency_score_mean"] - a["basic_consistency_score_mean"]
                ),
                "runtime_brake_mean": float(a["runtime_sec_mean"]),
                "runtime_drift0_mean": float(b["runtime_sec_mean"]),
                "delta_runtime_drift0_minus_brake": float(
                    b["runtime_sec_mean"] - a["runtime_sec_mean"]
                ),
            }
        )
    key_df = pd.DataFrame(key)
    key_df.to_csv(args.out_dir / "key_numbers.csv", index=False)

    paired.to_csv(args.out_dir / "paired_per_sequence_delta.csv", index=False)

    print(f"[OUT] {args.out_dir / 'fig_consistency_vs_reset.png'}")
    print(f"[OUT] {args.out_dir / 'fig_runtime_vs_reset.png'}")
    print(f"[OUT] {args.out_dir / 'fig_delta_distribution.png'}")
    print(f"[OUT] {args.out_dir / 'key_numbers.csv'}")
    print(f"[OUT] {args.out_dir / 'paired_per_sequence_delta.csv'}")


if __name__ == "__main__":
    main()
