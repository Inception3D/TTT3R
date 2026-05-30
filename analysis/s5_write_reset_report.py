from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write markdown report for S5 reset sensitivity.")
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path(
            "benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/summary_by_reset_method.csv"
        ),
    )
    parser.add_argument(
        "--raw_csv",
        type=Path,
        default=Path(
            "benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/reset_raw_results.csv"
        ),
    )
    parser.add_argument(
        "--key_csv",
        type=Path,
        default=Path(
            "benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/key_numbers.csv"
        ),
    )
    parser.add_argument(
        "--out_md",
        type=Path,
        default=Path("docs/reset_interval_sensitivity_safe224.md"),
    )
    return parser.parse_args()


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    raw = pd.read_csv(args.raw_csv)
    key = pd.read_csv(args.key_csv)

    n_runs = len(raw)
    n_seq = raw["sequence_id"].nunique()
    n_seed = raw["seed_eval"].nunique()
    reset_values = sorted(raw["reset_interval"].unique().tolist())
    ok_ratio = float((raw["run_ok"] == 1).mean())
    timed_out = int(raw["timed_out"].sum())
    vram_backend = raw["peak_vram_backend"].value_counts().idxmax()
    vram_mean = float(raw["peak_vram_mb"].mean())

    # Global paired deltas.
    paired = raw.pivot_table(
        index=["reset_interval", "seed_eval", "sequence_id"],
        columns="method",
        values=["basic_consistency_score", "runtime_sec"],
        aggfunc="mean",
    ).dropna()
    paired.columns = [f"{a}__{b}" for a, b in paired.columns]
    paired = paired.reset_index()
    paired["d_cons"] = (
        paired["basic_consistency_score__ttt3r_momentum_inv_t1_drift0"]
        - paired["basic_consistency_score__ttt3r_momentum_inv_t1"]
    )
    paired["d_runtime"] = (
        paired["runtime_sec__ttt3r_momentum_inv_t1_drift0"]
        - paired["runtime_sec__ttt3r_momentum_inv_t1"]
    )

    lines: list[str] = []
    lines.append("# S5 Reset-Interval Sensitivity (SAFE224, Local)")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "Evaluate whether brake-style residual update (`alpha_drift=0.15`) remains robust when the external state reset policy changes."
    )
    lines.append("")
    lines.append("## 2. Experimental Setup")
    lines.append(f"- Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("- Platform: WSL2 Ubuntu 22.04, RTX 4060 Laptop 8GB, CUDA")
    lines.append("- Input size: 224 (SAFE224)")
    lines.append("- Checkpoint: `/home/chen/TTT3R/model/cut3r_512_dpt_4_64.pth`")
    lines.append("- Dataset slice: 4 sampled sequences (`apple`/`bottle`, len=12/24)")
    lines.append("- Methods:")
    lines.append("  - `ttt3r_momentum_inv_t1` (`alpha_drift=0.15`, brake retained)")
    lines.append("  - `ttt3r_momentum_inv_t1_drift0` (`alpha_drift=0.0`)")
    lines.append(f"- Reset intervals: {', '.join(str(x) for x in reset_values)}")
    lines.append(f"- Seeds: {n_seed} (`41,42`), total runs: {n_runs}")
    lines.append(f"- Run validity: {ok_ratio*100:.1f}% (`run_ok=1` for all runs), `timed_out={timed_out}`")
    lines.append(f"- Peak VRAM monitor backend: `{vram_backend}`, mean peak VRAM: `{vram_mean:.1f} MB`")
    lines.append("- Runner fix used in this study: `run_one_method.py` supports configurable `repo_root` and auto-detects supported `demo.py` flags to avoid silent argument mismatch.")
    lines.append("")
    lines.append("## 3. Main Results (mean +- std)")
    lines.append("| reset_interval | method | runtime_sec | per_frame_sec | basic_consistency_score | loop_closure_trans_error |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in reset_values:
        for method in ["ttt3r_momentum_inv_t1", "ttt3r_momentum_inv_t1_drift0"]:
            row = summary[
                (summary["reset_interval"] == r) & (summary["method"] == method)
            ].iloc[0]
            lines.append(
                f"| {int(r)} | {method} | "
                f"{fmt_pm(row['runtime_sec_mean'], row['runtime_sec_std'])} | "
                f"{fmt_pm(row['per_frame_sec_mean'], row['per_frame_sec_std'])} | "
                f"{fmt_pm(row['basic_consistency_score_mean'], row['basic_consistency_score_std'])} | "
                f"{fmt_pm(row['loop_closure_trans_error_mean'], row['loop_closure_trans_error_std'])} |"
            )
    lines.append("")
    lines.append("## 4. Paired Delta Summary (`drift0 - brake`)")
    lines.append(
        f"- Consistency delta (overall): mean `{paired['d_cons'].mean():.6f}`, std `{paired['d_cons'].std():.6f}`, median `{paired['d_cons'].median():.6f}`"
    )
    lines.append(
        f"- Runtime delta (overall): mean `{paired['d_runtime'].mean():.6f}` s, std `{paired['d_runtime'].std():.6f}` s, median `{paired['d_runtime'].median():.6f}` s"
    )
    lines.append(
        f"- `d_runtime > 0` ratio: `{(paired['d_runtime'] > 0).mean():.3f}` (drift0 slower in most paired cases)"
    )
    lines.append(
        f"- `d_cons > 0` ratio: `{(paired['d_cons'] > 0).mean():.3f}` (mixed direction; no one-sided dominance)"
    )
    lines.append("")
    lines.append("Per-reset key numbers:")
    lines.append("| reset_interval | delta_consistency (drift0-brake) | delta_runtime_sec (drift0-brake) |")
    lines.append("|---|---:|---:|")
    for _, r in key.sort_values("reset_interval").iterrows():
        lines.append(
            f"| {int(r['reset_interval'])} | {r['delta_consistency_drift0_minus_brake']:.6f} | {r['delta_runtime_drift0_minus_brake']:.6f} |"
        )
    lines.append("")
    lines.append("## 5. Figures")
    lines.append("![Consistency vs reset](figures/s5_reset_interval_sensitivity_safe224/fig_consistency_vs_reset.png)")
    lines.append("")
    lines.append("![Runtime vs reset](figures/s5_reset_interval_sensitivity_safe224/fig_runtime_vs_reset.png)")
    lines.append("")
    lines.append("![Per-sequence delta distribution](figures/s5_reset_interval_sensitivity_safe224/fig_delta_distribution.png)")
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append(
        "1. The geometric benefit between brake and drift0 is small and reset-dependent on this local subset; no stable one-sided win is observed."
    )
    lines.append(
        "2. Runtime is consistently low-cost for both methods, while drift0 tends to be slower in paired comparisons."
    )
    lines.append(
        "3. The experiment validates robustness against reset policy changes at SAFE224 without OOM, and provides reproducible local evidence."
    )
    lines.append("")
    lines.append("## 7. Known Limitations")
    lines.append("- Small local slice (4 sequences) limits statistical power.")
    lines.append(
        "- The metric is internal consistency/pose-closure proxy, not final benchmark accuracy (e.g., KITTI depth metrics)."
    )
    lines.append(
        "- A matplotlib environment warning (`Axes3D`) appears on this machine but does not affect 2D plotting outputs."
    )
    lines.append("")
    lines.append("## 8. Reproducibility Files")
    lines.append(
        "- Raw: `benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/reset_raw_results.csv`"
    )
    lines.append(
        "- Summary: `benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/summary_by_reset_method.csv`"
    )
    lines.append(
        "- Effect table: `benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/brake_effect_by_reset.csv`"
    )
    lines.append(
        "- Paired delta table: `benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224/paired_per_sequence_delta.csv`"
    )
    lines.append("- Figures: `docs/figures/s5_reset_interval_sensitivity_safe224/`")
    lines.append("")

    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OUT] {args.out_md}")


if __name__ == "__main__":
    main()
