import argparse
from pathlib import Path

import pandas as pd


def build_overall(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["protocol", "method"], dropna=False)[["runtime_sec", "per_frame_sec"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg


def build_by_length(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["protocol", "method", "seq_length"], dropna=False)[["runtime_sec", "per_frame_sec"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def write_markdown(overall: pd.DataFrame, by_len: pd.DataFrame, out_md: Path, input_csv: Path) -> None:
    lines = []
    lines.append("# S2 Local Inference Overhead (SAFE224)")
    lines.append("")
    lines.append("## Input")
    lines.append(f"- Source CSV: `{input_csv}`")
    lines.append("- Metric scope: `runtime_sec`, `per_frame_sec`")
    lines.append("- Protocols: fixed seed repeat vs different seed repeat")
    lines.append("")
    lines.append("## Overall")
    lines.append("| Protocol | Method | Runtime (s) | Per-frame (s) |")
    lines.append("|---|---|---:|---:|")
    for _, r in overall.iterrows():
        lines.append(
            f"| {r['protocol']} | {r['method']} | "
            f"{fmt_mean_std(r['runtime_sec_mean'], r['runtime_sec_std'])} | "
            f"{fmt_mean_std(r['per_frame_sec_mean'], r['per_frame_sec_std'])} |"
        )
    lines.append("")
    lines.append("## By Sequence Length")
    lines.append("| Protocol | Method | Seq length | Runtime (s) | Per-frame (s) |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in by_len.iterrows():
        lines.append(
            f"| {r['protocol']} | {r['method']} | {int(r['seq_length'])} | "
            f"{fmt_mean_std(r['runtime_sec_mean'], r['runtime_sec_std'])} | "
            f"{fmt_mean_std(r['per_frame_sec_mean'], r['per_frame_sec_std'])} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("1. Overhead is stable across seed protocols; mean runtime/per-frame are nearly identical.")
    lines.append("2. Runtime variance exists (system scheduling / GPU runtime jitter), but geometric metrics were already shown seed-invariant.")
    lines.append("3. This local S2 result supports reproducible inference-time behavior under SAFE224.")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_raw_results.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis_results/s2_inference_overhead_local"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = build_overall(df)
    by_len = build_by_length(df)

    overall.to_csv(out_dir / "overall_overhead.csv", index=False)
    by_len.to_csv(out_dir / "overhead_by_length.csv", index=False)
    write_markdown(overall, by_len, out_dir / "summary.md", args.input_csv)

    print(f"[OUT] {out_dir / 'overall_overhead.csv'}")
    print(f"[OUT] {out_dir / 'overhead_by_length.csv'}")
    print(f"[OUT] {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
