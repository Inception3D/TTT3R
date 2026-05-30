from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import sys

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmark_single_object"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from scripts.run_one_method import run_one_experiment
from utils.sampling_utils import parse_length_from_seq_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local SAFE224 reset-interval sensitivity for brake (train-free)."
    )
    parser.add_argument(
        "--sequences_root",
        type=Path,
        default=Path("/home/chen/TTT3R/benchmark_single_object/outputs_ablation/sequences"),
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("/home/chen/TTT3R/model/cut3r_512_dpt_4_64.pth"),
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("/home/chen/TTT3R"),
    )
    parser.add_argument(
        "--predictions_root",
        type=Path,
        default=Path(
            "/home/chen/TTT3R/benchmark_single_object/outputs_ablation_safe/predictions/reset_interval_sensitivity_safe224"
        ),
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=Path(
            "benchmark_single_object/outputs_ablation_safe/metrics/reset_interval_sensitivity_safe224"
        ),
    )
    parser.add_argument("--reset_intervals", type=str, default="4,8,16,100")
    parser.add_argument("--seeds", type=str, default="41,42")
    parser.add_argument("--lengths", type=str, default="12,24")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--downsample_factor", type=int, default=100)
    parser.add_argument("--timeout_sec", type=int, default=7200)
    parser.add_argument("--retry_on_empty", type=int, default=1)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def discover_sequences(root: Path, allowed_lengths: set[int]) -> list[Path]:
    seqs: list[Path] = []
    for object_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for seq_dir in sorted([p for p in object_dir.iterdir() if p.is_dir()]):
            seq_len = parse_length_from_seq_name(seq_dir.name)
            if seq_len is None or seq_len not in allowed_lengths:
                continue
            if list(seq_dir.glob("*.jpg")):
                seqs.append(seq_dir.resolve())
    return seqs


def should_retry(row: dict[str, Any]) -> bool:
    if bool(row.get("timed_out", False)):
        return True
    if int(row.get("processed_frames", 0)) <= 0:
        return True
    runtime = float(row.get("runtime_sec", float("nan")))
    if math.isnan(runtime) or runtime <= 0.0:
        return True
    return False


def run_once(
    *,
    method_name: str,
    alpha_drift: float,
    seq_path: Path,
    seed: int,
    reset_interval: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ns = SimpleNamespace(
        method=method_name,
        model_update_type="ttt3r_momentum_inv_t1",
        alpha_drift=alpha_drift,
        seq_path=str(seq_path),
        model_path=str(args.model_path),
        output_root=str(args.predictions_root),
        python_exe=sys.executable,
        device=args.device,
        size=args.size,
        vis_threshold=6.0,
        frame_interval=args.frame_interval,
        reset_interval=reset_interval,
        downsample_factor=args.downsample_factor,
        seed=seed,
        timeout_sec=args.timeout_sec,
        repo_root=str(args.repo_root),
        skip_if_done=False,
    )
    return run_one_experiment(ns)


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [
        "runtime_sec",
        "per_frame_sec",
        "peak_vram_mb",
        "basic_consistency_score",
        "loop_closure_trans_error",
        "loop_closure_rot_error_deg",
    ]
    summary = (
        df.groupby(["method", "reset_interval"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary.to_csv(out_dir / "summary_by_reset_method.csv", index=False)

    # drift>0 vs drift0 gap by reset interval (negative means drift>0 better for "lower is better" metrics).
    pivot = (
        summary.set_index(["method", "reset_interval"])
        .sort_index()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for r in sorted(df["reset_interval"].unique().tolist()):
        a = summary[
            (summary["method"] == "ttt3r_momentum_inv_t1")
            & (summary["reset_interval"] == r)
        ]
        b = summary[
            (summary["method"] == "ttt3r_momentum_inv_t1_drift0")
            & (summary["reset_interval"] == r)
        ]
        if a.empty or b.empty:
            continue
        a = a.iloc[0]
        b = b.iloc[0]
        rows.append(
            {
                "reset_interval": int(r),
                "delta_runtime_sec_mean_drift0_minus_driftpos": float(
                    b["runtime_sec_mean"] - a["runtime_sec_mean"]
                ),
                "delta_per_frame_sec_mean_drift0_minus_driftpos": float(
                    b["per_frame_sec_mean"] - a["per_frame_sec_mean"]
                ),
                "delta_consistency_mean_drift0_minus_driftpos": float(
                    b["basic_consistency_score_mean"] - a["basic_consistency_score_mean"]
                ),
                "delta_loop_trans_mean_drift0_minus_driftpos": float(
                    b["loop_closure_trans_error_mean"] - a["loop_closure_trans_error_mean"]
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "brake_effect_by_reset.csv", index=False)


def main() -> None:
    args = parse_args()
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.predictions_root.mkdir(parents=True, exist_ok=True)

    reset_intervals = parse_int_list(args.reset_intervals)
    seeds = parse_int_list(args.seeds)
    lengths = set(parse_int_list(args.lengths))
    seqs = discover_sequences(args.sequences_root, lengths)
    if not seqs:
        raise RuntimeError(f"No valid sequences found in {args.sequences_root}")

    methods = [
        ("ttt3r_momentum_inv_t1", 0.15),
        ("ttt3r_momentum_inv_t1_drift0", 0.0),
    ]

    raw_csv = args.metrics_dir / "reset_raw_results.csv"
    rows: list[dict[str, Any]] = []

    total = len(reset_intervals) * len(seeds) * len(methods) * len(seqs)
    idx = 0
    for reset_interval in reset_intervals:
        for seed in seeds:
            for method_name, alpha_drift in methods:
                for seq in seqs:
                    idx += 1
                    print(
                        f"[{idx}/{total}] reset={reset_interval} seed={seed} "
                        f"method={method_name} seq={seq.name}"
                    )

                    row = run_once(
                        method_name=method_name,
                        alpha_drift=alpha_drift,
                        seq_path=seq,
                        seed=seed,
                        reset_interval=reset_interval,
                        args=args,
                    )

                    attempt = 0
                    while should_retry(row) and attempt < args.retry_on_empty:
                        attempt += 1
                        print(
                            f"  [retry {attempt}] timed_out={row.get('timed_out')} "
                            f"processed_frames={row.get('processed_frames')}"
                        )
                        time.sleep(2.0)
                        row = run_once(
                            method_name=method_name,
                            alpha_drift=alpha_drift,
                            seq_path=seq,
                            seed=seed,
                            reset_interval=reset_interval,
                            args=args,
                        )

                    row["reset_interval"] = int(reset_interval)
                    row["seed_eval"] = int(seed)
                    row["run_ok"] = int(not should_retry(row))
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(raw_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(raw_csv, index=False)
    summarize(df, args.metrics_dir)

    print(f"[DONE] reset sensitivity run finished: {len(df)} rows")
    print(f"[OUT] {raw_csv}")
    print(f"[OUT] {args.metrics_dir / 'summary_by_reset_method.csv'}")
    print(f"[OUT] {args.metrics_dir / 'brake_effect_by_reset.csv'}")


if __name__ == "__main__":
    main()
