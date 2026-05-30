import argparse
import csv
import re
from pathlib import Path


AVG_RE = re.compile(
    r"Average ATE:\s*([0-9.]+),\s*Average RPE trans:\s*([0-9.]+),\s*Average RPE rot:\s*([0-9.]+)"
)
SEQ_RE = re.compile(
    r"^(?P<dataset_seq>[^|]+)\|\s*ATE:\s*(?P<ate>[0-9.]+),\s*RPE trans:\s*(?P<rpe_t>[0-9.]+),\s*RPE rot:\s*(?P<rpe_r>[0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export relpose _error_log.txt files to summary/per-sequence CSV."
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        required=True,
        help="Root like eval_results/relpose/nuscenes_relpose",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="summary.csv",
        help="Summary CSV filename under eval_root.",
    )
    parser.add_argument(
        "--per_sequence_csv",
        type=str,
        default="per_sequence_results.csv",
        help="Per-sequence CSV filename under eval_root.",
    )
    parser.add_argument(
        "--summary_md",
        type=str,
        default="summary.md",
        help="Markdown summary filename under eval_root.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main():
    args = parse_args()
    eval_root = Path(args.eval_root)
    if not eval_root.exists():
        raise FileNotFoundError(f"eval_root does not exist: {eval_root}")

    summary_rows = []
    per_seq_rows = []

    model_dirs = sorted([p for p in eval_root.iterdir() if p.is_dir()])
    for model_dir in model_dirs:
        log_path = model_dir / "_error_log.txt"
        if not log_path.exists():
            continue
        text = read_text(log_path)

        avg_match = AVG_RE.search(text)
        if avg_match:
            summary_rows.append(
                {
                    "model": model_dir.name,
                    "avg_ate": float(avg_match.group(1)),
                    "avg_rpe_trans": float(avg_match.group(2)),
                    "avg_rpe_rot": float(avg_match.group(3)),
                    "log_path": str(log_path),
                }
            )

        for line in text.splitlines():
            line = line.strip()
            seq_match = SEQ_RE.match(line)
            if not seq_match:
                continue
            dataset_seq = seq_match.group("dataset_seq").strip()
            if "-" in dataset_seq:
                dataset_name, seq_name = dataset_seq.split("-", 1)
            else:
                dataset_name, seq_name = "", dataset_seq
            per_seq_rows.append(
                {
                    "model": model_dir.name,
                    "dataset": dataset_name.strip(),
                    "sequence": seq_name.strip(),
                    "ate": float(seq_match.group("ate")),
                    "rpe_trans": float(seq_match.group("rpe_t")),
                    "rpe_rot": float(seq_match.group("rpe_r")),
                }
            )

    summary_csv_path = eval_root / args.summary_csv
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "avg_ate", "avg_rpe_trans", "avg_rpe_rot", "log_path"]
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    per_seq_csv_path = eval_root / args.per_sequence_csv
    with per_seq_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "dataset", "sequence", "ate", "rpe_trans", "rpe_rot"]
        )
        writer.writeheader()
        for row in per_seq_rows:
            writer.writerow(row)

    summary_md_path = eval_root / args.summary_md
    lines = ["# Relpose Summary", ""]
    if summary_rows:
        lines.append("| model | avg_ate | avg_rpe_trans | avg_rpe_rot |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in sorted(summary_rows, key=lambda x: x["model"]):
            lines.append(
                f"| {row['model']} | {row['avg_ate']:.6f} | {row['avg_rpe_trans']:.6f} | {row['avg_rpe_rot']:.6f} |"
            )
    else:
        lines.append("No summary rows found. Check whether `_error_log.txt` exists.")
    lines.append("")
    lines.append(f"- summary_csv: `{summary_csv_path}`")
    lines.append(f"- per_sequence_csv: `{per_seq_csv_path}`")
    lines.append(f"- parsed_sequences: `{len(per_seq_rows)}`")
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved summary: {summary_csv_path}")
    print(f"saved per-sequence: {per_seq_csv_path}")
    print(f"saved markdown: {summary_md_path}")


if __name__ == "__main__":
    main()
