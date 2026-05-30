"""
Experiment A2 (proxy): cosine-variance versus convergence-improvement analysis.

This is a local, data-limited proxy for the full paper analysis:
1. Split local sequences into multiple temporal windows
2. For each window, run two update types with `forward_recurrent_analysis`
3. Measure state-dynamics variability via Var(cos(delta_t, delta_{t-1}))
4. Measure improvement as reduction in mean delta norm of method vs baseline
5. Plot variance-improvement scatter and report correlations

This script is intended as a reusable analysis pipeline. When formal per-scene
cosine logs become available for ScanNet/TUM, the same plotting logic can be
reused with formal relpose improvements.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from add_ckpt_path import add_path_to_dust3r


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy A2: correlate cosine variance with convergence improvement."
    )
    parser.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    parser.add_argument(
        "--seq_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more sequence directories containing images.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--baseline_type", type=str, default="cut3r")
    parser.add_argument("--method_type", type=str, default="ttt3r")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--window_stride", type=int, default=6)
    parser.add_argument("--max_windows_per_seq", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_img_paths(seq_path: str, frame_interval: int) -> list[str]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    all_paths = sorted(glob.glob(os.path.join(seq_path, "*")))
    return [p for p in all_paths if os.path.splitext(p)[1].lower() in img_exts][::frame_interval]


def build_views(img_paths: list[str], size: int) -> list[dict]:
    from dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []
    for i, img_dict in enumerate(images):
        views.append(
            {
                "img": img_dict["img"],
                "ray_map": torch.full(
                    (
                        img_dict["img"].shape[0],
                        6,
                        img_dict["img"].shape[-2],
                        img_dict["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(img_dict["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )
    return views


def compute_dynamics(state_history: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    deltas = []
    for t in range(1, len(state_history)):
        deltas.append((state_history[t] - state_history[t - 1]).reshape(-1))

    if not deltas:
        return np.zeros(1), np.asarray([np.nan], dtype=np.float64)

    delta_norms = [0.0]
    for d in deltas:
        delta_norms.append(float(d.norm().item()))

    cosines = [np.nan]
    for i in range(1, len(deltas)):
        prev_d = deltas[i - 1]
        curr_d = deltas[i]
        denom = prev_d.norm() * curr_d.norm()
        if float(denom.item()) < 1e-12:
            cosines.append(np.nan)
        else:
            cosines.append(float(torch.dot(prev_d, curr_d).item() / denom.item()))

    if len(cosines) < len(delta_norms):
        cosines.append(np.nan)

    return np.asarray(delta_norms), np.asarray(cosines[: len(delta_norms)])


def iter_windows(img_paths: list[str], window_size: int, window_stride: int, max_windows: int):
    yielded = 0
    for start in range(0, max(0, len(img_paths) - window_size + 1), window_stride):
        end = start + window_size
        if end > len(img_paths):
            break
        yield start, end, img_paths[start:end]
        yielded += 1
        if yielded >= max_windows:
            break


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return safe_corr(rankdata(x), rankdata(y))


def plot_scatter(x: np.ndarray, y: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(x, y, s=36, alpha=0.85, color="tab:blue")
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=7, alpha=0.8)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Variance of cos(delta_t, delta_{t-1})")
    ax.set_ylabel("Convergence improvement (%)")
    ax.set_title("A2 proxy: cosine variance vs convergence improvement")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(args.model_path).to(args.device)
    model.eval()

    rows: list[dict[str, object]] = []

    for seq_path in args.seq_paths:
        img_paths = load_img_paths(seq_path, args.frame_interval)
        seq_name = Path(seq_path).parent.name
        for window_idx, (start, end, window_paths) in enumerate(
            iter_windows(img_paths, args.window_size, args.window_stride, args.max_windows_per_seq)
        ):
            stats = {}
            for update_type in (args.baseline_type, args.method_type):
                model.config.model_update_type = update_type
                views = build_views(window_paths, args.size)
                with torch.no_grad():
                    _, analysis_data = model.forward_recurrent_analysis(views, device=args.device)
                delta_norms, cosines = compute_dynamics(analysis_data["state_history"])
                stats[update_type] = {
                    "mean_delta_norm": float(np.nanmean(delta_norms)),
                    "last_delta_norm": float(delta_norms[-1]),
                    "mean_delta_cosine": float(np.nanmean(cosines)),
                    "var_delta_cosine": float(np.nanvar(cosines)),
                }

            improve_pct = (
                (stats[args.baseline_type]["mean_delta_norm"] - stats[args.method_type]["mean_delta_norm"])
                / (stats[args.baseline_type]["mean_delta_norm"] + 1e-12)
                * 100.0
            )
            rows.append(
                {
                    "sequence": seq_name,
                    "window_idx": window_idx,
                    "start_frame": start,
                    "end_frame": end - 1,
                    "baseline_type": args.baseline_type,
                    "method_type": args.method_type,
                    "baseline_mean_delta_norm": stats[args.baseline_type]["mean_delta_norm"],
                    "method_mean_delta_norm": stats[args.method_type]["mean_delta_norm"],
                    "baseline_var_delta_cosine": stats[args.baseline_type]["var_delta_cosine"],
                    "method_var_delta_cosine": stats[args.method_type]["var_delta_cosine"],
                    "baseline_mean_delta_cosine": stats[args.baseline_type]["mean_delta_cosine"],
                    "method_mean_delta_cosine": stats[args.method_type]["mean_delta_cosine"],
                    "convergence_improve_pct": improve_pct,
                }
            )

    if not rows:
        raise RuntimeError("No windows were produced. Check sequence lengths and window settings.")

    x = np.asarray([float(r["baseline_var_delta_cosine"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["convergence_improve_pct"]) for r in rows], dtype=np.float64)
    labels = [f'{r["sequence"]}:{r["window_idx"]}' for r in rows]

    pearson = safe_corr(x, y)
    spearman = spearman_corr(x, y)

    with (output_dir / "a2_proxy_points.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_points": int(len(rows)),
        "baseline_type": args.baseline_type,
        "method_type": args.method_type,
        "mean_baseline_var_delta_cosine": float(np.mean(x)),
        "mean_convergence_improve_pct": float(np.mean(y)),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
    }
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    plot_scatter(x, y, labels, output_dir / "variance_vs_improvement.png")

    with (output_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write(
            "A2 proxy outputs:\n"
            "- a2_proxy_points.csv: per-window variance and improvement values\n"
            "- summary.csv: correlation summary\n"
            "- variance_vs_improvement.png: scatter plot\n"
            "\n"
            "This is a local proxy using available CO3D windows and convergence-improvement,\n"
            "not the final formal relpose-based A2 analysis.\n"
        )

    print(f"[OK] Wrote {output_dir / 'a2_proxy_points.csv'}")
    print(f"[OK] Wrote {output_dir / 'summary.csv'}")
    print(f"[OK] Wrote {output_dir / 'variance_vs_improvement.png'}")


if __name__ == "__main__":
    main()
