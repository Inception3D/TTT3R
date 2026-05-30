"""
Experiment A4: State convergence analysis.

Compare how different `model_update_type` settings affect:
1. State delta norm over time
2. Consecutive-delta cosine alignment over time

This script uses the existing `forward_recurrent_analysis` path and only
derives dynamics from the returned `state_history`.
"""

from __future__ import annotations

import argparse
import csv
import glob
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
    parser = argparse.ArgumentParser(description="Analyze state convergence dynamics.")
    parser.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--seq_path", type=str, required=True, help="Directory of images.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--update_types",
        type=str,
        nargs="+",
        default=["cut3r", "ttt3r"],
        help="Model update types to compare.",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_img_paths(seq_path: str, frame_interval: int, max_frames: int) -> list[str]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    all_paths = sorted(glob.glob(os.path.join(seq_path, "*")))
    img_paths = [p for p in all_paths if os.path.splitext(p)[1].lower() in img_exts]
    return img_paths[::frame_interval][:max_frames]


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
        return np.zeros(1), np.zeros(1)

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


def plot_series(frame_idx: np.ndarray, series: dict[str, np.ndarray], ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for label, values in series.items():
        ax.plot(frame_idx, values, linewidth=1.8, alpha=0.9, label=label)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = load_img_paths(args.seq_path, args.frame_interval, args.max_frames)
    if not img_paths:
        raise RuntimeError(f"No images found under {args.seq_path}")

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    result_rows: list[dict[str, object]] = []
    norm_series: dict[str, np.ndarray] = {}
    cosine_series: dict[str, np.ndarray] = {}

    for update_type in args.update_types:
        print(f"[run] update_type={update_type}")
        model = ARCroco3DStereo.from_pretrained(args.model_path).to(args.device)
        model.config.model_update_type = update_type
        model.eval()

        views = build_views(img_paths, args.size)
        with torch.no_grad():
            _, analysis_data = model.forward_recurrent_analysis(views, device=args.device)

        delta_norms, cosines = compute_dynamics(analysis_data["state_history"])
        norm_series[update_type] = delta_norms
        cosine_series[update_type] = cosines

        for idx in range(len(delta_norms)):
            result_rows.append(
                {
                    "update_type": update_type,
                    "frame_idx": idx,
                    "delta_norm": float(delta_norms[idx]),
                    "delta_cosine": float(cosines[idx]) if np.isfinite(cosines[idx]) else np.nan,
                }
            )

    with (output_dir / "state_convergence.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    frame_idx = np.arange(max(len(v) for v in norm_series.values()))
    plot_series(
        frame_idx,
        norm_series,
        "||delta_t||",
        "State delta norm over time",
        output_dir / "delta_norm_curve.png",
    )
    plot_series(
        frame_idx,
        cosine_series,
        "cos(delta_t, delta_{t-1})",
        "Consecutive delta cosine alignment",
        output_dir / "delta_cosine_curve.png",
    )

    summary_rows = []
    for update_type in args.update_types:
        norms = norm_series[update_type]
        cos = cosine_series[update_type]
        summary_rows.append(
            {
                "update_type": update_type,
                "frames": int(len(norms)),
                "mean_delta_norm": float(np.nanmean(norms)),
                "last_delta_norm": float(norms[-1]),
                "mean_delta_cosine": float(np.nanmean(cos)),
            }
        )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Wrote {output_dir / 'state_convergence.csv'}")
    print(f"[OK] Wrote {output_dir / 'summary.csv'}")
    print(f"[OK] Wrote {output_dir / 'delta_norm_curve.png'}")
    print(f"[OK] Wrote {output_dir / 'delta_cosine_curve.png'}")


if __name__ == "__main__":
    main()
