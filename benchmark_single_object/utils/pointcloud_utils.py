from __future__ import annotations

from pathlib import Path

import numpy as np


def count_output_points(output_dir: Path, conf_threshold: float = 1.0) -> int:
    conf_dir = output_dir / "conf"
    if not conf_dir.exists():
        return 0
    total = 0
    for conf_path in sorted(conf_dir.glob("*.npy")):
        conf = np.load(conf_path)
        total += int((conf > conf_threshold).sum())
    return total


def count_processed_frames(output_dir: Path) -> int:
    depth_dir = output_dir / "depth"
    if not depth_dir.exists():
        return 0
    return len(list(depth_dir.glob("*.npy")))


def compute_conf_stats(output_dir: Path) -> dict[str, float]:
    conf_dir = output_dir / "conf"
    if not conf_dir.exists():
        return {"mean_conf": float("nan"), "median_conf": float("nan")}
    values = []
    for conf_path in sorted(conf_dir.glob("*.npy")):
        conf = np.load(conf_path)
        values.append(conf.reshape(-1))
    if not values:
        return {"mean_conf": float("nan"), "median_conf": float("nan")}
    all_conf = np.concatenate(values, axis=0)
    return {
        "mean_conf": float(np.mean(all_conf)),
        "median_conf": float(np.median(all_conf)),
    }
