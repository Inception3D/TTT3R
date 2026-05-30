from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.trace(R)
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cos_theta)))


def compute_camera_consistency(camera_dir: Path) -> dict[str, float]:
    if not camera_dir.exists():
        return {
            "loop_closure_trans_error": float("nan"),
            "loop_closure_rot_error_deg": float("nan"),
            "pose_step_translation_mean": float("nan"),
            "pose_step_translation_std": float("nan"),
            "basic_consistency_score": float("nan"),
        }

    camera_paths = sorted(camera_dir.glob("*.npz"))
    if len(camera_paths) < 2:
        return {
            "loop_closure_trans_error": float("nan"),
            "loop_closure_rot_error_deg": float("nan"),
            "pose_step_translation_mean": float("nan"),
            "pose_step_translation_std": float("nan"),
            "basic_consistency_score": float("nan"),
        }

    poses = []
    for p in camera_paths:
        with np.load(p) as data:
            poses.append(data["pose"])
    poses = np.asarray(poses)

    centers = poses[:, :3, 3]
    rel_steps = centers[1:] - centers[:-1]
    step_norms = np.linalg.norm(rel_steps, axis=1)
    step_mean = float(np.mean(step_norms))
    step_std = float(np.std(step_norms))

    first_pose = poses[0]
    last_pose = poses[-1]
    loop_trans = float(np.linalg.norm(last_pose[:3, 3] - first_pose[:3, 3]))
    loop_rot = _rotation_angle_deg(last_pose[:3, :3] @ first_pose[:3, :3].T)

    # Lower is better: combines closure and trajectory smoothness.
    score = loop_trans + 0.01 * loop_rot + step_std

    return {
        "loop_closure_trans_error": loop_trans,
        "loop_closure_rot_error_deg": loop_rot,
        "pose_step_translation_mean": step_mean,
        "pose_step_translation_std": step_std,
        "basic_consistency_score": float(score),
    }

