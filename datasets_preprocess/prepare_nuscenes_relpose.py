import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert nuScenes camera streams to TTT3R relpose format."
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="nuScenes dataroot that contains samples/, sweeps/, maps/, v1.0-*/.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="nuScenes version, e.g. v1.0-mini or v1.0-trainval.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/nuscenes_relpose",
        help="Output root folder for converted sequences.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="CAM_FRONT",
        help="Camera channel name, e.g. CAM_FRONT / CAM_FRONT_LEFT.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=10,
        help="Maximum number of scenes to export.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Maximum exported frames per scene.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sample every N frames along the scene timeline.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=30,
        help="Keep only sequences with at least this many frames.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Copy image files or create symlinks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_root if it already exists.",
    )
    return parser.parse_args()


def _safe_copy_or_link(src: Path, dst: Path, copy_mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            # Fall back to file copy if symlink is unavailable.
            pass
    shutil.copy2(src, dst)


def _load_nuscenes(version: str, dataroot: str):
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise ImportError(
            "nuScenes devkit is missing. Install with: pip install nuscenes-devkit"
        ) from exc
    return NuScenes(version=version, dataroot=dataroot, verbose=True)


def _pose_matrix_to_row(mat44: np.ndarray) -> np.ndarray:
    return mat44[:3, :].reshape(-1)


def convert(args):
    nusc = _load_nuscenes(args.version, args.dataroot)
    try:
        from pyquaternion import Quaternion
        from nuscenes.utils.geometry_utils import transform_matrix
    except ImportError as exc:
        raise ImportError(
            "Missing nuScenes geometry dependencies. Install with: pip install pyquaternion nuscenes-devkit"
        ) from exc

    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scenes = sorted(nusc.scene, key=lambda x: x["name"])[: args.max_scenes]
    kept = 0
    skipped = 0

    for scene in scenes:
        scene_name = scene["name"]
        seq_root = output_root / scene_name
        rgb_dir = seq_root / "rgb_90"
        pose_path = seq_root / "pose_90.txt"
        rgb_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        sample_token = scene["first_sample_token"]
        source_frame_idx = 0
        export_frame_idx = 0

        while sample_token:
            sample = nusc.get("sample", sample_token)
            next_token = sample["next"] if sample["next"] != "" else None

            if source_frame_idx % args.stride != 0:
                source_frame_idx += 1
                sample_token = next_token
                continue

            camera_token = sample["data"].get(args.camera, None)
            if camera_token is None:
                source_frame_idx += 1
                sample_token = next_token
                continue

            sd = nusc.get("sample_data", camera_token)
            img_src = Path(args.dataroot) / sd["filename"]
            if not img_src.exists():
                raise FileNotFoundError(f"Missing image file: {img_src}")

            ext = img_src.suffix.lower() if img_src.suffix else ".jpg"
            img_dst = rgb_dir / f"frame_{export_frame_idx:06d}{ext}"
            _safe_copy_or_link(img_src, img_dst, args.copy_mode)

            calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ego = nusc.get("ego_pose", sd["ego_pose_token"])
            t_global_ego = transform_matrix(
                ego["translation"], Quaternion(ego["rotation"]), inverse=False
            )
            t_ego_cam = transform_matrix(
                calib["translation"], Quaternion(calib["rotation"]), inverse=False
            )
            t_global_cam = t_global_ego @ t_ego_cam
            rows.append(_pose_matrix_to_row(t_global_cam))

            export_frame_idx += 1
            if export_frame_idx >= args.max_frames:
                break

            source_frame_idx += 1
            sample_token = next_token

        if export_frame_idx < args.min_frames:
            skipped += 1
            if seq_root.exists():
                shutil.rmtree(seq_root)
            continue

        np.savetxt(pose_path, np.asarray(rows), fmt="%.8f")
        kept += 1
        print(
            f"[nuscenes] kept {scene_name}: frames={export_frame_idx}, camera={args.camera}, pose={pose_path}"
        )

    print(f"[nuscenes] done. kept={kept}, skipped={skipped}, output={output_root}")


if __name__ == "__main__":
    convert(parse_args())
