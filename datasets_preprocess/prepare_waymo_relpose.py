import argparse
import glob
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Waymo Open Dataset TFRecords to TTT3R relpose format."
    )
    parser.add_argument(
        "--tfrecord_glob",
        type=str,
        required=True,
        help="Glob for Waymo TFRecord files, e.g. '/data/waymo/training/*.tfrecord'.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/waymo_relpose",
        help="Output root folder for converted sequences.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="FRONT",
        choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        help="Camera stream to export.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=8,
        help="Maximum number of TFRecord segments to convert.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Maximum exported frames per segment.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sample every N frames per segment.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=30,
        help="Keep only sequences with at least this many frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_root if it already exists.",
    )
    return parser.parse_args()


def _load_waymo_modules():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is missing. Install a CUDA-matching build before conversion."
        ) from exc

    try:
        from waymo_open_dataset import dataset_pb2
    except ImportError as exc:
        raise ImportError(
            "Waymo Open Dataset API is missing. Install package waymo-open-dataset-tf-* matching your TF version."
        ) from exc

    return tf, dataset_pb2


def _camera_enum(dataset_pb2, name: str):
    mapping = {
        "FRONT": dataset_pb2.CameraName.FRONT,
        "FRONT_LEFT": dataset_pb2.CameraName.FRONT_LEFT,
        "FRONT_RIGHT": dataset_pb2.CameraName.FRONT_RIGHT,
        "SIDE_LEFT": dataset_pb2.CameraName.SIDE_LEFT,
        "SIDE_RIGHT": dataset_pb2.CameraName.SIDE_RIGHT,
    }
    return mapping[name]


def _pose_row_from_frame(frame) -> np.ndarray:
    mat = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    return mat[:3, :].reshape(-1)


def _camera_pose_row_from_frame(frame, camera_id: int) -> np.ndarray:
    t_global_vehicle = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)

    camera_extrinsic = None
    for calib in frame.context.camera_calibrations:
        if calib.name == camera_id:
            camera_extrinsic = np.asarray(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
            break

    if camera_extrinsic is None:
        return _pose_row_from_frame(frame)

    # Waymo camera extrinsic is camera->vehicle. Compose with vehicle->global.
    t_global_camera = t_global_vehicle @ camera_extrinsic
    return t_global_camera[:3, :].reshape(-1)


def convert(args):
    tf, dataset_pb2 = _load_waymo_modules()
    camera_id = _camera_enum(dataset_pb2, args.camera)

    # Use glob.glob directly to support absolute patterns on Linux.
    tfrecord_paths = [Path(p) for p in sorted(glob.glob(args.tfrecord_glob))]
    if not tfrecord_paths:
        raise FileNotFoundError(f"No TFRecords matched: {args.tfrecord_glob}")

    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    for tfrecord_path in tfrecord_paths[: args.max_segments]:
        seq_name = tfrecord_path.stem
        seq_root = output_root / seq_name
        rgb_dir = seq_root / "rgb_90"
        pose_path = seq_root / "pose_90.txt"
        rgb_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        export_frame_idx = 0
        source_frame_idx = 0

        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            if source_frame_idx % args.stride != 0:
                source_frame_idx += 1
                continue

            camera_image = None
            for img in frame.images:
                if img.name == camera_id:
                    camera_image = img
                    break

            if camera_image is None:
                source_frame_idx += 1
                continue

            img_dst = rgb_dir / f"frame_{export_frame_idx:06d}.jpg"
            with img_dst.open("wb") as f:
                f.write(camera_image.image)

            rows.append(_camera_pose_row_from_frame(frame, camera_id))

            export_frame_idx += 1
            if export_frame_idx >= args.max_frames:
                break

            source_frame_idx += 1

        if export_frame_idx < args.min_frames:
            skipped += 1
            if seq_root.exists():
                shutil.rmtree(seq_root)
            continue

        np.savetxt(pose_path, np.asarray(rows), fmt="%.8f")
        kept += 1
        print(
            f"[waymo] kept {seq_name}: frames={export_frame_idx}, camera={args.camera}, pose={pose_path}"
        )

    print(f"[waymo] done. kept={kept}, skipped={skipped}, output={output_root}")


if __name__ == "__main__":
    convert(parse_args())
