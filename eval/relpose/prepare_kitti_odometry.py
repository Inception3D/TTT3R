"""
Prepare KITTI odometry sequences for relpose evaluation.

Input structure (KITTI odometry dataset):
    <KITTI_ODO_ROOT>/
        sequences/
            00/image_2/*.png   (left color camera, 1241x376)
            02/image_2/*.png
            ...
        poses/
            00.txt             (N lines, each = 12 floats = 3x4 cam2world matrix)
            02.txt
            ...

Output structure (matches eval pipeline convention):
    data/long_kitti_odo_s1/
        00/
            image_200/         <- symlinks to first 200 frames
            pose_200.txt       <- TUM format: ts tx ty tz qx qy qz qw
            image_full/        <- symlinks to ALL frames (full sequence)
            pose_full.txt
        02/
            ...

Usage:
    python eval/relpose/prepare_kitti_odometry.py --kitti_root /path/to/kitti_odometry
    python eval/relpose/prepare_kitti_odometry.py --kitti_root /path/to/kitti_odometry --full
    python eval/relpose/prepare_kitti_odometry.py --kitti_root /path/to/kitti_odometry --seqs 00 02 05
"""

import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def kitti_pose_to_tum(row_12):
    """
    Convert one row of KITTI poses file (12 floats) to (t, q).
    KITTI format: row-major 3x4 world-to-camera matrix (cam0, left grayscale).
    We treat it as cam2world here — consistent with how evo computes ATE
    as long as all methods use the same convention.
    Returns:
        t: np.array [3]
        q: np.array [qx, qy, qz, qw]
    """
    mat34 = row_12.reshape(3, 4)
    mat44 = np.eye(4)
    mat44[:3, :] = mat34
    t = mat44[:3, 3]
    q = Rotation.from_matrix(mat44[:3, :3]).as_quat()  # [qx qy qz qw]
    return t, q


def _create_symlinks_and_poses(kitti_root, output_root, seq, all_frames, poses_raw, length, tag):
    """Create image symlinks and TUM pose file for a given length/tag."""
    img_src = os.path.join(kitti_root, "sequences", seq, "image_2")
    frames = all_frames[:length]

    # Create image directory with symlinks
    img_dst = os.path.join(output_root, seq, f"image_{tag}")
    os.makedirs(img_dst, exist_ok=True)
    for fname in frames:
        src = os.path.abspath(os.path.join(img_src, fname))
        dst = os.path.join(img_dst, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    # Write TUM pose file
    pose_dst = os.path.join(output_root, seq, f"pose_{tag}.txt")
    with open(pose_dst, "w") as f:
        for i in range(length):
            t, q = kitti_pose_to_tum(poses_raw[i])
            f.write(
                f"{i:06d} "
                f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n"
            )

    print(f"[OK] seq {seq}, tag {tag}: {length} frames -> {img_dst}")


def prepare_sequence(kitti_root, output_root, seq, lengths, full=False):
    img_src = os.path.join(kitti_root, "sequences", seq, "image_2")
    pose_src = os.path.join(kitti_root, "poses", f"{seq}.txt")

    if not os.path.isdir(img_src):
        print(f"[SKIP] image dir not found: {img_src}")
        return
    if not os.path.isfile(pose_src):
        print(f"[SKIP] pose file not found: {pose_src}")
        return

    all_frames = sorted(os.listdir(img_src))
    poses_raw = np.loadtxt(pose_src)  # (N, 12)
    n_available = min(len(all_frames), len(poses_raw))

    # Fixed-length configs
    for length in lengths:
        if n_available < length:
            print(f"[SKIP] seq {seq} has only {n_available} frames, need {length}")
            continue
        _create_symlinks_and_poses(kitti_root, output_root, seq, all_frames, poses_raw, length, str(length))

    # Full-length config
    if full:
        _create_symlinks_and_poses(kitti_root, output_root, seq, all_frames, poses_raw, n_available, "full")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_root",
        type=str,
        required=True,
        help="Path to KITTI odometry root (contains sequences/ and poses/)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/long_kitti_odo_s1",
        help="Output directory (default: data/long_kitti_odo_s1)",
    )
    parser.add_argument(
        "--seqs",
        nargs="+",
        default=["00", "02", "05", "07", "08"],
        help="Sequences to prepare (default: 00 02 05 07 08)",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[200, 500, 1000],
        help="Frame lengths to prepare (default: 200 500 1000)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Also prepare full-length sequences (all available frames)",
    )
    args = parser.parse_args()

    print(f"KITTI root : {args.kitti_root}")
    print(f"Output root: {args.output_root}")
    print(f"Sequences  : {args.seqs}")
    print(f"Lengths    : {args.lengths}")
    print(f"Full       : {args.full}")
    print()

    for seq in args.seqs:
        prepare_sequence(args.kitti_root, args.output_root, seq, args.lengths, full=args.full)


if __name__ == "__main__":
    main()
