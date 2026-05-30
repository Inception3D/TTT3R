import argparse
import glob
import os
import re
import shutil
from pathlib import Path


DEFAULT_TARGET_FRAMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
DEPTH_SELECTION_DEPTH_RE = re.compile(
    r"^(?P<seq>.+?)_groundtruth_depth_(?P<frame>\d+)_image_(?P<cam>\d{2})\.png$"
)
DEPTH_SELECTION_IMAGE_RE = re.compile(
    r"^(?P<seq>.+?)_image_(?P<frame>\d+)_image_(?P<cam>\d{2})\.png$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare long KITTI video-depth subsets for TTT3R evaluation."
    )
    parser.add_argument(
        "--source_root",
        type=str,
        required=True,
        help="Root containing KITTI val sequences with proj_depth/groundtruth/image_02.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/long_kitti_s1",
        help="Output root for gathered long KITTI subsets.",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_FRAMES,
        help="Target sequence lengths to export.",
    )
    return parser.parse_args()


def find_sequence_pairs(source_root):
    source_root = Path(source_root)

    # Official KITTI depth-selection package layout:
    # depth_selection/val_selection_cropped/{groundtruth_depth,image}/<seq_name>/*.png
    selection_root = source_root
    if (selection_root / "depth_selection" / "val_selection_cropped").exists():
        selection_root = selection_root / "depth_selection" / "val_selection_cropped"
    if (selection_root / "groundtruth_depth").exists() and (
        selection_root / "image"
    ).exists():
        # Some KITTI releases store each scene in its own directory.
        pairs = []
        for depth_dir in sorted((selection_root / "groundtruth_depth").glob("*")):
            if not depth_dir.is_dir():
                continue
            image_dir = selection_root / "image" / depth_dir.name
            if not image_dir.is_dir():
                print(f"missing image directory for {depth_dir.name}: {image_dir}")
                continue
            pairs.append((depth_dir.name, depth_dir, image_dir))
        if pairs:
            return pairs

        # Official val_selection_cropped is a flat directory where filenames encode
        # sequence id and camera id, e.g.
        # 2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png
        depth_files = sorted((selection_root / "groundtruth_depth").glob("*.png"))
        image_files = sorted((selection_root / "image").glob("*.png"))
        image_lookup = {}
        for image_file in image_files:
            match = DEPTH_SELECTION_IMAGE_RE.match(image_file.name)
            if not match:
                continue
            key = (
                match.group("seq"),
                match.group("cam"),
                match.group("frame"),
            )
            image_lookup[key] = image_file

        grouped_pairs = {}
        for depth_file in depth_files:
            match = DEPTH_SELECTION_DEPTH_RE.match(depth_file.name)
            if not match:
                continue
            seq_key = f"{match.group('seq')}_image_{match.group('cam')}"
            grouped_pairs.setdefault(seq_key, [])
            image_key = (
                match.group("seq"),
                match.group("cam"),
                match.group("frame"),
            )
            image_file = image_lookup.get(image_key)
            if image_file is None:
                print(f"missing paired image for {depth_file.name}")
                continue
            grouped_pairs[seq_key].append((depth_file, image_file))

        if grouped_pairs:
            return [
                (seq_key, file_pairs, None)
                for seq_key, file_pairs in sorted(grouped_pairs.items())
            ]

    # Legacy layout used by older project-local exports:
    # <seq>/proj_depth/groundtruth/image_02/*.png with sibling raw images
    pattern = os.path.join(source_root.as_posix(), "*", "proj_depth", "groundtruth", "image_02")
    pairs = []
    for depth_dir in sorted(glob.glob(pattern)):
        depth_path = Path(depth_dir)
        seq_root = depth_path.parents[3]
        seq_name = seq_root.name
        stem_parts = seq_root.name.split("_")
        raw_seq_name = "_".join(stem_parts[:3]) if len(stem_parts) >= 3 else seq_root.name
        image_dir = seq_root.parents[0] / raw_seq_name / "image_02" / "data"
        if not image_dir.is_dir():
            print(f"missing image directory for {seq_name}: {image_dir}")
            continue
        pairs.append((seq_name, depth_path, image_dir))
    return pairs


def copy_subset(seq_name, depth_path, image_dir, output_root, target_frames):
    output_root = Path(output_root)
    depth_out = (
        output_root
        / "depth_selection"
        / "val_selection_cropped"
        / f"groundtruth_depth_gathered_{target_frames}"
        / f"{seq_name}_02"
    )
    image_out = (
        output_root
        / "depth_selection"
        / "val_selection_cropped"
        / f"image_gathered_{target_frames}"
        / f"{seq_name}_02"
    )
    depth_out.mkdir(parents=True, exist_ok=True)
    image_out.mkdir(parents=True, exist_ok=True)

    all_depth_files = sorted(depth_path.glob("*.png"))
    actual_frames = min(len(all_depth_files), target_frames)
    print(
        f"sequence {seq_name}: target={target_frames}, "
        f"available={len(all_depth_files)}, processed={actual_frames}"
    )

    for depth_file in all_depth_files[:target_frames]:
        shutil.copy(depth_file, depth_out / depth_file.name)
        image_file = image_dir / depth_file.name
        if image_file.exists():
            shutil.copy(image_file, image_out / image_file.name)
        else:
            print(f"missing image: {image_file}")


def copy_flat_selection_subset(seq_name, file_pairs, output_root, target_frames):
    output_root = Path(output_root)
    depth_out = (
        output_root
        / "depth_selection"
        / "val_selection_cropped"
        / f"groundtruth_depth_gathered_{target_frames}"
        / seq_name
    )
    image_out = (
        output_root
        / "depth_selection"
        / "val_selection_cropped"
        / f"image_gathered_{target_frames}"
        / seq_name
    )
    depth_out.mkdir(parents=True, exist_ok=True)
    image_out.mkdir(parents=True, exist_ok=True)

    sorted_pairs = sorted(file_pairs, key=lambda pair: pair[0].name)
    actual_frames = min(len(sorted_pairs), target_frames)
    print(
        f"sequence {seq_name}: target={target_frames}, "
        f"available={len(sorted_pairs)}, processed={actual_frames}"
    )
    for depth_file, image_file in sorted_pairs[:target_frames]:
        shutil.copy(depth_file, depth_out / depth_file.name)
        shutil.copy(image_file, image_out / image_file.name)


def main():
    args = parse_args()
    sequence_pairs = find_sequence_pairs(args.source_root)
    if not sequence_pairs:
        raise FileNotFoundError(
            "No KITTI sequence directories found under "
            f"{args.source_root}. Expected either "
            "depth_selection/val_selection_cropped/{groundtruth_depth,image} "
            "or */proj_depth/groundtruth/image_02."
        )

    for target_frames in args.target_frames:
        for seq_name, depth_entry, image_entry in sequence_pairs:
            if isinstance(depth_entry, list):
                copy_flat_selection_subset(
                    seq_name, depth_entry, args.output_root, target_frames
                )
            else:
                copy_subset(
                    seq_name, depth_entry, image_entry, args.output_root, target_frames
                )


if __name__ == "__main__":
    main()
