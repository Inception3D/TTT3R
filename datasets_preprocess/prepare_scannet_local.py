"""Preprocess local ScanNet data for relpose evaluation.

Source: /mnt/sda/szy/research/dataset/scannetv2/{scene}/color/*.jpg
Target: data/long_scannet_s3/{scene}/color_{N}/ + pose_{N}.txt

Usage:
    python datasets_preprocess/prepare_scannet_local.py
"""
import glob
import os
import shutil
import numpy as np

SCANNET_ROOT = "/mnt/sda/szy/research/dataset/scannetv2"
OUTPUT_BASE = "./data/long_scannet_s3"
SAMPLE_INTERVAL = 3

TARGET_FRAMES_LIST = [1000]

seq_list = sorted([
    d for d in os.listdir(SCANNET_ROOT)
    if os.path.isdir(os.path.join(SCANNET_ROOT, d)) and d.startswith("scene")
])

print(f"Found {len(seq_list)} scenes")

for TARGET_FRAMES in TARGET_FRAMES_LIST:
    for seq in seq_list:
        seq_dir = os.path.join(SCANNET_ROOT, seq)
        img_paths = sorted(
            glob.glob(f"{seq_dir}/color/*.jpg"),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        depth_paths = sorted(
            glob.glob(f"{seq_dir}/depth/*.png"),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        pose_paths = sorted(
            glob.glob(f"{seq_dir}/pose/*.txt"),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        total_frames = min(len(img_paths), len(depth_paths), len(pose_paths))
        actual_target = min(TARGET_FRAMES, total_frames // SAMPLE_INTERVAL)

        if actual_target == 0:
            print(f"  {seq}: skipping (only {total_frames} frames)")
            continue

        new_color_dir = f"{OUTPUT_BASE}/{seq}/color_{TARGET_FRAMES}"
        new_depth_dir = f"{OUTPUT_BASE}/{seq}/depth_{TARGET_FRAMES}"

        sampled_imgs = img_paths[:actual_target * SAMPLE_INTERVAL:SAMPLE_INTERVAL]
        sampled_depths = depth_paths[:actual_target * SAMPLE_INTERVAL:SAMPLE_INTERVAL]
        sampled_poses = pose_paths[:actual_target * SAMPLE_INTERVAL:SAMPLE_INTERVAL]

        if os.path.exists(new_color_dir):
            shutil.rmtree(new_color_dir)
        if os.path.exists(new_depth_dir):
            shutil.rmtree(new_depth_dir)

        os.makedirs(new_color_dir, exist_ok=True)
        os.makedirs(new_depth_dir, exist_ok=True)

        for i, (img, dep) in enumerate(zip(sampled_imgs, sampled_depths)):
            shutil.copy(img, f"{new_color_dir}/frame_{i:04d}.jpg")
            shutil.copy(dep, f"{new_depth_dir}/frame_{i:04d}.png")

        pose_file = f"{OUTPUT_BASE}/{seq}/pose_{TARGET_FRAMES}.txt"
        with open(pose_file, 'w') as f:
            for pose_path in sampled_poses:
                pose = np.loadtxt(pose_path).reshape(-1)
                f.write(f"{' '.join(map(str, pose))}\n")

        print(f"  {seq}: {actual_target} frames (from {total_frames})")

print("Done.")
