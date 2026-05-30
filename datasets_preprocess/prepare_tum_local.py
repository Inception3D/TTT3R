"""Preprocess local TUM data for relpose evaluation.

Source: /mnt/sda/szy/research/dataset/tum/{seq}/rgb/ + groundtruth.txt
Target: data/long_tum_s1/{seq}/rgb_{N}/ + groundtruth_{N}.txt

Usage:
    python datasets_preprocess/prepare_tum_local.py
"""
import glob
import os
import shutil
import numpy as np


def read_file_list(filename):
    with open(filename) as f:
        data = f.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    lst = [
        [v.strip() for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    lst = [(float(l[0]), l[1:]) for l in lst if len(l) > 1]
    return dict(lst)


def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


TUM_ROOT = "/mnt/sda/szy/research/dataset/tum"
OUTPUT_BASE = "./data/long_tum_s1"
SAMPLE_INTERVAL = 1
TARGET_FRAMES_LIST = [1000]

os.makedirs(OUTPUT_BASE, exist_ok=True)

dirs = sorted(glob.glob(f"{TUM_ROOT}/*/"))
print(f"Found {len(dirs)} TUM sequences")

for TARGET_FRAMES in TARGET_FRAMES_LIST:
    for d in dirs:
        first_file = os.path.join(d, "rgb.txt")
        second_file = os.path.join(d, "groundtruth.txt")

        if not os.path.exists(first_file) or not os.path.exists(second_file):
            print(f"  Skipping {d}: missing rgb.txt or groundtruth.txt")
            continue

        first_list = read_file_list(first_file)
        second_list = read_file_list(second_file)
        matches = associate(first_list, second_list)

        frames = []
        gt = []
        for a, b in matches:
            frames.append(os.path.join(d, first_list[a][0]))
            gt.append([b] + second_list[b])

        frames = frames[::SAMPLE_INTERVAL][:TARGET_FRAMES]
        gt_sampled = gt[::SAMPLE_INTERVAL][:TARGET_FRAMES]

        dir_name = os.path.basename(os.path.dirname(d))
        new_dir = os.path.join(OUTPUT_BASE, dir_name, f"rgb_{TARGET_FRAMES}")

        os.makedirs(new_dir, exist_ok=True)
        for frame in frames:
            shutil.copy(frame, new_dir)

        gt_file = os.path.join(OUTPUT_BASE, dir_name, f"groundtruth_{TARGET_FRAMES}.txt")
        with open(gt_file, "w") as f:
            for pose in gt_sampled:
                f.write(f"{' '.join(map(str, pose))}\n")

        print(f"  {dir_name}: {len(frames)} frames")

print("Done.")
