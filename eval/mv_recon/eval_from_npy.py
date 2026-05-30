#!/usr/bin/env python3
"""
eval_from_npy.py — Evaluate 7scenes from saved .npy files.
Uses subprocess (not multiprocessing.Pool) to avoid Open3D fork issues.
Evaluation logic is identical to launch.py: ICP + KDTree Acc/Comp/NC.
"""
import os, sys, argparse, time, subprocess, re
import numpy as np
import open3d as o3d
from pathlib import Path


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    from scipy.spatial import cKDTree as KDTree
    gt_kd = KDTree(gt_points)
    distances, idx = gt_kd.query(rec_points, workers=-1)
    acc, acc_med = np.mean(distances), np.median(distances)
    if gt_normals is not None and rec_normals is not None:
        nd = np.abs(np.sum(gt_normals[idx] * rec_normals, axis=-1))
        return acc, acc_med, np.mean(nd), np.median(nd)
    return acc, acc_med


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    from scipy.spatial import cKDTree as KDTree
    rec_kd = KDTree(rec_points)
    distances, idx = rec_kd.query(gt_points, workers=-1)
    comp, comp_med = np.mean(distances), np.median(distances)
    if gt_normals is not None and rec_normals is not None:
        nd = np.abs(np.sum(gt_normals * rec_normals[idx], axis=-1))
        return comp, comp_med, np.mean(nd), np.median(nd)
    return comp, comp_med


def eval_single_config(results_root, cfg, threshold=0.1):
    """Evaluate all scenes for one config."""
    cfg_dir = Path(results_root) / cfg / '7scenes'
    npys = sorted(cfg_dir.glob('*.npy'))
    if not npys:
        print(f"SKIP: {cfg} (no .npy)")
        return

    log_file = cfg_dir / 'logs_0.txt'
    if log_file.exists():
        existing = sum(1 for l in open(log_file) if 'Idx:' in l)
        if existing >= len(npys):
            print(f"SKIP: {cfg} ({existing}/{len(npys)} already evaluated)")
            return

    print(f"EVAL: {cfg} ({len(npys)} scenes)", flush=True)
    lines = []
    t0 = time.time()

    for i, npy_path in enumerate(npys):
        try:
            data = np.load(str(npy_path), allow_pickle=True).item()
            pts_all = data['pts_all']
            pts_gt_all = data['pts_gt_all']
            images_all = data['images_all']
            masks_all = data['masks_all']

            pts_masked = pts_all[masks_all > 0].reshape(-1, 3)
            pts_gt_masked = pts_gt_all[masks_all > 0].reshape(-1, 3)
            imgs_masked = images_all[masks_all > 0].reshape(-1, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_masked)
            pcd.colors = o3d.utility.Vector3dVector(imgs_masked)
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_masked)
            pcd_gt.colors = o3d.utility.Vector3dVector(imgs_masked)

            # ICP — identical to launch.py
            reg = o3d.pipelines.registration.registration_icp(
                pcd, pcd_gt, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            pcd = pcd.transform(reg.transformation)
            pcd.estimate_normals()
            pcd_gt.estimate_normals()

            gt_n = np.asarray(pcd_gt.normals)
            pred_n = np.asarray(pcd.normals)

            acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_n, pred_n)
            comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_n, pred_n)

            scene_id = npy_path.stem
            line = (
                f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                f"Acc_med: {acc_med}, Compc_med: {comp_med}, "
                f"NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
            )
            lines.append(line)
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(npys)}] {scene_id} "
                f"Acc={acc:.4f} Comp={comp:.4f} NC={(nc1+nc2)/2:.3f} ({elapsed:.0f}s)",
                flush=True,
            )
        except Exception as e:
            print(f"  ERROR: {npy_path.name}: {e}", flush=True)

    # Write logs (same format as launch.py)
    with open(cfg_dir / 'logs_0.txt', 'w') as f:
        for line in sorted(lines):
            f.write(line + '\n')

    pattern = re.compile(
        r"Idx:\s*(?P<sid>[^,]+),\s*Acc:\s*(?P<acc>[^,]+),\s*Comp:\s*(?P<comp>[^,]+),\s*"
        r"NC1:\s*(?P<nc1>[^,]+),\s*NC2:\s*(?P<nc2>[^,]+)\s*-\s*"
        r"Acc_med:\s*(?P<acc_med>[^,]+),\s*Compc_med:\s*(?P<comp_med>[^,]+),\s*"
        r"NC1c_med:\s*(?P<nc1_med>[^,]+),\s*NC2c_med:\s*(?P<nc2_med>[^,]+)"
    )
    metrics = {}
    for line in lines:
        m = pattern.match(line)
        if m:
            d = m.groupdict()
            for k, v in d.items():
                if k != 'sid':
                    metrics.setdefault(k, []).append(float(v))
    mean_str = "mean                : "
    for k, vals in metrics.items():
        mean_str += f"{k}: {np.mean(vals):.3f} | "

    with open(cfg_dir / 'logs_all.txt', 'w') as f:
        for line in sorted(lines):
            f.write(line + '\n')
        f.write(mean_str + '\n')

    elapsed = time.time() - t0
    print(f"DONE: {cfg} ({len(lines)} scenes in {elapsed:.0f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', required=True)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--configs', nargs='*', default=None)
    parser.add_argument('--single_config', type=str, default=None,
                        help='Evaluate a single config (used internally by subprocess)')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel config evaluations via subprocess')
    args = parser.parse_args()

    # Single config mode (called as subprocess)
    if args.single_config:
        eval_single_config(args.results_root, args.single_config, args.threshold)
        return

    # Multi-config mode: launch subprocesses
    root = Path(args.results_root)
    configs = args.configs or sorted([d.name for d in root.iterdir() if d.is_dir()])

    todo = []
    for cfg in configs:
        cfg_dir = root / cfg / '7scenes'
        if not cfg_dir.exists():
            continue
        npys = list(cfg_dir.glob('*.npy'))
        if not npys:
            continue
        log_file = cfg_dir / 'logs_0.txt'
        if log_file.exists():
            existing = sum(1 for l in open(log_file) if 'Idx:' in l)
            if existing >= len(npys):
                print(f"SKIP: {cfg} (already done)")
                continue
        todo.append(cfg)

    print(f"\n=== Evaluating {len(todo)} configs, {args.parallel} parallel ===\n", flush=True)

    script = os.path.abspath(__file__)
    running = []

    for cfg in todo:
        cmd = [
            sys.executable, script,
            '--results_root', str(root),
            '--single_config', cfg,
            '--threshold', str(args.threshold),
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running.append((cfg, p))
        print(f"LAUNCHED: {cfg} (PID {p.pid})", flush=True)

        while len(running) >= args.parallel:
            for j, (c, proc) in enumerate(running):
                if proc.poll() is not None:
                    out = proc.stdout.read()
                    print(out, end='', flush=True)
                    running.pop(j)
                    break
            else:
                time.sleep(2)

    for c, proc in running:
        proc.wait()
        out = proc.stdout.read()
        print(out, end='', flush=True)

    print(f"\n=== All evaluations complete ===", flush=True)


if __name__ == '__main__':
    main()
