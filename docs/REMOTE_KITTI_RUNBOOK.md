# Remote KITTI Runbook

This is the minimal server checklist for the current `zjc` brake-focused branch.

## What Is Ready

- `ttt3r_momentum_inv_t1` is wired into the model code
- KITTI video-depth runner includes both `ttt3r` and `ttt3r_momentum_inv_t1`
- KITTI preprocessing script is parameterized and no longer tied to someone else's path
- server helper scripts exist:
  - `scripts/server/setup_remote_env.sh`
  - `scripts/server/run_brake_kitti_eval.sh`

## What Must Exist On The Server

1. This repo checked out on branch `zjc`
2. Checkpoint:
   - `src/cut3r_512_dpt_4_64.pth`
3. KITTI validation data unpacked so that this pattern exists:
   - `data/kitti/val/*/proj_depth/groundtruth/image_02`

## Recommended Server Steps

```bash
git pull origin zjc
bash scripts/server/setup_remote_env.sh
```

If the weight is missing, place it at:

```bash
src/cut3r_512_dpt_4_64.pth
```

Then run:

```bash
KITTI_VAL_ROOT=$PWD/data/kitti/val \
KITTI_LONG_ROOT=$PWD/data/long_kitti_s1 \
WEIGHTS_PATH=$PWD/src/cut3r_512_dpt_4_64.pth \
TARGET_FRAMES=500 \
bash scripts/server/run_brake_kitti_eval.sh
```

## Expected Output

Results should appear under:

- `eval_results/video_depth/kitti_s1_500/ttt3r/`
- `eval_results/video_depth/kitti_s1_500/ttt3r_momentum_inv_t1/`

Useful files:

- `result_metric.json`
- `result_scale.json`
- `result_scale&shift.json`

## Notes

- KITTI download itself is not automated here because it depends on the dataset package you provide on the remote machine.
- If the remote image already ships with PyTorch and CUDA, `setup_remote_env.sh` will reuse that via the virtualenv.
