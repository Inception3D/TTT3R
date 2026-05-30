# H200 Server Runlog (2026-03-29)

## 1) 环境与硬件

- 服务器: `NVIDIA H200 x1` (`143771 MiB`)
- CPU/内存: `8 vCPU / 141 GB RAM`
- 仓库分支: `zjc` (commit `c7ba452`)
- Python 环境: `python3.10 + .venv`
- 关键运行参数:
  - `NUM_PROCESSES=1`
  - `SIZE=512`
  - `AMP_DTYPE=bf16`
  - `TF32=1`
  - `CUDNN_BENCHMARK=1`
  - `INFERENCE_MODE=1`

## 2) 数据准备

- 已下载并校验模型权重:
  - `src/cut3r_512_dpt_4_64.pth` (`~3.0G`)
- 已下载并解压:
  - `nuScenes v1.0-mini` 到 `/root/datasets/nuscenes`
- Waymo 数据状态:
  - 当前服务器未检测到任何 `*.tfrecord`
  - 公开直链访问返回 `403`，说明需账号授权后的 Waymo 数据源

## 3) nuScenes 实验（已完成）

执行脚本:

```bash
bash scripts/server/run_nuscenes_relpose_pipeline.sh
```

数据转换输出:

- `data/nuscenes_relpose/` (10 个序列，单序列约 39~41 帧)

评测输出:

- `eval_results/relpose/nuscenes_relpose/summary.csv`
- `eval_results/relpose/nuscenes_relpose/per_sequence_results.csv`
- `eval_results/relpose/nuscenes_relpose/summary.md`
- `eval_results/relpose/nuscenes_pipeline.log`

### 3.1 结果摘要 (有效组)

| model | avg_ate | avg_rpe_trans | avg_rpe_rot |
|---|---:|---:|---:|
| cut3r | 2.57783 | 1.25918 | 0.90599 |
| ttt3r | 10.04719 | 4.88576 | 1.31527 |
| ttt3r_momentum_inv_t1 | 17.99197 | 7.99655 | 10.64272 |

备注:

- 在本次 nuScenes mini 子集上，`cut3r` 明显优于当前 TTT3R 配置。
- 历史导出的 `ttt3r_momentum_inv_t1_drift0` 与 `ttt3r_momentum_inv_t1` 完全一致，后续确认是 `alpha_drift` 未生效导致；因此这里从主比较中移除 `drift0` 行，避免误读为有效独立实验组。

## 4) Waymo 状态与下一步

当前阻塞不是代码错误，而是 **数据访问权限**:

1. Waymo 数据未挂载到服务器（当前 `tfrecord` 数量为 0）。
2. 公开 URL 返回 403，说明必须使用有权限的账号/数据盘。

拿到可用 Waymo 数据后可直接运行:

```bash
export WAYMO_TFRECORD_GLOB="/path/to/waymo/*.tfrecord"
export NUM_PROCESSES=1
export SIZE=512
export AMP_DTYPE=bf16
export TF32=1
export CUDNN_BENCHMARK=1
export INFERENCE_MODE=1
bash scripts/server/run_waymo_relpose_pipeline.sh
```

## 5) nuScenes Full Trainval（CAM_FRONT，850 scenes，已完成）

在同一台 H200 服务器上，已完成 `v1.0-trainval` 全量 `CAM_FRONT` 跑数（`850/850`）。

执行入口:

```bash
NUSCENES_DATAROOT=/root/datasets/nuscenes_trainval_camfront \
NUSCENES_VERSION=v1.0-trainval \
NUSCENES_CAMERA=CAM_FRONT \
NUSCENES_OUTPUT_ROOT=/root/TTT3R/data/nuscenes_relpose_full_camfront \
MAX_SCENES=850 \
MAX_FRAMES=500 \
SIZE=512 \
NUM_PROCESSES=1 \
AMP_DTYPE=bf16 \
TF32=1 \
CUDNN_BENCHMARK=1 \
INFERENCE_MODE=1 \
bash scripts/server/run_nuscenes_relpose_pipeline.sh
```

产物:

- `eval_results/relpose/nuscenes_relpose/summary.csv`
- `eval_results/relpose/nuscenes_relpose/per_sequence_results.csv`（3401 行，含表头）
- `logs/nuscenes_full_h200.log`

### 5.1 Full 结果摘要 (有效组)

| model | avg_ate | avg_rpe_trans | avg_rpe_rot |
|---|---:|---:|---:|
| cut3r | 2.32265 | 0.85829 | 0.72078 |
| ttt3r | 5.02525 | 2.07429 | 1.16555 |
| ttt3r_momentum_inv_t1 | 11.83113 | 4.72726 | 3.73936 |

### 5.2 Full 原始 4 组结果（含历史 drift0 组）

| model | avg_ate | avg_rpe_trans | avg_rpe_rot |
|---|---:|---:|---:|
| cut3r | 2.32265 | 0.85829 | 0.72078 |
| ttt3r | 5.02525 | 2.07429 | 1.16555 |
| ttt3r_momentum_inv_t1 | 11.83113 | 4.72726 | 3.73936 |
| ttt3r_momentum_inv_t1_drift0 | 11.83113 | 4.72726 | 3.73936 |

说明:

- `drift0` 行与 `ttt3r_momentum_inv_t1` 完全一致是历史实现问题（当时 `alpha_drift` 分支未生效）导致，不能作为独立有效对比组。
- 当前仓库已修复该实现问题（后续新实验应重跑后再用于结论）。

### 5.3 序列分布统计（补齐均值以外信息）

下面统计基于 `850` 个 scene 的 `per_sequence_results.csv`：

| model | ATE(mean/median/p90) | RPE_trans(mean/median/p90) | RPE_rot(mean/median/p90) |
|---|---:|---:|---:|
| cut3r | 2.3227 / 1.8415 / 4.9125 | 0.8583 / 0.7671 / 1.5677 | 0.7208 / 0.5739 / 1.2828 |
| ttt3r | 5.0252 / 2.4177 / 8.5885 | 2.0743 / 1.2470 / 3.5543 | 1.1655 / 0.6104 / 1.7978 |
| ttt3r_momentum_inv_t1 | 11.8311 / 5.4329 / 35.7223 | 4.7273 / 2.7427 / 12.0787 | 3.7394 / 0.9007 / 10.6717 |

### 5.4 运行时统计（由完整日志自动抽取）

| output_tag | model_name | alpha_drift | total_runtime_min | fps_mean | fps_median |
|---|---|---:|---:|---:|---:|
| cut3r | cut3r | 0.15 | 53.417 | 19.391 | 19.670 |
| ttt3r | ttt3r | 0.15 | 52.167 | 19.278 | 19.610 |
| ttt3r_momentum_inv_t1 | ttt3r_momentum_inv_t1 | 0.15 | 53.717 | 19.079 | 19.600 |
| ttt3r_momentum_inv_t1_drift0 | ttt3r_momentum_inv_t1 | 0.00 | 53.950 | 18.806 | 19.200 |

显存说明:

- 本次导出的 `nuscenes_full_h200.log` 未包含 NVML 连续监控行，因此**不能**从当前导出文件可靠复原峰值 VRAM。
- 可确认信息是：本次 `SIZE=512` 全流程未报 OOM，4 组均完成 `850/850`。

### 5.5 关于“是否只测了 3 个指标”

- 对 `relpose` 任务，评估器本身定义的核心几何指标确实是 `ATE / RPE_trans / RPE_rot` 三项。
- 本次补齐的内容是：`per-sequence` 分布统计（median/std/p90 等）和运行效率统计（total runtime/FPS），用于增强可信度与可复核性。

## 6) Video Depth 全指标补齐（metric/scale/scale&shift）

已将 `eval_results_export/video_depth/` 下全部可用 JSON（`36` 个）统一汇总到单表。

### 6.1 各数据集在不同对齐方式下的最优项（按 Abs Rel）

| dataset | alignment | best_model | Abs Rel | RMSE | δ < 1.25 |
|---|---|---|---:|---:|---:|
| bonn_s1_500 | metric | ttt3r_joint | 0.094081 | 0.323577 | 0.934312 |
| bonn_s1_500 | scale | ttt3r_joint | 0.069972 | 0.273363 | 0.954001 |
| bonn_s1_500 | scale_shift | ttt3r_joint | 0.065464 | 0.264115 | 0.960074 |
| kitti_s1_500 | metric | ttt3r | 0.128815 | 5.700562 | 0.850601 |
| kitti_s1_500 | scale | ttt3r_joint | 0.117087 | 5.092210 | 0.893164 |
| kitti_s1_500 | scale_shift | ttt3r_joint | 0.106017 | 5.201986 | 0.901254 |
| kitti_s1_500_bugfix_final | metric | ttt3r_momentum_inv_t1 | 0.115049 | 5.672172 | 0.866680 |
| kitti_s1_500_bugfix_final | scale | ttt3r_momentum_inv_t1 | 0.118438 | 5.463623 | 0.880861 |
| kitti_s1_500_bugfix_final | scale_shift | ttt3r_momentum_inv_t1 | 0.106303 | 5.566821 | 0.889503 |
| sintel | metric | ttt3r_joint | 0.917253 | 6.549427 | 0.247228 |
| sintel | scale | ttt3r | 0.404878 | 4.349745 | 0.490798 |
| sintel | scale_shift | ttt3r_joint | 0.398702 | 5.020374 | 0.570762 |

### 6.2 KITTI bugfix final（ttt3r vs brake 版）完整指标

| model | alignment | Abs Rel | Sq Rel | RMSE | Log RMSE | δ < 1.25 |
|---|---|---:|---:|---:|---:|---:|
| ttt3r | metric | 0.128815 | 0.912491 | 5.700562 | 0.180974 | 0.850601 |
| ttt3r | scale | 0.125868 | 0.853534 | 5.495092 | 0.173581 | 0.867252 |
| ttt3r | scale_shift | 0.116942 | 0.835753 | 5.547695 | 0.171391 | 0.873662 |
| ttt3r_momentum_inv_t1 | metric | 0.115049 | 0.845235 | 5.672172 | 0.171253 | 0.866680 |
| ttt3r_momentum_inv_t1 | scale | 0.118438 | 0.805025 | 5.463623 | 0.165685 | 0.880861 |
| ttt3r_momentum_inv_t1 | scale_shift | 0.106303 | 0.795042 | 5.566821 | 0.162461 | 0.889503 |

## 7) 新增导出文件（本次补齐）

- `eval_results_export/relpose/nuscenes_relpose_h200_full_20260329/summary_distribution_stats.csv`
- `eval_results_export/relpose/nuscenes_relpose_h200_full_20260329/summary_runtime_fps_from_log.csv`
- `eval_results_export/video_depth/summary_all_metrics.csv`

简要结论:

1. 你说得对：`avg_ate / avg_rpe_trans / avg_rpe_rot` 只是 relpose 的核心均值，不足以完整表达实验质量；本次已补齐分布和效率统计。
2. `drift0` 历史结果与 `t1` 完全一致属于实现未生效问题，不应再当独立实验结论引用。
3. 在现有可复核数据上，`kitti_s1_500_bugfix_final` 中 `ttt3r_momentum_inv_t1` 相比 `ttt3r` 在三种对齐方式下均有改善。
