# Server Quickstart (nuScenes / Waymo)

这份文档对应“明天上服务器后立即开跑”的最短流程。  
目标：一条命令完成 `环境 -> 数据转换 -> relpose实验 -> CSV结果导出`。

## 0. 进入仓库

```bash
cd ~/TTT3R
```

## 1. 检查权重

默认权重路径：

```bash
ls src/cut3r_512_dpt_4_64.pth
```

如果不在这里，后续命令里通过 `WEIGHTS_PATH=...` 指定即可。

## 2. 跑 nuScenes（推荐先跑 mini 验证流程）

```bash
export NUSCENES_DATAROOT=/path/to/nuscenes
export NUSCENES_VERSION=v1.0-mini
export MAX_SCENES=10
export MAX_FRAMES=300
export NUM_PROCESSES=1
export OVERWRITE_DATA=1

bash scripts/server/run_nuscenes_relpose_pipeline.sh
```

输出结果：

- `eval_results/relpose/nuscenes_relpose/summary.csv`
- `eval_results/relpose/nuscenes_relpose/per_sequence_results.csv`
- `eval_results/relpose/nuscenes_relpose/summary.md`

## 3. 跑 Waymo（先小规模）

```bash
export WAYMO_TFRECORD_GLOB="/path/to/waymo/training/*.tfrecord"
export MAX_SEGMENTS=4
export MAX_FRAMES=300
export NUM_PROCESSES=1
export OVERWRITE_DATA=1

bash scripts/server/run_waymo_relpose_pipeline.sh
```

输出结果：

- `eval_results/relpose/waymo_relpose/summary.csv`
- `eval_results/relpose/waymo_relpose/per_sequence_results.csv`
- `eval_results/relpose/waymo_relpose/summary.md`

## 4. 常见参数

- `NUM_PROCESSES=1`：更稳，显存压力小。
- 不设置 `NUM_PROCESSES`：脚本会自动探测 GPU 数量并并行。
- `MAX_FRAMES`：先用 200~300 验证流程，再拉到 500。
- `STRIDE=2`：可以减轻耗时和存储。
- `WEIGHTS_PATH=/your/path/cut3r_512_dpt_4_64.pth`：自定义权重路径。
- `AMP_DTYPE=bf16`（默认）：H200 推荐，速度更快。
- `TF32=1`（默认）：开启 Tensor Core TF32 加速。

## 5. 说明

- 当前这套是 **relpose评测链路**（因为 Waymo/nuScenes 原始格式与项目现有 depth 基准接口不同，先走最稳的 pose 验证）。
- 数据转换输出格式：
  - `<dataset>/<sequence>/rgb_90/frame_xxxxxx.jpg`
  - `<dataset>/<sequence>/pose_90.txt`（每行 3x4 相机位姿矩阵）
