# Experiment 1: State Token Frequency Visualization

## 前置准备

```bash
# 进入项目根目录
cd /path/to/TTT3R

# 拉取最新代码（lxl 分支）
git fetch origin
git checkout lxl
git pull

# 激活环境
conda activate ttt3r
```

---

## 运行实验

### 基本命令

```bash
python analysis/state_freq_analysis.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path /path/to/your/sequence \
    --output_dir analysis_results/exp1 \
    --model_update_type ttt3r \
    --size 512 \
    --max_frames 200
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型权重路径 | `src/cut3r_512_dpt_4_64.pth` |
| `--seq_path` | 输入序列：视频文件 或 图片目录 | 必填 |
| `--output_dir` | 结果保存目录 | `analysis_results/exp1` |
| `--model_update_type` | `ttt3r` 或 `cut3r` | `ttt3r` |
| `--size` | 输入图像短边缩放尺寸 | `512` |
| `--frame_interval` | 每隔 N 帧取一帧 | `1` |
| `--max_frames` | 最多处理帧数 | `200` |
| `--top_k_tokens` | 可视化的高/低方差 token 数量 | `20` |
| `--window_size` | 计算滑动窗口方差的窗口大小（帧） | `10` |
| `--device` | `cuda` 或 `cpu` | `cuda` |

---

## 推荐实验配置

### ScanNet（室内，慢速运动）

```bash
CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/home/szy/research/TTT3R/src:$PYTHONPATH \
    python analysis/state_freq_analysis.py \
    --model_path model/cut3r_512_dpt_4_64.pth \
    --seq_path /home/szy/research/TTT3R/tmp/taylor/color \
    --output_dir analysis_results/westlake \
    --model_update_type ttt3r \
    --size 512 \
    --frame_interval 5 \
    --max_frames 200 \
    --device cuda
```

### TUM-dynamics（室内，含动态物体）

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/state_freq_analysis.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path /data/tum/rgbd_dataset_freiburg3_walking_xyz/rgb \
    --output_dir analysis_results/exp1_tum_walking \
    --model_update_type ttt3r \
    --size 512 \
    --frame_interval 2 \
    --max_frames 200 \
    --device cuda
```

### 对比实验：cut3r vs ttt3r

```bash
# cut3r baseline
CUDA_VISIBLE_DEVICES=0 python analysis/state_freq_analysis.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path /data/scannet/scans/scene0000_00/color \
    --output_dir analysis_results/exp1_scannet_cut3r \
    --model_update_type cut3r \
    --size 512 --frame_interval 5 --max_frames 200

# ttt3r
CUDA_VISIBLE_DEVICES=0 python analysis/state_freq_analysis.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path /data/scannet/scans/scene0000_00/color \
    --output_dir analysis_results/exp1_scannet_ttt3r \
    --model_update_type ttt3r \
    --size 512 --frame_interval 5 --max_frames 200
```

### 用视频文件作为输入

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/state_freq_analysis.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/taylor.mp4 \
    --output_dir analysis_results/exp1_taylor \
    --model_update_type ttt3r \
    --size 512 \
    --frame_interval 1 \
    --max_frames 100
```

---

## 输出文件说明

```
analysis_results/exp1/
├── state_freq_data.npz               # 原始数据（可用 numpy 加载做进一步分析）
│     token_var   [n_state]           # 每个 state token 的全局时间方差
│     running_var [T, n_state]        # 滑动窗口方差（用于绘制演化曲线）
│     state_stack [T, n_state, D]     # state token 轨迹
│     cross_attn  [T, n_state, N_p]   # 每帧 cross-attention（已平均 layers/heads）
│     img_shapes  [T, 2]              # 每帧 patch 网格尺寸 (H_p, W_p)
│
├── plots/
│   ├── token_variance_hist.png       # state token 方差分布直方图
│   ├── token_variance_evolution.png  # top-K / bottom-K token 方差随时间变化
│   └── high_low_token_attention.png  # 高频 vs 低频 token 的时间平均 attention 图
│
└── freq_heatmaps/
    ├── frame_000000.png              # 原图 + 频率热力图叠加（直接用于论文）
    ├── frame_000001.png
    └── ...
```

---

## 用 npz 原始数据做额外分析

```python
import numpy as np

data = np.load("analysis_results/exp1/state_freq_data.npz")

token_var   = data["token_var"]    # [n_state]  每个 token 的方差
running_var = data["running_var"]  # [T, n_state]  随时间变化
cross_attn  = data["cross_attn"]   # [T, n_state, N_patches]
state_stack = data["state_stack"]  # [T, n_state, D]

# 找方差最高的 token
top_tokens = np.argsort(token_var)[::-1][:10]
print("最不稳定的 10 个 state token 索引:", top_tokens)
print("对应方差:", token_var[top_tokens])

# 某一帧的频率热力图（patch 级别）
t = 50
freq_map = (token_var[:, None] * cross_attn[t]).sum(axis=0)  # [N_patches]
H_p, W_p = data["img_shapes"][t]
freq_map_2d = freq_map.reshape(H_p, W_p)
```

---

# Experiment A: Frequency vs Reconstruction Error Correlation

## 目标

验证 state token 的时间频率（temporal variance）和重建误差（depth error）之间的相关性。
如果高频 token 关注的区域误差更大 → 说明频率可以作为自监督的质量信号。

## 运行命令

### ScanNet（需要 GT depth）

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/freq_error_analysis.py \
    --model_path /home/szy/research/TTT3R/model/cut3r_512_dpt_4_64.pth \
    --seq_path /mnt/sda/szy/research/dataset/scannetv2/scene0707_00/color \
    --output_dir analysis_results/expA_scannet_scene0707_ttt3r \
    --model_update_type ttt3r \
    --size 512 \
    --frame_interval 5 \
    --max_frames 200 \
    --device cuda
```

### 对比实验：cut3r vs ttt3r

```bash
# ttt3r
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/freq_error_analysis.py \
    --model_path /home/szy/research/TTT3R/model/cut3r_512_dpt_4_64.pth \
    --seq_path /mnt/sda/szy/research/dataset/scannetv2/scene0707_00/color \
    --output_dir analysis_results/expA_scannet_ttt3r \
    --model_update_type ttt3r \
    --size 512 --frame_interval 5 --max_frames 200

# cut3r
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/freq_error_analysis.py \
    --model_path /home/szy/research/TTT3R/model/cut3r_512_dpt_4_64.pth \
    --seq_path /mnt/sda/szy/research/dataset/scannetv2/scene0707_00/color \
    --output_dir analysis_results/expA_scannet_cut3r \
    --model_update_type cut3r \
    --size 512 --frame_interval 5 --max_frames 200
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--seq_path` | ScanNet scene 的 `color/` 目录 | 必填 |
| `--depth_scale` | GT depth 从 uint16 转 meters 的除数 | `1000.0` |
| `--max_depth` | 忽略超过此深度的 GT（meters） | `10.0` |
| `--patch_size` | ViT patch 大小 | `16` |

## 输出文件

```
analysis_results/expA/
├── correlation_summary.txt          # Pearson/Spearman 汇总
├── freq_error_data.npz              # 原始数据
└── plots/
    ├── scatter_freq_vs_error.png    # 频率 vs 误差散点图
    ├── correlation_over_time.png    # 逐帧相关系数变化
    └── side_by_side/
        └── frame_XXXXXX.png        # 4-panel: RGB | 频率 | 误差 | 叠加
```
