# SAFE224 Visual Evidence (Local)

本页汇总本地可复现实验（SAFE224）生成的 3 张论文友好图。

## Figure 1: Metric vs `alpha_drift`

文件：
- `docs/figures/safe224/fig_alpha_drift_curve.png`

含义：
- 横轴是 `alpha_drift`（当前有 `0.0` 和 `0.15` 两个点）。
- 纵轴是 `basic_consistency_score`（越低越好）。
- 按 `seq_length` 分线展示（`12` / `24`）。

## Figure 2: Per-sequence Improvement Distribution

文件：
- `docs/figures/safe224/fig_sequence_improvement_distribution.png`

含义：
- 每根柱子对应一个序列（`apple/bottle` × `len12/len24`）。
- 定义：`improvement = score(drift0) - score(inv_t1)`。
- 柱子 > 0 表示 `inv_t1` 更好，< 0 表示 `drift0` 更好。

## Figure 3: Typical Before/After Visualization

文件：
- `docs/figures/safe224/fig_typical_before_after_depth.png`
- `docs/figures/safe224/fig_typical_before_after_depth_frame008.png`（推荐网页查看，避免缓存）

含义：
- 示例序列：`apple/540_79043_153212_len024`，第 `8` 帧。
- 展示内容：`Input RGB`、`Depth(drift0)`、`Depth(inv_t1)`、`|Depth diff|`。
- 用于直观展示 brake 配置变化带来的深度图差异。

## Reproduce

在 WSL 中运行：

```bash
python3 /mnt/c/Users/Chen/Desktop/codes/TTT3R/analysis/plot_safe224_paper_figures.py
```

默认输出目录：

```text
/mnt/c/Users/Chen/Desktop/codes/TTT3R/docs/figures/safe224
```
