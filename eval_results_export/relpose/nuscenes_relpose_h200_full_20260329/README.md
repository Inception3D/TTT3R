# nuScenes Full (H200, 2026-03-29)

本目录是本次 **nuScenes v1.0-trainval (CAM_FRONT, 850 scenes)** 完整评测的导出结果（精简版）。

包含文件:

- `summary.csv`：4 组模型的总体平均指标
- `summary_effective_models.csv`：去除无效重复组后的主比较表（推荐引用）
- `per_sequence_results.csv`：逐序列指标明细
- `summary.md`：评测脚本自动生成的 Markdown 摘要
- `nuscenes_full_h200.log`：完整运行日志（含进度、速度、结束标记）
- `summary_distribution_stats.csv`：逐模型分布统计（mean / median / std / p90 / min / max）
- `summary_runtime_fps_from_log.csv`：从完整日志自动提取的每组运行时长和 FPS 统计

关键结果（有效组，avg_ate / avg_rpe_trans / avg_rpe_rot）:

- `cut3r`: `2.32265 / 0.85829 / 0.72078`
- `ttt3r`: `5.02525 / 2.07429 / 1.16555`
- `ttt3r_momentum_inv_t1`: `11.83113 / 4.72726 / 3.73936`

说明:

- 历史导出的 `ttt3r_momentum_inv_t1_drift0` 与 `ttt3r_momentum_inv_t1` 完全一致。
- 后续代码排查确认：当时 `alpha_drift` 在 `stability brake` 路径未真正生效，因此不将 `drift0` 作为有效独立组进行主比较。
- 运行日志未包含 NVML 连续监控输出，因此当前导出中不提供可复核峰值 VRAM（但 4 组均已完整跑通 `850/850`，未 OOM）。
