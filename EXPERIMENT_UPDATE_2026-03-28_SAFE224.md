# Experiment Update (2026-03-28, SAFE224)

## Scope
- 目标：在本机避免爆显存的前提下，完成可运行的 brake 消融小实验。
- 设置：`CUDA + size=224 + 单卡 + 单进程 + 短序列优先`。
- 对比方法：
  - `ttt3r_momentum_inv_t1` (`alpha_drift=0.15`)
  - `ttt3r_momentum_inv_t1_drift0` (`alpha_drift=0.0`)

## Completed Runs
- 序列总数：8/8 完成（apple/bottle × len012/len024 × 2 methods）。
- 输出目录：
  - `benchmark_single_object/outputs_ablation_safe/predictions/`
  - `benchmark_single_object/outputs_ablation_safe/metrics/`

## Key Results

### Summary (mean over objects)
- `len=12`
  - `ttt3r_momentum_inv_t1`: `basic_consistency_score_mean = 1.0513`
  - `ttt3r_momentum_inv_t1_drift0`: `basic_consistency_score_mean = 1.0510`
- `len=24`
  - `ttt3r_momentum_inv_t1`: `basic_consistency_score_mean = 1.1656`
  - `ttt3r_momentum_inv_t1_drift0`: `basic_consistency_score_mean = 1.1655`

### Representative Per-object (len024)
- `apple`
  - `inv_t1`: `0.5921`
  - `drift0`: `0.5970`
- `bottle`
  - `inv_t1`: `1.7391`
  - `drift0`: `1.7341`

## Takeaway
- SAFE224 配置在本机可稳定跑完，不再出现 OOM。
- 本轮小规模消融中，`alpha_drift=0.15` 与 `alpha_drift=0` 在该指标上差异很小；后续需要更大数据量或更多指标（如 ATE/深度误差）继续验证“drift 不能完全去掉”的主结论。

## Result Files
- `benchmark_single_object/outputs_ablation_safe/metrics/per_sequence_results.csv`
- `benchmark_single_object/outputs_ablation_safe/metrics/per_object_results.csv`
- `benchmark_single_object/outputs_ablation_safe/metrics/summary_results.csv`
