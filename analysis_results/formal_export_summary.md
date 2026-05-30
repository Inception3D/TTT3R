# Exported Results Summary

This summary is generated from `eval_results_export/` pulled from `origin/szy` on 2026-03-27.

## A3 per-scene relpose

### ScanNet: `ttt3r_random` vs `ttt3r_momentum_inv_t1`

- Common scenes: 65
- Improved scenes: 31
- Degraded scenes: 34
- Median ATE: `0.20304 -> 0.19217`
- Mean relative improvement: `+0.92%`
- Median relative improvement: `-1.35%`

Outputs:
- `analysis_results/a3_scannet_momentum_inv_t1/per_scene_comparison.csv`
- `analysis_results/a3_scannet_momentum_inv_t1/summary.csv`
- `analysis_results/a3_scannet_momentum_inv_t1/ate_scatter.png`
- `analysis_results/a3_scannet_momentum_inv_t1/improvement_hist.png`

### TUM: `ttt3r_random` vs `ttt3r_momentum_inv_t1`

- Common scenes: 8
- Improved scenes: 7
- Degraded scenes: 1
- Median ATE: `0.08224 -> 0.065545`
- Mean relative improvement: `+14.90%`
- Median relative improvement: `+10.21%`

Outputs:
- `analysis_results/a3_tum_momentum_inv_t1/per_scene_comparison.csv`
- `analysis_results/a3_tum_momentum_inv_t1/summary.csv`
- `analysis_results/a3_tum_momentum_inv_t1/ate_scatter.png`
- `analysis_results/a3_tum_momentum_inv_t1/improvement_hist.png`

### ScanNet: `ttt3r_random` vs `ttt3r_brake_geo`

- Common scenes: 65
- Improved scenes: 20
- Degraded scenes: 45
- Median ATE: `0.20304 -> 0.24746`
- Mean relative improvement: `-35.37%`
- Median relative improvement: `-21.93%`

Outputs:
- `analysis_results/a3_scannet_brake_geo/per_scene_comparison.csv`
- `analysis_results/a3_scannet_brake_geo/summary.csv`
- `analysis_results/a3_scannet_brake_geo/ate_scatter.png`
- `analysis_results/a3_scannet_brake_geo/improvement_hist.png`

### TUM: `ttt3r_random` vs `ttt3r_brake_geo`

- Common scenes: 8
- Improved scenes: 5
- Degraded scenes: 3
- Median ATE: `0.08224 -> 0.054865`
- Mean relative improvement: `+3.84%`
- Median relative improvement: `+10.02%`

Outputs:
- `analysis_results/a3_tum_brake_geo/per_scene_comparison.csv`
- `analysis_results/a3_tum_brake_geo/summary.csv`
- `analysis_results/a3_tum_brake_geo/ate_scatter.png`
- `analysis_results/a3_tum_brake_geo/improvement_hist.png`

## S3 tau sensitivity

Only exported `tau=1` and `tau=2` relpose results are currently available.

### ScanNet

- `tau=1`: median ATE `0.19217`, mean ATE `0.26147`
- `tau=2`: median ATE `0.26213`, mean ATE `0.31068`

Outputs:
- `analysis_results/s3_scannet_momentum_inv/tau_sensitivity_summary.csv`
- `analysis_results/s3_scannet_momentum_inv/tau_sensitivity_curve.png`

### TUM

- `tau=1`: median ATE `0.065545`, mean ATE `0.06339`
- `tau=2`: median ATE `0.05592`, mean ATE `0.08219`

Outputs:
- `analysis_results/s3_tum_momentum_inv/tau_sensitivity_summary.csv`
- `analysis_results/s3_tum_momentum_inv/tau_sensitivity_curve.png`

## Exported video depth

Metric files are under `eval_results_export/video_depth/*/*/result_metric.json`.

### KITTI

- `cut3r`: Abs Rel `0.15153`, RMSE `5.66694`, delta<1.25 `0.80434`
- `ttt3r`: Abs Rel `0.13192`, RMSE `5.42614`, delta<1.25 `0.86530`
- `ttt3r_joint`: Abs Rel `0.13437`, RMSE `5.38475`, delta<1.25 `0.85774`

### Bonn

- `cut3r`: Abs Rel `0.09900`, RMSE `0.34637`, delta<1.25 `0.90612`
- `ttt3r`: Abs Rel `0.09974`, RMSE `0.33887`, delta<1.25 `0.92143`
- `ttt3r_joint`: Abs Rel `0.09408`, RMSE `0.32358`, delta<1.25 `0.93431`

### Sintel

- `cut3r`: Abs Rel `1.02167`, RMSE `6.88020`, delta<1.25 `0.23766`
- `ttt3r`: Abs Rel `0.97764`, RMSE `6.67607`, delta<1.25 `0.23245`
- `ttt3r_joint`: Abs Rel `0.91725`, RMSE `6.54943`, delta<1.25 `0.24723`

## Exported 7scenes video recon

Mean values parsed from `eval_results_export/video_recon/7scenes_200/*/7scenes/logs_all.txt`.

- `cut3r`: acc `0.092`, comp `0.048`, nc1 `0.582`, nc2 `0.545`
- `ttt3r`: acc `0.027`, comp `0.023`, nc1 `0.600`, nc2 `0.561`
- `ttt3r_joint`: acc `0.021`, comp `0.022`, nc1 `0.594`, nc2 `0.565`
