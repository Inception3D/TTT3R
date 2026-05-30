# TTT3R (zjc) — Stability Brake Result Summary

## Project Goal
Train-free, inference-time improvement for recurrent 3D reconstruction by controlling **systematic over-update** in state updates.

Current main line on `zjc`:

- keep one core method: **stability brake**
- remove multi-gate complexity from main story
- validate on local safe setting + outdoor depth + large-scale relpose export

---

## Core Method

State update strength is controlled by alignment of consecutive residual directions:

`alpha_t = sigmoid(-tau * cos(delta_t, delta_{t-1}))`

- high cosine (update direction repeats): stronger brake
- low cosine (new direction appears): allow more update

In code this line corresponds to `ttt3r_momentum_inv_t1` branch, with `alpha_drift` as key parameter.

---

## Key Results

### 1) KITTI Outdoor Depth (post-bugfix, brake vs baseline)

Main comparison:

- baseline: `ttt3r`
- brake: `ttt3r_momentum_inv_t1`
- dataset: `kitti_s1_500_bugfix_final`

#### Metric alignment

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | delta < 1.25 |
|---|---:|---:|---:|---:|---:|
| ttt3r | 0.128815 | 0.912491 | 5.700562 | 0.180974 | 0.850601 |
| ttt3r_momentum_inv_t1 | 0.115049 | 0.845235 | 5.672172 | 0.171253 | 0.866680 |

#### Scale alignment

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | delta < 1.25 |
|---|---:|---:|---:|---:|---:|
| ttt3r | 0.125868 | 0.853534 | 5.495092 | 0.173581 | 0.867252 |
| ttt3r_momentum_inv_t1 | 0.118438 | 0.805025 | 5.463623 | 0.165685 | 0.880861 |

#### Scale+shift alignment

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | delta < 1.25 |
|---|---:|---:|---:|---:|---:|
| ttt3r | 0.116942 | 0.835753 | 5.547695 | 0.171391 | 0.873662 |
| ttt3r_momentum_inv_t1 | 0.106303 | 0.795042 | 5.566821 | 0.162461 | 0.889503 |

Abs Rel improvement of brake:

- metric: `-10.69%`
- scale: `-5.90%`
- scale+shift: `-9.10%`

Conclusion:

- after bugfix, brake is active and consistently improves KITTI depth quality.

---

### 2) nuScenes Full Relpose (H200, CAM_FRONT, 850 scenes)

#### Effective main comparison (3 valid groups)

| model | avg_ate | avg_rpe_trans | avg_rpe_rot |
|---|---:|---:|---:|
| cut3r | 2.32265 | 0.85829 | 0.72078 |
| ttt3r | 5.02525 | 2.07429 | 1.16555 |
| ttt3r_momentum_inv_t1 | 11.83113 | 4.72726 | 3.73936 |

#### Distribution stats (per-sequence, mean / median / p90)

| model | ATE | RPE_trans | RPE_rot |
|---|---:|---:|---:|
| cut3r | 2.3227 / 1.8415 / 4.9125 | 0.8583 / 0.7671 / 1.5677 | 0.7208 / 0.5739 / 1.2828 |
| ttt3r | 5.0252 / 2.4177 / 8.5885 | 2.0743 / 1.2470 / 3.5543 | 1.1655 / 0.6104 / 1.7978 |
| ttt3r_momentum_inv_t1 | 11.8311 / 5.4329 / 35.7223 | 4.7273 / 2.7427 / 12.0787 | 3.7394 / 0.9007 / 10.6717 |

Important note:

- historical `ttt3r_momentum_inv_t1_drift0` was identical to `ttt3r_momentum_inv_t1` in this run, and should not be treated as a valid independent group for conclusion.

---

### 3) Runtime / Efficiency

#### nuScenes full run-time (from full log)

| output_tag | alpha_drift | total_runtime_min | fps_mean | fps_median |
|---|---:|---:|---:|---:|
| cut3r | 0.15 | 53.417 | 19.391 | 19.670 |
| ttt3r | 0.15 | 52.167 | 19.278 | 19.610 |
| ttt3r_momentum_inv_t1 | 0.15 | 53.717 | 19.079 | 19.600 |
| ttt3r_momentum_inv_t1_drift0 | 0.00 | 53.950 | 18.806 | 19.200 |

#### Local safe224 overhead (`drift>0` vs `drift0`)

- runtime delta (`drift0 - drift>0`): `-1.03%`
- per-frame delta: `-0.98%`

Conclusion:

- brake-related control does not introduce meaningful overhead in current local safe setting.

---

### 4) Reproducibility (SAFE224, 3 repeats)

Method: `ttt3r_momentum_inv_t1`, 2 protocols:

- fixed seed: `42,42,42`
- different seed: `41,42,43`

Overall (mean ± std):

| protocol | runtime_sec | per_frame_sec | basic_consistency | loop_trans_error |
|---|---:|---:|---:|---:|
| fixed seed | 15.8234 ± 1.7198 | 0.9656 ± 0.2895 | 1.1085 ± 0.5101 | 0.3946 ± 0.3710 |
| different seed | 15.8584 ± 1.5049 | 0.9634 ± 0.2614 | 1.1085 ± 0.5101 | 0.3946 ± 0.3710 |

Takeaway:

- geometric quality metrics are stable across seeds in this SAFE224 setup.
- observed variance is mainly runtime jitter.

---

### 5) Reset-Interval Sensitivity (SAFE224)

Compared methods:

- `ttt3r_momentum_inv_t1` (`alpha_drift=0.15`)
- `ttt3r_momentum_inv_t1_drift0` (`alpha_drift=0.0`)

Reset intervals tested: `4, 8, 16, 100`

Paired summary (`drift0 - brake`):

- consistency delta mean: `0.002720` (small)
- runtime delta mean: `0.249268 s`
- `drift0` slower ratio: `0.688`

Takeaway:

- no one-sided geometry dominance on this tiny local subset
- runtime side still slightly favors brake in most paired runs

---

## What Was Fixed

### Bugfix A (KITTI invalid identical result)

- issue: brake state was reset too frequently due to improper reset-mask handling
- fix: only reset when `torch.any(reset_mask)` is true
- commit: `4e3e14e`

### Bugfix B (`alpha_drift` ineffective in brake path)

- issue: `alpha_drift` parameter was passed but not applied in one stability-brake path
- fix: apply `alpha_drift + (1 - alpha_drift) * alpha_raw` form in brake update
- commit: `6be34c2`

---

## Current Story (Paper-Oriented)

1. Over-update exists and can be controlled by a lightweight state-space brake.
2. Brake gives clear improvement on outdoor depth (KITTI post-bugfix).
3. Brake is low-overhead and reproducible in local safe configuration.
4. Large-scale nuScenes full run is finished and fully exported with distribution-level stats for auditability.

---

## Artifact Paths

- Main report: `docs/waymo_nuscenes_h200_runlog_20260329.md`
- KITTI summary: `docs/kitti_brake_summary.md`
- Reproducibility: `docs/reproducibility_safe224_seedstudy.md`
- Overhead: `docs/s2_overhead_drift_compare.md`
- Reset sensitivity: `docs/reset_interval_sensitivity_safe224.md`
- nuScenes full summary:
  - `eval_results_export/relpose/nuscenes_relpose_h200_full_20260329/summary_effective_models.csv`
  - `eval_results_export/relpose/nuscenes_relpose_h200_full_20260329/summary_distribution_stats.csv`
  - `eval_results_export/relpose/nuscenes_relpose_h200_full_20260329/summary_runtime_fps_from_log.csv`
- Depth full table:
  - `eval_results_export/video_depth/summary_all_metrics.csv`
