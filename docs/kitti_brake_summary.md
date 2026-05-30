# KITTI Brake Summary

## Scope

This note summarizes the completed KITTI outdoor validation for the brake-based recurrent update in TTT3R.

Main comparison:

- baseline: `ttt3r`
- brake variant: `ttt3r_momentum_inv_t1`

Dataset:

- `kitti_s1_500`

## Main Story

The outdoor KITTI experiment was designed to answer one question:

- after focusing the method story on the state-space brake, does the brake still help outside indoor relpose benchmarks?

The answer is yes.

After fixing the recurrent reset bug, the brake variant consistently improves the main depth metrics over plain `ttt3r` on KITTI.

## Why A Bugfix Was Needed

The first `kitti_s1_500` run produced identical numbers for `ttt3r` and `ttt3r_momentum_inv_t1`, which was treated as invalid.

Root cause:

- brake-side state was reset whenever `reset_mask` existed
- but `reset_mask` exists every step, even when no real reset is triggered
- this caused the brake to be cleared repeatedly
- as a result, `ttt3r_momentum_inv_t1` degenerated to plain `ttt3r`

Fix:

- only reset brake-side state when `torch.any(reset_mask)` is true

Fix commit:

- `4e3e14e`

## Final Post-Fix Results

Final outputs are stored under:

- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/ttt3r/`
- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/ttt3r_momentum_inv_t1/`

### Metric Alignment

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | delta < 1.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ttt3r` | 0.128815 | 0.912491 | 5.700562 | 0.180974 | 0.850601 |
| `ttt3r_momentum_inv_t1` | 0.115049 | 0.845235 | 5.672172 | 0.171253 | 0.866680 |

### Scale Alignment

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | delta < 1.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ttt3r` | 0.125868 | 0.853534 | 5.495092 | 0.173581 | 0.867252 |
| `ttt3r_momentum_inv_t1` | 0.118438 | 0.805025 | 5.463623 | 0.165685 | 0.880861 |

### Scale And Shift Alignment

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | delta < 1.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ttt3r` | 0.116942 | 0.835753 | 5.547695 | 0.171391 | 0.873662 |
| `ttt3r_momentum_inv_t1` | 0.106303 | 0.795042 | 5.566821 | 0.162461 | 0.889503 |

## Relative Improvement

Abs Rel reduction of brake over baseline:

- `metric`: `-10.69%`
- `scale`: `-5.90%`
- `scale&shift`: `-9.10%`

Delta accuracy gain:

- `metric`: `+0.01608`
- `scale`: `+0.01361`
- `scale&shift`: `+0.01584`

Log RMSE also improves in all three evaluation settings.

## Interpretation

This KITTI result is important for the paper story for three reasons:

1. it confirms the brake is not only helping on indoor relpose datasets such as TUM and ScanNet
2. it shows the state-space brake still helps on outdoor driving scenes
3. it supports the argument that the main problem is over-update, not a narrow benchmark artifact

The result also matches the broader project narrative:

- constant dampening already hinted that recurrent updates were too aggressive
- the adaptive brake provides a cleaner train-free control rule
- the post-fix KITTI result validates that this control rule has real effect at inference time

## Final Conclusion

For the current branch, the KITTI conclusion should be stated as:

- the brake-only method is active after bugfix
- the brake-only method improves outdoor video depth on KITTI
- `ttt3r_momentum_inv_t1` should remain the main candidate for the paper story

## Related Files

- `docs/kitti_brake_summary.md`
- `analysis_results/kitti_brake_20260327_summary.md`
- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/ttt3r/result_metric.json`
- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/ttt3r/result_scale.json`
- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/ttt3r/result_scale&shift.json`
- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/ttt3r_momentum_inv_t1/result_metric.json`
- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/ttt3r_momentum_inv_t1/result_scale.json`
- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/ttt3r_momentum_inv_t1/result_scale&shift.json`
