# KITTI Outdoor Validation and Bugfix Report (March 27, 2026)

## 1) Initial result on `kitti_s1_500` (before bugfix)

The first comparison between `ttt3r` and `ttt3r_momentum_inv_t1` produced identical numbers:

- `result_metric.json`: same
- `result_scale.json`: same
- `result_scale&shift.json`: same

This was treated as a bug signal (not a valid method conclusion).

## 2) Root cause

In `src/dust3r/model.py`, reset-related side states (`brake_state` and spectral state) were being reset whenever `reset_mask` existed, instead of only when reset was actually triggered.

Since `reset_mask` tensor exists on every frame, this caused:

- `brake_state` to be cleared every step
- `_stability_brake()` to repeatedly hit `prev_delta is None`
- `alpha = 1` effectively on every frame

So `ttt3r_momentum_inv_t1` degenerated to `ttt3r` behavior.

## 3) Fix

Committed fix:

- `fix: only reset brake state on true reset mask`
- commit: `4e3e14e`

Implemented condition:

- `has_reset = reset_mask is not None and bool(torch.any(reset_mask).item())`
- reset side states only when `has_reset` is true

Applied in three inference paths:

- `_forward_impl`
- `forward_recurrent_lighter`
- `forward_recurrent_analysis`

## 4) Post-fix sanity check (`kitti_s1_50`, align=`scale&shift`)

After fix, results are no longer identical:

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | δ < 1.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ttt3r` | 0.098075 | 0.553454 | 4.513370 | 0.146409 | 0.914084 |
| `ttt3r_momentum_inv_t1` | 0.093137 | 0.543719 | 4.538340 | 0.143319 | 0.917186 |

Observations:

- `Abs Rel` improved with brake (`0.0981 -> 0.0931`)
- `Sq Rel` improved with brake (`0.5535 -> 0.5437`)
- `Log RMSE` improved with brake (`0.1464 -> 0.1433`)
- `δ < 1.25` improved with brake (`0.9141 -> 0.9172`)
- `RMSE` is slightly worse (`4.5134 -> 4.5383`)

This confirms the brake path is active after the fix.

## 5) Current status

- Bug is fixed in branch `zjc`.
- The old identical-result conclusion on `kitti_s1_500` should be considered invalid due to the reset bug.
- Next recommended step: rerun the full `kitti_s1_500` comparison after fix for final outdoor conclusions.
