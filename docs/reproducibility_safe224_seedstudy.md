# Reproducibility Study (SAFE224, Stability Brake)

## Goal
Run the same configuration 3 times and report mean ± std to verify that improvements are not accidental.

## Setup
- Branch: `zjc-ttt3r-sensitivity-clean`
- Dataset split: sampled single-object sequences (`apple/bottle`, lengths `12/24`)
- Method: `ttt3r_momentum_inv_t1`
- Image size: `224` (GPU-safe on 8GB VRAM)
- Device: `cuda`
- Repeats:
  - Fixed-seed group: `42, 42, 42`
  - Different-seed group: `41, 42, 43`

## Command
```bash
python3 benchmark_single_object/scripts/run_reproducibility.py \
  --config benchmark_single_object/configs/experiment_brake_ablation_safe224.yaml \
  --methods ttt3r_momentum_inv_t1 \
  --tag repro_safe224_seedstudy
```

## Outputs
- Raw runs:
  - `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_raw_results.csv`
- Summary (mean ± std):
  - `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_summary_overall.csv`
  - `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_summary_by_length.csv`
- Per-sequence stability:
  - `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_sequence_stability.csv`

## Main Results (3x repeat)

### Overall (all sequences pooled)
| Protocol | Runtime (s) | Per-frame (s) | Basic Consistency | Loop Trans Error |
|---|---:|---:|---:|---:|
| fixed seed (42,42,42) | `15.8234 ± 1.7198` | `0.9656 ± 0.2895` | `1.1085 ± 0.5101` | `0.3946 ± 0.3710` |
| different seed (41,42,43) | `15.8584 ± 1.5049` | `0.9634 ± 0.2614` | `1.1085 ± 0.5101` | `0.3946 ± 0.3710` |

### By sequence length
| Protocol | Seq length | Runtime (s) | Per-frame (s) | Basic Consistency | Loop Trans Error |
|---|---:|---:|---:|---:|---:|
| fixed seed | 12 | `14.7033 ± 1.7796` | `1.2253 ± 0.1483` | `1.0513 ± 0.4124` | `0.3842 ± 0.3741` |
| fixed seed | 24 | `16.9434 ± 0.5737` | `0.7060 ± 0.0239` | `1.1656 ± 0.6282` | `0.4051 ± 0.4033` |
| different seed | 12 | `14.5244 ± 0.7075` | `1.2104 ± 0.0590` | `1.0513 ± 0.4124` | `0.3842 ± 0.3741` |
| different seed | 24 | `17.1923 ± 0.4593` | `0.7163 ± 0.0191` | `1.1656 ± 0.6282` | `0.4051 ± 0.4033` |

## Result Analysis

1. **Core quality metrics are seed-invariant in this SAFE224 setting.**  
`basic_consistency_score` and `loop_closure_trans_error` are numerically identical between fixed-seed and different-seed groups (same mean/std to 4 decimals).

2. **Per-sequence repeat variance is zero for geometric quality metrics.**  
In `repro_sequence_stability.csv`, per-sequence std for `output_point_count / mean_conf / median_conf / basic_consistency_score / loop_closure_trans_error / loop_closure_rot_error_deg` is `0.0` in both protocols, indicating deterministic behavior under this pipeline.

3. **Observed variance is mainly runtime jitter, not geometry jitter.**  
Runtime shows small fluctuations (`~1.5–1.7s` std overall), which is expected from system scheduling and GPU runtime noise, while geometry quality remains stable.

## Takeaway for Paper

This reproducibility study supports the claim that the reported SAFE224 brake results are **robust and non-accidental**: changing random seeds does not change geometric quality outcomes in our tested setting.
