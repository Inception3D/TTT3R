# Local Work Derived from `szy` Recent TODOs

This note maps the latest `szy`-branch TODOs to what is feasible on the current local PC (8GB VRAM, SAFE224 workflow), and records completed local actions.

## Source TODOs (from `origin/szy` latest `CLAUDE.md`)
- `[P0]` Adaptive ortho ScanNet result analysis (match / threshold)
- `[P1]` Inference overhead (S2): wall-clock time + peak GPU memory
- `[P1]` Theory update
- `[P2]` Paper writing

## Local Feasibility Assessment
1. Adaptive ortho full ScanNet rerun: **Not suitable locally** (long jobs, higher risk of OOM/time cost).
2. Inference overhead (S2): **Suitable locally** using existing SAFE224 repeated-run CSVs.
3. Theory writing / paper text: **Can be drafted locally**, but not the current priority.

## Work Completed in This Local Session
1. Added local S2 analysis script:
   - `analysis/s2_inference_overhead_local.py`
2. Generated local overhead outputs:
   - `analysis_results/s2_inference_overhead_local/overall_overhead.csv`
   - `analysis_results/s2_inference_overhead_local/overhead_by_length.csv`
   - `analysis_results/s2_inference_overhead_local/summary.md`
3. Result:
   - Runtime/per-frame overhead is stable across fixed-seed and different-seed protocols.
   - This closes a local, low-cost part of TODO `[P1]`.

## Suggested Next Local Step
If we continue locally, the highest-value low-risk step is:
- add `ttt3r_momentum_inv_t1_drift0` to the same S2 pipeline for direct runtime-overhead comparison between `drift0` and `drift>0`.
