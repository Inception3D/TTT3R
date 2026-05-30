# DDD3R on `zjc` Branch

## 1. What This Repository Is

This repository is still named `TTT3R`, and the upstream / baseline identity comes from the original TTT3R project on the `szy` branch:

- upstream topic: train-free, inference-time improvement for recurrent 3D reconstruction built on CUT3R
- core model class: `src/dust3r/model.py`
- main entry points: `demo.py`, `eval/relpose/launch.py`, `eval/video_depth/launch.py`, `eval/mv_recon/launch.py`

On the `zjc` branch, however, the research story has changed:

- the active project is **DDD3R**
- DDD3R is a unified state-update control framework for recurrent 3D reconstruction
- the current paper direction is no longer “TTT3R gate variants” as the main contribution
- old TTT3R / brake-only naming still exists in code and artifacts, but many of those names are now treated as historical aliases or intermediate stages

In short:

- **repository lineage**: TTT3R / CUT3R
- **current branch identity**: DDD3R
- **main research question**: how to diagnose and control recurrent state over-update / drift in a unified way

## 2. Read This Branch In The Right Order

If you need to understand `zjc`, trust files in this order:

1. `docs/DD3R_Training.md`
2. `src/dust3r/model.py`
3. `eval/relpose/launch.py`, `eval/video_depth/launch.py`, `eval/mv_recon/launch.py`
4. `analysis/` and `analysis_results/`
5. exported summaries under `eval_results_export/`

Treat these as historical or stage-specific, not as the final branch-wide truth:

- `docs/kitti_brake_summary.md`
- `docs/waymo_nuscenes_h200_runlog_20260329.md`
- `CLAUDE_zjc.md`
- old `ttt3r_*` naming in result folders

Reason:

- `zjc` evolved from “brake-only” story toward “DDD3R unified spectrum”
- some docs describe an earlier paper narrative
- code already contains backward-compatibility aliases from old names to DDD3R names

## 3. Current Project Narrative

The working DDD3R narrative on `zjc` is:

- recurrent 3D reconstruction suffers from **systematic update mis-regulation**
- this appears as three linked issues:
  - **M1**: update magnitude is too large and accumulates with sequence length
  - **M2**: existing adaptive scalar gates collapse toward near-constant behavior
  - **M3**: update direction contains structured drift / redundancy, not just random noise
- DDD3R provides one unified update rule that covers:
  - constant dampening
  - temporal brake-like behavior
  - directional decomposition
  - drift-adaptive interpolation between them

The branch’s main method statement is therefore:

`DDD3R = unified train-free state update control for recurrent 3D reconstruction`

## 4. Canonical Method View

The most important implementation file is:

- `src/dust3r/model.py`

Important facts from the current implementation:

- old names are aliased to DDD3R paper names near the top of the file
- for example:
  - `ttt3r_random` -> `ddd3r_constant`
  - `ttt3r_momentum` -> `ddd3r_brake`
  - `ttt3r_ortho` -> `ddd3r`
- DDD3R-related update types are implemented directly in recurrent forward paths
- orthogonal / drift decomposition, adaptive gamma, warmup, and alpha settings all live in this file

For branch understanding, think in **canonical DDD3R names first**, and only map back to old TTT3R names when reading historical logs.

## 5. Main Code Areas Added Or Expanded On `zjc`

### 5.1 Method / model logic

- `src/dust3r/model.py`
- branch-specific recurrent update logic lives here
- includes aliases, brake-style control, orthogonal decomposition, constant dampening, and adaptive DDD3R variants

### 5.2 Relpose evaluation

- `eval/relpose/launch.py`
- `eval/relpose/metadata.py`
- `eval/relpose/prepare_kitti_odometry.py`
- `eval/relpose/run_*.sh`
- `scripts/server/run_nuscenes_relpose_pipeline.sh`
- `scripts/server/run_waymo_relpose_pipeline.sh`

This area was expanded heavily on `zjc` for large-scale relpose evaluation and export.

### 5.3 Video depth evaluation

- `eval/video_depth/launch.py`
- `eval/video_depth/run_kitti.sh`
- large exported results under `eval_results_export/video_depth/`

This includes the outdoor KITTI bugfix-era comparison that was important during the brake-only stage.

### 5.4 Multi-view reconstruction

- `eval/mv_recon/launch.py`
- `eval/mv_recon/generate_7scenes_report.py`
- `eval/mv_recon/generate_dtu_report.py`
- `eval/mv_recon/run_7scenes_allconfigs.sh`
- `eval/mv_recon/run_dtu_allconfigs.sh`

This area now supports 7Scenes and DTU-style reporting for DDD3R ablations.

### 5.5 Analysis pipeline

- `analysis/`
- `analysis_results/`
- `docs/figures/`

This is where much of the paper-facing evidence and plots live:

- per-scene improvement analysis
- variance proxies
- state convergence analysis
- reset sensitivity
- overhead / runtime studies
- paper figure generation

### 5.6 Single-object local benchmark

- `benchmark_single_object/`

This is a local experimental sandbox for geometry / stability style comparisons.

### 5.7 Dataset preparation and remote workflow

- `datasets_preprocess/`
- `scripts/server/`
- `docs/REMOTE_KITTI_RUNBOOK.md`
- `docs/server_quickstart_waymo_nuscenes.md`
- `WINDOWS_DEV.md`

`zjc` is not only a method branch; it also adds a practical remote-eval workflow.

## 6. Core Documents Worth Reading

### Branch-level method / paper framing

- `docs/DD3R_Training.md`

This is the closest thing to the current branch manifesto.

### Historical branch summaries

- `CLAUDE_zjc.md`
- `docs/kitti_brake_summary.md`
- `docs/waymo_nuscenes_h200_runlog_20260329.md`

Useful for archaeology, but not authoritative over the newer DDD3R framing.

### Generated experiment reports

- `docs/dtu_experiment_report.md`
- `eval/relpose/kitti_odo_report.md`
- `eval/relpose/kitti_odo_full_report.md`
- `docs/reproducibility_safe224_seedstudy.md`
- `docs/reset_interval_sensitivity_safe224.md`

### Figures / paper export

- `analysis/paper_figures.py`
- `analysis_results/paper_figures/`
- `docs/figures/`

## 7. Practical Working Rules For This Branch

### Naming

When editing or documenting `zjc`, prefer:

- `DDD3R`
- `ddd3r_constant`
- `ddd3r_brake`
- `ddd3r`
- adaptive gamma language

Avoid presenting these as the main project identity:

- “TTT3R joint”
- “brake-only final story”
- “TTT3R gate paper”

Those names may still appear in code or exports, but they are not the best top-level framing anymore.

### Trust the code over historical markdown

If a markdown document conflicts with:

- `src/dust3r/model.py`
- current eval scripts
- current aliases / argument names

then the code wins.

### Be careful with old exported results

Some exported result folders preserve old method names for continuity.
When summarizing them:

- explain the old name
- map it to the current DDD3R interpretation
- note if the result belongs to an older narrative stage

### Reproducibility caveat

Some reports were generated from server-side logs and then copied back locally.
Before claiming a report is reproducible from the current branch state, verify:

- branch commit
- eval script scene list
- output path conventions
- whether the report was log-derived or directory-derived

## 8. Important Branch Caveats

### Caveat A: repository name and project name differ

- repo / upstream label: `TTT3R`
- current `zjc` paper identity: `DDD3R`

Do not “fix” all old names mechanically; many are intentionally kept for compatibility.

### Caveat B: not all docs are synchronized

There are at least three narrative layers coexisting:

1. upstream TTT3R project description from `szy`
2. intermediate brake-only story
3. current DDD3R unified-spectrum story

When writing new documentation, align with layer 3 unless the task is explicitly historical.

### Caveat C: old bug-era results exist

Some brake-era outputs and summaries were affected by implementation bugs later fixed.
If using those results, check whether the document already labels them as pre-fix or historical.

## 9. Useful Entry Commands

Typical commands still follow upstream TTT3R structure:

```bash
conda activate ttt3r

# demo
python demo.py --model_path MODEL_PATH --seq_path SEQ_PATH --output_dir OUT_DIR

# relpose
PYTHONPATH=src accelerate launch eval/relpose/launch.py ...

# video depth
PYTHONPATH=src accelerate launch eval/video_depth/launch.py ...

# multi-view reconstruction
PYTHONPATH=src accelerate launch eval/mv_recon/launch.py ...
```

But when selecting methods, prefer current DDD3R names or confirm which aliases the script accepts.

## 10. If You Need A One-Paragraph Summary

`szy` is the original TTT3R / CUT3R-based project branch. `zjc` has grown into a much larger experimental branch whose real research identity is now DDD3R: a unified, train-free framework for controlling recurrent state updates via constant dampening, brake-like control, directional decomposition, and adaptive interpolation on one spectrum. The most reliable branch-level references are `docs/DD3R_Training.md`, `src/dust3r/model.py`, and the current eval / analysis pipelines; many older brake-only summaries remain useful, but they should be read as intermediate history rather than the final project definition.
