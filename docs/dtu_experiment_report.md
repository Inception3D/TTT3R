# DTU Fine 3D Reconstruction Experiment Report

**Generated:** 2026-04-03 23:57:21
**Project:** DDD3R (Directional Decomposition and Dampening for Recurrent 3D Reconstruction)
**Target:** NeurIPS submission

---

## 1. Experiment Setup

### 1.1 Dataset

| Property | Value |
|----------|-------|
| Dataset | DTU MVSNet Evaluation Split |
| Scenes | 22 (scan1, scan4, scan9, scan10, scan11, scan12, scan13, scan15, scan23, scan24, scan29, scan32, scan33, scan34, scan48, scan49, scan62, scan75, scan77, scan110, scan114, scan118) |
| Views per scene | 49 |
| Total frames | 1,078 |
| Data format | MVSNet-style: images/*.jpg, depths/*.npy, cams/*_cam.txt, binary_masks/*.png, pair.txt |
| GT source | Official DTU SampleSet.zip (ObsMask + Plane) + Points.zip (STL reference point clouds) |
| Benchmark lineage | DUSt3R (CVPR'24), MASt3R (ECCV'24), Spann3R (NeurIPS'24), CUT3R |

### 1.2 Evaluation Protocol

1. Recurrent model processes 49 views **sequentially** per scene (simulating video input)
2. Predicted 3D points undergo scale-shift alignment to GT via `Regr3D_t_ScaleShiftInv`
3. ICP point-to-point registration (threshold=100) aligns predicted point cloud to GT
4. Center crop 224x224 applied before metric computation
5. Normals estimated via Open3D after ICP alignment

**Metrics:**

| Metric | Definition | Direction |
|--------|-----------|-----------|
| Accuracy (Acc) | Mean L2 distance from each predicted point to its nearest GT point | Lower = better |
| Completeness (Comp) | Mean L2 distance from each GT point to its nearest predicted point | Lower = better |
| NC (Normal Consistency) | Mean of NC1 (GT normal vs pred normal at nearest) and NC2 (pred vs GT) | Higher = better |

### 1.3 Model & Infrastructure

| Property | Value |
|----------|-------|
| Base model | CUT3R (ARCroco3DStereo) |
| Weights | cut3r_512_dpt_4_64.pth (3.0 GB) |
| Architecture | ViT-L encoder (24 layers) + DPT decoder (12 layers) |
| Input resolution | 512 x 384 |
| GPU | NVIDIA A100-PCIE-40GB |
| CPU | 80 cores |
| Framework | PyTorch + Accelerate |

### 1.4 Configurations (14 total)

| # | Config | update_type | Parameters | Role |
|---|--------|-------------|------------|------|
| 1 | CUT3R (baseline) | cut3r | mask1=1.0 (no gate) | Baseline |
| 2 | TTT3R (baseline) | ttt3r | mask1=sigmoid(cross_attn) | Baseline |
| 3 | Constant Dampening | ttt3r_random | alpha_perp=alpha_parallel=0.5 | M1 evidence |
| 4 | Temporal Brake | ttt3r_momentum | tau=2.0 | M2 baseline |
| 5 | DDD3R (gamma=0, pure ortho) | ddd3r | gamma=0, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95 | DDD3R variant |
| 6 | DDD3R (gamma=1) | ddd3r | gamma=1, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95 | DDD3R variant |
| 7 | DDD3R (gamma=2) | ddd3r | gamma=2, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95 | DDD3R variant |
| 8 | DDD3R (gamma=3) | ddd3r | gamma=3, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95 | DDD3R variant |
| 9 | DDD3R (gamma=4) | ddd3r | gamma=4, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95 | DDD3R variant |
| 10 | DDD3R (gamma=5) | ddd3r | gamma=5, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95 | DDD3R variant |
| 11 | DDD3R (auto: steep_clamp) | ddd3r | auto_gamma=steep_clamp, lo=0.3, hi=0.6, max=3.0 | DDD3R variant |
| 12 | DDD3R (auto: steep_sigmoid) | ddd3r | auto_gamma=steep_sigmoid, k=10.0, max=3.0 | DDD3R variant |
| 13 | DDD3R (auto: warmup_linear) | ddd3r | auto_gamma=warmup_linear, warmup=30, max=3.0 | DDD3R variant |
| 14 | DDD3R (auto: warmup_threshold) | ddd3r | auto_gamma=warmup_threshold, warmup=30, max=3.0 | DDD3R variant |

### 1.5 DDD3R Unified Update Rule

All DDD3R variants are special cases of:

```
S_t = S_{t-1} + beta_t * (alpha_perp * delta_perp + alpha_parallel * delta_parallel)
```

| Setting | Equivalent Method |
|---------|-------------------|
| alpha_perp = alpha_parallel = alpha | Constant dampening (no directional awareness) |
| alpha_perp > alpha_parallel, gamma=0 | Fixed directional decomposition |
| alpha_perp > alpha_parallel, gamma>0 | Drift-adaptive (auto ortho-isotropic sliding) |

### 1.6 Reproducibility

```bash
# Environment
conda activate ttt3r

# Single config
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch \
    --num_processes 1 --main_process_port 29570 \
    eval/mv_recon/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth \
    --output_dir eval_results/mv_recon/dtu/<config> \
    --eval_dataset dtu --dtu_root ./data/dtu --size 512 \
    --model_update_type <type> [--gamma <value>]

# All 14 configs
bash eval/mv_recon/run_dtu_allconfigs.sh

# Generate this report
python3 generate_dtu_report.py
```

**Note on scene list**: This experiment uses the **MVSNet 22-scene evaluation split**
(scan1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118),
following DUSt3R/MASt3R/Spann3R/CUT3R conventions. The `launch.py` on the experiment server
was patched to auto-discover all `scan*` directories under `--dtu_root` instead of using the
hardcoded 15-scene benchmark list. To reproduce, either use the same 22-scene DTU data layout
or apply the equivalent patch (replace `DTU_TEST_SCENES` list with
`sorted([d for d in os.listdir(dtu_root) if d.startswith("scan")])`
in `eval/mv_recon/launch.py`).

---

## 2. Main Results

### 2.1 Overall Summary (Mean +/- Std Over 22 Scenes)

| Config | N | Acc ↓ | Comp ↓ | NC ↑ | vs CUT3R Acc | vs CUT3R Comp |
|--------|---|-------|--------|------|-------------|---------------|
| CUT3R (baseline) | 22 | 3.657 +/- 1.268 | 0.989 +/- 0.432 | 0.621 +/- 0.023 | +0.0% | +0.0% |
| TTT3R (baseline) | 22 | 3.495 +/- 1.963 | 0.997 +/- 0.467 | 0.617 +/- 0.023 | -4.4% | +0.9% |
| Constant Dampening | 22 | 4.227 +/- 3.756 | 1.079 +/- 0.587 | 0.613 +/- 0.027 | +15.6% | +9.2% |
| Temporal Brake | 22 | 5.801 +/- 5.114 | 1.205 +/- 0.699 | 0.612 +/- 0.031 | +58.6% | +21.9% |
| DDD3R (gamma=0, pure ortho) | 22 | 6.146 +/- 5.000 | 1.233 +/- 0.746 | 0.615 +/- 0.033 | +68.1% | +24.7% |
| DDD3R (gamma=1) | 22 | 4.354 +/- 4.078 | 1.086 +/- 0.606 | 0.611 +/- 0.027 | +19.0% | +9.8% |
| DDD3R (gamma=2) | 22 | 4.543 +/- 4.241 | 1.100 +/- 0.650 | 0.612 +/- 0.028 | +24.2% | +11.3% |
| DDD3R (gamma=3) | 22 | 4.752 +/- 4.356 | 1.113 +/- 0.655 | 0.612 +/- 0.027 | +29.9% | +12.6% |
| DDD3R (gamma=4) | 22 | 4.989 +/- 4.544 | 1.136 +/- 0.668 | 0.613 +/- 0.027 | +36.4% | +14.9% |
| DDD3R (gamma=5) | 22 | 5.175 +/- 4.650 | 1.147 +/- 0.680 | 0.613 +/- 0.028 | +41.5% | +16.1% |
| DDD3R (auto: steep_clamp) | 22 | 4.208 +/- 3.958 | 1.065 +/- 0.598 | 0.611 +/- 0.027 | +15.1% | +7.8% |
| DDD3R (auto: steep_sigmoid) | 22 | 4.269 +/- 4.028 | 1.067 +/- 0.592 | 0.611 +/- 0.026 | +16.7% | +8.0% |
| DDD3R (auto: warmup_linear) | 22 | 4.201 +/- 3.761 | 1.064 +/- 0.579 | 0.613 +/- 0.027 | +14.9% | +7.6% |
| DDD3R (auto: warmup_threshold) | 22 | 4.238 +/- 3.755 | 1.069 +/- 0.577 | 0.613 +/- 0.027 | +15.9% | +8.1% |

### 2.2 Robust Statistics: Median Accuracy & Per-Scene Win-Rate

Mean Accuracy is sensitive to outlier scenes (e.g., scan110 with Acc > 10 across most configs).
Median and per-scene win-rate provide a more robust picture:

| Config | Mean Acc ↓ | Median Acc ↓ | Win-rate vs CUT3R (Acc) | Median Comp ↓ |
|--------|-----------|-------------|------------------------|--------------|
| CUT3R (baseline) | 3.657 | 3.343 | — | 0.846 |
| TTT3R | 3.495 | 3.137 | 16/22 (73%) | 0.904 |
| Constant | 4.227 | 2.732 | **14/22 (64%)** | 0.895 |
| Brake | 5.801 | 3.578 | 9/22 (41%) | 0.869 |
| Ortho (gamma=0) | 6.146 | 4.173 | 7/22 (32%) | 1.086 |
| DDD3R gamma=1 | 4.354 | 2.620 | **14/22 (64%)** | 0.920 |
| DDD3R gamma=2 | 4.543 | 2.820 | **14/22 (64%)** | 0.915 |
| DDD3R gamma=3 | 4.752 | 3.040 | 12/22 (55%) | 0.905 |
| DDD3R gamma=4 | 4.989 | 3.309 | 10/22 (45%) | 0.916 |
| DDD3R gamma=5 | 5.175 | 3.414 | 10/22 (45%) | 0.926 |
| Auto: steep_clamp | 4.208 | **2.492** | **14/22 (64%)** | 0.867 |
| Auto: steep_sigmoid | 4.269 | 2.509 | **14/22 (64%)** | 0.883 |
| Auto: warmup_linear | 4.201 | 2.692 | **14/22 (64%)** | 0.897 |
| Auto: warmup_threshold | 4.238 | 2.749 | **14/22 (64%)** | 0.895 |

**Key finding**: While mean Accuracy favors CUT3R, several DDD3R variants (constant, gamma=1-2,
all auto-gamma) **improve Accuracy on the majority of scenes (14/22)** and achieve substantially
lower **median Accuracy** (2.49-2.73 vs CUT3R's 3.34). The mean is inflated by a few outlier
scenes (notably scan24, scan110) where dampening amplifies reconstruction errors.

### 2.3 Gamma Spectrum Ablation

gamma controls the ortho-isotropic spectrum: gamma->inf = pure ortho, gamma->0 = isotropic.

| gamma | Acc ↓ | Comp ↓ | NC ↑ | Behavior |
|-------|-------|--------|------|----------|
| alpha_perp=alpha_parallel (isotropic) | 4.227 | 1.079 | 0.613 | Isotropic baseline |
| 0 (pure ortho) | 6.146 | 1.233 | 0.615 | Full directional decomposition |
| 1 | 4.354 | 1.086 | 0.611 | Light drift-adaptive |
| 2 | 4.543 | 1.100 | 0.612 | Moderate drift-adaptive |
| 3 | 4.752 | 1.113 | 0.612 | Strong drift-adaptive |
| 4 | 4.989 | 1.136 | 0.613 | Stronger drift-adaptive |
| 5 | 5.175 | 1.147 | 0.613 | Near pure ortho |

---

## 3. Per-Scene Results

<details>
<summary><b>CUT3R (baseline)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 3.093 | 0.712 | 0.682 | 0.602 | 0.642 |
| scan4 | 3.273 | 0.948 | 0.634 | 0.587 | 0.610 |
| scan9 | 2.796 | 0.705 | 0.631 | 0.572 | 0.601 |
| scan10 | 3.453 | 0.627 | 0.657 | 0.602 | 0.630 |
| scan11 | 2.042 | 0.649 | 0.646 | 0.594 | 0.620 |
| scan12 | 2.579 | 0.640 | 0.649 | 0.583 | 0.616 |
| scan13 | 5.366 | 1.076 | 0.722 | 0.629 | 0.676 |
| scan15 | 3.413 | 0.818 | 0.630 | 0.569 | 0.599 |
| scan23 | 3.686 | 1.632 | 0.629 | 0.608 | 0.619 |
| scan24 | 5.068 | 0.755 | 0.644 | 0.572 | 0.608 |
| scan29 | 5.923 | 1.718 | 0.661 | 0.620 | 0.640 |
| scan32 | 3.149 | 0.873 | 0.683 | 0.610 | 0.646 |
| scan33 | 4.034 | 1.096 | 0.635 | 0.588 | 0.611 |
| scan34 | 3.032 | 0.745 | 0.686 | 0.610 | 0.648 |
| scan48 | 3.545 | 0.702 | 0.680 | 0.613 | 0.646 |
| scan49 | 2.772 | 1.088 | 0.621 | 0.566 | 0.594 |
| scan62 | 4.055 | 0.947 | 0.670 | 0.602 | 0.636 |
| scan75 | 5.835 | 1.887 | 0.665 | 0.603 | 0.634 |
| scan77 | 3.071 | 1.948 | 0.618 | 0.566 | 0.592 |
| scan110 | 6.199 | 1.124 | 0.635 | 0.560 | 0.598 |
| scan114 | 2.235 | 0.468 | 0.649 | 0.577 | 0.613 |
| scan118 | 1.839 | 0.592 | 0.599 | 0.561 | 0.580 |
| **Mean** | **3.657** | **0.989** | — | — | **0.621** |

</details>

<details>
<summary><b>TTT3R (baseline)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 2.526 | 0.676 | 0.680 | 0.602 | 0.641 |
| scan4 | 2.874 | 0.903 | 0.632 | 0.583 | 0.607 |
| scan9 | 1.453 | 0.549 | 0.610 | 0.560 | 0.585 |
| scan10 | 3.689 | 0.584 | 0.654 | 0.597 | 0.625 |
| scan11 | 1.693 | 0.579 | 0.641 | 0.592 | 0.616 |
| scan12 | 2.187 | 0.580 | 0.638 | 0.579 | 0.609 |
| scan13 | 5.937 | 1.108 | 0.715 | 0.627 | 0.671 |
| scan15 | 2.431 | 0.801 | 0.625 | 0.567 | 0.596 |
| scan23 | 3.459 | 1.705 | 0.636 | 0.616 | 0.626 |
| scan24 | 7.420 | 0.983 | 0.641 | 0.577 | 0.609 |
| scan29 | 4.881 | 1.792 | 0.659 | 0.618 | 0.639 |
| scan32 | 3.112 | 1.057 | 0.676 | 0.600 | 0.638 |
| scan33 | 3.422 | 0.925 | 0.649 | 0.590 | 0.619 |
| scan34 | 1.982 | 0.592 | 0.666 | 0.591 | 0.628 |
| scan48 | 3.433 | 0.743 | 0.680 | 0.611 | 0.646 |
| scan49 | 3.161 | 1.117 | 0.633 | 0.563 | 0.598 |
| scan62 | 3.174 | 0.906 | 0.662 | 0.597 | 0.630 |
| scan75 | 4.309 | 1.840 | 0.658 | 0.594 | 0.626 |
| scan77 | 2.561 | 2.006 | 0.613 | 0.556 | 0.584 |
| scan110 | 9.500 | 1.394 | 0.625 | 0.565 | 0.595 |
| scan114 | 1.462 | 0.449 | 0.637 | 0.573 | 0.605 |
| scan118 | 2.213 | 0.648 | 0.603 | 0.564 | 0.584 |
| **Mean** | **3.495** | **0.997** | — | — | **0.617** |

</details>

<details>
<summary><b>Constant Dampening</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.702 | 0.534 | 0.656 | 0.588 | 0.622 |
| scan4 | 2.572 | 0.845 | 0.628 | 0.583 | 0.605 |
| scan9 | 1.691 | 0.590 | 0.612 | 0.561 | 0.587 |
| scan10 | 8.700 | 1.265 | 0.664 | 0.619 | 0.642 |
| scan11 | 1.485 | 0.557 | 0.634 | 0.589 | 0.612 |
| scan12 | 2.253 | 0.641 | 0.640 | 0.579 | 0.610 |
| scan13 | 7.300 | 1.241 | 0.738 | 0.637 | 0.687 |
| scan15 | 2.124 | 0.701 | 0.619 | 0.560 | 0.590 |
| scan23 | 4.163 | 1.647 | 0.642 | 0.618 | 0.630 |
| scan24 | 12.209 | 1.362 | 0.627 | 0.578 | 0.603 |
| scan29 | 5.467 | 1.664 | 0.654 | 0.621 | 0.638 |
| scan32 | 2.684 | 0.945 | 0.662 | 0.595 | 0.629 |
| scan33 | 3.514 | 0.987 | 0.647 | 0.590 | 0.619 |
| scan34 | 1.616 | 0.508 | 0.656 | 0.585 | 0.620 |
| scan48 | 2.780 | 0.708 | 0.667 | 0.602 | 0.635 |
| scan49 | 3.233 | 1.111 | 0.628 | 0.562 | 0.595 |
| scan62 | 2.673 | 0.795 | 0.660 | 0.593 | 0.626 |
| scan75 | 3.472 | 1.899 | 0.647 | 0.584 | 0.616 |
| scan77 | 4.135 | 2.113 | 0.605 | 0.558 | 0.581 |
| scan110 | 16.058 | 2.592 | 0.579 | 0.539 | 0.559 |
| scan114 | 1.111 | 0.429 | 0.620 | 0.570 | 0.595 |
| scan118 | 2.049 | 0.614 | 0.603 | 0.562 | 0.582 |
| **Mean** | **4.227** | **1.079** | — | — | **0.613** |

</details>

<details>
<summary><b>Temporal Brake</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 2.382 | 0.561 | 0.668 | 0.588 | 0.628 |
| scan4 | 4.054 | 0.896 | 0.638 | 0.584 | 0.611 |
| scan9 | 2.687 | 0.589 | 0.621 | 0.561 | 0.591 |
| scan10 | 10.892 | 1.724 | 0.655 | 0.623 | 0.639 |
| scan11 | 1.336 | 0.599 | 0.629 | 0.587 | 0.608 |
| scan12 | 3.223 | 0.807 | 0.647 | 0.582 | 0.614 |
| scan13 | 9.587 | 1.440 | 0.758 | 0.652 | 0.705 |
| scan15 | 2.831 | 0.655 | 0.629 | 0.563 | 0.596 |
| scan23 | 6.762 | 1.743 | 0.642 | 0.614 | 0.628 |
| scan24 | 19.890 | 1.730 | 0.603 | 0.574 | 0.588 |
| scan29 | 7.476 | 2.025 | 0.644 | 0.611 | 0.628 |
| scan32 | 2.599 | 0.842 | 0.657 | 0.592 | 0.624 |
| scan33 | 9.822 | 1.381 | 0.639 | 0.594 | 0.617 |
| scan34 | 4.859 | 0.667 | 0.663 | 0.589 | 0.626 |
| scan48 | 2.882 | 0.678 | 0.660 | 0.598 | 0.629 |
| scan49 | 3.316 | 1.093 | 0.627 | 0.566 | 0.597 |
| scan62 | 2.533 | 0.765 | 0.655 | 0.592 | 0.624 |
| scan75 | 3.841 | 1.986 | 0.645 | 0.579 | 0.612 |
| scan77 | 4.892 | 2.437 | 0.606 | 0.558 | 0.582 |
| scan110 | 18.286 | 2.907 | 0.561 | 0.522 | 0.541 |
| scan114 | 1.146 | 0.393 | 0.618 | 0.566 | 0.592 |
| scan118 | 2.329 | 0.598 | 0.602 | 0.560 | 0.581 |
| **Mean** | **5.801** | **1.205** | — | — | **0.612** |

</details>

<details>
<summary><b>DDD3R (gamma=0, pure ortho)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 2.691 | 0.588 | 0.677 | 0.592 | 0.634 |
| scan4 | 5.494 | 1.116 | 0.636 | 0.585 | 0.610 |
| scan9 | 3.501 | 0.584 | 0.626 | 0.561 | 0.594 |
| scan10 | 12.372 | 1.825 | 0.665 | 0.620 | 0.643 |
| scan11 | 1.334 | 0.586 | 0.629 | 0.584 | 0.607 |
| scan12 | 2.678 | 0.717 | 0.648 | 0.583 | 0.616 |
| scan13 | 10.646 | 1.558 | 0.762 | 0.658 | 0.710 |
| scan15 | 2.483 | 0.571 | 0.621 | 0.560 | 0.590 |
| scan23 | 8.801 | 2.021 | 0.632 | 0.609 | 0.620 |
| scan24 | 18.456 | 1.383 | 0.612 | 0.584 | 0.598 |
| scan29 | 8.503 | 2.086 | 0.646 | 0.597 | 0.622 |
| scan32 | 4.253 | 1.086 | 0.691 | 0.607 | 0.649 |
| scan33 | 7.089 | 1.152 | 0.629 | 0.593 | 0.611 |
| scan34 | 5.723 | 0.714 | 0.692 | 0.612 | 0.652 |
| scan48 | 2.975 | 0.713 | 0.661 | 0.596 | 0.628 |
| scan49 | 3.350 | 1.086 | 0.626 | 0.566 | 0.596 |
| scan62 | 2.588 | 0.763 | 0.657 | 0.593 | 0.625 |
| scan75 | 4.092 | 1.828 | 0.644 | 0.583 | 0.613 |
| scan77 | 5.383 | 2.396 | 0.607 | 0.555 | 0.581 |
| scan110 | 18.751 | 3.311 | 0.562 | 0.526 | 0.544 |
| scan114 | 1.311 | 0.395 | 0.627 | 0.566 | 0.597 |
| scan118 | 2.741 | 0.654 | 0.601 | 0.564 | 0.582 |
| **Mean** | **6.146** | **1.233** | — | — | **0.615** |

</details>

<details>
<summary><b>DDD3R (gamma=1)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.314 | 0.460 | 0.643 | 0.583 | 0.613 |
| scan4 | 2.559 | 0.834 | 0.630 | 0.582 | 0.606 |
| scan9 | 1.890 | 0.591 | 0.616 | 0.561 | 0.588 |
| scan10 | 9.241 | 1.454 | 0.664 | 0.619 | 0.641 |
| scan11 | 1.346 | 0.581 | 0.629 | 0.588 | 0.608 |
| scan12 | 2.303 | 0.660 | 0.640 | 0.580 | 0.610 |
| scan13 | 7.489 | 1.232 | 0.737 | 0.639 | 0.688 |
| scan15 | 2.117 | 0.654 | 0.618 | 0.559 | 0.589 |
| scan23 | 4.418 | 1.591 | 0.642 | 0.617 | 0.630 |
| scan24 | 14.073 | 1.471 | 0.625 | 0.582 | 0.603 |
| scan29 | 5.091 | 1.520 | 0.649 | 0.619 | 0.634 |
| scan32 | 2.680 | 1.007 | 0.661 | 0.593 | 0.627 |
| scan33 | 3.712 | 1.011 | 0.646 | 0.589 | 0.618 |
| scan34 | 1.718 | 0.477 | 0.657 | 0.583 | 0.620 |
| scan48 | 2.517 | 0.680 | 0.659 | 0.600 | 0.630 |
| scan49 | 3.177 | 1.078 | 0.627 | 0.567 | 0.597 |
| scan62 | 2.434 | 0.766 | 0.656 | 0.593 | 0.624 |
| scan75 | 3.314 | 1.907 | 0.644 | 0.580 | 0.612 |
| scan77 | 4.745 | 2.432 | 0.598 | 0.557 | 0.577 |
| scan110 | 16.513 | 2.443 | 0.579 | 0.537 | 0.558 |
| scan114 | 1.043 | 0.417 | 0.614 | 0.569 | 0.592 |
| scan118 | 2.090 | 0.617 | 0.603 | 0.561 | 0.582 |
| **Mean** | **4.354** | **1.086** | — | — | **0.611** |

</details>

<details>
<summary><b>DDD3R (gamma=2)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.319 | 0.454 | 0.642 | 0.580 | 0.611 |
| scan4 | 2.927 | 0.824 | 0.635 | 0.581 | 0.608 |
| scan9 | 2.091 | 0.586 | 0.618 | 0.560 | 0.589 |
| scan10 | 9.619 | 1.524 | 0.664 | 0.620 | 0.642 |
| scan11 | 1.261 | 0.601 | 0.627 | 0.588 | 0.607 |
| scan12 | 2.382 | 0.674 | 0.642 | 0.581 | 0.612 |
| scan13 | 7.940 | 1.315 | 0.741 | 0.646 | 0.693 |
| scan15 | 2.151 | 0.623 | 0.620 | 0.559 | 0.589 |
| scan23 | 4.957 | 1.614 | 0.643 | 0.614 | 0.628 |
| scan24 | 14.793 | 1.295 | 0.630 | 0.586 | 0.608 |
| scan29 | 5.153 | 1.487 | 0.650 | 0.622 | 0.636 |
| scan32 | 2.712 | 1.005 | 0.662 | 0.594 | 0.628 |
| scan33 | 3.981 | 1.006 | 0.645 | 0.590 | 0.618 |
| scan34 | 2.207 | 0.497 | 0.669 | 0.584 | 0.627 |
| scan48 | 2.491 | 0.663 | 0.657 | 0.599 | 0.628 |
| scan49 | 3.257 | 1.079 | 0.628 | 0.569 | 0.598 |
| scan62 | 2.379 | 0.771 | 0.654 | 0.593 | 0.624 |
| scan75 | 3.327 | 1.889 | 0.641 | 0.577 | 0.609 |
| scan77 | 4.806 | 2.509 | 0.599 | 0.553 | 0.576 |
| scan110 | 17.011 | 2.764 | 0.578 | 0.538 | 0.558 |
| scan114 | 1.068 | 0.405 | 0.616 | 0.568 | 0.592 |
| scan118 | 2.111 | 0.616 | 0.604 | 0.561 | 0.582 |
| **Mean** | **4.543** | **1.100** | — | — | **0.612** |

</details>

<details>
<summary><b>DDD3R (gamma=3)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.473 | 0.471 | 0.648 | 0.582 | 0.615 |
| scan4 | 3.286 | 0.849 | 0.637 | 0.583 | 0.610 |
| scan9 | 2.283 | 0.579 | 0.618 | 0.560 | 0.589 |
| scan10 | 9.929 | 1.579 | 0.662 | 0.617 | 0.640 |
| scan11 | 1.225 | 0.611 | 0.624 | 0.587 | 0.605 |
| scan12 | 2.440 | 0.683 | 0.643 | 0.580 | 0.612 |
| scan13 | 8.319 | 1.384 | 0.744 | 0.645 | 0.694 |
| scan15 | 2.190 | 0.602 | 0.620 | 0.559 | 0.590 |
| scan23 | 5.622 | 1.630 | 0.639 | 0.612 | 0.625 |
| scan24 | 15.687 | 1.355 | 0.628 | 0.590 | 0.609 |
| scan29 | 5.582 | 1.535 | 0.647 | 0.621 | 0.634 |
| scan32 | 2.727 | 0.961 | 0.663 | 0.592 | 0.628 |
| scan33 | 4.217 | 1.019 | 0.645 | 0.590 | 0.617 |
| scan34 | 2.795 | 0.541 | 0.679 | 0.592 | 0.635 |
| scan48 | 2.458 | 0.666 | 0.656 | 0.598 | 0.627 |
| scan49 | 3.322 | 1.083 | 0.628 | 0.570 | 0.599 |
| scan62 | 2.380 | 0.766 | 0.653 | 0.594 | 0.623 |
| scan75 | 3.380 | 1.863 | 0.641 | 0.578 | 0.609 |
| scan77 | 4.894 | 2.515 | 0.601 | 0.553 | 0.577 |
| scan110 | 17.062 | 2.779 | 0.581 | 0.540 | 0.561 |
| scan114 | 1.117 | 0.400 | 0.619 | 0.568 | 0.593 |
| scan118 | 2.159 | 0.619 | 0.603 | 0.561 | 0.582 |
| **Mean** | **4.752** | **1.113** | — | — | **0.612** |

</details>

<details>
<summary><b>DDD3R (gamma=4)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.634 | 0.486 | 0.653 | 0.584 | 0.619 |
| scan4 | 3.622 | 0.878 | 0.638 | 0.584 | 0.611 |
| scan9 | 2.434 | 0.576 | 0.621 | 0.560 | 0.590 |
| scan10 | 10.463 | 1.669 | 0.664 | 0.618 | 0.641 |
| scan11 | 1.215 | 0.616 | 0.625 | 0.586 | 0.606 |
| scan12 | 2.477 | 0.690 | 0.644 | 0.580 | 0.612 |
| scan13 | 8.719 | 1.438 | 0.744 | 0.643 | 0.693 |
| scan15 | 2.238 | 0.590 | 0.621 | 0.560 | 0.590 |
| scan23 | 6.169 | 1.697 | 0.639 | 0.613 | 0.626 |
| scan24 | 16.920 | 1.398 | 0.619 | 0.585 | 0.602 |
| scan29 | 5.945 | 1.649 | 0.645 | 0.616 | 0.631 |
| scan32 | 2.829 | 0.953 | 0.665 | 0.593 | 0.629 |
| scan33 | 4.489 | 1.040 | 0.644 | 0.590 | 0.617 |
| scan34 | 3.284 | 0.566 | 0.681 | 0.595 | 0.638 |
| scan48 | 2.511 | 0.678 | 0.657 | 0.600 | 0.629 |
| scan49 | 3.334 | 1.081 | 0.629 | 0.569 | 0.599 |
| scan62 | 2.399 | 0.770 | 0.653 | 0.595 | 0.624 |
| scan75 | 3.448 | 1.847 | 0.641 | 0.578 | 0.609 |
| scan77 | 5.034 | 2.545 | 0.600 | 0.553 | 0.577 |
| scan110 | 17.207 | 2.801 | 0.583 | 0.537 | 0.560 |
| scan114 | 1.168 | 0.394 | 0.622 | 0.567 | 0.594 |
| scan118 | 2.209 | 0.625 | 0.601 | 0.561 | 0.581 |
| **Mean** | **4.989** | **1.136** | — | — | **0.613** |

</details>

<details>
<summary><b>DDD3R (gamma=5)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.767 | 0.495 | 0.657 | 0.585 | 0.621 |
| scan4 | 3.901 | 0.905 | 0.638 | 0.586 | 0.612 |
| scan9 | 2.611 | 0.572 | 0.622 | 0.559 | 0.590 |
| scan10 | 10.943 | 1.708 | 0.663 | 0.623 | 0.643 |
| scan11 | 1.211 | 0.621 | 0.624 | 0.586 | 0.605 |
| scan12 | 2.519 | 0.699 | 0.645 | 0.581 | 0.613 |
| scan13 | 9.007 | 1.434 | 0.749 | 0.643 | 0.696 |
| scan15 | 2.261 | 0.583 | 0.621 | 0.560 | 0.591 |
| scan23 | 6.710 | 1.774 | 0.638 | 0.612 | 0.625 |
| scan24 | 17.345 | 1.394 | 0.617 | 0.583 | 0.600 |
| scan29 | 6.258 | 1.674 | 0.645 | 0.613 | 0.629 |
| scan32 | 2.861 | 0.947 | 0.665 | 0.591 | 0.628 |
| scan33 | 4.825 | 1.047 | 0.642 | 0.588 | 0.615 |
| scan34 | 3.737 | 0.589 | 0.684 | 0.598 | 0.641 |
| scan48 | 2.551 | 0.668 | 0.656 | 0.599 | 0.627 |
| scan49 | 3.323 | 1.089 | 0.629 | 0.569 | 0.599 |
| scan62 | 2.423 | 0.773 | 0.652 | 0.595 | 0.624 |
| scan75 | 3.505 | 1.837 | 0.641 | 0.578 | 0.610 |
| scan77 | 5.149 | 2.536 | 0.602 | 0.552 | 0.577 |
| scan110 | 17.466 | 2.879 | 0.579 | 0.537 | 0.558 |
| scan114 | 1.217 | 0.393 | 0.624 | 0.566 | 0.595 |
| scan118 | 2.266 | 0.629 | 0.601 | 0.561 | 0.581 |
| **Mean** | **5.175** | **1.147** | — | — | **0.613** |

</details>

<details>
<summary><b>DDD3R (auto: steep_clamp)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.324 | 0.471 | 0.646 | 0.584 | 0.615 |
| scan4 | 2.610 | 0.818 | 0.629 | 0.580 | 0.605 |
| scan9 | 1.897 | 0.573 | 0.615 | 0.561 | 0.588 |
| scan10 | 9.138 | 1.454 | 0.667 | 0.620 | 0.643 |
| scan11 | 1.304 | 0.599 | 0.629 | 0.588 | 0.608 |
| scan12 | 2.236 | 0.642 | 0.639 | 0.579 | 0.609 |
| scan13 | 7.082 | 1.198 | 0.739 | 0.638 | 0.689 |
| scan15 | 2.164 | 0.633 | 0.620 | 0.560 | 0.590 |
| scan23 | 4.293 | 1.602 | 0.640 | 0.614 | 0.627 |
| scan24 | 13.312 | 1.397 | 0.629 | 0.582 | 0.605 |
| scan29 | 5.275 | 1.472 | 0.646 | 0.617 | 0.632 |
| scan32 | 2.347 | 0.930 | 0.653 | 0.593 | 0.623 |
| scan33 | 3.104 | 0.916 | 0.643 | 0.588 | 0.616 |
| scan34 | 1.630 | 0.485 | 0.655 | 0.584 | 0.619 |
| scan48 | 2.374 | 0.650 | 0.657 | 0.600 | 0.629 |
| scan49 | 3.187 | 1.099 | 0.626 | 0.570 | 0.598 |
| scan62 | 2.373 | 0.777 | 0.658 | 0.591 | 0.624 |
| scan75 | 3.221 | 1.944 | 0.641 | 0.579 | 0.610 |
| scan77 | 4.477 | 2.324 | 0.601 | 0.554 | 0.577 |
| scan110 | 16.228 | 2.445 | 0.580 | 0.536 | 0.558 |
| scan114 | 1.037 | 0.419 | 0.614 | 0.570 | 0.592 |
| scan118 | 1.961 | 0.590 | 0.603 | 0.561 | 0.582 |
| **Mean** | **4.208** | **1.065** | — | — | **0.611** |

</details>

<details>
<summary><b>DDD3R (auto: steep_sigmoid)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.283 | 0.455 | 0.642 | 0.580 | 0.611 |
| scan4 | 2.598 | 0.828 | 0.631 | 0.581 | 0.606 |
| scan9 | 1.921 | 0.584 | 0.616 | 0.560 | 0.588 |
| scan10 | 9.383 | 1.406 | 0.665 | 0.618 | 0.642 |
| scan11 | 1.280 | 0.603 | 0.627 | 0.587 | 0.607 |
| scan12 | 2.270 | 0.648 | 0.639 | 0.580 | 0.609 |
| scan13 | 7.214 | 1.210 | 0.738 | 0.637 | 0.688 |
| scan15 | 2.146 | 0.639 | 0.619 | 0.559 | 0.589 |
| scan23 | 4.385 | 1.605 | 0.641 | 0.616 | 0.628 |
| scan24 | 13.726 | 1.417 | 0.626 | 0.580 | 0.603 |
| scan29 | 5.167 | 1.461 | 0.647 | 0.619 | 0.633 |
| scan32 | 2.421 | 0.978 | 0.656 | 0.594 | 0.625 |
| scan33 | 3.278 | 0.937 | 0.645 | 0.589 | 0.617 |
| scan34 | 1.710 | 0.481 | 0.657 | 0.583 | 0.620 |
| scan48 | 2.402 | 0.650 | 0.656 | 0.600 | 0.628 |
| scan49 | 3.188 | 1.089 | 0.627 | 0.570 | 0.598 |
| scan62 | 2.364 | 0.789 | 0.655 | 0.593 | 0.624 |
| scan75 | 3.242 | 1.933 | 0.642 | 0.580 | 0.611 |
| scan77 | 4.600 | 2.406 | 0.600 | 0.554 | 0.577 |
| scan110 | 16.308 | 2.343 | 0.579 | 0.539 | 0.559 |
| scan114 | 1.039 | 0.417 | 0.614 | 0.569 | 0.592 |
| scan118 | 2.003 | 0.602 | 0.603 | 0.560 | 0.582 |
| **Mean** | **4.269** | **1.067** | — | — | **0.611** |

</details>

<details>
<summary><b>DDD3R (auto: warmup_linear)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.584 | 0.525 | 0.653 | 0.589 | 0.621 |
| scan4 | 2.517 | 0.848 | 0.627 | 0.581 | 0.604 |
| scan9 | 1.739 | 0.588 | 0.613 | 0.562 | 0.587 |
| scan10 | 8.812 | 1.175 | 0.663 | 0.619 | 0.641 |
| scan11 | 1.458 | 0.561 | 0.634 | 0.590 | 0.612 |
| scan12 | 2.215 | 0.646 | 0.639 | 0.580 | 0.609 |
| scan13 | 7.029 | 1.080 | 0.737 | 0.638 | 0.687 |
| scan15 | 2.132 | 0.682 | 0.618 | 0.560 | 0.589 |
| scan23 | 4.212 | 1.610 | 0.641 | 0.616 | 0.629 |
| scan24 | 12.222 | 1.321 | 0.627 | 0.580 | 0.603 |
| scan29 | 5.472 | 1.672 | 0.654 | 0.622 | 0.638 |
| scan32 | 2.686 | 0.945 | 0.663 | 0.595 | 0.629 |
| scan33 | 3.547 | 0.996 | 0.647 | 0.591 | 0.619 |
| scan34 | 1.541 | 0.506 | 0.657 | 0.587 | 0.622 |
| scan48 | 2.697 | 0.722 | 0.667 | 0.603 | 0.635 |
| scan49 | 3.223 | 1.108 | 0.627 | 0.563 | 0.595 |
| scan62 | 2.603 | 0.797 | 0.660 | 0.593 | 0.627 |
| scan75 | 3.393 | 1.907 | 0.646 | 0.583 | 0.614 |
| scan77 | 4.191 | 2.150 | 0.605 | 0.559 | 0.582 |
| scan110 | 16.025 | 2.524 | 0.576 | 0.539 | 0.558 |
| scan114 | 1.077 | 0.428 | 0.617 | 0.570 | 0.593 |
| scan118 | 2.051 | 0.607 | 0.603 | 0.561 | 0.582 |
| **Mean** | **4.201** | **1.064** | — | — | **0.613** |

</details>

<details>
<summary><b>DDD3R (auto: warmup_threshold)</b> (22 scenes)</summary>

| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |
|-------|-------|--------|-------|-------|------|
| scan1 | 1.702 | 0.532 | 0.656 | 0.588 | 0.622 |
| scan4 | 2.569 | 0.845 | 0.628 | 0.583 | 0.605 |
| scan9 | 1.695 | 0.590 | 0.612 | 0.561 | 0.587 |
| scan10 | 8.779 | 1.169 | 0.664 | 0.619 | 0.641 |
| scan11 | 1.463 | 0.560 | 0.634 | 0.589 | 0.611 |
| scan12 | 2.214 | 0.646 | 0.638 | 0.579 | 0.609 |
| scan13 | 7.300 | 1.243 | 0.738 | 0.637 | 0.688 |
| scan15 | 2.135 | 0.680 | 0.618 | 0.559 | 0.589 |
| scan23 | 4.222 | 1.608 | 0.642 | 0.616 | 0.629 |
| scan24 | 12.223 | 1.316 | 0.628 | 0.580 | 0.604 |
| scan29 | 5.479 | 1.669 | 0.654 | 0.621 | 0.637 |
| scan32 | 2.722 | 0.945 | 0.664 | 0.595 | 0.630 |
| scan33 | 3.607 | 1.000 | 0.647 | 0.590 | 0.618 |
| scan34 | 1.615 | 0.508 | 0.656 | 0.584 | 0.620 |
| scan48 | 2.777 | 0.709 | 0.667 | 0.603 | 0.635 |
| scan49 | 3.233 | 1.108 | 0.627 | 0.562 | 0.594 |
| scan62 | 2.674 | 0.796 | 0.660 | 0.593 | 0.626 |
| scan75 | 3.474 | 1.898 | 0.648 | 0.584 | 0.616 |
| scan77 | 4.184 | 2.144 | 0.605 | 0.559 | 0.582 |
| scan110 | 16.006 | 2.508 | 0.578 | 0.541 | 0.559 |
| scan114 | 1.111 | 0.429 | 0.620 | 0.570 | 0.595 |
| scan118 | 2.047 | 0.613 | 0.603 | 0.562 | 0.583 |
| **Mean** | **4.238** | **1.069** | — | — | **0.613** |

</details>

---

## 4. Analysis

### 4.1 DTU as Short-Sequence Validation

DTU scenes contain only **49 frames** each. Per the DDD3R diagnostic framework:

- **M1 (Over-update accumulation)**: Scales with sequence length.
  - ScanNet: 1000f/90f degradation ratio = 8.5x
  - TUM: 1000f/90f degradation ratio = 5.0x
  - Sintel (~20-50f): No over-update observed; dampening provides no benefit
  - **DTU (49f)**: Similar regime to Sintel. Over-update has barely begun to accumulate.

- **Observed behavior**: DDD3R methods show **higher mean Accuracy** (worse) than CUT3R,
  but the picture is nuanced — several variants (constant, gamma=1-2, all auto-gamma)
  **improve Accuracy on the majority of scenes (14/22)** and achieve lower median Accuracy
  (2.49-2.73 vs CUT3R's 3.34). The mean is inflated by a few outlier scenes (scan24, scan110)
  where dampening amplifies errors.

### 4.2 Key Observations

1. **Mean vs median divergence**: CUT3R's mean Acc (3.66) is pulled up by outliers but its
   median (3.34) is already moderate. DDD3R variants like auto_steep_clamp achieve median Acc
   of 2.49 (25% lower than CUT3R), yet their mean is worse due to outlier sensitivity.
   This suggests DDD3R typically helps but can catastrophically fail on specific scenes.

2. **TTT3R improves on both mean and majority of scenes** (mean -4.4%, 16/22 win-rate),
   confirming the learned gate has some value even at short sequences.

3. **Several DDD3R variants improve on the majority of scenes**: constant, gamma=1-2, and
   all four auto-gamma variants each win on 14/22 scenes (64% win-rate). These methods
   improve typical-case Accuracy while increasing worst-case Accuracy on outlier scenes.

4. **Stronger ortho = worse robustness**: Brake (9/22) and pure ortho (7/22) have poor
   win-rates. Degradation increases monotonically with decomposition strength in both
   mean and win-rate, confirming that aggressive direction-aware suppression is
   counterproductive when over-update has not yet accumulated.

5. **Auto-gamma self-correction works partially**: Auto variants recover to 14/22 win-rate
   (vs pure ortho's 7/22), but cannot prevent outlier-driven mean degradation.

6. **Gamma spectrum monotonicity holds**: More ortho consistently worsens both mean Acc and
   win-rate, validating the framework's theoretical prediction for short sequences.

### 4.3 Cross-Dataset Comparison

| Dataset | Frames | Over-update severity | Constant vs CUT3R (mean) | Brake vs CUT3R (mean) | Ortho vs CUT3R (mean) |
|---------|--------|---------------------|--------------------------|----------------------|----------------------|
| Sintel | ~20-50 | None | +5% | +14% | +13% |
| **DTU** | **49** | **Minimal** | **+15.6%** | **+58.6%** | **+68.1%** |
| TUM 90f | 90 | Moderate | -53% | -53% | -55% |
| TUM 1000f | 1000 | Severe | -60% | -62% | -66% |
| ScanNet 1000f | 1000 | Severe | -66% | -68% | -40% |

**Note**: DTU mean-based comparison overstates the degradation — median Acc and win-rate
(Section 2.2) show several DDD3R variants improve on the majority of DTU scenes.
The cross-dataset pattern is nonetheless clear: dampening methods provide progressively
larger benefits as sequence length increases, consistent with the M1 diagnostic that
over-update accumulates with sequence length.

### 4.4 Paper Narrative Value

DTU results serve as a **boundary condition** in the paper:
- They are consistent with M1 (over-update scales with length): at 49 frames, the benefit
  of dampening is marginal at best and sensitive to outlier scenes
- Robust statistics (median, win-rate) show DDD3R variants do improve typical-case
  reconstruction, but the improvement is not strong enough to overcome outlier degradation
- Combined with other datasets, DTU helps characterize when and how much DDD3R helps:
  strong and consistent on long sequences (TUM/ScanNet 1000f), marginal and outlier-sensitive
  on short sequences (DTU 49f, Sintel ~50f)

### 4.5 Variance Analysis

High per-scene variance is expected on DTU because:
1. Scene complexity varies greatly (simple objects vs complex geometry)
2. Only 49 frames — less statistical averaging than 1000f sequences
3. ICP alignment sensitivity — different initial conditions per scene
4. Outlier scenes (e.g., scan110 with Acc > 16 across most configs) amplify variance

---

## 5. Experiment Completeness

| Config | Scenes | Status |
|--------|--------|--------|
| CUT3R (baseline) | 22/22 | COMPLETE |
| TTT3R (baseline) | 22/22 | COMPLETE |
| Constant Dampening | 22/22 | COMPLETE |
| Temporal Brake | 22/22 | COMPLETE |
| DDD3R (gamma=0, pure ortho) | 22/22 | COMPLETE |
| DDD3R (gamma=1) | 22/22 | COMPLETE |
| DDD3R (gamma=2) | 22/22 | COMPLETE |
| DDD3R (gamma=3) | 22/22 | COMPLETE |
| DDD3R (gamma=4) | 22/22 | COMPLETE |
| DDD3R (gamma=5) | 22/22 | COMPLETE |
| DDD3R (auto: steep_clamp) | 22/22 | COMPLETE |
| DDD3R (auto: steep_sigmoid) | 22/22 | COMPLETE |
| DDD3R (auto: warmup_linear) | 22/22 | COMPLETE |
| DDD3R (auto: warmup_threshold) | 22/22 | COMPLETE |

**Overall: 14/14 configurations complete.**

---

## 6. Output Artifacts

```
eval_results/mv_recon/dtu/
  <config>/
    DTU/
      logs_0.txt              # per-process log
      logs_all.txt            # merged log with mean metrics
      <scene>.npy             # raw predictions (images, pts, gt, masks)
      <scene>-mask.ply        # predicted point cloud (after masking)
      <scene>-gt.ply          # ground truth point cloud
```
