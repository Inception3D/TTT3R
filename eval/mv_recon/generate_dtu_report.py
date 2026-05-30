#!/usr/bin/env python3
"""
generate_dtu_report.py — Generate comprehensive DTU experiment report
Run after all 14 configs complete:
    python3 generate_dtu_report.py
Can also run mid-experiment to see partial results.
"""
import os, re, sys
from collections import defaultdict
from datetime import datetime

EVAL_ROOT = "/root/TTT3R/eval_results/mv_recon/dtu"
LOG_A = "/root/TTT3R/dtu_group_a.log"
LOG_B = "/root/TTT3R/dtu_group_b.log"
REPORT_PATH = "/root/TTT3R/dtu_experiment_report.md"

PATTERN = re.compile(
    r"Idx:\s*(?P<scene>\S+),\s*"
    r"Acc:\s*(?P<acc>[\d.e+-]+),\s*"
    r"Comp:\s*(?P<comp>[\d.e+-]+),\s*"
    r"NC1:\s*(?P<nc1>[\d.e+-]+),\s*"
    r"NC2:\s*(?P<nc2>[\d.e+-]+)\s*-\s*"
    r"Acc_med:\s*(?P<acc_med>[\d.e+-]+),\s*"
    r"Compc_med:\s*(?P<comp_med>[\d.e+-]+),\s*"
    r"NC1c_med:\s*(?P<nc1_med>[\d.e+-]+),\s*"
    r"NC2c_med:\s*(?P<nc2_med>[\d.e+-]+)"
)

CONFIG_ORDER = [
    "cut3r", "ttt3r", "constant", "brake", "ortho",
    "ddd3r_g1", "ddd3r_g2", "ddd3r_g3", "ddd3r_g4", "ddd3r_g5",
    "auto_steep_clamp", "auto_steep_sigmoid",
    "auto_warmup_linear", "auto_warmup_threshold",
]

CONFIG_LABELS = {
    "cut3r": "CUT3R (baseline)",
    "ttt3r": "TTT3R (baseline)",
    "constant": "Constant Dampening",
    "brake": "Temporal Brake",
    "ortho": "DDD3R (gamma=0, pure ortho)",
    "ddd3r_g1": "DDD3R (gamma=1)",
    "ddd3r_g2": "DDD3R (gamma=2)",
    "ddd3r_g3": "DDD3R (gamma=3)",
    "ddd3r_g4": "DDD3R (gamma=4)",
    "ddd3r_g5": "DDD3R (gamma=5)",
    "auto_steep_clamp": "DDD3R (auto: steep_clamp)",
    "auto_steep_sigmoid": "DDD3R (auto: steep_sigmoid)",
    "auto_warmup_linear": "DDD3R (auto: warmup_linear)",
    "auto_warmup_threshold": "DDD3R (auto: warmup_threshold)",
}

CONFIG_PARAMS = {
    "cut3r": "mask1=1.0 (no gate)",
    "ttt3r": "mask1=sigmoid(cross_attn)",
    "constant": "alpha_perp=alpha_parallel=0.5",
    "brake": "tau=2.0",
    "ortho": "gamma=0, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95",
    "ddd3r_g1": "gamma=1, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95",
    "ddd3r_g2": "gamma=2, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95",
    "ddd3r_g3": "gamma=3, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95",
    "ddd3r_g4": "gamma=4, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95",
    "ddd3r_g5": "gamma=5, alpha_perp=0.5, alpha_parallel=0.05, beta_ema=0.95",
    "auto_steep_clamp": "auto_gamma=steep_clamp, lo=0.3, hi=0.6, max=3.0",
    "auto_steep_sigmoid": "auto_gamma=steep_sigmoid, k=10.0, max=3.0",
    "auto_warmup_linear": "auto_gamma=warmup_linear, warmup=30, max=3.0",
    "auto_warmup_threshold": "auto_gamma=warmup_threshold, warmup=30, max=3.0",
}


def parse_log(log_path):
    results = {}
    current_config = None
    if not os.path.exists(log_path):
        return results
    with open(log_path) as f:
        for line in f:
            run_match = re.search(r"\[(?:A|B)\]\s+(\S+)", line)
            if run_match and "[run]" not in line and "done" not in line:
                current_config = run_match.group(1)
                if current_config not in results:
                    results[current_config] = {}
                continue
            # Also detect original format
            run_match2 = re.search(r"\[run\]\s+(\S+)\s*->", line)
            if run_match2:
                current_config = run_match2.group(1)
                if current_config not in results:
                    results[current_config] = {}
                continue
            m = PATTERN.search(line)
            if m and current_config:
                d = m.groupdict()
                scene = d["scene"].rstrip(",")
                results[current_config][scene] = {
                    k: float(v) for k, v in d.items() if k != "scene"
                }
    return results


def merge_results(*logs):
    merged = {}
    for log in logs:
        data = parse_log(log)
        for cfg, scenes in data.items():
            if cfg not in merged:
                merged[cfg] = {}
            merged[cfg].update(scenes)
    return merged


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def generate_report(results):
    lines = []
    L = lines.append

    L("# DTU Fine 3D Reconstruction Experiment Report")
    L("")
    L(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"**Project:** DDD3R (Directional Decomposition and Dampening for Recurrent 3D Reconstruction)")
    L(f"**Target:** NeurIPS submission")
    L("")
    L("---")
    L("")

    # ================================================================
    # Section 1: Experiment Setup
    # ================================================================
    L("## 1. Experiment Setup")
    L("")
    L("### 1.1 Dataset")
    L("")
    L("| Property | Value |")
    L("|----------|-------|")
    L("| Dataset | DTU MVSNet Evaluation Split |")
    L("| Scenes | 22 (scan1, scan4, scan9, scan10, scan11, scan12, scan13, scan15, scan23, scan24, scan29, scan32, scan33, scan34, scan48, scan49, scan62, scan75, scan77, scan110, scan114, scan118) |")
    L("| Views per scene | 49 |")
    L("| Total frames | 1,078 |")
    L("| Data format | MVSNet-style: images/*.jpg, depths/*.npy, cams/*_cam.txt, binary_masks/*.png, pair.txt |")
    L("| GT source | Official DTU SampleSet.zip (ObsMask + Plane) + Points.zip (STL reference point clouds) |")
    L("| Benchmark lineage | DUSt3R (CVPR'24), MASt3R (ECCV'24), Spann3R (NeurIPS'24), CUT3R |")
    L("")
    L("### 1.2 Evaluation Protocol")
    L("")
    L("1. Recurrent model processes 49 views **sequentially** per scene (simulating video input)")
    L("2. Predicted 3D points undergo scale-shift alignment to GT via `Regr3D_t_ScaleShiftInv`")
    L("3. ICP point-to-point registration (threshold=100) aligns predicted point cloud to GT")
    L("4. Center crop 224x224 applied before metric computation")
    L("5. Normals estimated via Open3D after ICP alignment")
    L("")
    L("**Metrics:**")
    L("")
    L("| Metric | Definition | Direction |")
    L("|--------|-----------|-----------|")
    L("| Accuracy (Acc) | Mean L2 distance from each predicted point to its nearest GT point | Lower = better |")
    L("| Completeness (Comp) | Mean L2 distance from each GT point to its nearest predicted point | Lower = better |")
    L("| NC (Normal Consistency) | Mean of NC1 (GT normal vs pred normal at nearest) and NC2 (pred vs GT) | Higher = better |")
    L("")
    L("### 1.3 Model & Infrastructure")
    L("")
    L("| Property | Value |")
    L("|----------|-------|")
    L("| Base model | CUT3R (ARCroco3DStereo) |")
    L("| Weights | cut3r_512_dpt_4_64.pth (3.0 GB) |")
    L("| Architecture | ViT-L encoder (24 layers) + DPT decoder (12 layers) |")
    L("| Input resolution | 512 x 384 |")
    L("| GPU | NVIDIA A100-PCIE-40GB |")
    L("| CPU | 80 cores |")
    L("| Framework | PyTorch + Accelerate |")
    L("")
    L("### 1.4 Configurations (14 total)")
    L("")
    L("| # | Config | update_type | Parameters | Role |")
    L("|---|--------|-------------|------------|------|")
    for i, cfg in enumerate(CONFIG_ORDER, 1):
        L(f"| {i} | {CONFIG_LABELS[cfg]} | {cfg if cfg in ('cut3r','ttt3r') else 'ttt3r_random' if cfg=='constant' else 'ttt3r_momentum' if cfg=='brake' else 'ddd3r'} | {CONFIG_PARAMS[cfg]} | {'Baseline' if cfg in ('cut3r','ttt3r') else 'M1 evidence' if cfg=='constant' else 'M2 baseline' if cfg=='brake' else 'DDD3R variant'} |")
    L("")
    L("### 1.5 DDD3R Unified Update Rule")
    L("")
    L("All DDD3R variants are special cases of:")
    L("")
    L("```")
    L("S_t = S_{t-1} + beta_t * (alpha_perp * delta_perp + alpha_parallel * delta_parallel)")
    L("```")
    L("")
    L("| Setting | Equivalent Method |")
    L("|---------|-------------------|")
    L("| alpha_perp = alpha_parallel = alpha | Constant dampening (no directional awareness) |")
    L("| alpha_perp > alpha_parallel, gamma=0 | Fixed directional decomposition |")
    L("| alpha_perp > alpha_parallel, gamma>0 | Drift-adaptive (auto ortho-isotropic sliding) |")
    L("")
    L("### 1.6 Reproducibility")
    L("")
    L("```bash")
    L("# Environment")
    L("conda activate ttt3r")
    L("")
    L("# Single config")
    L("CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch \\")
    L("    --num_processes 1 --main_process_port 29570 \\")
    L("    eval/mv_recon/launch.py \\")
    L("    --weights model/cut3r_512_dpt_4_64.pth \\")
    L("    --output_dir eval_results/mv_recon/dtu/<config> \\")
    L("    --eval_dataset dtu --dtu_root ./data/dtu --size 512 \\")
    L("    --model_update_type <type> [--gamma <value>]")
    L("")
    L("# All 14 configs")
    L("bash eval/mv_recon/run_dtu_allconfigs.sh")
    L("")
    L("# Generate this report")
    L("python3 generate_dtu_report.py")
    L("```")
    L("")
    L("---")
    L("")

    # ================================================================
    # Section 2: Main Results
    # ================================================================
    L("## 2. Main Results")
    L("")
    L("### 2.1 Overall Summary (Mean +/- Std Over 22 Scenes)")
    L("")
    L("| Config | N | Acc ↓ | Comp ↓ | NC ↑ | vs CUT3R Acc | vs CUT3R Comp |")
    L("|--------|---|-------|--------|------|-------------|---------------|")

    baseline_acc = None
    baseline_comp = None
    summary = {}

    for cfg in CONFIG_ORDER:
        if cfg not in results or len(results[cfg]) == 0:
            L(f"| {CONFIG_LABELS.get(cfg, cfg)} | 0 | — | — | — | — | — |")
            continue

        scenes = results[cfg]
        acc_vals = [s["acc"] for s in scenes.values()]
        comp_vals = [s["comp"] for s in scenes.values()]
        nc_vals = [(s["nc1"] + s["nc2"]) / 2 for s in scenes.values()]

        m_acc = mean(acc_vals)
        m_comp = mean(comp_vals)
        m_nc = mean(nc_vals)
        s_acc = std(acc_vals)
        s_comp = std(comp_vals)
        s_nc = std(nc_vals)

        summary[cfg] = {"acc": m_acc, "comp": m_comp, "nc": m_nc, "n": len(scenes)}

        if cfg == "cut3r":
            baseline_acc = m_acc
            baseline_comp = m_comp

        if baseline_acc and baseline_acc > 0:
            d_acc = f"{(m_acc - baseline_acc) / baseline_acc * 100:+.1f}%"
            d_comp = f"{(m_comp - baseline_comp) / baseline_comp * 100:+.1f}%"
        else:
            d_acc = "—"
            d_comp = "—"

        n = len(scenes)
        L(f"| {CONFIG_LABELS.get(cfg, cfg)} | {n} | {m_acc:.3f} +/- {s_acc:.3f} | {m_comp:.3f} +/- {s_comp:.3f} | {m_nc:.3f} +/- {s_nc:.3f} | {d_acc} | {d_comp} |")

    L("")

    # ================================================================
    # Spectrum ablation
    # ================================================================
    L("### 2.2 Gamma Spectrum Ablation")
    L("")
    L("gamma controls the ortho-isotropic spectrum: gamma->inf = pure ortho, gamma->0 = isotropic.")
    L("")
    L("| gamma | Acc ↓ | Comp ↓ | NC ↑ | Behavior |")
    L("|-------|-------|--------|------|----------|")

    gamma_map = [
        ("constant", "alpha_perp=alpha_parallel (isotropic)", "Isotropic baseline"),
        ("ortho", "0 (pure ortho)", "Full directional decomposition"),
        ("ddd3r_g1", "1", "Light drift-adaptive"),
        ("ddd3r_g2", "2", "Moderate drift-adaptive"),
        ("ddd3r_g3", "3", "Strong drift-adaptive"),
        ("ddd3r_g4", "4", "Stronger drift-adaptive"),
        ("ddd3r_g5", "5", "Near pure ortho"),
    ]
    for cfg, glabel, desc in gamma_map:
        if cfg in summary:
            s = summary[cfg]
            L(f"| {glabel} | {s['acc']:.3f} | {s['comp']:.3f} | {s['nc']:.3f} | {desc} |")
        else:
            L(f"| {glabel} | — | — | — | {desc} |")
    L("")

    # ================================================================
    # Section 3: Per-Scene
    # ================================================================
    L("---")
    L("")
    L("## 3. Per-Scene Results")
    L("")

    for cfg in CONFIG_ORDER:
        if cfg not in results or len(results[cfg]) == 0:
            continue
        scenes = results[cfg]
        L(f"<details>")
        L(f"<summary><b>{CONFIG_LABELS.get(cfg, cfg)}</b> ({len(scenes)} scenes)</summary>")
        L(f"")
        L(f"| Scene | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |")
        L(f"|-------|-------|--------|-------|-------|------|")

        sorted_scenes = sorted(scenes.keys(), key=lambda x: int(x.replace("scan", "")))
        for sc in sorted_scenes:
            m = scenes[sc]
            nc = (m["nc1"] + m["nc2"]) / 2
            L(f"| {sc} | {m['acc']:.3f} | {m['comp']:.3f} | {m['nc1']:.3f} | {m['nc2']:.3f} | {nc:.3f} |")

        acc_vals = [s["acc"] for s in scenes.values()]
        comp_vals = [s["comp"] for s in scenes.values()]
        nc_vals = [(s["nc1"] + s["nc2"]) / 2 for s in scenes.values()]
        L(f"| **Mean** | **{mean(acc_vals):.3f}** | **{mean(comp_vals):.3f}** | — | — | **{mean(nc_vals):.3f}** |")
        L(f"")
        L(f"</details>")
        L(f"")

    # ================================================================
    # Section 4: Analysis
    # ================================================================
    L("---")
    L("")
    L("## 4. Analysis")
    L("")
    L("### 4.1 DTU as Short-Sequence Validation")
    L("")
    L("DTU scenes contain only **49 frames** each. Per the DDD3R diagnostic framework:")
    L("")
    L("- **M1 (Over-update accumulation)**: Scales with sequence length.")
    L("  - ScanNet: 1000f/90f degradation ratio = 8.5x")
    L("  - TUM: 1000f/90f degradation ratio = 5.0x")
    L("  - Sintel (~20-50f): No over-update observed; dampening provides no benefit")
    L("  - **DTU (49f)**: Similar regime to Sintel. Over-update has barely begun to accumulate.")
    L("")
    L("- **Expected behavior**: DDD3R methods should show **marginal or no improvement** over baselines.")
    L("  This is **not a weakness** but a validation of the core thesis: over-update is a long-sequence phenomenon.")
    L("")
    L("- **Key validation criterion**: DDD3R should **not degrade** performance (training-free, zero-overhead guarantee).")
    L("")
    L("### 4.2 Cross-Dataset Comparison")
    L("")
    L("| Dataset | Frames | Over-update severity | Constant vs CUT3R | Brake vs CUT3R | Ortho vs CUT3R |")
    L("|---------|--------|---------------------|-------------------|----------------|----------------|")
    L("| Sintel | ~20-50 | None | +5% (hurts) | +14% (hurts) | +13% (hurts) |")

    # Fill DTU row from our results
    dtu_const = f"{(summary['constant']['acc'] - summary['cut3r']['acc']) / summary['cut3r']['acc'] * 100:+.1f}%" if "constant" in summary and "cut3r" in summary else "—"
    dtu_brake = f"{(summary['brake']['acc'] - summary['cut3r']['acc']) / summary['cut3r']['acc'] * 100:+.1f}%" if "brake" in summary and "cut3r" in summary else "—"
    dtu_ortho = f"{(summary['ortho']['acc'] - summary['cut3r']['acc']) / summary['cut3r']['acc'] * 100:+.1f}%" if "ortho" in summary and "cut3r" in summary else "—"
    L(f"| **DTU** | **49** | **Minimal** | **{dtu_const}** | **{dtu_brake}** | **{dtu_ortho}** |")
    L("| TUM 90f | 90 | Moderate | -53% | -53% | -55% |")
    L("| TUM 1000f | 1000 | Severe | -60% | -62% | -66% |")
    L("| ScanNet 1000f | 1000 | Severe | -66% | -68% | -40% |")
    L("")
    L("### 4.3 Possible Outcomes & Interpretation")
    L("")
    L("| Outcome | Interpretation | Paper narrative |")
    L("|---------|---------------|-----------------|")
    L("| DDD3R ~ CUT3R | Over-update is length-dependent; method is safe for short sequences | Supports M1: over-update hasn't accumulated at 49f |")
    L("| DDD3R slightly better | Even 49f shows some over-update; DDD3R helps universally | Strengthens the contribution — method is always beneficial |")
    L("| DDD3R slightly worse | Directional decomposition removes useful early updates | Consistent with Sintel; short sequences need no dampening |")
    L("| Constant helps but ortho doesn't | Drift at 49f is mostly useful refinement (like ScanNet) | Supports M3: drift properties are scene-dependent |")
    L("")
    L("### 4.4 Variance Analysis")
    L("")
    L("High per-scene variance is expected on DTU because:")
    L("1. Scene complexity varies greatly (simple objects vs complex geometry)")
    L("2. Only 49 frames — less statistical averaging than 1000f sequences")
    L("3. ICP alignment sensitivity — different initial conditions per scene")
    L("")

    # ================================================================
    # Section 5: Completeness
    # ================================================================
    L("---")
    L("")
    L("## 5. Experiment Completeness")
    L("")
    L("| Config | Scenes | Status |")
    L("|--------|--------|--------|")
    for cfg in CONFIG_ORDER:
        n = len(results.get(cfg, {}))
        status = "COMPLETE" if n == 22 else f"IN PROGRESS ({n}/22)" if n > 0 else "NOT STARTED"
        L(f"| {CONFIG_LABELS.get(cfg, cfg)} | {n}/22 | {status} |")
    L("")

    total = sum(1 for c in CONFIG_ORDER if len(results.get(c, {})) == 22)
    L(f"**Overall: {total}/14 configurations complete.**")
    L("")

    # ================================================================
    # Section 6: Artifacts
    # ================================================================
    L("---")
    L("")
    L("## 6. Output Artifacts")
    L("")
    L("```")
    L("eval_results/mv_recon/dtu/")
    L("  <config>/")
    L("    DTU/")
    L("      logs_0.txt              # per-process log")
    L("      logs_all.txt            # merged log with mean metrics")
    L("      <scene>.npy             # raw predictions (images, pts, gt, masks)")
    L("      <scene>-mask.ply        # predicted point cloud (after masking)")
    L("      <scene>-gt.ply          # ground truth point cloud")
    L("```")
    L("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Report: {REPORT_PATH}")
    print(f"Configs: {list(results.keys())}")
    print(f"Scenes total: {sum(len(v) for v in results.values())}")


if __name__ == "__main__":
    results = merge_results(LOG_A, LOG_B)
    generate_report(results)
