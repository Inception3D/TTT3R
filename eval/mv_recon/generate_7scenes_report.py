#!/usr/bin/env python3
"""
generate_7scenes_report.py — Comprehensive 7scenes 3D Reconstruction Report
Run after all 14 configs complete:
    python3 generate_7scenes_report.py
Can also run mid-experiment for partial results.
"""
import os, re
from datetime import datetime

EVAL_ROOT = "/root/TTT3R/eval_results/video_recon/7scenes_200"
REPORT_PATH = "/root/TTT3R/7scenes_experiment_report.md"

PATTERN = re.compile(
    r"Idx:\s*(?P<scene>[^,]+),\s*"
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

SCENE_CATEGORIES = {
    "chess": ["chess/seq-03", "chess/seq-05"],
    "fire": ["fire/seq-03", "fire/seq-04"],
    "heads": ["heads/seq-01"],
    "office": ["office/seq-02", "office/seq-06", "office/seq-07", "office/seq-09"],
    "pumpkin": ["pumpkin/seq-01", "pumpkin/seq-07"],
    "redkitchen": ["redkitchen/seq-03", "redkitchen/seq-04", "redkitchen/seq-06",
                    "redkitchen/seq-12", "redkitchen/seq-14"],
    "stairs": ["stairs/seq-01", "stairs/seq-04"],
}

TOTAL_SEQUENCES = 18


def parse_logs_dir(config_name):
    """Parse logs from eval_results directory."""
    results = {}
    # Try multiple log locations
    candidates = [
        os.path.join(EVAL_ROOT, config_name, "7scenes", "logs_all.txt"),
        os.path.join(EVAL_ROOT, config_name, "7scenes", "logs_0.txt"),
    ]
    for log_path in candidates:
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            for line in f:
                m = PATTERN.search(line)
                if m:
                    d = m.groupdict()
                    scene = d["scene"].strip().rstrip(",")
                    results[scene] = {
                        k: float(v) for k, v in d.items() if k != "scene"
                    }
    return results


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def generate_report():
    # Collect all results
    all_results = {}
    for cfg in CONFIG_ORDER:
        data = parse_logs_dir(cfg)
        if data:
            all_results[cfg] = data

    lines = []
    L = lines.append

    L("# 7scenes Fine 3D Reconstruction Experiment Report")
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
    L("| Dataset | 7scenes (Microsoft RGB-D 7-Scenes) |")
    L("| Split | Test (18 sequences from 7 indoor scenes) |")
    L("| Scenes | chess (2), fire (2), heads (1), office (4), pumpkin (2), redkitchen (5), stairs (2) |")
    L("| Total test sequences | 18 |")
    L("| Frames per sequence | 500-1000 (raw) |")
    L("| Keyframe sampling | kf_every=2 (every 2nd frame) |")
    L("| Max frames | 200 (capped) |")
    L("| Effective views per sequence | ~200 |")
    L("| Sensor | Kinect RGB-D, 640x480 |")
    L("| GT depth | Pseudo GT from SimpleRecon (depth.proj.png) |")
    L("| GT pose | frame-XXXXXX.pose.txt (Kinect Fusion) |")
    L("| Intrinsics | fx=fy=525, cx=320, cy=240 (SimpleRecon standard) |")
    L("| Benchmark lineage | Spann3R (NeurIPS'24), CUT3R, TTT3R |")
    L("")
    L("### 1.2 Evaluation Protocol")
    L("")
    L("1. Recurrent model processes up to **200 views sequentially** per sequence")
    L("2. Predicted 3D points undergo scale-shift alignment to GT via `Regr3D_t_ScaleShiftInv`")
    L("3. Center crop 224x224 applied to each view before point extraction")
    L("4. ICP point-to-point registration (threshold=0.1m) aligns predicted point cloud to GT")
    L("5. Normals estimated via Open3D after ICP alignment")
    L("")
    L("**Metrics:**")
    L("")
    L("| Metric | Definition | Direction | Unit |")
    L("|--------|-----------|-----------|------|")
    L("| Accuracy (Acc) | Mean L2 distance: predicted -> nearest GT | Lower = better | meters |")
    L("| Completeness (Comp) | Mean L2 distance: GT -> nearest predicted | Lower = better | meters |")
    L("| NC | Mean of NC1 (GT->pred normal) and NC2 (pred->GT normal) dot products | Higher = better | [0,1] |")
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
    L("| Framework | PyTorch + Accelerate |")
    L("")
    L("### 1.4 Configurations (14 total)")
    L("")
    L("| # | Config | update_type | Parameters | Role |")
    L("|---|--------|-------------|------------|------|")
    for i, cfg in enumerate(CONFIG_ORDER, 1):
        ut = cfg if cfg in ("cut3r", "ttt3r") else "ttt3r_random" if cfg == "constant" else "ttt3r_momentum" if cfg == "brake" else "ddd3r"
        role = "Baseline" if cfg in ("cut3r", "ttt3r") else "M1 evidence" if cfg == "constant" else "M2 baseline" if cfg == "brake" else "DDD3R variant"
        L(f"| {i} | {CONFIG_LABELS[cfg]} | {ut} | {CONFIG_PARAMS[cfg]} | {role} |")
    L("")

    L("### 1.5 DDD3R Unified Update Rule")
    L("")
    L("```")
    L("S_t = S_{t-1} + beta_t * (alpha_perp * delta_perp + alpha_parallel * delta_parallel)")
    L("```")
    L("")
    L("Where `delta_perp` and `delta_parallel` are the orthogonal and drift-aligned components of the state update delta, decomposed via EMA-tracked drift direction.")
    L("")

    L("### 1.6 Why 7scenes (not DTU)")
    L("")
    L("DDD3R addresses over-update accumulation in **long sequences**:")
    L("")
    L("| Dataset | Frames | Over-update | DDD3R benefit |")
    L("|---------|--------|-------------|---------------|")
    L("| DTU | 49 | None (too short) | None / harmful |")
    L("| Sintel | 20-50 | None | None / harmful |")
    L("| **7scenes** | **200** | **Moderate** | **Significant** |")
    L("| TUM | 90-1000 | Moderate-Severe | Significant |")
    L("| ScanNet | 90-1000 | Moderate-Severe | Significant |")
    L("")
    L("7scenes at 200 frames is long enough for over-update to accumulate, making it the appropriate 3D reconstruction benchmark for DDD3R evaluation.")
    L("")

    L("### 1.7 Reproducibility")
    L("")
    L("```bash")
    L("# Environment")
    L("conda activate ttt3r")
    L("")
    L("# Single config example (e.g., brake)")
    L("CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch \\")
    L("    --num_processes 1 --main_process_port 29590 \\")
    L("    eval/mv_recon/launch.py \\")
    L("    --weights model/cut3r_512_dpt_4_64.pth \\")
    L("    --output_dir eval_results/video_recon/7scenes_200/brake \\")
    L("    --max_frames 200 \\")
    L("    --model_update_type ttt3r_momentum")
    L("")
    L("# All 14 configs")
    L("bash eval/mv_recon/run_7scenes_allconfigs.sh")
    L("")
    L("# Generate this report")
    L("python3 generate_7scenes_report.py")
    L("```")
    L("")
    L("---")
    L("")

    # ================================================================
    # Section 2: Main Results
    # ================================================================
    L("## 2. Main Results")
    L("")
    L("### 2.1 Overall Summary (Mean over 18 Test Sequences)")
    L("")
    L("| Config | N | Acc ↓ | Comp ↓ | NC ↑ | vs CUT3R Acc | vs CUT3R Comp | vs CUT3R NC |")
    L("|--------|---|-------|--------|------|-------------|---------------|-------------|")

    baseline = {}
    summary = {}

    for cfg in CONFIG_ORDER:
        if cfg not in all_results or not all_results[cfg]:
            L(f"| {CONFIG_LABELS[cfg]} | 0 | — | — | — | — | — | — |")
            continue

        scenes = all_results[cfg]
        acc = [s["acc"] for s in scenes.values()]
        comp = [s["comp"] for s in scenes.values()]
        nc = [(s["nc1"] + s["nc2"]) / 2 for s in scenes.values()]

        m_acc, m_comp, m_nc = mean(acc), mean(comp), mean(nc)
        s_acc, s_comp, s_nc = std(acc), std(comp), std(nc)
        n = len(scenes)

        summary[cfg] = {"acc": m_acc, "comp": m_comp, "nc": m_nc, "n": n}

        if cfg == "cut3r":
            baseline = {"acc": m_acc, "comp": m_comp, "nc": m_nc}

        if baseline:
            d_acc = f"{(m_acc - baseline['acc']) / baseline['acc'] * 100:+.1f}%" if baseline["acc"] > 0 else "—"
            d_comp = f"{(m_comp - baseline['comp']) / baseline['comp'] * 100:+.1f}%" if baseline["comp"] > 0 else "—"
            d_nc = f"{(m_nc - baseline['nc']) / baseline['nc'] * 100:+.1f}%" if baseline["nc"] > 0 else "—"
        else:
            d_acc = d_comp = d_nc = "—"

        tag = f" ({n}/18)" if n < TOTAL_SEQUENCES else ""
        L(f"| {CONFIG_LABELS[cfg]}{tag} | {n} | {m_acc:.4f}±{s_acc:.4f} | {m_comp:.4f}±{s_comp:.4f} | {m_nc:.3f}±{s_nc:.3f} | {d_acc} | {d_comp} | {d_nc} |")

    L("")

    # ================================================================
    # Gamma Spectrum
    # ================================================================
    L("### 2.2 Gamma Spectrum Ablation")
    L("")
    L("gamma controls the ortho-isotropic spectrum:")
    L("- gamma -> inf: pure ortho (aggressive drift suppression)")
    L("- gamma -> 0: isotropic (no directional awareness)")
    L("")
    L("| gamma | Acc ↓ | Comp ↓ | NC ↑ | vs CUT3R Acc | Behavior |")
    L("|-------|-------|--------|------|-------------|----------|")

    gamma_map = [
        ("constant", "iso (alpha_perp=alpha_parallel)", "Isotropic baseline"),
        ("ortho", "0 (pure ortho)", "Full directional decomposition"),
        ("ddd3r_g1", "1", "Light drift-adaptive"),
        ("ddd3r_g2", "2", "Moderate drift-adaptive"),
        ("ddd3r_g3", "3", "Strong drift-adaptive"),
        ("ddd3r_g4", "4", "Stronger drift-adaptive"),
        ("ddd3r_g5", "5", "Near pure ortho"),
    ]
    for cfg_key, glabel, desc in gamma_map:
        if cfg_key in summary:
            s = summary[cfg_key]
            d_acc = f"{(s['acc'] - baseline['acc']) / baseline['acc'] * 100:+.1f}%" if baseline else "—"
            L(f"| {glabel} | {s['acc']:.4f} | {s['comp']:.4f} | {s['nc']:.3f} | {d_acc} | {desc} |")
        else:
            L(f"| {glabel} | — | — | — | — | {desc} |")
    L("")

    # ================================================================
    # Method comparison (paper table format)
    # ================================================================
    L("### 2.3 Paper Table Format (Acc / Comp / NC)")
    L("")
    L("| Config | Acc ↓ | Comp ↓ | NC ↑ |")
    L("|--------|-------|--------|------|")
    for cfg in ["cut3r", "ttt3r", "constant", "brake", "ortho"]:
        if cfg in summary:
            s = summary[cfg]
            bold = "**" if cfg in ("brake", "ortho") else ""
            L(f"| {bold}{CONFIG_LABELS[cfg]}{bold} | {bold}{s['acc']:.4f}{bold} | {bold}{s['comp']:.4f}{bold} | {bold}{s['nc']:.3f}{bold} |")
    L("")

    # ================================================================
    # Section 3: Per-Scene Breakdown by Category
    # ================================================================
    L("---")
    L("")
    L("## 3. Per-Scene Results")
    L("")

    for cfg in CONFIG_ORDER:
        if cfg not in all_results or not all_results[cfg]:
            continue
        scenes = all_results[cfg]
        L(f"<details>")
        L(f"<summary><b>{CONFIG_LABELS[cfg]}</b> ({len(scenes)}/18 sequences)</summary>")
        L(f"")
        L(f"| Scene | Sequence | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | NC ↑ |")
        L(f"|-------|----------|-------|--------|-------|-------|------|")

        for cat_name, cat_seqs in SCENE_CATEGORIES.items():
            for seq in cat_seqs:
                # Try different key formats
                key = None
                for k in [seq, seq.replace("/", "_"), seq.split("/")[-1]]:
                    if k in scenes:
                        key = k
                        break
                if key:
                    m = scenes[key]
                    nc = (m["nc1"] + m["nc2"]) / 2
                    L(f"| {cat_name} | {seq} | {m['acc']:.4f} | {m['comp']:.4f} | {m['nc1']:.3f} | {m['nc2']:.3f} | {nc:.3f} |")

        acc_vals = [s["acc"] for s in scenes.values()]
        comp_vals = [s["comp"] for s in scenes.values()]
        nc_vals = [(s["nc1"] + s["nc2"]) / 2 for s in scenes.values()]
        L(f"| | **Mean** | **{mean(acc_vals):.4f}** | **{mean(comp_vals):.4f}** | — | — | **{mean(nc_vals):.3f}** |")
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
    L("### 4.1 Long-Sequence Over-Update on 7scenes")
    L("")
    L("7scenes uses 200-frame sequences (10x longer than DTU's 49 frames),")
    L("entering the regime where over-update accumulation becomes significant:")
    L("")
    L("- **M1 validation**: If constant dampening improves over CUT3R, over-update exists at 200f")
    L("- **M2 validation**: If TTT3R ~ CUT3R, the sigmoid gate has degenerated (consistent with A1-A3 analysis)")
    L("- **M3 validation**: Comparing ortho vs constant reveals whether drift is harmful or useful refinement")
    L("")

    # Automated interpretation
    if "cut3r" in summary and "constant" in summary:
        d = (summary["constant"]["acc"] - summary["cut3r"]["acc"]) / summary["cut3r"]["acc"] * 100
        if d < -10:
            L(f"**M1 confirmed**: Constant dampening reduces Acc by {d:.1f}% vs CUT3R — over-update is significant at 200f.")
        elif d < 0:
            L(f"**M1 marginal**: Constant dampening shows {d:.1f}% Acc change — moderate over-update at 200f.")
        else:
            L(f"**M1 not observed**: Constant dampening shows {d:+.1f}% Acc change — over-update may not be dominant at 200f.")
        L("")

    if "cut3r" in summary and "ttt3r" in summary:
        d = abs(summary["ttt3r"]["acc"] - summary["cut3r"]["acc"]) / summary["cut3r"]["acc"] * 100
        if d < 5:
            L(f"**M2 confirmed**: TTT3R Acc differs by only {d:.1f}% from CUT3R — sigmoid gate effectively degenerated.")
        L("")

    if "ortho" in summary and "constant" in summary:
        ortho_better = summary["ortho"]["acc"] < summary["constant"]["acc"]
        if ortho_better:
            L("**M3 insight**: Ortho outperforms constant — drift at 200f is predominantly harmful (like TUM).")
        else:
            L("**M3 insight**: Constant outperforms ortho — drift at 200f contains useful refinement (like ScanNet).")
        L("")

    L("### 4.2 Cross-Dataset Consistency")
    L("")
    L("| Dataset | Frames | Task | Constant vs CUT3R | Brake vs CUT3R | Ortho vs CUT3R |")
    L("|---------|--------|------|-------------------|----------------|----------------|")
    L("| Sintel | ~20-50 | Pose | +5% | +14% | +13% |")

    if baseline:
        const_d = f"{(summary['constant']['acc'] - baseline['acc']) / baseline['acc'] * 100:+.1f}%" if "constant" in summary else "—"
        brake_d = f"{(summary['brake']['acc'] - baseline['acc']) / baseline['acc'] * 100:+.1f}%" if "brake" in summary else "—"
        ortho_d = f"{(summary['ortho']['acc'] - baseline['acc']) / baseline['acc'] * 100:+.1f}%" if "ortho" in summary else "—"
        L(f"| **7scenes** | **200** | **3D Recon** | **{const_d}** | **{brake_d}** | **{ortho_d}** |")

    L("| TUM 90f | 90 | Pose | -53% | -53% | -55% |")
    L("| TUM 1000f | 1000 | Pose | -60% | -62% | -66% |")
    L("| ScanNet 1000f | 1000 | Pose | -66% | -68% | -40% |")
    L("| 7scenes (prior) | 200 | 3D Recon | — | -77% (Acc) | -72% (Acc) |")
    L("")
    L("*Prior 7scenes results from CLAUDE.md (brake Acc: 0.021, ortho: 0.026 vs cut3r: 0.092)*")
    L("")

    # ================================================================
    # Section 5: Completeness
    # ================================================================
    L("---")
    L("")
    L("## 5. Experiment Completeness")
    L("")
    L("| Config | Sequences | Status |")
    L("|--------|-----------|--------|")
    for cfg in CONFIG_ORDER:
        n = len(all_results.get(cfg, {}))
        status = "COMPLETE" if n == TOTAL_SEQUENCES else f"IN PROGRESS ({n}/{TOTAL_SEQUENCES})" if n > 0 else "NOT STARTED"
        L(f"| {CONFIG_LABELS[cfg]} | {n}/{TOTAL_SEQUENCES} | {status} |")
    L("")
    total = sum(1 for c in CONFIG_ORDER if len(all_results.get(c, {})) == TOTAL_SEQUENCES)
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
    L("eval_results/video_recon/7scenes_200/")
    L("  <config>/")
    L("    7scenes/")
    L("      logs_0.txt               # per-process metrics log")
    L("      logs_all.txt             # merged log with mean metrics")
    L("      <scene>_<seq>.npy        # raw predictions")
    L("      <scene>_<seq>-mask.ply   # predicted point cloud")
    L("      <scene>_<seq>-gt.ply     # ground truth point cloud")
    L("```")
    L("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Report: {REPORT_PATH}")
    print(f"Configs found: {list(all_results.keys())}")
    print(f"Total sequences: {sum(len(v) for v in all_results.values())}")


if __name__ == "__main__":
    generate_report()
