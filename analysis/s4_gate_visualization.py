"""
S4: Gate activation visualization for joint gating variants.

This file is copied into the local branch so the gate visualization path
is available even when the remote branch results are not yet synced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from add_ckpt_path import add_path_to_dust3r
from analysis.spectral_analysis import load_img_paths, build_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--scannet_root", default="data/long_scannet_s3")
    parser.add_argument("--output_dir", default="analysis_results/s4_gate_viz")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--scenes", default="auto")
    return parser.parse_args()


def discover_scenes(scannet_root: str) -> list[tuple[str, str]]:
    scenes = []
    if not os.path.isdir(scannet_root):
        return scenes
    for d in sorted(os.listdir(scannet_root)):
        for cname in ("color_1000", "color"):
            color_dir = os.path.join(scannet_root, d, cname)
            if os.path.isdir(color_dir):
                scenes.append((d, color_dir))
                break
    return scenes


def select_scenes(scenes: list[tuple[str, str]], spec: str) -> list[tuple[str, str]]:
    if spec != "auto":
        wanted = [x.strip() for x in spec.split(",") if x.strip()]
        return [item for item in scenes if item[0] in wanted]
    if len(scenes) <= 3:
        return scenes
    idx = [0, len(scenes) // 2, len(scenes) - 1]
    return [scenes[i] for i in idx]


def plot_scene(records: list[dict[str, float]], scene_name: str, out_dir: str) -> None:
    frames = [r["frame"] for r in records]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Gate activations: {scene_name}", fontsize=14)

    specs = [
        ("ttt3r_mean", "ttt3r_mask", "tab:blue"),
        ("alpha_mean", "alpha", "tab:red"),
        ("g_geo", "g_geo", "tab:green"),
        ("effective_mean", "effective", "black"),
    ]
    for ax, (key, title, color) in zip(axes, specs):
        vals = [r[key] for r in records]
        ax.plot(frames, vals, color=color, linewidth=1.4)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{scene_name}_gates.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    scenes = select_scenes(discover_scenes(args.scannet_root), args.scenes)
    if not scenes:
        raise RuntimeError(f"No scenes found under {args.scannet_root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()
    model.config.model_update_type = "ttt3r_joint"
    model.config.spectral_temperature = 1.0
    model.config.geo_gate_tau = 2.0
    model.config.geo_gate_freq_cutoff = 4

    all_records: dict[str, list[dict[str, float]]] = {}
    for scene_name, color_dir in scenes:
        img_paths = load_img_paths(color_dir, args.frame_interval, args.max_frames)
        if len(img_paths) < 4:
            continue
        views = build_views(img_paths, args.size)
        model._gate_log = []
        with torch.no_grad():
            model.forward_recurrent_lighter(views, device=device)
        gate_log = getattr(model, "_gate_log", [])
        if not gate_log:
            continue
        records = []
        for entry in gate_log:
            ttt3r = entry["ttt3r_mask"].squeeze(-1).squeeze(0)
            alpha = entry["alpha"].squeeze(-1).squeeze(0)
            eff = entry["effective"].squeeze(-1).squeeze(0)
            g_geo = entry["g_geo"]
            if isinstance(g_geo, torch.Tensor):
                g_geo = g_geo.item() if g_geo.numel() == 1 else g_geo.mean().item()
            records.append(
                {
                    "frame": int(entry["frame"]),
                    "ttt3r_mean": float(ttt3r.mean().item()),
                    "alpha_mean": float(alpha.mean().item()),
                    "g_geo": float(g_geo),
                    "effective_mean": float(eff.mean().item()),
                }
            )
        all_records[scene_name] = records
        plot_scene(records, scene_name, args.output_dir)

    with open(os.path.join(args.output_dir, "gate_log_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"[OK] Wrote {os.path.join(args.output_dir, 'gate_log_summary.json')}")


if __name__ == "__main__":
    main()
