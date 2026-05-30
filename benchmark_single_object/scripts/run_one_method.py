from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
BENCHMARK_ROOT = THIS_FILE.parent.parent
REPO_ROOT = BENCHMARK_ROOT.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from utils.io_utils import ensure_dir, write_json
from utils.metrics_utils import compute_camera_consistency
from utils.pointcloud_utils import compute_conf_stats, count_output_points, count_processed_frames
from utils.sampling_utils import parse_length_from_seq_name

_DEMO_FLAGS_CACHE: dict[tuple[str, str], set[str]] = {}


class GpuMemoryMonitor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread = None
        self.peak_mb = float("nan")
        self.backend = "disabled"

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        peak = 0.0
        backend = "none"
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            backend = "pynvml"
            while not self._stop.is_set():
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mb = mem_info.used / (1024.0 * 1024.0)
                peak = max(peak, used_mb)
                time.sleep(0.2)
            pynvml.nvmlShutdown()
        except Exception:
            # Fallback: no pynvml available, poll nvidia-smi.
            backend = "nvidia-smi"
            while not self._stop.is_set():
                try:
                    proc = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=2,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        # Take GPU0 by default.
                        first = proc.stdout.strip().splitlines()[0].strip()
                        used_mb = float(first)
                        peak = max(peak, used_mb)
                except Exception:
                    pass
                time.sleep(0.25)
        self.peak_mb = peak if peak > 0 else float("nan")
        self.backend = backend if not (backend == "nvidia-smi" and peak == 0) else "unavailable"

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one method on one sampled sequence.")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--model_update_type", type=str, default=None)
    parser.add_argument("--alpha_drift", type=float, default=None)
    parser.add_argument("--seq_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--vis_threshold", type=float, default=6.0)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=100)
    parser.add_argument("--downsample_factor", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout_sec", type=int, default=7200)
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--skip_if_done", action="store_true")
    return parser.parse_args()


def get_demo_supported_flags(python_exe: str, repo_root: Path) -> set[str]:
    key = (str(Path(python_exe).resolve()), str(repo_root.resolve()))
    if key in _DEMO_FLAGS_CACHE:
        return _DEMO_FLAGS_CACHE[key]

    try:
        proc = subprocess.run(
            [str(Path(python_exe).resolve()), "demo.py", "-h"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
        text = proc.stdout or ""
    except Exception:
        text = ""

    flags = {
        "--model_path",
        "--seq_path",
        "--device",
        "--size",
        "--vis_threshold",
        "--output_dir",
        "--port",
        "--model_update_type",
        "--frame_interval",
        "--reset_interval",
        "--downsample_factor",
    }
    if "--alpha_drift" in text:
        flags.add("--alpha_drift")
    if "--seed" in text:
        flags.add("--seed")

    _DEMO_FLAGS_CACHE[key] = flags
    return flags


def run_one_experiment(args: argparse.Namespace) -> dict[str, Any]:
    seq_path = Path(args.seq_path).resolve()
    repo_root = Path(getattr(args, "repo_root", None) or REPO_ROOT).resolve()
    object_id = seq_path.parent.name
    sequence_id = seq_path.name
    seq_length = parse_length_from_seq_name(sequence_id)

    run_output_dir = Path(args.output_root).resolve() / args.method / object_id / sequence_id
    ensure_dir(run_output_dir)
    logs_dir = ensure_dir(Path(args.output_root).resolve() / "logs")
    log_path = logs_dir / f"{args.method}__{object_id}__{sequence_id}.log"
    result_json = run_output_dir / "run_metrics.json"

    if args.skip_if_done and result_json.exists():
        from utils.io_utils import read_json

        payload = read_json(result_json)
        payload["skipped"] = True
        return payload

    model_update_type = args.model_update_type or args.method
    command = [
        str(Path(args.python_exe).resolve()),
        "demo.py",
        "--model_path",
        args.model_path,
        "--seq_path",
        str(seq_path),
        "--device",
        args.device,
        "--size",
        str(args.size),
        "--vis_threshold",
        str(args.vis_threshold),
        "--output_dir",
        str(run_output_dir),
        "--port",
        "8080",
        "--model_update_type",
        model_update_type,
        "--frame_interval",
        str(args.frame_interval),
        "--reset_interval",
        str(args.reset_interval),
        "--downsample_factor",
        str(args.downsample_factor),
    ]
    supported_flags = get_demo_supported_flags(args.python_exe, repo_root)
    if "--seed" in supported_flags:
        command.extend(["--seed", str(args.seed)])
    if args.alpha_drift is not None and "--alpha_drift" in supported_flags:
        command.extend(["--alpha_drift", str(args.alpha_drift)])

    start = time.time()
    timed_out = False
    launch_seen = False

    mem_monitor = GpuMemoryMonitor(enabled=(args.device.lower() == "cuda"))
    mem_monitor.start()

    with log_path.open("w", encoding="utf-8", buffering=1) as log_f:
        log_f.write(f"[CMD] {' '.join(command)}\n")
        proc = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            while True:
                if proc.stdout is None:
                    break
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    if (time.time() - start) > args.timeout_sec:
                        timed_out = True
                        proc.kill()
                        break
                    continue

                log_f.write(line)
                if "Launching point cloud viewer..." in line:
                    launch_seen = True
                    proc.terminate()
                    break

            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            mem_monitor.stop()

    elapsed = time.time() - start

    frames = count_processed_frames(run_output_dir)
    points = count_output_points(run_output_dir, conf_threshold=1.0)
    conf_stats = compute_conf_stats(run_output_dir)
    consistency = compute_camera_consistency(run_output_dir / "camera")

    payload: dict[str, Any] = {
        "method": args.method,
        "model_update_type": model_update_type,
        "alpha_drift": args.alpha_drift,
        "object_id": object_id,
        "sequence_id": sequence_id,
        "seq_path": str(seq_path),
        "seq_length": seq_length if seq_length is not None else -1,
        "model_path": args.model_path,
        "device": args.device,
        "size": args.size,
        "frame_interval": args.frame_interval,
        "reset_interval": args.reset_interval,
        "downsample_factor": args.downsample_factor,
        "seed": args.seed,
        "runtime_sec": elapsed,
        "per_frame_sec": (elapsed / frames) if frames > 0 else float("nan"),
        "peak_vram_mb": mem_monitor.peak_mb,
        "peak_vram_backend": mem_monitor.backend,
        "processed_frames": frames,
        "output_point_count": points,
        "launch_seen": launch_seen,
        "timed_out": timed_out,
        "output_dir": str(run_output_dir),
        "log_path": str(log_path),
    }
    payload.update(conf_stats)
    payload.update(consistency)

    write_json(result_json, payload)
    return payload


def main() -> None:
    args = parse_args()
    result = run_one_experiment(args)
    print("[DONE] run_one_method")
    for k in [
        "method",
        "object_id",
        "sequence_id",
        "runtime_sec",
        "per_frame_sec",
        "peak_vram_mb",
        "output_point_count",
        "basic_consistency_score",
    ]:
        print(f"{k}: {result.get(k)}")


if __name__ == "__main__":
    main()
