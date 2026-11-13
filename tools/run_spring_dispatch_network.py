"""Launch the native Nodus four-worker trial from a Turing shader contract.

This only loads and verifies the built product; scheduling and tensor movement
are entirely in the native tool rows and FIFO trial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def run(manifest_path: Path, native_exe: Path, llvm_dll: Path, rounds: int,
        mode: str = "direct", cpu_workers: int = 1,
        ray_spv: Path | None = None) -> int:
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = record["resident_work_contract"]
    if record["schema"] != "turing.spring-network-shader.v1" or \
       contract["schema"] != "nodus.shader-work.v1" or \
       contract["resident_state_feedback"] is not True:
        raise ValueError("resident shader work contract is incompatible")
    module = manifest_path.parent / record["resident_spirv"]
    observed = hashlib.sha256(module.read_bytes()).hexdigest()
    if observed != record["sha256"][module.name]:
        raise ValueError("resident SPIR-V differs from its published contract")
    args = [str(native_exe), str(module), str(llvm_dll), str(rounds),
            str(record["local_size_x"]), str(contract["steps_per_loadout"]),
            str(contract["target_completion_ns"]), mode, str(cpu_workers)]
    if mode == "canvas-mixed-ray":
        if ray_spv is None:
            raise ValueError("mixed-ray mode requires --ray-spv")
        args.append(str(ray_spv))
    return subprocess.run(args, check=False).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("native_exe", type=Path)
    parser.add_argument("llvm_dll", type=Path)
    parser.add_argument("rounds", type=int)
    parser.add_argument("--mode", choices=("direct", "canvas-warm", "canvas-cold",
                                           "canvas-deadline",
                                           "canvas-overload", "canvas-mixed-ray",
                                           "canvas-mixed-light",
                                           "canvas-cpu-only", "canvas-gpu0",
                                           "canvas-gpu1", "canvas-gpu2"),
                        default="direct")
    parser.add_argument("--cpu-workers", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--ray-spv", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, args.native_exe, args.llvm_dll,
                         args.rounds, args.mode, args.cpu_workers,
                         args.ray_spv))
