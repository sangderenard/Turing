"""Sustain the native spring row at its full 8×256-node capacity.

Runs the existing Nodus row host in one or four concurrent processes. The
host's own tick clock, complete UDP publication and two backend implementations
remain in use. Build products and all profile files stay in the supplied user
directory. This measures host wall time and completed row ticks, not hardware
occupancy or a shared multi-device dispatcher.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time


NETWORKS, SIDE, NODES, FIELDS = 8, 16, 256, 38
METRICS = re.compile(
    r"spring_host_steps=(\d+) mean_us=([\d.]+) median_us=([\d.]+) "
    r"p95_us=([\d.]+) max_us=([\d.]+)")


def configuration(path: Path) -> None:
    data = [0.0] * (NETWORKS * NODES * FIELDS)
    for network in range(NETWORKS):
        data[(network * NODES) * FIELDS + 37] = 100 + network
        for y in range(SIDE):
            for x in range(SIDE):
                node = y * SIDE + x
                at = (network * NODES + node) * FIELDS
                position = (x * .1, (SIDE - y) * .1, 0.0)
                for axis in range(3):
                    data[at + axis] = position[axis]
                    data[at + 8 + axis] = position[axis]
                data[at + 6] = 1
                data[at + 7] = float(y == 0)
                data[at + 11] = 1
                data[at + 12] = 100
                data[at + 13] = .35
                data[at + 15] = -9.8
                for edge, nx, ny in ((0, x-1, y), (1, x+1, y),
                                     (2, x, y-1), (3, x, y+1)):
                    if not (0 <= nx < SIDE and 0 <= ny < SIDE):
                        continue
                    other = ny * SIDE + nx
                    data[at + 17 + edge] = other
                    data[at + 33 + edge] = 1
                    data[at + 21 + 3*edge] = (nx-x)*.1
                    data[at + 22 + 3*edge] = (y-ny)*.1
    path.write_bytes(struct.pack("<" + "d" * len(data), *data))


def run_case(name: str, backends: list[str], *, host: Path, plugin: Path,
             core: Path, output: Path, duration: float, substeps: int = 1,
             target_ms: float = 0, tick_us: int = 0) -> dict:
    case_dir = output / name
    case_dir.mkdir(parents=True, exist_ok=True)
    config = case_dir / "full_8x256.bin"
    configuration(config)
    workers = []
    started = time.monotonic()
    for index, backend in enumerate(backends):
        env = os.environ.copy()
        env["PATH"] = str(core) + os.pathsep + env.get("PATH", "")
        env["NODUS_FABRIC_BACKEND"] = backend
        env["NODUS_FABRIC_TICK_US"] = str(tick_us)
        log = open(case_dir / f"worker_{index}.stderr.log", "wb")
        process = subprocess.Popen(
            [str(host), str(plugin), str(config), "127.0.0.1", str(48800+index)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        process.stdin.write(f"substeps {substeps}\ntarget {target_ms}\n".encode())
        process.stdin.flush()
        workers.append((process, log, backend))
    time.sleep(duration)
    for process, _log, _backend in workers:
        if process.poll() is None:
            process.stdin.write(b"quit\n")
            process.stdin.flush()
    rows = []
    for process, log, backend in workers:
        try:
            stdout, _ = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            stdout, _ = process.communicate(timeout=5)
        log.close()
        match = METRICS.search(stdout.decode(errors="replace"))
        rows.append({"backend": backend, "exit_code": process.returncode,
                     "metrics": None if match is None else {
                         key: (int(value) if key == "steps" else float(value))
                         for key, value in zip(
                             ("steps", "mean_us", "median_us", "p95_us", "max_us"),
                             match.groups())},
                     "stderr": str(case_dir / f"worker_{len(rows)}.stderr.log")})
    elapsed = time.monotonic() - started
    result = {"case": name, "duration_s": elapsed, "active_networks_per_row": 8,
              "nodes_per_network": 256, "substeps": substeps, "tick_us": tick_us,
              "target_ms": target_ms, "workers": rows,
              "completed_row_ticks_per_s": sum(
                  row["metrics"]["steps"] for row in rows if row["metrics"]
              ) / elapsed}
    (case_dir / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=4)
    parser.add_argument("--tick-us", type=int, default=0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--target-ms", type=float, default=0)
    args = parser.parse_args()
    results = [run_case(name, backends, host=args.host, plugin=args.plugin,
                        core=args.core, output=args.output, duration=args.duration,
                        tick_us=args.tick_us, substeps=args.substeps,
                        target_ms=args.target_ms)
               for name, backends in (
                   ("one_cpu", ["cpu"]), ("one_vulkan", ["vulkan"]),
                   ("four_cpu", ["cpu"]*4), ("four_vulkan", ["vulkan"]*4),
                   ("two_cpu_two_vulkan", ["cpu", "cpu", "vulkan", "vulkan"]),
               )]
    print(json.dumps(results, indent=2))
