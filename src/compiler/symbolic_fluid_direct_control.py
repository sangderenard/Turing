"""Direct ProcessGraph -> repository-SSA build with external memory telemetry.

The worker uses ``lower_ast_source_to_ssa`` and therefore never captures a
FusedProgram.  The parent observes private bytes from outside the compiler
process, so a GIL-heavy graph pass cannot hide growth. Observation is the
default; only an explicit positive CLI value enables the emergency clamp.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _process_memory(pid: int) -> tuple[int, int]:
    if os.name != "nt":
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024
        return rss, rss
    process = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not process:
        raise ProcessLookupError(pid)
    try:
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error())
        return int(counters.WorkingSetSize), int(counters.PrivateUsage)
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def _worker(output: Path) -> int:
    from ..common.dt_system.dt_controller import run_superstep
    from ..common.dt_system.dt_scaler import Metrics
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE
    from .symbolic_fluid_model import compile_symbolic_fluid_step

    output.mkdir(parents=True, exist_ok=True)
    symbolic = compile_symbolic_fluid_step()

    def progress(message: str) -> None:
        # lower_ast_source_to_ssa intentionally redirects stdout while the
        # graph builder runs. Diagnostics use stderr so the parent log retains
        # the exact dependency and phase history.
        print(message, file=sys.stderr, flush=True)

    module, outputs, exports = lower_ast_source_to_ssa(
        SYMBOLIC_FLUID_DT_SOURCE,
        "symbolic_fluid_frame",
        python_bindings={
            "Metrics": Metrics,
            "run_superstep": run_superstep,
        },
        linked_process_graphs={
            "symbolic_fluid_step": symbolic.process_graph,
        },
        name="symbolic_fluid_control",
        runtime_closure_only=True,
        extraction_contract=(
            Path(__file__).resolve().parents[2]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        progress=progress,
    )
    with (output / "control_repository_ssa.pkl").open("wb") as stream:
        pickle.dump((module, outputs, exports), stream, protocol=5)
    lines = []
    for function_name, function in module.functions.items():
        lines.append(f"function {function_name}")
        for block_name, block in function.blocks.items():
            lines.append(f"  block {block_name}")
            lines.extend(f"    {instruction}" for instruction in block.instrs)
    (output / "control_repository_ssa.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    (output / "control_summary.json").write_text(
        json.dumps({
            "schema": "turing.symbolic-fluid-direct-control.v1",
            "fused_program_used": False,
            "functions": list(module.functions),
            "exports": list(exports),
            "outputs": {
                name: [int(value.id) for value in values]
                for name, values in outputs.items()
            },
        }, indent=2),
        encoding="utf-8",
    )
    return 0


def bounded_compile(
    output: str | Path,
    *,
    max_memory_gb: float = 0.0,
) -> dict[str, Any]:
    """Run the direct compiler with external memory telemetry.

    ``max_memory_gb=0`` is observational only. A positive value is an explicit
    emergency operator choice, not a normal compilation limit.
    """

    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "control_compile.log"
    command = [
        sys.executable,
        "-m",
        "src.compiler.symbolic_fluid_direct_control",
        "--worker",
        "--output",
        str(destination),
    ]
    ceiling = int(float(max_memory_gb) * 1024 ** 3)
    started = time.monotonic()
    peak_working = 0
    peak_private = 0
    killed = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            try:
                working, private = _process_memory(process.pid)
            except ProcessLookupError:
                break
            peak_working = max(peak_working, working)
            peak_private = max(peak_private, private)
            if ceiling > 0 and private > ceiling:
                killed = True
                process.kill()
                break
            time.sleep(0.5)
        return_code = process.wait()
    report = {
        "schema": "turing.bounded-direct-control-compile.v1",
        "return_code": int(return_code),
        "emergency_clamp": killed,
        "max_memory_gb": float(max_memory_gb),
        "peak_working_gb": peak_working / 1024 ** 3,
        "peak_private_gb": peak_private / 1024 ** 3,
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log_path),
        "completed": (
            return_code == 0
            and (destination / "control_repository_ssa.pkl").is_file()
        ),
    }
    (destination / "bounded_compile_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=0.0,
        help="explicit emergency private-byte ceiling; zero only records telemetry",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args.output)
    report = bounded_compile(args.output, max_memory_gb=args.max_memory_gb)
    print(json.dumps(report, indent=2))
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
