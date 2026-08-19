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
    from .work_contract import active_contract

    # The identity policy is baked into the lowering itself; record it beside
    # the pickle so a cache produced under one contract is never served under
    # another. Mtime staleness cannot see an environment change.
    (output / "control_repository_ssa.contract.json").write_text(
        json.dumps(
            {"inexact_identities": active_contract().inexact_identities}
        ),
        encoding="utf-8",
    )
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
    # A zero exit and a file on disk is not evidence the program was built.
    #
    # A change to node identity once left this reporting "completed": true
    # for a module that had fallen from 45 functions to 2 -- every callsite
    # shell silently unattributed, the whole control program missing, the
    # pickle present and well formed. Six builds were spent before anyone
    # looked inside it. So the report now says what is in there, and the
    # named entry points this program is FOR have to be among them.
    report.update(_structure_report(destination))
    if report.get("missing_entry_points"):
        report["completed"] = False
    return report


#: The functions this compile exists to produce. Absent any one of them the
#: build did not build the program, whatever the exit code says.
REQUIRED_ENTRY_POINTS = (
    "symbolic_fluid_control__symbolic_fluid_advance",
    "symbolic_fluid_control__symbolic_fluid_step",
    "symbolic_fluid_control__symbolic_fluid_frame",
)


def _structure_report(destination: Path) -> dict:
    """What the lowering actually contains, for the report to carry."""
    lowering = destination / "control_repository_ssa.pkl"
    if not lowering.is_file():
        return {"function_count": 0, "missing_entry_points": list(
            REQUIRED_ENTRY_POINTS
        )}
    try:
        with lowering.open("rb") as stream:
            module, _outputs, _exports = pickle.load(stream)
        names = set(getattr(module, "functions", {}) or {})
    except Exception as error:  # a pickle that will not load is a failure
        return {
            "function_count": 0,
            "structure_error": f"{type(error).__name__}: {error}",
            "missing_entry_points": list(REQUIRED_ENTRY_POINTS),
        }
    return {
        "function_count": len(names),
        "missing_entry_points": [
            name for name in REQUIRED_ENTRY_POINTS if name not in names
        ],
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
