"""Lift one dependency-validated binary-machine tape lineage to trace SSA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.compiler.binary_machine_program import BinaryMachineProgram
from src.compiler.machine_system_tape import MachineSystemTape
from src.compiler.machine_trace_ssa import lift_tape_lineage_to_trace_ssa


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--slice-resource", action="append", default=[])
    parser.add_argument(
        "--ignore-control", action="store_true",
        help="make a data/effect slice without retaining the executed control chain",
    )
    options = parser.parse_args(argv)

    tape = MachineSystemTape.read(options.tape)
    machine = BinaryMachineProgram.load_system_tape(
        tape, maximum_file_size=128 * 1024 * 1024,
    )
    instructions = machine.machine.cores[options.core].executor.instructions
    trace = lift_tape_lineage_to_trace_ssa(
        tape,
        core=options.core,
        sequence=options.sequence,
        instruction_lookup=instructions.get,
    )
    result = trace
    summary = None
    if options.slice_resource:
        result = trace.backward_slice(
            options.slice_resource,
            include_control=not options.ignore_control,
        )
        summary = dict(trace.reduction_summary(result))
    payload = result.to_mapping()
    if summary is not None:
        payload["reduction_summary"] = summary
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"operations={len(result.operations)} source={len(trace.operations)} "
        f"output={options.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
