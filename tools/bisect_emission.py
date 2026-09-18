"""Find the FIRST instruction whose emitted result leaves the SSA's meaning.

Q5c routes a defect to a layer. This narrows it inside that layer, for the
case where the layer is emission: it runs the same function twice, once as
repository SSA through the calibrated reference evaluator and once as the
compiled artifact, on identical inputs, and compares EVERY value both can
see -- then reports the earliest disagreement in program order.

The earliest one is the whole point. Once a value is wrong, everything
downstream of it is wrong too, and a report sorted by magnitude puts the
loudest consequence first and the cause somewhere in the middle. Program
order puts the cause first.

Two properties make the comparison trustworthy, and neither should be
weakened:

* the evaluator is calibrated (`tests/test_ssa_reference_evaluator.py`);
  if that is failing, nothing here means anything;
* the artifact is read through `watch=`, which exposes an internal value
  by copying one the program already computed. It adds no arithmetic and
  reorders nothing, so the run being measured is the run that would have
  happened anyway.

    python tools/bisect_emission.py
    python tools/bisect_emission.py --grid 8 --dt 0.05
    python tools/bisect_emission.py --limit 40
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

ADVANCE = "symbolic_fluid_control__symbolic_fluid_advance"
TOLERANCE = 1e-9


def newest_lowering() -> Path:
    candidates = sorted(
        (ROOT / "build").glob("*/control_repository_ssa.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "no lowered SSA under build/; run "
            "`python -m src.compiler.symbolic_fluid_direct_control "
            "--output build/<name>` first"
        )
    return candidates[0]


def program_order(function: Any) -> list[tuple[int, str, str]]:
    """(value id, opcode, block) for each result, in scheduled order."""
    ordered: list[tuple[int, str, str]] = []
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            if instruction.res is not None:
                ordered.append(
                    (int(instruction.res.id), str(instruction.op), block_name)
                )
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=25)
    arguments = parser.parse_args()

    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
    )
    from src.compiler.ssa_reference_evaluator import (
        SSAReferenceEvaluator, bind_program_abi_arguments,
    )
    from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState
    from src.compiler.symbolic_fluid_native_runtime import (
        NativeSymbolicFluidAdvance,
    )

    lowering = newest_lowering()
    print(f"reading {lowering.parent.name} ...")
    with lowering.open("rb") as stream:
        module, outputs, _exports = pickle.load(stream)
    function = module.functions[ADVANCE]
    names = {
        int(value): str(label)
        for label, value in (function.metadata.get("value_names") or ())
    }

    # -- the SSA's own answer ------------------------------------------
    state = SymbolicFluidGridState.initial(arguments.grid, arguments.grid)
    bound, _unbound = bind_program_abi_arguments(
        function,
        record=state,
        named={
            "dt": arguments.dt,
            "height_count": arguments.grid,
            "width_count": arguments.grid,
        },
        functions=module.functions,
    )
    evaluated = SSAReferenceEvaluator(module).run(ADVANCE, bound).values
    print(f"evaluated {len(evaluated)} SSA values")

    # -- the artifact's answer, for the same values --------------------
    candidates = [value_id for value_id, _op, _block in program_order(function)]
    artifact = emit_ssa_function_to_llvm(module, ADVANCE, watch=candidates)
    if not artifact.complete:
        print("emission incomplete:", artifact.shortfalls[:3])
        return 2
    compile_artifact(artifact, directory=ROOT / "build" / "bisect")
    native = NativeSymbolicFluidAdvance(
        artifact, function, dict(module.functions), outputs.get(ADVANCE),
    )
    native_state = SymbolicFluidGridState.initial(
        arguments.grid, arguments.grid,
    )
    native(native_state, arguments.dt)
    watched = set(artifact.watched)
    print(f"watched {len(watched)} of them in the artifact")
    if artifact.watch_shortfalls:
        print(f"  ({len(artifact.watch_shortfalls)} could not be watched)")

    # -- compare, in program order -------------------------------------
    findings = []
    compared = 0
    for value_id, opcode, block_name in program_order(function):
        if value_id not in watched or value_id not in evaluated:
            continue
        if not native.observable(value_id):
            continue
        held = evaluated[value_id]
        # Addresses and aggregates are not numbers. Comparing them would
        # mean comparing an evaluator bookkeeping object against whatever
        # bytes the artifact's slot happens to hold, which is noise rather
        # than evidence.
        if isinstance(held, (list, tuple)) or type(held).__name__ == "_Pointer":
            continue
        try:
            left = np.asarray(held, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        right = np.asarray(
            native.execution.buffers[value_id], dtype=float,
        ).reshape(-1)
        if not left.size or not right.size:
            continue
        compared += 1
        gap = float(abs(left[0] - right[0]))
        if gap > TOLERANCE:
            findings.append((value_id, opcode, block_name, left[0], right[0], gap))

    print(f"compared {compared} values both sides can see\n")
    if not findings:
        print("no disagreement among the values both sides expose.")
        print("  The defect is in something neither exposes -- widen the")
        print("  comparison rather than concluding emission is faithful.")
        return 0

    print(f"FIRST DIVERGENCE IN PROGRAM ORDER (of {len(findings)}):")
    value_id, opcode, block_name, left, right, gap = findings[0]
    label = names.get(value_id)
    print(f"  value {value_id}" + (f" ({label})" if label else ""))
    print(f"  produced by {opcode} in block {block_name}")
    print(f"  ssa      = {left!r}")
    print(f"  artifact = {right!r}")
    print(f"  gap      = {gap:.6e}")
    print("\n  Everything below is downstream of it and is expected to")
    print("  disagree; fix this one first and re-run.\n")

    print("the rest, in program order:")
    for value_id, opcode, block_name, left, right, gap in findings[
        1:arguments.limit
    ]:
        label = names.get(value_id)
        tag = f" ({label})" if label else ""
        print(
            f"  {value_id:5}{tag:22} {opcode:14} {block_name:14} "
            f"gap={gap:.3e}"
        )
    if len(findings) > arguments.limit:
        print(f"  ... {len(findings) - arguments.limit} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
