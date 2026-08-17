"""Follow every authored SymPy symbol down to the value the program passes.

The authored mathematics is a set of SymPy equations over named symbols.
Everything below it -- ProcessGraph, planner regions, repository SSA, a
backend -- is a transformation of that, and each transformation is supposed
to PRESERVE the symbol's identity while changing its representation. This
prints that chain as data, one row per symbol, so the preservation can be
read rather than assumed.

That makes a whole class of defect visible at a glance, because the authored
symbols divide cleanly into two kinds and the two must not mix:

* a **gather** -- ``height_east``, ``tracer_center`` -- names a neighbour
  element and should arrive as an indexed read of a field;
* a **parameter** -- ``viscosity``, ``gravity`` -- names a scalar of the
  record and should arrive as an unresolved FORMAL, hoisted out of the loop.

The division is derived from the model itself (``STATE_FIELDS`` crossed with
``NEIGHBORS``), never hardcoded, so it stays correct if the equations change.

A parameter arriving as a gather is a defect and this says so: it means a
scalar was resolved to an array element, which produces a plausible number
and no error. That is how `viscosity` came to be 1.0 -- it was gathering
height -- inflating the momentum diffusion 5000-fold with nothing raised
anywhere.

    python tools/symbol_provenance.py
    python tools/symbol_provenance.py --values      # also run it, per cell
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
STEP = "symbolic_fluid_control__symbolic_fluid_step"


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


def authored_symbol_kinds() -> dict[str, str]:
    """{symbol: 'gather'|'parameter'}, derived from the model."""
    from src.compiler.symbolic_fluid_model import NEIGHBORS, STATE_FIELDS

    gathers = {
        f"{field}_{neighbor}"
        for field in STATE_FIELDS
        for neighbor in NEIGHBORS
    }
    from src.compiler.symbolic_fluid_model import (
        compile_symbolic_fluid_step,
    )
    ordered = tuple(
        compile_symbolic_fluid_step().function.metadata["argument_names"]
    )
    return {
        name: ("gather" if name in gathers else "parameter")
        for name in ordered
    }


def step_call(function: Any) -> Any:
    """The advance's call into the step, found by callee not by position."""
    for block in function.blocks.values():
        for instruction in block.instrs:
            if str(instruction.op) != "Call":
                continue
            if str(instruction.attributes.get("callee") or "") == STEP:
                return instruction
    raise SystemExit(f"no call to {STEP} in {function.name}")


def describe(function: Any, value_id: int) -> tuple[str, str]:
    """(how the advance produces this value, detail)."""
    formals = {int(a.id): a for a in function.args}
    if value_id in formals:
        accounting = dict(formals[value_id].accounting or {})
        field = accounting.get("program_abi_field")
        if field:
            return "formal", f"record field {field!r}"
        return "formal", "no program_abi accounting"
    produced = {
        int(i.res.id): i
        for block in function.blocks.values()
        for i in block.instrs
        if i.res is not None
    }
    instruction = produced.get(value_id)
    if instruction is None:
        return "absent", "neither a formal nor produced in this frame"
    if str(instruction.op) == "Load":
        source = produced.get(int(instruction.args[0].id))
        if source is not None and str(source.op) == "GetElementPtr":
            index = source.attributes.get("aggregate_index")
            return "gather", f"load of region aggregate[{index}]"
        return "gather", "load"
    if str(instruction.op) == "Const":
        return "const", repr(instruction.attributes.get("value"))
    return str(instruction.op).lower(), ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", action="store_true")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.2)
    arguments = parser.parse_args()

    lowering = newest_lowering()
    print(f"reading {lowering.parent.name}\n")
    with lowering.open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    advance = module.functions[ADVANCE]
    kinds = authored_symbol_kinds()
    passed = [int(a.id) for a in step_call(advance).args]
    ordered = list(kinds)
    if len(passed) != len(ordered):
        print(
            f"WARNING: the call passes {len(passed)} arguments but the "
            f"model declares {len(ordered)} symbols; the pairing below is "
            "positional and cannot be trusted."
        )

    observed: dict[int, float] = {}
    if arguments.values:
        from src.compiler.ssa_reference_evaluator import (
            SSAReferenceEvaluator, bind_program_abi_arguments,
        )
        from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState

        state = SymbolicFluidGridState.initial(arguments.grid, arguments.grid)
        bound, _unbound = bind_program_abi_arguments(
            advance, record=state,
            named={
                "dt": arguments.dt,
                "height_count": arguments.grid,
                "width_count": arguments.grid,
            },
            functions=module.functions,
        )
        evaluator = SSAReferenceEvaluator(module, history=tuple(passed))
        evaluator.run(ADVANCE, bound)
        for value_id in passed:
            series = evaluator.history.get(value_id) or []
            held = series[0] if series else bound.get(value_id)
            if held is not None:
                observed[value_id] = float(
                    np.asarray(held, dtype=float).reshape(-1)[0]
                )

    header = f"{'symbol':22} {'expects':10} {'arrives as':9} {'detail':34}"
    print(header + ("value" if arguments.values else ""))
    print("-" * (len(header) + (7 if arguments.values else 0)))
    wrong = []
    for symbol, value_id in zip(ordered, passed):
        expects = kinds[symbol]
        arrives, detail = describe(advance, value_id)
        suspect = expects == "parameter" and arrives != "formal"
        shown = ""
        if arguments.values and value_id in observed:
            shown = f"{observed[value_id]:.6g}"
        mark = "  <-- a scalar arriving as data" if suspect else ""
        print(
            f"{symbol:22} {expects:10} {arrives:9} {detail:34}{shown}{mark}"
        )
        if suspect:
            wrong.append(symbol)

    if wrong:
        print(
            f"\n{len(wrong)} parameter(s) do not arrive as formals: "
            f"{', '.join(wrong)}"
        )
        print(
            "  A parameter must reach the step as an unresolved formal. One\n"
            "  resolved to an array read is silently wrong: it produces a\n"
            "  plausible number, so nothing downstream can notice."
        )
        return 1
    print("\nevery authored symbol arrives in the form its kind requires.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
