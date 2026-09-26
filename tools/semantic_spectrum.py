"""View the final SSA as semantic groups on a pure-math to pure-action axis.

Everything in this tree enters somewhere on a spectrum. At one end is pure
mathematics -- a SymPy expression, referentially transparent, no notion of
when or where. At the other is pure action -- a store to a caller's array, a
branch, a call across a boundary. Compilation does not move a program from
one end to the other; it *decomposes* it, so that authored mathematics
becomes math instructions plus the addressing and control needed to place
them in time and space.

This prints that decomposition three ways:

1. **Spectrum** -- every function's instruction mix, ordered from the most
   mathematical to the most active. A planner region full of arithmetic and
   a traversal full of branches are different KINDS of thing, and the axis
   makes that visible rather than a matter of impression.

2. **Symbolic attribution** -- for each authored SymPy equation, the SSA it
   actually became: how many values, and how many are UNIQUE to it versus
   shared with sibling equations. The unique count is the equation's own
   footprint; the shared pool is common subexpression, which is exactly
   what a Rusanov flux with a repeated wave-speed term should produce. An
   equation whose unique footprint collapses to nothing has been folded
   away, and that is worth knowing before a backend disagrees about it.

3. **Provenance coverage** -- how much of the final form can still be
   traced back to an authored name. This is the honest half: it reports
   what identity SURVIVED, and by subtraction what was lost. A value with
   no route back to a symbol is not a bug on its own -- scheduling
   necessarily invents addresses and indices -- but a NAMED authored
   quantity that cannot be found is.

The op classification is derived from the backend's own likeness tables,
never a private list, so it cannot drift from what the compiler means. An
op the tables do not cover is reported as unclassified rather than folded
into a default bucket -- a silent default here would quietly describe the
program as more mathematical than it is.

    python tools/semantic_spectrum.py
    python tools/semantic_spectrum.py --function symbolic_fluid_control__symbolic_fluid_step
"""
from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEP = "symbolic_fluid_control__symbolic_fluid_step"
REGION = "symbolic_fluid_control__symbolic_fluid_step__planned_region_0"

#: Classes on the axis, ordered from pure math to pure action. `math` is
#: filled from the backend's likeness tables at import; the rest name
#: structural roles that no likeness table covers.
CONTROL = {"Br", "br", "CondBr", "condbr", "Phi", "phi", "Ret", "ret",
           "Return", "return", "Switch", "switch"}
MEMORY = {"Load", "load", "Store", "store"}
ADDRESS = {"GetElementPtr", "getelementptr", "extent"}
LITERAL = {"Const", "const", "StaticRef"}
BOUNDARY = {"Call", "call"}

ORDER = ("math", "literal", "address", "memory", "control", "boundary")

# There is deliberately no weighting of these classes into a single
# "spectrum position", and there must not be one. Assigning math 0.0 and a
# call 1.0 and averaging produces a number that looks measured and is
# invented -- it would order functions by a scale nobody defined, and the
# ordering would then be quoted as a finding. The only division here is
# the one the compiler itself makes: which ops its likeness tables cover.
# Everything printed below is an exact count or an exact set operation.
# Ops no table covers are named, never bucketed and never guessed at.


def math_ops() -> frozenset[str]:
    """Every op the backend treats as scalar arithmetic, from its tables."""
    from src.compiler.ssa_llvm_backend import _BINARY, _UNARY

    names = set(_BINARY) | set(_UNARY)
    return frozenset(names | {name.casefold() for name in names})


def classify(op: str, known_math: frozenset[str]) -> str | None:
    if op in known_math:
        return "math"
    if op in CONTROL:
        return "control"
    if op in MEMORY:
        return "memory"
    if op in ADDRESS:
        return "address"
    if op in LITERAL:
        return "literal"
    if op in BOUNDARY:
        return "boundary"
    return None


def newest_lowering() -> Path:
    candidates = sorted(
        (ROOT / "build").glob("*/control_repository_ssa.pkl"),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if not candidates:
        raise SystemExit("no lowered SSA under build/")
    return candidates[0]


def cone(defs: dict[int, Any], root: int) -> set[int]:
    """Every value the root transitively depends on, root included."""
    seen: set[int] = set()
    pending = [root]
    while pending:
        value_id = pending.pop()
        if value_id in seen or value_id not in defs:
            continue
        seen.add(value_id)
        pending.extend(int(a.id) for a in defs[value_id].args)
    return seen


def spectrum(module: Any, known_math: frozenset[str]) -> tuple[list, set]:
    rows = []
    unclassified: set[str] = set()
    for name, function in module.functions.items():
        counts: collections.Counter = collections.Counter()
        uncounted = 0
        for block in function.blocks.values():
            for instruction in block.instrs:
                op = str(instruction.op)
                group = classify(op, known_math)
                if group is None:
                    unclassified.add(op)
                    uncounted += 1
                    continue
                counts[group] += 1
        # Unclassified instructions stay in the denominator. Excluding them
        # would shrink n and raise every proportion, describing the program
        # as more arithmetic than it is.
        total = sum(counts.values()) + uncounted
        if not total:
            continue
        rows.append((name, counts, total, uncounted))
    # Ordered by the exact count of instructions the likeness tables do NOT
    # cover -- a fact, not a score. Functions that are entirely arithmetic
    # sort first because they contain zero of them.
    rows.sort(key=lambda row: (row[2] - row[1]["math"], row[0]))
    return rows, unclassified


def attribution(module: Any, known_math: frozenset[str]) -> None:
    step = module.functions.get(STEP)
    region = module.functions.get(REGION)
    if step is None or region is None:
        print("  (the fluid step regions are absent from this lowering)")
        return
    named = tuple(step.metadata.get("named_outputs") or ())
    defs = {
        int(i.res.id): i
        for block in region.blocks.values()
        for i in block.instrs if i.res is not None
    }
    cones = {
        str(label): cone(defs, int(value))
        for label, value in named if int(value) in defs
    }
    if not cones:
        print("  (no authored output resolves into the region's own frame)")
        return
    membership: collections.Counter = collections.Counter()
    for values in cones.values():
        membership.update(values)

    print(f"  {'authored equation':22} {'values':>7} {'unique':>7} "
          f"{'shared':>7}  composition")
    for label, values in cones.items():
        unique = sum(1 for v in values if membership[v] == 1)
        shared = len(values) - unique
        mix: collections.Counter = collections.Counter()
        for value_id in values:
            group = classify(str(defs[value_id].op), known_math)
            if group is not None:
                mix[group] += 1
        shape = " ".join(
            f"{group}:{mix[group]}" for group in ORDER if mix[group]
        )
        # Zero unique values means the equation contributes nothing
        # exclusively -- every value it needs, a sibling needs too. For a
        # small cone (a division reused everywhere) that is ordinary common
        # subexpression and expected. It is only worth a second look when a
        # LARGE cone has almost none of its own, which says the equation is
        # nearly a restatement of a sibling rather than its own computation.
        # Stated as the exact counts and nothing else. Zero unique values
        # means no value in this equation's cone is absent from every
        # sibling's -- a set fact. Whether that is ordinary common
        # subexpression or a collapse is a reading of the program, and the
        # reader makes it; a threshold here would invent the answer.
        flag = "   (no exclusive values)" if not unique else ""
        print(f"  {label:22} {len(values):>7} {unique:>7} {shared:>7}  "
              f"{shape}{flag}")


def provenance(module: Any) -> None:
    total = named = accounted = 0
    for name, function in module.functions.items():
        labels = {
            int(value) for _label, value in
            (function.metadata.get("value_names") or ())
        }
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is None:
                    continue
                total += 1
                value_id = int(instruction.res.id)
                if value_id in labels:
                    named += 1
                    continue
                keys = set(instruction.attributes or {})
                if keys & {
                    "tensor_operation", "source_output_id",
                    "program_abi_field", "aggregate_index",
                }:
                    accounted += 1
    anonymous = total - named - accounted
    print(f"  {total} produced values")
    print(f"    {named:>6} carry an authored name")
    print(f"    {accounted:>6} carry accounting that routes back "
          f"(tensor_operation / source_output_id / aggregate_index)")
    print(f"    {anonymous:>6} have neither -- scheduling's own addresses "
          f"and indices")
    print(f"\n  The {anonymous} with no route back are not a property of the")
    print("  program. They are a property of the PIPELINE: provenance is")
    print("  not written down where it is created.")
    print("  Identity survives a transformation only if the pass records")
    print("  the exchange AT THE POINT IT MAKES IT -- one thing for one")
    print("  thing, as it substitutes. Recovering it afterwards from the")
    print("  finished form is inference, and inference here is invention.")
    print("  This count is therefore a measure of what the passes fail to")
    print("  record, not of what the program forgot.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default=None)
    arguments = parser.parse_args()

    lowering = newest_lowering()
    with lowering.open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    known_math = math_ops()
    print(f"reading {lowering.parent.name}\n")

    print("INSTRUCTION CLASSES PER FUNCTION")
    print("  math = covered by the backend's likeness tables; the rest are")
    print("  structural. Ordered by how many instructions are NOT math.\n")
    rows, unclassified = spectrum(module, known_math)
    shown = [
        row for row in rows
        if arguments.function is None or arguments.function in row[0]
    ]
    for name, counts, total, uncounted in shown:
        shape = " ".join(
            f"{group}:{counts[group]}" for group in ORDER if counts[group]
        )
        if uncounted:
            shape += f" unclassified:{uncounted}"
        label = name if len(name) <= 46 else "..." + name[-43:]
        print(f"  {label:46} n={total:<5} {shape}")
    if unclassified:
        print(f"\n  UNCLASSIFIED OPS (counted nowhere above): "
              f"{sorted(unclassified)}")
        print("   Left out rather than bucketed: a default would describe")
        print("   the program as more mathematical than it is.")

    print("\nSYMBOLIC ATTRIBUTION  (what each authored equation became)")
    attribution(module, known_math)

    print("\nPROVENANCE COVERAGE  (what identity survived)")
    provenance(module)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
