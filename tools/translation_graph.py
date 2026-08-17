"""One graph across all languages; colour diffuses from ingestion.

The earlier approach built a separate field per representation and then
correlated them by name afterwards. That is backwards, and it shows: a
name lookup is one-to-one, so an authored quantity lit exactly one token
and nothing downstream, and the panes had to be patched with cone walks
to fake the spreading.

Here there is ONE graph. Its nodes are textual objects in every
representation and its edges include the ones that cross a language
boundary -- the moment a thing becomes another thing. Sources sit at
ingestion, which is the causal focus: colour originates where the author
wrote something and diffuses forward through every translation that
carried it.

Then spreading and mixing are not features anyone implements. They are
what diffusion on a graph already does: one authored token reaching many
instructions IS spreading, several tokens reaching one instruction and
adding their spectra IS mixing. And because every representation reads
the same field, the layers cannot disagree -- they are one syntactic
spectral map seen from different sides, rather than several maps
correlated and hoped to line up.

    python tools/translation_graph.py
"""
from __future__ import annotations

import argparse
import ast
import pickle
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ADVANCE = "symbolic_fluid_control__symbolic_fluid_advance"
STEP = "symbolic_fluid_control__symbolic_fluid_step"
REGION = STEP + "__planned_region_0"


def newest_lowering() -> Path:
    found = sorted(
        (ROOT / "build").glob("*/control_repository_ssa.pkl"),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if not found:
        raise SystemExit("no lowered SSA under build/")
    return found[0]


def build(contract: Any):
    """One field over python -> sympy -> ssa, with the crossings included."""
    from src.compiler.influence_field import DYNAMIC, InfluenceField
    from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE
    from src.compiler.symbolic_fluid_model import (
        symbolic_viscous_shallow_water_equations,
    )

    field = InfluenceField(contract)
    model = symbolic_viscous_shallow_water_equations()
    with newest_lowering().open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)

    # -- ingestion: one node per authored identifier occurrence ---------
    #
    # These are the sources. The causal focus is the point of ingestion,
    # so every colour in every representation ultimately originates at a
    # place the author actually wrote, and a node's colour says which
    # authored text reached it.
    tree = ast.parse(SYMBOLIC_FLUID_DT_SOURCE)
    occurrences: list[tuple[Any, str, int, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            end = node.end_col_offset or 0
            name, row, start = node.attr, node.lineno - 1, max(0, end - len(node.attr))
        elif isinstance(node, ast.Name):
            name, row, start = node.id, node.lineno - 1, node.col_offset
            end = node.end_col_offset or start
        else:
            continue
        key = ("py", row, start, end)
        field.add_node(key)
        occurrences.append((key, name, row, start, end))

    # -- sympy: expression tree, nodes keyed by the expression ----------
    def visit(expression: Any) -> None:
        field.add_node(("sy", expression))
        for operand in getattr(expression, "args", ()) or ():
            visit(operand)
            field.add_edge(("sy", operand), ("sy", expression), role="data")

    for equation in model.equations:
        visit(equation.rhs)
        field.add_node(("sy", equation.lhs))
        field.add_edge(("sy", equation.rhs), ("sy", equation.lhs), role="data")

    # -- ssa: def-use inside the region that holds the arithmetic -------
    region = module.functions.get(REGION)
    home: dict[int, str] = {}
    if region is not None:
        for block_name, block in region.blocks.items():
            for instruction in block.instrs:
                if instruction.res is None:
                    continue
                value_id = int(instruction.res.id)
                home[value_id] = block_name
                field.add_node(("ssa", value_id))
        for block in region.blocks.values():
            for instruction in block.instrs:
                if instruction.res is None:
                    continue
                for argument in instruction.args:
                    # Formals are operands too, and they are precisely where
                    # the authored inputs cross in. Requiring the operand to
                    # be an instruction RESULT dropped every edge leaving a
                    # formal, so colour entered the region and stopped.
                    field.add_node(("ssa", int(argument.id)))
                    field.add_edge(
                        ("ssa", int(argument.id)),
                        ("ssa", int(instruction.res.id)),
                        role="data",
                    )

    # -- the crossings -------------------------------------------------
    #
    # These edges are the whole point: they are where one thing becomes
    # another, and without them the representations are three unrelated
    # graphs that can only be compared by matching strings afterwards.
    crossings = 0
    symbols = {str(symbol): symbol for symbol in model.symbols.values()}
    for key, name, _row, _start, _end in occurrences:
        symbol = symbols.get(name)
        if symbol is not None:
            field.add_edge(key, ("sy", symbol), role="data")
            crossings += 1

    named = tuple(
        module.functions[STEP].metadata.get("named_outputs") or ()
    ) if STEP in module.functions else ()

    # The crossing belongs at the INPUTS, not at the results.
    #
    # An authored equation causes every instruction its expansion produced,
    # and in a def-use graph those instructions are downstream of the
    # region's formals -- not downstream of the named output, which is
    # where the computation ENDS. Attaching at the output let colour reach
    # eight values out of 290, because there is nothing after a result.
    # Attaching at the formals lets it flow through the whole expansion,
    # which is what "spread" actually is.
    step_function = module.functions.get(STEP)
    parameters = dict(
        (step_function.metadata.get("parameter_names") or ())
        if step_function is not None else ()
    )
    for name, value in parameters.items():
        symbol = symbols.get(str(name))
        if symbol is None:
            continue
        field.add_node(("ssa", int(value)))
        field.add_edge(("sy", symbol), ("ssa", int(value)), role="data")
        crossings += 1

    # Results still cross back, so a named output is reachable as itself
    # rather than only as the confluence of its operands.
    by_name = {str(equation.lhs): equation.lhs for equation in model.equations}
    for label, value in named:
        symbol = by_name.get(str(label))
        if symbol is not None and int(value) in home:
            field.add_edge(("sy", symbol), ("ssa", int(value)), role="data")
            crossings += 1

    # An authored result is also written back into the traversal, so the
    # python token that receives it is downstream of the SSA value. Without
    # this the python side is only ever a source and never shows what came
    # back to it.
    for label, value in named:
        if int(value) not in home:
            continue
        for key, name, _row, _start, _end in occurrences:
            if name == str(label):
                field.add_edge(("ssa", int(value)), key, role="data")
                crossings += 1

    entries = [
        (key, DYNAMIC, ordinal, name, "")
        for ordinal, (key, name, _row, _start, _end) in enumerate(occurrences)
    ]
    field.add_sources(entries)
    return field, occurrences, home, module, model, crossings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=12)
    arguments = parser.parse_args()

    from src.compiler.influence_field import InfluenceContract

    contract = InfluenceContract(enabled=True, spectral=True)
    field, occurrences, home, _module, _model, crossings = build(contract)
    readings = list(field.table())
    print(f"one field over three representations")
    print(f"  ingestion sources (authored identifier occurrences): "
          f"{len(occurrences)}")
    print(f"  cross-language edges: {crossings}")
    print(f"  ssa values in the region: {len(home)}")
    print(f"  nodes with a reading: {len(readings)}")

    kinds: dict[str, int] = {}
    for reading in readings:
        kind = reading.key[0] if isinstance(reading.key, tuple) else "?"
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"  readings by representation: {kinds}")

    lit = [
        reading for reading in readings
        if isinstance(reading.key, tuple) and reading.key[0] == "ssa"
        and reading.value > 0.0
    ]
    print(f"\nSSA values reached by diffusion from ingestion: {len(lit)}"
          f" of {len(home)}")
    lit.sort(key=lambda reading: -reading.value)
    for reading in lit[:arguments.top]:
        accumulator = field.moments(reading.key).get("dynamic")
        lines = len(getattr(accumulator, "lines", ()) or ())
        print(f"  t{reading.key[1]:<5} weight={reading.value:.4f} "
              f"origins={lines}")
    print("\nOrigins is the count of DISTINCT authored occurrences whose")
    print("influence reached that instruction. More than one is mixing, and")
    print("it is what diffusion produces on its own -- no cone walk, no")
    print("name matching, and nothing that has to be kept in step by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
