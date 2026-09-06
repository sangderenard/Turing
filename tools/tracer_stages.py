"""One quantity, all four representations, through the same tracer.

SymPy equations, the Python/AST dependency graph, the Dual IR and the
repository SSA are the same computation described four ways. The influence
field already builds over each of them; what was missing is a view that
puts them side by side under ONE contract, so a reader follows a single
authored quantity across the whole pipeline instead of comparing four
unrelated pictures.

The correlation key is the authored name, because that is the only
identity every stage genuinely shares. SSA value ids do not survive a
stage boundary -- they are not even unique across frames -- and node ids
are per-graph. A name is what the author wrote, and each stage records it:
SymPy in the equation's left-hand side, SSA in `named_outputs` and
`value_names`. Where a stage cannot produce a name, the row says so rather
than being filled in by matching on position or on value, which is how
this tree has repeatedly paired the wrong two things.

    python tools/tracer_stages.py
    python tools/tracer_stages.py --name tracer_next
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ADVANCE = "symbolic_fluid_control__symbolic_fluid_advance"
STEP = "symbolic_fluid_control__symbolic_fluid_step"
REGION = STEP + "__planned_region_0"
DYNAMIC = "dynamic"


def newest_lowering() -> Path:
    found = sorted(
        (ROOT / "build").glob("*/control_repository_ssa.pkl"),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if not found:
        raise SystemExit("no lowered SSA under build/")
    return found[0]


def reading_of(field: Any, key: Any):
    for reading in field.table():
        if reading.key == key:
            return reading
    return None


def describe(reading: Any) -> str:
    if reading is None:
        return f"{'-':>8} {'-':>6} {'-':>6}"
    category = (reading.categories or {}).get(DYNAMIC)
    hue = getattr(category, "hue", 0.0)
    saturation = getattr(category, "saturation", 0.0)
    return f"{hue:>8.4f} {saturation:>6.3f} {reading.value:>6.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)
    arguments = parser.parse_args()

    from src.compiler.influence_field import (
        InfluenceContract, field_from_process_graph, field_from_ssa,
        field_from_sympy,
    )
    from src.compiler.symbolic_fluid_model import (
        symbolic_viscous_shallow_water_equations,
    )

    # ONE contract for every stage: the same policy, so the fields are
    # comparable. Separate contracts would silently allow different
    # categories or transport rules per stage and the colours would stop
    # meaning the same thing.
    contract = InfluenceContract(enabled=True)
    model = symbolic_viscous_shallow_water_equations()

    stages: dict[str, Any] = {}
    missing: dict[str, str] = {}

    stages["sympy"] = field_from_sympy(model.equations, contract)

    # AST dependency: the authored traversal, ingested.
    try:
        import ast

        from src.compiler.ast_process_graph import build_semantic_ast
        from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE
        from src.transmogrifier.graph.graph_express2 import ProcessGraph

        tree = ast.parse(SYMBOLIC_FLUID_DT_SOURCE)
        function = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "symbolic_fluid_advance"
        )
        graph = ProcessGraph(0, False, materialize_memory=False)
        build_semantic_ast(graph, function, filename="<tracer>")
        stages["ast"] = field_from_process_graph(graph, contract)
    except Exception as error:
        missing["ast"] = f"{type(error).__name__}: {error}"

    with newest_lowering().open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    # The region is where the arithmetic actually is. In the step's own
    # frame every named output is a Load off one call, so all eleven share
    # an identical reading -- true, and useless for telling them apart.
    stages["ssa"] = field_from_ssa(
        module, contract, functions=[ADVANCE, STEP, REGION],
    )

    # Dual IR needs the shell object, which the lowering pickle does not
    # carry. Reported, not faked: an empty column that looks like a stage
    # would be worse than an absent one.
    missing.setdefault(
        "dual_ir",
        "needs the DualIRShell; the lowering pickle carries SSA only. "
        "Build with the aot_compile path to include it.",
    )

    print(f"stages traced under one contract: {', '.join(sorted(stages))}")
    for stage, why in sorted(missing.items()):
        print(f"  {stage}: NOT TRACED -- {why}")

    # -- correlate by authored name -----------------------------------
    step = module.functions.get(STEP)
    region = module.functions.get(REGION)
    ssa_names: dict[str, tuple] = {}
    if step is not None and region is not None:
        # named_outputs records the region-local value each authored name
        # denotes, so the name resolves in the region's frame -- the one
        # place the value is computed rather than merely forwarded.
        for label, value in tuple(step.metadata.get("named_outputs") or ()):
            for block_name, block in region.blocks.items():
                for instruction in block.instrs:
                    if (
                        instruction.res is not None
                        and int(instruction.res.id) == int(value)
                    ):
                        ssa_names[str(label)] = (REGION, block_name, int(value))

    equations = {str(equation.lhs): equation.lhs for equation in model.equations}
    wanted = (
        [arguments.name] if arguments.name else sorted(equations)
    )
    print(
        f"\n{'authored quantity':22} "
        f"{'sympy hue':>8} {'sat':>6} {'val':>6}   "
        f"{'ssa hue':>8} {'sat':>6} {'val':>6}"
    )
    for name in wanted:
        symbol = equations.get(name)
        if symbol is None:
            print(f"{name:22}  no such authored equation")
            continue
        left = reading_of(stages["sympy"], symbol)
        key = ssa_names.get(name)
        right = reading_of(stages["ssa"], key) if key else None
        note = "" if key else "   (no SSA value carries this name)"
        print(f"{name:22} {describe(left)}   {describe(right)}{note}")

    print(
        "\nHue is a centroid on the spectral arc, so it reads as "
        "depth-of-origin;\nsaturation is 1 - dispersion, so a low value "
        "means the influence arriving\nhere came from many places rather "
        "than one. The two stages allocate their\nown arcs, so compare "
        "ORDERING and spread across a column, not absolute hue\nbetween "
        "columns -- those are different arcs and equating them would be "
        "the\nsame mistake as equating value ids across frames."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
