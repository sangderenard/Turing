"""Colour Python source by ANOTHER stage's influence field.

The source is the substrate everyone can read; the colour comes from a
representation further down the pipeline. So "what does this code look
like to the SSA" is a question you can answer by looking at the code
itself, rather than by reading a graph and holding the correspondence in
your head.

Correlation is by authored name, and only by authored name. Every stage
records the names the author wrote -- SymPy in each equation's left-hand
side and its free symbols, SSA in `named_outputs`, `value_names` and
`parameter_names` -- and an identifier in the source is the same name. No
positional matching and no matching on value: those are how this tree has
repeatedly paired the wrong two things.

An identifier with no reading in the chosen stage is left uncoloured and
counted. That is the interesting output, not a gap to fill: it means the
quantity the author named does not survive into that stage under that
name.

    python tools/source_by_stage.py --by sympy
    python tools/source_by_stage.py --by ssa
    python tools/source_by_stage.py --by ssa --source path.py --function name
"""
from __future__ import annotations

import argparse
import ast
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


def readings_by_name(stage: str) -> tuple[dict[str, Any], str]:
    """{authored name: reading} for the chosen stage."""
    from src.compiler.influence_field import (
        InfluenceContract, field_from_ssa, field_from_sympy,
    )

    contract = InfluenceContract(enabled=True)

    if stage == "sympy":
        from src.compiler.symbolic_fluid_model import (
            symbolic_viscous_shallow_water_equations,
        )
        model = symbolic_viscous_shallow_water_equations()
        field = field_from_sympy(model.equations, contract)
        table = {reading.key: reading for reading in field.table()}
        found: dict[str, Any] = {}
        # Both sides of every equation: the named results AND the free
        # symbols the author wrote, since both appear in the source.
        for equation in model.equations:
            if equation.lhs in table:
                found[str(equation.lhs)] = table[equation.lhs]
        for symbol in model.symbols.values():
            if symbol in table:
                found[str(symbol)] = table[symbol]
        return found, "authored SymPy equations, keyed by expression"

    with newest_lowering().open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    field = field_from_ssa(
        module, contract, functions=[ADVANCE, STEP, REGION],
    )
    table = {reading.key: reading for reading in field.table()}
    found = {}
    for function_name in (REGION, STEP, ADVANCE):
        function = module.functions.get(function_name)
        if function is None:
            continue
        # named_outputs points at the frame where the value is COMPUTED,
        # which for the fluid step is the region, not the step's own
        # forwarding Loads.
        pairs = list(function.metadata.get("value_names") or ())
        pairs += list(function.metadata.get("parameter_names") or ())
        if function_name == STEP:
            pairs += list(function.metadata.get("named_outputs") or ())
        home = REGION if function_name == STEP else function_name
        target = module.functions.get(home)
        if target is None:
            continue
        for label, value in pairs:
            for block_name, block in target.blocks.items():
                for instruction in block.instrs:
                    if (
                        instruction.res is not None
                        and int(instruction.res.id) == int(value)
                    ):
                        reading = table.get((home, block_name, int(value)))
                        if reading is not None:
                            found.setdefault(str(label), reading)
    return found, "lowered SSA, keyed (function, block, value id)"


def identifiers(source: str, function_name: str | None):
    """(line, col, end_col, name) for every identifier in the source."""
    tree = ast.parse(source)
    roots = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (function_name is None or node.name == function_name)
    ] or [tree]
    seen = []
    for root in roots:
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                seen.append((
                    node.lineno, node.col_offset, node.end_col_offset,
                    node.id,
                ))
            elif isinstance(node, ast.Attribute):
                # `state.viscosity` -- the authored name is the attribute,
                # and it sits at the end of the span.
                end = node.end_col_offset or 0
                seen.append((
                    node.lineno, max(0, end - len(node.attr)), end, node.attr,
                ))
    return seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--by", choices=("sympy", "ssa"), default="ssa")
    parser.add_argument("--source", default=None)
    parser.add_argument("--function", default=None)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    from src.rendering.influence_field_image import dye_rgb

    if arguments.source:
        source = Path(arguments.source).read_text(encoding="utf-8")
    else:
        from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE
        source = SYMBOLIC_FLUID_DT_SOURCE

    found, provenance = readings_by_name(arguments.by)
    print(f"colouring by {arguments.by}: {provenance}")
    print(f"{len(found)} authored names carry a reading in that stage")

    lines = source.splitlines()
    grid: list[list[Any]] = [[None] * len(line) for line in lines]
    hit = miss = 0
    unmatched: set[str] = set()
    for line_number, start, end, name in identifiers(source, arguments.function):
        row = line_number - 1
        if row < 0 or row >= len(lines):
            continue
        reading = found.get(name)
        if reading is None:
            miss += 1
            unmatched.add(name)
            continue
        hit += 1
        category = (reading.categories or {}).get(DYNAMIC)
        colour = dye_rgb(
            getattr(category, "hue", 0.0),
            getattr(category, "saturation", 0.0),
            max(0.25, min(1.0, float(reading.value))),
        )
        for column in range(max(0, start), min(end, len(lines[row]))):
            grid[row][column] = colour

    print(f"identifiers coloured {hit}, unmatched {miss}")
    if unmatched:
        listed = ", ".join(sorted(unmatched)[:14])
        print(f"  no reading under that name: {listed}"
              + (" ..." if len(unmatched) > 14 else ""))
        print("  Left uncoloured. A name absent here did not survive into "
              "that stage under\n  the author's name -- which is the finding, "
              "not a gap to fill.")

    body = []
    for row, line in enumerate(lines):
        cells, run, text = [], None, []

        def flush(run=None, text=None):
            if not text:
                return
            payload = "".join(text).replace("&", "&amp;")
            payload = payload.replace("<", "&lt;").replace(">", "&gt;")
            if run is None:
                cells.append(f"<span>{payload}</span>")
            else:
                red, green, blue = (int(round(c * 255)) for c in run)
                cells.append(
                    f'<span style="background:rgb({red},{green},{blue});'
                    f'color:#111">{payload}</span>'
                )
        for column, character in enumerate(line):
            colour = grid[row][column]
            if colour != run:
                flush(run, text)
                run, text = colour, []
            text.append(character)
        flush(run, text)
        body.append(
            f'<div class="l"><span class="n">{row + 1}</span>'
            + "".join(cells) + "</div>"
        )

    destination = ROOT / (
        arguments.out or f"build/source_by_{arguments.by}.html"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>source by {arguments.by}</title>"
        "<style>body{background:#0e0e12;color:#ddd;font:13px/1.5 ui-monospace,"
        "Consolas,monospace;padding:18px}.l{white-space:pre}.n{color:#445;"
        "display:inline-block;width:3.5em;text-align:right;padding-right:1em;"
        "user-select:none}h2{font:600 14px ui-sans-serif,system-ui;margin:0 0 "
        ".2em}p{color:#889;margin:.2em 0 1.4em}</style>"
        f"<h2>source coloured by the {arguments.by} field</h2>"
        f"<p>{provenance}. Correlated by authored name. "
        f"{hit} identifiers coloured, {miss} with no reading in this stage "
        "(left uncoloured).</p>"
        + "".join(body),
        encoding="utf-8",
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
