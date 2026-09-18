"""Show source coloured by the layer flags it carries.

Each translation stamps a colour flag -- one continuous float -- onto the
values it produces, building outward from the ingested form. A flag is a
frequency, so the honest way to look at it is as light: map the flag to a
wavelength, and where a construct carries several layers, ADD the spectra
the way light adds. A region that has been through two translations is
literally the mix of their colours, and two constructs that share a
history share a hue because they share the numbers.

This is a viewer, not an analysis. It computes no grouping and infers no
structure: it reads the flags already on the nodes and renders them. If a
construct is uncoloured here, nothing stamped it -- which is a finding
about the pipeline, and the legend says so rather than filling it in.

    python tools/spectral_view.py                       # the fluid source
    python tools/spectral_view.py --source path.py --function name
    python tools/spectral_view.py --out build/view.html
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Visible band. A flag is uniform in [0, 1), so this spreads flags across
#: the spectrum evenly -- adjacent flags are unrelated, which is correct:
#: a digest carries no ordering and the view must not imply one.
VIOLET_NM, RED_NM = 380.0, 700.0


def wavelength_rgb(nanometres: float) -> tuple[float, float, float]:
    """Approximate sRGB for a single wavelength (Bruton's piecewise fit)."""
    w = float(nanometres)
    if w < 440:
        red, green, blue = -(w - 440) / (440 - 380), 0.0, 1.0
    elif w < 490:
        red, green, blue = 0.0, (w - 440) / (490 - 440), 1.0
    elif w < 510:
        red, green, blue = 0.0, 1.0, -(w - 510) / (510 - 490)
    elif w < 580:
        red, green, blue = (w - 510) / (580 - 510), 1.0, 0.0
    elif w < 645:
        red, green, blue = 1.0, -(w - 645) / (645 - 580), 0.0
    else:
        red, green, blue = 1.0, 0.0, 0.0
    # Eye response falls off at the ends; without this the extremes read as
    # bright pure hues and dominate a view they should not.
    if w < 420:
        falloff = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif w > 700:
        falloff = 0.3
    elif w > 645:
        falloff = 0.3 + 0.7 * (700 - w) / (700 - 645)
    else:
        falloff = 1.0
    return (red * falloff, green * falloff, blue * falloff)


def flag_rgb(flag: float) -> tuple[float, float, float]:
    return wavelength_rgb(VIOLET_NM + (float(flag) % 1.0) * (RED_NM - VIOLET_NM))


def mix(layers: tuple) -> tuple[int, int, int]:
    """Collapse layers to colour using the tree's own dye mapping.

    Deliberately NOT a second colour model. `influence_field` already fixes
    how this tree turns influence into colour -- power sums over a spectral
    ARC that terminates at violet and never wraps, so a mean hue is a real
    centroid instead of the meaningless average two opposed hues give on a
    circle -- and `influence_field_image.dye_rgb` already renders that in
    OKLCH. Re-deriving either here would be a parallel mechanism that
    disagrees with the shader for the same field.

    So layers reduce the same way the field does: mean hue as the centroid,
    dispersion as saturation, so a construct carrying one layer is vivid and
    one carrying several reads as mixed.
    """
    from src.rendering.influence_field_image import dye_rgb

    if not layers:
        return (28, 28, 32)
    weight = float(len(layers))
    first = sum(float(flag) for flag in layers) / weight
    second = sum(float(flag) ** 2 for flag in layers) / weight
    spread = max(0.0, second - first * first) ** 0.5
    red, green, blue = dye_rgb(first, max(0.0, 1.0 - 2.0 * spread), 1.0)
    return (int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)))


def ingest(source: str, function_name: str | None):
    """Run the real ingestion and collect (span, layers) for every node."""
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    from src.compiler.ast_process_graph import build_semantic_ast

    tree = ast.parse(source)
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (function_name is None or node.name == function_name)
    ]
    if not functions:
        raise SystemExit(f"no function {function_name!r} in the source")

    painted = []
    names: dict[float, str] = {}
    ingested: list[str] = []
    refused: list[tuple[str, str]] = []
    for function in functions:
        graph = ProcessGraph(0, False, materialize_memory=False)
        try:
            build_semantic_ast(graph, function, filename="<spectral>")
        except Exception as error:
            # A function the ingestion declines is reported, not skipped
            # silently: an uncoloured region in the view must never be
            # ambiguous between "nothing stamped it" and "never looked at".
            refused.append((function.name, f"{type(error).__name__}: {error}"))
            continue
        ingested.append(function.name)
        names.update(graph.G.graph.get("layer_names") or {})
        for _node_id, data in graph.G.nodes(data=True):
            span = data.get("source_span") or {}
            layers = tuple(span.get("layers") or ())
            if not layers or span.get("line") is None:
                continue
            painted.append((span, layers, str(data.get("op"))))
    return painted, names, ingested, refused


def paint(source: str, painted) -> list[list[tuple]]:
    """Per-character layers; the narrowest span covering a character wins."""
    lines = source.splitlines()
    grid: list[list[tuple]] = [[() for _ in line] for line in lines]

    def extent(span):
        start_line = int(span["line"])
        end_line = int(span.get("end_line") or start_line)
        return (end_line - start_line, int(span.get("end_column") or 0))

    for span, layers, _op in sorted(
        painted, key=lambda item: extent(item[0]), reverse=True,
    ):
        start_line = int(span["line"]) - 1
        end_line = int(span.get("end_line") or span["line"]) - 1
        start_col = int(span.get("column") or 0)
        end_col = int(span.get("end_column") or 0)
        for row in range(start_line, min(end_line, len(lines) - 1) + 1):
            if row < 0 or row >= len(lines):
                continue
            first = start_col if row == start_line else 0
            last = end_col if row == end_line else len(lines[row])
            for column in range(max(0, first), min(last, len(lines[row]))):
                grid[row][column] = layers
    return grid


def render_terminal(source: str, grid) -> None:
    lines = source.splitlines()
    for row, line in enumerate(lines):
        out = []
        for column, character in enumerate(line):
            red, green, blue = mix(grid[row][column])
            out.append(f"\x1b[48;2;{red};{green};{blue}m\x1b[38;2;250;250;250m{character}")
        print(f"\x1b[0m{row + 1:>4} " + "".join(out) + "\x1b[0m")


def render_html(source: str, grid, names: dict, destination: Path) -> None:
    lines = source.splitlines()
    body = []
    for row, line in enumerate(lines):
        cells = []
        run_layers, run_text = None, []
        def flush():
            if not run_text:
                return
            red, green, blue = mix(run_layers or ())
            text = "".join(run_text).replace("&", "&amp;")
            text = text.replace("<", "&lt;").replace(">", "&gt;")
            cells.append(
                f'<span style="background:rgb({red},{green},{blue})">{text}</span>'
            )
        for column, character in enumerate(line):
            layers = grid[row][column]
            if layers != run_layers:
                flush()
                run_layers, run_text = layers, []
            run_text.append(character)
        flush()
        body.append(
            f'<div class="l"><span class="n">{row + 1}</span>'
            + "".join(cells) + "</div>"
        )
    legend = "".join(
        f'<div><span class="sw" style="background:rgb(%d,%d,%d)"></span>'
        % mix((flag,)) + f"<code>{flag:.12f}</code> &mdash; {label}</div>"
        for flag, label in sorted(names.items(), key=lambda kv: kv[1])
    )
    destination.write_text(
        "<!doctype html><meta charset='utf-8'><title>spectral view</title>"
        "<style>body{background:#101014;color:#eee;font:13px/1.45 ui-monospace,"
        "Consolas,monospace;padding:18px}.l{white-space:pre}.n{color:#556;"
        "display:inline-block;width:3.5em;text-align:right;padding-right:1em;"
        "user-select:none}.sw{display:inline-block;width:1.1em;height:1.1em;"
        "vertical-align:-2px;margin-right:.6em;border-radius:2px}"
        "h2{font:600 14px ui-sans-serif,system-ui;margin:1.6em 0 .6em}"
        "code{color:#9cf}</style>"
        f"<h2>source, coloured by the layers each construct carries</h2>"
        + "".join(body)
        + "<h2>flags present, and what each one was at ingestion</h2>"
        + legend,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--function", default=None,
        help="one function; omit to render every function in the source",
    )
    parser.add_argument("--out", default="build/spectral_view.html")
    parser.add_argument("--no-terminal", action="store_true")
    arguments = parser.parse_args()

    if arguments.source:
        source = Path(arguments.source).read_text(encoding="utf-8")
    else:
        from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE
        source = SYMBOLIC_FLUID_DT_SOURCE

    painted, names, ingested, refused = ingest(source, arguments.function)
    grid = paint(source, painted)
    coloured = sum(1 for row in grid for cell in row if cell)
    total = sum(len(row) for row in grid)
    print(f"ingested {len(ingested)} function(s): {', '.join(ingested)}")
    if refused:
        print(f"  {len(refused)} function(s) the ingestion declined "
              "(their source stays uncoloured for that reason, not for "
              "lack of a stamp):")
        for name, why in refused:
            print(f"    {name}: {why}")
    print(f"{len(painted)} stamped nodes; "
          f"{coloured}/{total} source characters carry a layer")
    if coloured < total:
        print(f"  {total - coloured} characters carry none -- nothing stamped "
              "them. Left uncoloured rather than filled in.")
    print(f"distinct flags: {len(names)}")

    if not arguments.no_terminal:
        render_terminal(source, grid)
    destination = ROOT / arguments.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_html(source, grid, names, destination)
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
