"""Whole-source Mandelbrot-to-JFIF ProcessGraph entry point.

The numerical function composes the original Mandelbrot, palette, and JPEG
functions.  ``build_mandelbrot_encoder_process_graph`` imports this file, the
original demo file, and every compression source file as one semantic program.
It does not execute or record an AbstractTensor tape.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from ..abstraction import AbstractTensor


def mandelbrot_display_master(
    unit_x: AbstractTensor,
    unit_y: AbstractTensor,
    center_x: AbstractTensor,
    center_y: AbstractTensor,
    span: AbstractTensor,
    family_mix: AbstractTensor,
    julia_x: AbstractTensor,
    julia_y: AbstractTensor,
    palette_phase: AbstractTensor,
    color_drive: AbstractTensor,
    *,
    iterations: int,
):
    """Return the resident solve and display/encoder color planes."""

    from .demo_mandelbrot_fusion import (
        mandelbrot_jpeg_planes,
        parametric_mandelbrot_escape,
    )

    counts = parametric_mandelbrot_escape(
        unit_x,
        unit_y,
        center_x,
        center_y,
        span,
        family_mix,
        julia_x,
        julia_y,
        iterations,
        clamp=1e18,
    )
    luminance, blue_difference, red_difference = mandelbrot_jpeg_planes(
        counts,
        iterations,
        palette_phase,
        color_drive,
    )
    return counts, luminance, blue_difference, red_difference


def mandelbrot_jpeg_master(
    unit_x: AbstractTensor,
    unit_y: AbstractTensor,
    center_x: AbstractTensor,
    center_y: AbstractTensor,
    span: AbstractTensor,
    family_mix: AbstractTensor,
    julia_x: AbstractTensor,
    julia_y: AbstractTensor,
    palette_phase: AbstractTensor,
    color_drive: AbstractTensor,
    *,
    width: int,
    height: int,
    iterations: int,
    resources,
):
    """Compose the original solve and complete 4:4:4 JFIF encoder."""

    # Avoid eagerly importing the demo/UI module from this source-compiler
    # entrypoint. The AST resolves these original definitions from the supplied
    # source bundle; ordinary execution imports them when the master is called.
    from .demo_mandelbrot_fusion import (
        mandelbrot_jpeg_planes,
        parametric_mandelbrot_escape,
    )
    from ..compression.jpeg.frame import encode_ycbcr_jfif

    counts = parametric_mandelbrot_escape(
        unit_x,
        unit_y,
        center_x,
        center_y,
        span,
        family_mix,
        julia_x,
        julia_y,
        iterations,
    )
    luminance, blue_difference, red_difference = mandelbrot_jpeg_planes(
        counts,
        iterations,
        palette_phase,
        color_drive,
    )
    planes = (
        luminance.reshape(height, width),
        blue_difference.reshape(height, width),
        red_difference.reshape(height, width),
    )
    encoded = encode_ycbcr_jfif(
        planes,
        mcu_rows_per_batch=(height + 7) // 8,
        resources=resources,
    )
    return counts, encoded


def mandelbrot_encoder_source_files() -> tuple[Path, ...]:
    """Return the original source bundle constituting the master program."""

    tensor_root = Path(__file__).resolve().parents[1]
    compression = tensor_root / "compression"
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("demo_mandelbrot_fusion.py"),
    ]
    sources.extend(sorted(compression.rglob("*.py")))
    return tuple(sources)


def mandelbrot_recording_function_ast() -> ast.Module:
    """Put the complete recording program inside one extracted function AST.

    The source files remain authoritative. A source-level dependency walk
    collects the definitions referenced by ``animate_glsl``; no ProcessGraph
    is built during this step. Those original definitions are copied into the
    body of one outer function, followed by the original recording entrypoint
    body. The returned module has exactly one top-level definition and is the
    sole input to ProcessGraph ingestion.
    """

    parsed_sources: list[tuple[str, ast.Module]] = []
    candidates: dict[
        str,
        list[
            tuple[
                str,
                ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
            ]
        ],
    ] = {}
    for source_path in mandelbrot_encoder_source_files():
        resolved = str(source_path.resolve())
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=resolved,
        )
        parsed_sources.append((resolved, tree))
        for statement in tree.body:
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                candidates.setdefault(statement.name, []).append(
                    (resolved, statement)
                )

    roots = [
        item for item in candidates.get("animate_glsl", ())
        if Path(item[0]).name == "demo_mandelbrot_fusion.py"
    ]
    if len(roots) != 1:
        raise RuntimeError("animate_glsl source definition is ambiguous")

    selected: dict[
        tuple[str, str, int | None],
        ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ] = {}
    pending = [roots[0]]

    def select_candidate(
        name: str,
        *,
        referring_file: str,
    ) -> tuple[
        str,
        ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ] | None:
        options = candidates.get(name, ())
        local = [item for item in options if item[0] == referring_file]
        if len(local) == 1:
            return local[0]
        if len(options) == 1:
            return options[0]
        return None

    while pending:
        source_name, definition = pending.pop()
        key = (
            source_name,
            definition.name,
            getattr(definition, "lineno", None),
        )
        if key in selected:
            continue
        selected[key] = definition
        referenced_names = {
            node.id
            for node in ast.walk(definition)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for call in (
            node for node in ast.walk(definition)
            if isinstance(node, ast.Call)
        ):
            if isinstance(call.func, ast.Name):
                referenced_names.add(call.func.id)
            elif isinstance(call.func, ast.Attribute):
                referenced_names.add(call.func.attr)
            for keyword in call.keywords:
                if (
                    keyword.arg == "entrypoint"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    referenced_names.add(keyword.value.value)
        for name in referenced_names:
            candidate = select_candidate(
                name,
                referring_file=source_name,
            )
            if candidate is not None:
                pending.append(candidate)

    selected_files = {key[0] for key in selected}

    root: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    dependencies: list[ast.stmt] = []
    support: list[ast.stmt] = []
    for resolved, tree in parsed_sources:
        if resolved in selected_files:
            support.extend(
                copy.deepcopy(statement)
                for statement in tree.body
                if isinstance(
                    statement,
                    (ast.Import, ast.Assign, ast.AnnAssign),
                )
                or (
                    isinstance(statement, ast.ImportFrom)
                    and statement.module != "__future__"
                )
            )
        for statement in tree.body:
            if not isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            key = (resolved, statement.name, getattr(statement, "lineno", None))
            if key not in selected:
                continue
            if statement.name == "animate_glsl":
                if not isinstance(
                    statement, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    raise TypeError("animate_glsl must be a function")
                root = copy.deepcopy(statement)
            else:
                dependencies.append(copy.deepcopy(statement))

    if root is None:
        raise RuntimeError("animate_glsl source definition was not found")

    master = ast.FunctionDef(
        name="mandelbrot_recording_program",
        args=copy.deepcopy(root.args),
        body=[*support, *dependencies, *copy.deepcopy(root.body)],
        decorator_list=[],
        returns=copy.deepcopy(root.returns),
        type_comment=getattr(root, "type_comment", None),
    )
    ast.copy_location(master, root)
    module = ast.Module(body=[master], type_ignores=[])
    return ast.fix_missing_locations(module)


def build_mandelbrot_encoder_process_graph(
    *,
    profile: str = "tensor_control",
    entrypoint: str = "mandelbrot_jpeg_master",
):
    """Ingest the source bundle without executing the program.

    ``tensor_control`` is the numerical compiler projection, ``program`` keeps
    every node in the transitive entrypoint program, and ``complete`` retains
    the entire source bundle for syntax-coverage and archaeology audits.
    """

    from ....transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        mandelbrot_encoder_source_files(),
        entrypoint=entrypoint,
        profile=profile,
    )
    return graph


def build_mandelbrot_recording_process_graph(*, profile: str = "program"):
    """Ingest one function AST containing the complete recording program."""

    from ....transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        mandelbrot_recording_function_ast(),
        filename="<mandelbrot_recording_program>",
        entrypoint="mandelbrot_recording_program",
        profile=profile,
    )
    graph.G.graph.update(
        program_name="mandelbrot_recording",
        program_entrypoint="mandelbrot_recording_program",
        source_entrypoint="animate_glsl",
        single_function_ast=True,
    )
    return graph


__all__ = [
    "build_mandelbrot_encoder_process_graph",
    "build_mandelbrot_recording_process_graph",
    "mandelbrot_display_master",
    "mandelbrot_encoder_source_files",
    "mandelbrot_jpeg_master",
    "mandelbrot_recording_function_ast",
]
