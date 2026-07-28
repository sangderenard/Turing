"""One explicit AbstractTensor Mandelbrot-to-JFIF ProcessGraph entry point."""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import inspect
import re
import textwrap
from pathlib import Path

from ..abstraction import AbstractTensor


_COMPILER_RUNTIME_MODULES = (
    "src.common.tensors.abstraction",
    "src.common.tensors.autograd",
    "src.common.tensors.accelerator_backends.glsl_backend",
    "src.common.tensors.accelerator_backends.glsl_tensor_backend",
)


def _resolve_ast_reference(
    expression: ast.AST,
    bindings: dict[str, object],
):
    if isinstance(expression, ast.Name):
        return bindings.get(expression.id)
    if not isinstance(expression, ast.Attribute):
        return None
    owner = _resolve_ast_reference(expression.value, bindings)
    if owner is None:
        return None
    try:
        return getattr(owner, expression.attr)
    except AttributeError:
        return None


def _callable_source_node(value) -> ast.AST | None:
    if not (inspect.isfunction(value) or inspect.isclass(value)):
        return None
    try:
        source = textwrap.dedent(inspect.getsource(value))
    except (OSError, TypeError):
        return None
    parsed = ast.parse(
        source,
        filename=str(inspect.getsourcefile(value) or "<helper>"),
    )
    return next(
        (
            statement
            for statement in parsed.body
            if isinstance(
                statement,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            )
        ),
        None,
    )


def _is_program_helper(value) -> bool:
    module = str(getattr(value, "__module__", ""))
    return (
        module.startswith("src.common.tensors")
        and not module.startswith(_COMPILER_RUNTIME_MODULES)
        and (inspect.isfunction(value) or inspect.isclass(value))
    )


def _definition_bindings(value, definition: ast.AST) -> dict[str, object]:
    module = inspect.getmodule(value)
    bindings = dict(vars(module)) if module is not None else {}
    package = str(getattr(module, "__package__", "") or "")
    for statement in ast.walk(definition):
        if isinstance(statement, ast.ImportFrom):
            module_name = (
                "." * int(statement.level)
                + str(statement.module or "")
            )
            try:
                imported_module = importlib.import_module(
                    module_name,
                    package=package,
                )
            except (ImportError, TypeError, ValueError):
                continue
            for imported in statement.names:
                try:
                    bindings[imported.asname or imported.name] = getattr(
                        imported_module,
                        imported.name,
                    )
                except AttributeError:
                    continue
        elif isinstance(statement, ast.Import):
            for imported in statement.names:
                try:
                    bindings[
                        imported.asname or imported.name.split(".")[0]
                    ] = importlib.import_module(imported.name)
                except ImportError:
                    continue
    return bindings


def mandelbrot_recording_program_ast_closure() -> tuple[
    ast.Module,
    dict[str, object],
]:
    """Collect every project-owned helper body used by solve-through-AVI.

    AbstractTensor and backend runtime methods remain compiler primitives.
    Compression, entropy, container, and program helpers are ordinary source
    definitions and enter the same AST module/function table as the root.
    """

    from ..compression.containers.avi import MJPEGAVIWriter

    queue = [mandelbrot_recording_program, MJPEGAVIWriter]
    definitions: list[tuple[object, ast.AST, dict[str, object]]] = []
    merged_bindings = dict(mandelbrot_recording_program.__globals__)
    seen: set[tuple[str, str]] = set()
    while queue:
        value = queue.pop()
        identity = (
            str(getattr(value, "__module__", "")),
            str(getattr(value, "__qualname__", "")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        definition = _callable_source_node(value)
        if definition is None:
            continue
        bindings = _definition_bindings(value, definition)
        definitions.append((value, definition, bindings))
        merged_bindings.update(bindings)
        for call in (
            node for node in ast.walk(definition)
            if isinstance(node, ast.Call)
        ):
            target = _resolve_ast_reference(call.func, bindings)
            if _is_program_helper(target):
                queue.append(target)

    name_counts: dict[str, int] = {}
    for _, definition, _ in definitions:
        name = str(getattr(definition, "name", ""))
        name_counts[name] = name_counts.get(name, 0) + 1

    compiled_names: dict[int, str] = {}
    for value, definition, _ in definitions:
        original_name = str(getattr(definition, "name", ""))
        compiled_name = original_name
        if name_counts[original_name] > 1:
            module_name = re.sub(
                r"\W+",
                "_",
                str(getattr(value, "__module__", "")),
            ).strip("_")
            compiled_name = f"{module_name}__{original_name}"
        compiled_names[id(value)] = compiled_name

    class _RewriteCollectedCalls(ast.NodeTransformer):
        def __init__(self, bindings: dict[str, object]):
            self.bindings = bindings

        def visit_Call(self, node: ast.Call):
            self.generic_visit(node)
            target = _resolve_ast_reference(node.func, self.bindings)
            compiled_name = compiled_names.get(id(target))
            if compiled_name is not None:
                node.func = ast.copy_location(
                    ast.Name(id=compiled_name, ctx=ast.Load()),
                    node.func,
                )
            return node

    assembled: list[ast.AST] = []
    for value, definition, bindings in definitions:
        compiled_name = compiled_names[id(value)]
        definition.name = compiled_name
        merged_bindings[compiled_name] = value
        assembled.append(_RewriteCollectedCalls(bindings).visit(definition))

    assembled.sort(
        key=lambda node: (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "mandelbrot_recording_program",
            getattr(node, "name", ""),
        )
    )
    module = ast.Module(body=assembled, type_ignores=[])
    ast.fix_missing_locations(module)
    return module, merged_bindings


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


# ============================================================================
# AST-ROOT INTEGRITY CONTRACT
#
# This function is the semantic root of the compiled Mandelbrot recording
# demo.  EVERY operation required to move from its declared inputs to a
# finished recording belongs in this function's AST closure.  That includes
# numerical solving, color construction, JPEG transforms and entropy coding,
# AVI/OpenDML writer construction, packet ordering, audio interleaving, index
# finalization, error-safe closure, and every helper those operations require.
#
# It is NEVER acceptable to make the demo appear complete by performing one
# of those stages in its Python caller, in a CLI-only helper, in a test sink,
# or in an object constructed outside this AST root.  In particular:
#
#   * Do not restore FrameCollector or any equivalent frame-catching proxy.
#   * Do not pass a preconstructed AVI writer into this function.
#   * Do not encode or frame the returned tensors in the surrounding demo.
#   * Do not let profiling/capture execute a shortened pipeline while the
#     ordinary Python caller silently completes the file afterward.
#
# Host and operating-system primitives may remain explicit terminal runtime
# boundaries until their lowerings exist, but the calls to those boundaries,
# their ordering, their state transitions, and their cleanup must remain
# represented by this root ProcessGraph.  If some construct cannot yet lower,
# the correct result is a visible compiler frontier and profiler traceback—
# never an out-of-graph substitute that changes what is being demonstrated.
# ============================================================================
def mandelbrot_recording_program(
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
    avi_path=None,
    avi_fps=30,
    avi_opendml=True,
    avi_segment_bytes=1 << 30,
    audio_samples=None,
    resources=None,
):
    """Solve and encode one frame, optionally producing a complete AVI."""

    from .demo_mandelbrot_fusion import (
        mandelbrot_jpeg_planes,
        parametric_mandelbrot_escape,
    )
    from ..autograd import autograd
    from ..compression.block_transform import (
        block_view_2d,
        dct_2d_blocks,
    )
    from ..compression.coefficient_events import (
        collect_component_block_coefficient_events,
        slice_block_coefficient_events,
    )
    from ..compression.jpeg.frame import (
        _color_header,
        finalize_entropy_scan,
        prepare_jpeg_encoding_resources,
    )
    from ..compression.jpeg.scan import (
        encode_baseline_color_component_scan,
    )
    from ..compression.containers.avi import MJPEGAVIWriter

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
    resources = resources or prepare_jpeg_encoding_resources(planes[0])

    with autograd.no_grad():
        # Keep every numerical compression stage in this submitted function:
        # component batching/blocking -> DCT -> quantization -> rounding ->
        # zigzag ordering.  Each operation remains ordinary AbstractTensor
        # composition and therefore stays visible to ProcessGraph lowering.
        component_samples = AbstractTensor.stack(planes, dim=0)
        blocks = block_view_2d(
            component_samples - 128.0,
            block_height=8,
            block_width=8,
        )
        transformed = dct_2d_blocks(
            blocks,
            basis=resources.dct_basis,
        )
        scaled = transformed / resources.ycbcr_quantization
        quantized = scaled.sign() * ((scaled.abs() + 0.500001) // 1)
        flattened = quantized.reshape(
            *(quantized.shape[:-2] + (64,))
        )
        coefficients = flattened[..., resources.zigzag]

        # Coefficient events are the explicit boundary between the numerical
        # tensor transform and the canonical JPEG Huffman tables.
        events = collect_component_block_coefficient_events(
            coefficients,
            max_magnitude_bits=11,
            previous_dc=(0, 0, 0),
        )
        block_count = coefficients[0].reshape(-1, 64).shape[0]
        y_events = slice_block_coefficient_events(
            events,
            0,
            block_count,
        )
        chroma_events = slice_block_coefficient_events(
            events,
            block_count,
            block_count * 3,
        )
        huffman_scan = encode_baseline_color_component_scan(
            y_events,
            chroma_events,
            luma_dc_table=resources.luma_dc_table,
            luma_ac_table=resources.luma_ac_table,
            chroma_dc_table=resources.chroma_dc_table,
            chroma_ac_table=resources.chroma_ac_table,
        )
        entropy_bytes = finalize_entropy_scan(huffman_scan)
        trailing_bytes = b""

    jpeg_frame = b"".join(
        (
            _color_header(height, width),
            entropy_bytes,
            trailing_bytes,
            b"\xFF\xD9",
        )
    )

    # Writer construction, RIFF/OpenDML framing, packet insertion, index
    # finalization, and closure are all inside the submitted AST.  No caller
    # supplied object is permitted to stand in for this part of the program.
    if avi_path is not None:
        avi_writer = MJPEGAVIWriter(
            avi_path,
            width=width,
            height=height,
            fps=avi_fps,
            opendml=avi_opendml,
            segment_bytes=avi_segment_bytes,
        )
        try:
            avi_writer.append_frame(jpeg_frame)
            if audio_samples is not None:
                avi_writer.append_audio_tensor(audio_samples)
        finally:
            avi_writer.close()

    return counts, planes, coefficients, jpeg_frame


# Compatibility name for callers that used the former entrypoint spelling.
mandelbrot_jpeg_master = mandelbrot_recording_program


def mandelbrot_recording_function_ast() -> ast.Module:
    """Load only the saved, explicit AbstractTensor program into Python AST."""

    source = textwrap.dedent(inspect.getsource(mandelbrot_recording_program))
    return ast.parse(
        source,
        filename=str(Path(__file__).resolve()),
    )


def mandelbrot_display_function_ast() -> ast.Module:
    """Collect the complete solve/display/encode entrypoint program AST."""

    from .demo_mandelbrot_fusion import (
        mandelbrot_jpeg_planes,
        parametric_mandelbrot_escape,
    )

    source = "\n\n".join(
        textwrap.dedent(inspect.getsource(function))
        for function in (
            parametric_mandelbrot_escape,
            mandelbrot_jpeg_planes,
            mandelbrot_display_master,
            mandelbrot_recording_program,
        )
    )
    return ast.parse(source, filename=str(Path(__file__).resolve()))


def build_mandelbrot_encoder_process_graph(
    *,
    profile: str = "tensor_control",
    entrypoint: str = "mandelbrot_recording_program",
):
    """Build the complete structural solve/JPEG/AVI ProcessGraph."""

    from ....transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    # The legacy structural importer still prints its node walk
    # unconditionally.  That diagnostic is not part of this demo's compiler
    # contract, so keep it out of the live render/profile stream.
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(mandelbrot_display_function_ast())
    graph.G.graph.update(
        program_name="mandelbrot_display",
        program_entrypoint=entrypoint,
        semantic_profile=profile,
    )
    return graph


def build_mandelbrot_recording_process_graph():
    """Ingest solve-through-AVI AST, discovering source parents in place."""

    # Do not assemble a smaller "GPU portion" here and finish the recording in
    # the caller.  This builder must begin at mandelbrot_recording_program and
    # recursively ingest its complete source dependency closure.  A missing
    # lowering is compiler work; it is not permission to move that operation
    # outside the graph.
    from ....transmogrifier.graph.graph_express2 import ProcessGraph
    from ..topological_reducer import reduce_abstract_tensor_topology

    program_ast = mandelbrot_recording_function_ast()
    program_bindings = dict(mandelbrot_recording_program.__globals__)
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = program_bindings
    graph.python_package = __package__
    # The structural importer currently prints its node walk
    # unconditionally. That diagnostic is not part of deployment output.
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            program_ast,
            resolve_unresolved_parents=True,
            parent_include=_is_program_helper,
        )
    graph.G.graph.update(
        program_name="mandelbrot_recording_program",
        program_entrypoint="mandelbrot_recording_program",
        source_kind="python_ast",
        source_scope="solve_through_avi_ingested_parent_closure",
    )
    reduced_graph = reduce_abstract_tensor_topology(graph)
    # The writer class and every method body are present in the ingested AST.
    # Until ClassDef/object-layout lowering exists, only construction crosses
    # the Python runtime boundary; it is still invoked by the ProcessGraph,
    # never supplied as a preconstructed caller-owned object.
    reduced_graph.external_function_table.resolve_imports(
        package=__package__,
    )
    reduced_graph.G.graph["external_runtime_boundaries"] = tuple(
        entry.qualified_name
        for entry in reduced_graph.external_function_table
    )
    return reduced_graph


__all__ = [
    "build_mandelbrot_encoder_process_graph",
    "build_mandelbrot_recording_process_graph",
    "mandelbrot_display_function_ast",
    "mandelbrot_display_master",
    "mandelbrot_jpeg_master",
    "mandelbrot_recording_function_ast",
    "mandelbrot_recording_program_ast_closure",
    "mandelbrot_recording_program",
]
