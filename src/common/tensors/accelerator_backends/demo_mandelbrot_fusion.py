"""Mandelbrot/JPEG demo awaiting a real ProcessGraph-to-GLSL compiler.

The former structural-AST reinterpretation shortcut is deliberately removed.
The mathematical, compression, and rendering helpers remain here, but GLSL
compilation must not resume until it consumes the ProcessGraph's scheduled
operation and control nodes directly.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
import os
import time
from functools import lru_cache
from pathlib import Path
import sys

import ctypes

import numpy as np


_ControlBands = namedtuple(
    "_ControlBands",
    ("loudness", "bass", "low_mid", "high_mid", "treble"),
)


class _ProceduralControlStream:
    """Audio-compatible controls for recordings that provide no audio file."""

    def __init__(self, duration: float, sample_rate: int = 48_000):
        self.sample_rate = int(sample_rate)
        self.path = None
        self.samples = np.zeros(
            max(1, int(np.ceil(float(duration) * self.sample_rate))),
            dtype=np.float32,
        )

    def sample(self, logical_time: float):
        logical_time = float(logical_time)
        return _ControlBands(
            0.38 + 0.22 * np.sin(logical_time * 1.31),
            0.5 + 0.5 * np.sin(logical_time * 0.83),
            0.5 + 0.5 * np.sin(logical_time * 0.57 + 0.8),
            0.5 + 0.5 * np.sin(logical_time * 1.07 + 1.7),
            0.5 + 0.5 * np.sin(logical_time * 1.73 + 2.2),
        )

    def close(self):
        return None


def _open_control_stream(audio_path, *, gain, duration):
    if audio_path is None:
        return _ProceduralControlStream(duration)
    pluck = Path(__file__).resolve().parents[5] / "spectral-analyzer"
    if str(pluck) not in sys.path:
        sys.path.insert(0, str(pluck))
    from audio_reactive_controls import AudioReactiveControlStream

    return AudioReactiveControlStream(audio_path, gain=gain)


# ---------------------------------------------------------------------------
# the program
# ---------------------------------------------------------------------------

# Escaped orbits diverge without bound. In float32 they reach inf within a few
# more iterations, then inf-inf produces NaN, and the two backends round that
# boundary differently -- the first version of this demo agreed with numpy on
# only 99.9577% of pixels, max |diff| = 18, purely from overflow timing.
#
# Pinning |zx|,|zy| to ORBIT_CLAMP fixes it exactly. Points still inside the set
# satisfy |z| <= 2 by definition, so clamping at 1e18 cannot affect them; escaped
# points freeze at a magnitude whose square (1e36) is still finite in float32 and
# still far above the escape radius, so their count stays frozen either way.
# minimum/maximum are in both backends' vocabularies already.
ORBIT_CLAMP = 1e18


def mandelbrot_escape(cx, cy, iterations: int, clamp: float = ORBIT_CLAMP):
    """Ordinary backend-agnostic AbstractTensor Mandelbrot computation."""

    zx = cx * 0.0
    zy = cx * 0.0
    count = cx * 0.0
    clamp_value = cx * 0.0 + clamp
    for _ in range(iterations):
        zx2, zy2 = zx * zx, zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        zx, zy = zx2 - zy2 + cx, 2.0 * zx * zy + cy
        zx = zx.minimum(clamp_value).maximum(-clamp_value)
        zy = zy.minimum(clamp_value).maximum(-clamp_value)
    return count


def parametric_mandelbrot_escape(
    unit_x,
    unit_y,
    center_x,
    center_y,
    span,
    family_mix,
    julia_x,
    julia_y,
    iterations: int,
    clamp: float = ORBIT_CLAMP,
):
    """Continuous Mandelbrot-to-Julia quadratic-family solve.

    Complex values remain paired real/imaginary AbstractTensors. This is
    algebraically the complex recurrence while retaining the scalar canonical
    operator vocabulary understood by every lowering target.
    """
    cx = center_x + unit_x * span
    cy = center_y + unit_y * span
    zx = cx * family_mix
    zy = cy * family_mix
    constant_x = cx + family_mix * (julia_x - cx)
    constant_y = cy + family_mix * (julia_y - cy)
    count = cx * 0.0
    clamp_value = cx * 0.0 + clamp
    for _ in range(iterations):
        zx2, zy2 = zx * zx, zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        zx, zy = zx2 - zy2 + constant_x, 2.0 * zx * zy + constant_y
        zx = zx.minimum(clamp_value).maximum(-clamp_value)
        zy = zy.minimum(clamp_value).maximum(-clamp_value)
    return count


def capture_mandelbrot(cx: np.ndarray, cy: np.ndarray, iterations: int):
    """Reject the retired execution-tape compiler shortcut."""

    raise RuntimeError(
        "Mandelbrot GradTape capture is disabled; compile the complete "
        "program through AST -> ProcessGraph instead"
    )


def capture_parametric_mandelbrot(iterations: int):
    """Reject the retired execution-tape compiler shortcut."""

    raise RuntimeError(
        "Parametric Mandelbrot GradTape capture is disabled; compile the "
        "complete program through AST -> ProcessGraph instead"
    )


def mandelbrot_jpeg_planes(
    counts,
    iterations: int,
    palette_phase,
    color_drive,
):
    """Compose the display palette and JPEG 4:4:4 planes elementwise.

    This is ordinary AbstractTensor math. A ProcessGraph backend optimizer may
    keep it beside the shared Mandelbrot producer and expose count/Y/Cb/Cr as
    four outputs of one dispatch.
    """

    phase = (
        (counts / max(iterations, 1)).minimum(1.0).maximum(0.0).sqrt()
        + palette_phase
    )
    drive = color_drive.minimum(1.0).maximum(0.0)
    exponent = 1.65 + (0.62 - 1.65) * drive

    def channel(offset: float):
        wave = (
            0.5
            + 0.5
            * (6.283185307179586 * (phase + offset)).cos()
        ) ** exponent
        return ((wave * 255.0 + 0.5) // 1).minimum(255.0).maximum(0.0)

    red = channel(0.0)
    green = channel(0.21)
    blue = channel(0.43)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    blue_difference = (
        -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0
    )
    red_difference = (
        0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0
    )
    return (
        luminance.minimum(255.0).maximum(0.0),
        blue_difference.minimum(255.0).maximum(0.0),
        red_difference.minimum(255.0).maximum(0.0),
    )


def capture_parametric_mandelbrot_encoder(iterations: int):
    """Reject the partial tape capture previously presented as compilation."""

    raise RuntimeError(
        "Partial Mandelbrot/JPEG GradTape capture is disabled; the complete "
        "encoder must enter through AST -> ProcessGraph before lowering"
    )


def build_parametric_mandelbrot_glsl_deployment(
    iterations: int,
    *,
    profiling: bool = False,
    verbose_profile: bool = False,
    entrypoint: str = "mandelbrot_recording_program",
    legacy_fused_network: bool = False,
):
    """Plan and select an ingested Mandelbrot function's GLSL shell."""

    from ....compiler.glsl_deployment_strategy import (
        strategize_shell_deployment,
    )
    from .mandelbrot_encoder_program import (
        build_mandelbrot_recording_process_graph,
    )

    graph = build_mandelbrot_recording_process_graph(
        profile_verbose=verbose_profile,
    )
    module_shell_type = strategize_shell_deployment(graph)
    module_shell = module_shell_type(
        iterations=int(iterations),
        profiling=profiling,
        verbose_profile=verbose_profile,
        legacy_fused_network=legacy_fused_network,
    )
    reference = graph.function_table.reference(entrypoint)
    if reference is None:
        raise RuntimeError(
            f"Mandelbrot ProcessGraph entrypoint {entrypoint!r} is undeclared"
        )
    try:
        deployment = module_shell.function_shells[reference.address]
    except KeyError as exc:
        raise RuntimeError(
            f"Mandelbrot ProcessGraph entrypoint {entrypoint!r} has no "
            "deployment shell"
        ) from exc
    deployment.module_shell = module_shell
    deployment.entry_reference = reference
    deployment.refresh_hierarchy_plan()
    return deployment, graph


def compile_parametric_mandelbrot_glsl(iterations: int):
    """Compatibility spelling for the deployment-shell construction stage."""

    return build_parametric_mandelbrot_glsl_deployment(iterations)


def run_abstract_numpy(cx: np.ndarray, cy: np.ndarray, iterations: int):
    """Run the exact same AbstractTensor function on the NumPy backend."""

    from ..numpy_backend import NumPyTensorOperations

    result = mandelbrot_escape(
        NumPyTensorOperations.tensor(cx),
        NumPyTensorOperations.tensor(cy),
        iterations,
    )
    return np.asarray(result.tolist(), dtype=cx.dtype)


def run_abstract_backend(
    backend: str,
    cx: np.ndarray,
    cy: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Run the ordinary tensor program under one selected CPU backend."""

    from ..abstraction import AbstractTensor

    with AbstractTensor.use_backend(backend):
        result = mandelbrot_escape(
            AbstractTensor.tensor(cx),
            AbstractTensor.tensor(cy),
            iterations,
        )
    payload = getattr(result, "data", result)
    if hasattr(payload, "tolist"):
        return np.asarray(payload.tolist(), dtype=np.float32)
    return np.asarray(result.tolist(), dtype=np.float32)


@lru_cache(maxsize=None)
def _compiled_c_mandelbrot_shell():
    """Compile the complete CPU Mandelbrot control loop as one C function."""

    from ....compiler.control_source import (
        ControlProgram,
        ControlTarget,
        LoopBlock,
        RegionCode,
        SequenceBlock,
        StatementBlock,
        compile_cffi_shell,
    )

    logical = ControlProgram(
        SequenceBlock((
            StatementBlock((
                "for (int element = 0; element < count; ++element) {",
                "    zx[element] = 0.0f;",
                "    zy[element] = 0.0f;",
                "    output[element] = 0.0f;",
                "}",
            )),
            LoopBlock(
                "iteration",
                "0",
                "iterations",
                "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
        )),
        region_indices=(0,),
    )
    return compile_cffi_shell(
        logical,
        (RegionCode(
            0,
            ControlTarget.C,
            StatementBlock((
                "for (int element = 0; element < count; ++element) {",
                "    float old_x = zx[element];",
                "    float old_y = zy[element];",
                "    float x2 = old_x * old_x;",
                "    float y2 = old_y * old_y;",
                "    output[element] += (x2 + y2 <= 4.0f);",
                "    float next_x = x2 - y2 + cx[element];",
                "    float next_y = 2.0f * old_x * old_y + cy[element];",
                "    zx[element] = fmaxf(-clamp, fminf(clamp, next_x));",
                "    zy[element] = fmaxf(-clamp, fminf(clamp, next_y));",
                "}",
            )),
        ),),
        function_name="mandelbrot_c_shell",
        parameters=(
            "const float *cx",
            "const float *cy",
            "float *zx",
            "float *zy",
            "float *output",
            "int count",
            "int iterations",
            "float clamp",
        ),
        c_declaration=(
            "void mandelbrot_c_shell("
            "const float *cx, const float *cy, float *zx, float *zy, "
            "float *output, int count, int iterations, float clamp);"
        ),
        preamble="#include <math.h>",
        extra_compile_args=(
            ("/fp:strict",)
            if os.name == "nt"
            else ("-ffp-contract=off",)
        ),
    )


def run_compiled_c_shell(
    cx: np.ndarray,
    cy: np.ndarray,
    iterations: int,
) -> np.ndarray:
    compiled = _compiled_c_mandelbrot_shell()
    cx = np.ascontiguousarray(cx, dtype=np.float32)
    cy = np.ascontiguousarray(cy, dtype=np.float32)
    zx = np.empty_like(cx)
    zy = np.empty_like(cy)
    output = np.empty_like(cx)
    pointer = lambda array: compiled.ffi.cast(
        "float *", array.ctypes.data
    )
    compiled(
        compiled.ffi.cast("const float *", cx.ctypes.data),
        compiled.ffi.cast("const float *", cy.ctypes.data),
        pointer(zx),
        pointer(zy),
        pointer(output),
        cx.size,
        int(iterations),
        np.float32(ORBIT_CLAMP),
    )
    return output


# =====================================================================
# NOTE: NO SIMULTANEOUS TUPLE ASSIGNMENT IN THIS LOOP -- ON PURPOSE.
# =====================================================================
# The obvious way to write a Mandelbrot step is
#
#     zx, zy = zx2 - zy2 + cx, 2.0 * zx * zy + cy
#
# and that is what this source used to say. It does not lower: a tuple
# assignment binds each name to a tuple temporary, so the loop's carried
# update names that temporary instead of a value produced in the body, and
# lower_precompile_and_control_to_ssa fails with
#
#     carried update value N has no producer inside the loop body
#
# The loop is still discovered and the ControlProgram still contains a real
# LoopBlock, so the failure looks unrelated to the assignment. Each carried
# name is therefore assigned exactly once per iteration, from a named value
# computed in the body. Same computation; see aot_compile.py's docstring.
# Do not "simplify" this back into a tuple assignment.
_MANDELBROT_FORTRAN_AOT_SOURCE = f"""
def mandelbrot_escape_aot(cx, cy, iterations):
    zx = cx * 0.0
    zy = cx * 0.0
    count = cx * 0.0
    clamp_value = cx * 0.0 + {ORBIT_CLAMP}
    for _ in range(iterations):
        zx2 = zx * zx
        zy2 = zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        next_zx = zx2 - zy2 + cx
        next_zy = 2.0 * zx * zy + cy
        zx = next_zx.minimum(clamp_value).maximum(-clamp_value)
        zy = next_zy.minimum(clamp_value).maximum(-clamp_value)
    return count


def run_mandelbrot_aot(cx, cy, iterations):
    return mandelbrot_escape_aot(cx, cy, iterations)
"""


def _values_by_id(function):
    # A function's outputs must be named, or its results stay dead locals
    # instead of becoming intent(out) arguments (ssa_fortran_backend.py).
    values = {value.id: value for value in function.args}
    for block in function.blocks.values():
        for instr in block.instrs:
            if instr.res is not None:
                values[instr.res.id] = instr.res
    return values


def _aot_compile_mandelbrot_fortran(count: int, iterations: int, out_path: Path):
    from ....transmogrifier.ssa import IRModule
    from .aot_compile import compile_ast_aot
    from ....compiler.precompile_to_ssa import lower_precompile_and_control_to_ssa
    from ....compiler.ssa_fortran_backend import emit_module, compile_module

    placeholder = np.zeros(count, dtype=np.float32)
    feeds = {
        "cx": placeholder,
        "cy": placeholder,
        "iterations": int(iterations),
    }
    # precompile_only: this route wants the backend-agnostic
    # FusedProgram/ControlProgram pair, not a GLSL emission and execution.
    aot = compile_ast_aot(
        _MANDELBROT_FORTRAN_AOT_SOURCE,
        "run_mandelbrot_aot",
        feeds,
        backend="fortran",
        precompile_only=True,
    )
    numerical_name = "mandelbrot_fortran_numerical"
    control_name = "mandelbrot_fortran_control"
    if aot.shell_control_program is None:
        raise RuntimeError(
            "the AOT run produced no control program, so the iteration loop "
            "was not captured; a Fortran shell built from that would be one "
            "unrolled step rather than the loop"
        )
    lowering = lower_precompile_and_control_to_ssa(
        aot.compiled_shell_program,
        aot.shell_control_program,
        region_programs=aot.region_programs,
        hierarchy_plan=getattr(aot, "hierarchy_plan", None),
        numerical_name=numerical_name,
        control_name=control_name,
    )
    if not lowering.complete:
        raise RuntimeError(
            "Fortran AOT lowering incomplete: " + lowering.shortfall_report()
        )

    program = getattr(aot.compiled_shell_program, "program", aot.compiled_shell_program)
    values = _values_by_id(lowering.module.functions[numerical_name])
    # The loop's result is carried in the control subroutine, not the
    # numerical one -- the numerical body is a single iteration. Naming those
    # values as control outputs is what turns them into intent(out) dummies;
    # without it the control subroutine computes the answer into a local and
    # the caller has nothing to read.
    control_values = _values_by_id(lowering.module.functions[control_name])
    emit_outputs = {
        numerical_name: [
            values[int(value_id)]
            for value_id in program.outputs.values()
            if int(value_id) in values
        ],
        control_name: [
            control_values[int(value_id)]
            for value_id in program.outputs.values()
            if int(value_id) in control_values
        ],
    }
    # The control subroutine calls one numerical_region_N per scheduled
    # region, so those must travel with it; emitting only the numerical and
    # control pair produced source that compiled and then failed to link
    # with "undefined reference to numerical_region_0_".
    region_function_names = tuple(
        name
        for name in lowering.module.functions
        if name.startswith("numerical_region_")
    )
    program_module = IRModule({
        name: lowering.module.functions[name]
        for name in (numerical_name, control_name, *region_function_names)
    })
    module = emit_module(
        program_module,
        # compile_module names the .f90/.dll from the module name, not from
        # out_path, so this is what actually keys the artifact on disk. It
        # must carry the specialization or one build overwrites another's
        # library -- and a loaded DLL then blocks the next link.
        name=out_path.stem,
        outputs=emit_outputs,
    )
    if not module.complete:
        raise RuntimeError(
            "Fortran emission incomplete: "
            + "; ".join(s.format() for s in module.shortfalls)
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(module.source, encoding="utf-8")
    library_path = compile_module(module, directory=out_path.parent)
    return ctypes.CDLL(str(library_path))


def _fortran_artifact_path(count: int, iterations: int) -> Path:
    """Where the emitted .f90 (and, beside it, the .api.yaml) lives.

    Keyed by count and iterations because the emitted program is specialized
    to both -- extents are compiled into the signature and the iteration
    bound is a control uniform of this build. A single fixed path let one
    configuration silently overwrite another's artifact, and left the loaded
    DLL locked against the next compile.

    Absolute, because compile_module runs the Fortran compiler as a
    subprocess and a relative cache path is only correct while the working
    directory happens to be the repository root.
    """

    return (
        Path(__file__).resolve().parents[4]
        / ".turing-cache"
        / f"mandelbrot_fortran_aot_n{int(count)}_i{int(iterations)}.f90"
    )


@lru_cache(maxsize=None)
def _compiled_fortran_mandelbrot_library(count: int, iterations: int):
    out_path = _fortran_artifact_path(count, iterations)
    library_path = out_path.with_suffix(".dll" if sys.platform == "win32" else ".so")
    descriptor_path = out_path.with_suffix(".api.yaml")
    if library_path.is_file() and descriptor_path.is_file():
        # Reuse the artifact rather than recompiling on every process. The
        # lru_cache above is per-process only, so without this each run
        # relinked -- and a library already loaded elsewhere holds the file
        # open, which fails the link with "Permission denied" rather than
        # anything that names the real cause.
        return ctypes.CDLL(str(library_path))
    return _aot_compile_mandelbrot_fortran(count, iterations, out_path)


def run_compiled_fortran_shell(cx: np.ndarray, cy: np.ndarray, iterations: int) -> np.ndarray:
    """Call the compiled program through its emitted API descriptor.

    Nothing here is hardcoded about the signature. The descriptor written
    beside the artifact (``*.api.yaml``, see compiler/compiled_program_api.py)
    names the entry point, the argument order, each argument's element type,
    and whether it is passed by value or by reference -- all of which this
    demo previously guessed, and got wrong in three separate ways at once.
    """

    import yaml

    count = cx.size
    lib = _compiled_fortran_mandelbrot_library(count, iterations)
    descriptor_path = _fortran_artifact_path(count, iterations).with_suffix(
        ".api.yaml"
    )
    api = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    entry_name = api["entry"]
    entry = next(e for e in api["entry_points"] if e["name"] == entry_name)

    outputs = [p for p in entry["parameters"] if p["role"] == "output"]
    if not outputs:
        raise NotImplementedError(
            f"{entry_name} declares no output parameter, so the iteration "
            "result cannot be read back. The loop is captured and the "
            "program compiles and links, but the control subroutine's "
            "loop-carried result is still a local rather than an "
            "intent(out) argument -- emit_module needs to be told which "
            "control values are outputs. Descriptor: " + str(descriptor_path)
        )

    ctypes_by_name = {
        "c_float": ctypes.c_float, "c_double": ctypes.c_double,
        "c_int32": ctypes.c_int32, "c_int64": ctypes.c_int64,
        "c_bool": ctypes.c_bool,
    }
    function = getattr(lib, entry["symbol"])
    function.restype = None

    supplied = {"cx": cx, "cy": cy}
    feed_arrays = [np.ascontiguousarray(v.reshape(-1)) for v in supplied.values()]
    arguments = []
    argument_types = []
    feeds = iter(feed_arrays)
    results: list[np.ndarray] = []
    for parameter in entry["parameters"]:
        element = ctypes_by_name[parameter["ctypes"]]
        if parameter["role"] == "extent":
            arguments.append(element(count))
            argument_types.append(element)
        elif parameter["passing"] == "value":
            # A by-value scalar input is the iteration count uniform.
            arguments.append(element(iterations))
            argument_types.append(element)
        elif parameter["role"] == "input":
            array = next(feeds).astype(
                np.float32 if parameter["ctypes"] == "c_float" else np.float64,
                copy=False,
            )
            arguments.append(array.ctypes.data_as(ctypes.POINTER(element)))
            argument_types.append(ctypes.POINTER(element))
        else:
            extent = int(np.prod(parameter.get("shape") or (count,)))
            buffer = np.zeros(
                extent,
                dtype=np.float32 if parameter["ctypes"] == "c_float" else np.float64,
            )
            results.append(buffer)
            arguments.append(buffer.ctypes.data_as(ctypes.POINTER(element)))
            argument_types.append(ctypes.POINTER(element))

    function.argtypes = argument_types
    function(*arguments)
    return results[0].reshape(cx.shape)


def benchmark_cpu_mandelbrot(
    *,
    shell: str,
    backend: str,
    width: int,
    height: int,
    iterations: int,
    center: complex,
    span: float,
    repeats: int,
) -> tuple[np.ndarray, tuple[float, ...]]:
    cx, cy = complex_plane(width, height, center, span)
    if shell == "python":
        execute = lambda: run_abstract_backend(backend, cx, cy, iterations)
    elif shell == "c":
        if backend != "c":
            raise ValueError("the C shell requires --backend c")
        execute = lambda: run_compiled_c_shell(cx, cy, iterations)
    elif shell == "fortran":
        if backend != "fortran":
            raise ValueError("the Fortran shell requires --backend fortran")
        execute = lambda: run_compiled_fortran_shell(cx, cy, iterations)
    else:
        raise ValueError(f"{shell!r} is not a CPU shell")
    # Materialize imports, backend registries, and CFFI compilation before
    # measuring steady-state execution.  Mixing compilation into the first
    # sample makes backend comparisons meaningless.
    result = execute()
    timings = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        result = execute()
        timings.append((time.perf_counter() - started) * 1e3)
    return result, tuple(timings)


def complex_plane(width: int, height: int, center: complex, span: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Row-major cx/cy grids, flattened, float32."""
    aspect = width / height
    xs = np.linspace(center.real - span * aspect / 2,
                     center.real + span * aspect / 2, width, dtype=np.float32)
    ys = np.linspace(center.imag - span / 2,
                     center.imag + span / 2, height, dtype=np.float32)
    cx, cy = np.meshgrid(xs, ys)
    return (np.ascontiguousarray(cx.ravel()),
            np.ascontiguousarray(cy.ravel()))


def normalized_plane(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Dimensionless pixel coordinates consumed by the parametric solve."""
    aspect = width / height
    xs = np.linspace(-0.5 * aspect, 0.5 * aspect, width, dtype=np.float32)
    ys = np.linspace(-0.5, 0.5, height, dtype=np.float32)
    unit_x, unit_y = np.meshgrid(xs, ys)
    return (
        np.ascontiguousarray(unit_x.ravel()),
        np.ascontiguousarray(unit_y.ravel()),
    )


def animated_camera(
    center: complex, span: float, phase: float
) -> tuple[complex, float]:
    """A visibly varied but continuous tour around the requested view.

    The exponential span modulation covers roughly an order of magnitude,
    while incommensurate lateral frequencies avoid a short repeating orbit.
    """
    phase = float(phase)
    log_zoom = (
        1.25 * np.sin(0.71 * phase)
        + 0.45 * np.sin(1.93 * phase)
    )
    animated_span = float(span) * float(np.exp(log_zoom))
    dx = float(span) * (
        0.58 * np.sin(0.83 * phase)
        + 0.22 * np.sin(2.17 * phase)
    )
    dy = float(span) * (
        0.48 * (np.sin(0.97 * phase + 0.61) - np.sin(0.61))
        + 0.19 * np.sin(1.67 * phase)
    )
    return center + complex(dx, dy), animated_span


def dream_parameters(
    center: complex,
    span: float,
    travel: float,
    *,
    bass: float,
    low_mid: float,
    high_mid: float,
    reaction: float,
    zoom_rate: float,
) -> tuple[complex, float, float, complex]:
    """Map restrained spectral controls into a detailed complex-family view."""
    reaction = max(0.0, float(reaction))
    # Keep this particular tour in the Mandelbrot-detailed portion of the
    # continuous family. Larger family excursions need a different camera
    # chart; applying them to a deep Mandelbrot view erases its structure.
    family_mix = 0.04 + 0.18 * (
        0.5 + 0.5 * np.sin(
        0.24 * travel + reaction * 0.38 * (low_mid - 0.5)
        )
    )

    # c = mu/2 - mu^2/4 parameterizes the Mandelbrot main cardioid. Keeping
    # |mu| < 1 produces connected Julia sets instead of mostly empty dust.
    mu_radius = np.clip(
        0.58 + reaction * 0.08 * (low_mid - 0.5), 0.46, 0.72
    )
    mu_angle = 0.31 * travel + reaction * 0.42 * (high_mid - 0.5)
    mu = mu_radius * np.exp(1j * mu_angle)
    julia_constant = 0.5 * mu - 0.25 * mu * mu

    mandelbrot_center, mandelbrot_span = animated_camera(
        center, span, travel
    )
    # Preserve the detailed target c-plane exactly under the family transform:
    # (1-mix)*pixel + mix*julia == mandelbrot_pixel.
    family_scale = max(1.0 - family_mix, 1e-6)
    animated_center = (
        mandelbrot_center - family_mix * julia_constant
    ) / family_scale
    animated_span = float(np.exp(
        np.log(max(mandelbrot_span / family_scale, 1e-15))
        - zoom_rate * travel
        + reaction * 0.08 * (0.5 - bass)
    ))
    return animated_center, animated_span, float(family_mix), julia_constant


def detail_state_features(
    center: complex,
    span: float,
    travels: np.ndarray,
    *,
    bass: np.ndarray,
    low_mid: np.ndarray,
    high_mid: np.ndarray,
    reaction: float,
    zoom_rate: float,
) -> tuple[np.ndarray, list[tuple[complex, float, float, complex]]]:
    """Describe candidate camera states for the learned detail controller."""
    from .mandelbrot_detail_network import dream_features

    travels = np.asarray(travels, dtype=np.float64)
    bass = np.broadcast_to(np.asarray(bass, dtype=np.float64), travels.shape)
    low_mid = np.broadcast_to(
        np.asarray(low_mid, dtype=np.float64), travels.shape
    )
    high_mid = np.broadcast_to(
        np.asarray(high_mid, dtype=np.float64), travels.shape
    )
    states = [
        dream_parameters(
            center,
            span,
            float(travel),
            bass=float(b),
            low_mid=float(lm),
            high_mid=float(hm),
            reaction=reaction,
            zoom_rate=zoom_rate,
        )
        for travel, b, lm, hm in zip(travels, bass, low_mid, high_mid)
    ]
    return (
        dream_features(
            travels,
            bass,
            low_mid,
            high_mid,
            np.asarray([state[1] for state in states]),
            np.asarray([state[2] for state in states]),
        ),
        states,
    )


def build_detail_controller(
    center: complex,
    span: float,
    *,
    iterations: int,
    samples: int,
    epochs: int,
    reaction: float,
    zoom_rate: float,
):
    """Train AbstractNN on batched low-resolution AbstractTensor solves."""
    from ..autograd import autograd
    from ..numpy_backend import NumPyTensorOperations as NT
    from .mandelbrot_detail_network import (
        detail_scores,
        train_detail_controller,
    )

    travels = np.linspace(0.0, 36.0, samples, endpoint=False)
    bass = 0.5 + 0.5 * np.sin(0.83 * travels + 0.2)
    low_mid = 0.5 + 0.5 * np.sin(0.57 * travels + 0.8)
    high_mid = 0.5 + 0.5 * np.sin(0.73 * travels + 1.7)
    features, states = detail_state_features(
        center,
        span,
        travels,
        bass=bass,
        low_mid=low_mid,
        high_mid=high_mid,
        reaction=reaction,
        zoom_rate=zoom_rate,
    )

    train_width, train_height = 48, 30
    unit_x, unit_y = normalized_plane(train_width, train_height)
    unit_x = np.broadcast_to(unit_x, (samples, unit_x.size)).copy()
    unit_y = np.broadcast_to(unit_y, (samples, unit_y.size)).copy()
    as_column = lambda values: np.asarray(values, dtype=np.float32)[:, None]
    with autograd.no_grad():
        field = parametric_mandelbrot_escape(
            NT.tensor(unit_x),
            NT.tensor(unit_y),
            NT.tensor(as_column([state[0].real for state in states])),
            NT.tensor(as_column([state[0].imag for state in states])),
            NT.tensor(as_column([state[1] for state in states])),
            NT.tensor(as_column([state[2] for state in states])),
            NT.tensor(as_column([state[3].real for state in states])),
            NT.tensor(as_column([state[3].imag for state in states])),
            min(iterations, 40),
        )
    fields = np.asarray(field.tolist(), dtype=np.float32).reshape(
        samples, train_height, train_width
    )
    scores = detail_scores(fields, min(iterations, 40))
    return train_detail_controller(features, scores, epochs=epochs), scores


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

def _replacement_feeds(captured, cx, cy):
    """Bind replacement arrays by matching the two captured root identities."""

    feed_ids = list(captured.feeds)
    if len(feed_ids) != 2:
        raise ValueError(f"expected cx/cy capture roots, found {len(feed_ids)}")
    return {feed_ids[0]: cx, feed_ids[1]: cy}


def run_glsl(captured, cx, cy):
    from .glsl_backend import execute_program

    return execute_program(
        captured.program, _replacement_feeds(captured, cx, cy)
    ).numpy()


def run_glsl_frame_batch(
    program,
    roles,
    unit_x,
    unit_y,
    *,
    centers,
    spans,
    family_mixes,
    julia_constants,
):
    """Solve an outer batch of animation frames in one fused GLSL dispatch.

    The captured program is unchanged. AbstractTensor broadcasting supplies
    the extra axis: coordinates are ``(1, pixels)`` and per-frame controls are
    ``(frames, 1)``, producing a resident ``(frames, pixels)`` result.
    """

    from .glsl_backend import GLChunk, execute_program

    centers = np.asarray(centers, dtype=np.complex64).reshape(-1)
    spans = np.asarray(spans, dtype=np.float32).reshape(-1)
    family_mixes = np.asarray(family_mixes, dtype=np.float32).reshape(-1)
    julia_constants = np.asarray(
        julia_constants, dtype=np.complex64
    ).reshape(-1)
    frame_count = centers.size
    if frame_count < 1:
        raise ValueError("frame batch must contain at least one frame")
    if any(
        values.size != frame_count
        for values in (spans, family_mixes, julia_constants)
    ):
        raise ValueError("all frame-batch controls must have equal length")
    unit_x = np.asarray(unit_x, dtype=np.float32).reshape(1, -1)
    unit_y = np.asarray(unit_y, dtype=np.float32).reshape(1, -1)
    if unit_x.shape != unit_y.shape:
        raise ValueError("unit_x and unit_y must contain equal pixel counts")

    column = lambda values: np.asarray(values, dtype=np.float32).reshape(-1, 1)
    feeds = {
        roles["unit_x"]: GLChunk.from_numpy(unit_x).to_gpu(),
        roles["unit_y"]: GLChunk.from_numpy(unit_y).to_gpu(),
        roles["center_x"]: GLChunk.from_numpy(column(centers.real)).to_gpu(),
        roles["center_y"]: GLChunk.from_numpy(column(centers.imag)).to_gpu(),
        roles["span"]: GLChunk.from_numpy(column(spans)).to_gpu(),
        roles["family_mix"]: GLChunk.from_numpy(
            column(family_mixes)
        ).to_gpu(),
        roles["julia_x"]: GLChunk.from_numpy(
            column(julia_constants.real)
        ).to_gpu(),
        roles["julia_y"]: GLChunk.from_numpy(
            column(julia_constants.imag)
        ).to_gpu(),
    }
    try:
        return execute_program(program, feeds)
    finally:
        for chunk in feeds.values():
            chunk.release()


def run_c(captured, cx, cy):
    from .c_backend import CTensor
    from .c_primitive_program import execute_fused_program

    feeds = {
        feed_id: CTensor.from_list(array.tolist(), array.shape)
        for feed_id, array in _replacement_feeds(captured, cx, cy).items()
    }
    return np.asarray(
        execute_fused_program(captured.program, feeds).tolist(), dtype=np.float32
    )


def c_workspace_bytes(program, elements: int) -> int:
    """The C interpreter allocates one full slot array per instruction result."""
    return (len(program.feeds) + len(program.steps)) * elements * 8


def _retired_display_only_animate_glsl(
    *,
    width: int,
    height: int,
    iterations: int,
    center: complex,
    span: float,
    speed: float = 1.0,
    zoom_rate: float = 0.0,
    audio_path: str | Path | None = None,
    audio_gain: float = 1.0,
    reaction: float = 0.20,
    detail_network: bool = True,
    detail_samples: int = 60,
    detail_epochs: int = 20,
    play_audio: bool = True,
    max_frames: int | None = None,
    batch_size: int = 8,
    timeline_fps: float = 60.0,
    profile: bool = False,
    verbose_profile: bool = False,
) -> None:
    """Removed: animation must execute the complete AVI-producing root."""
    raise RuntimeError(
        "display-only animation was removed; use the complete batched "
        "AVI recording path"
    )

    # Unreachable historical body retained only until the surrounding
    # in-progress compiler work is consolidated.
    import pygame
    from OpenGL import GL
    from OpenGL.GL.shaders import compileProgram, compileShader

    from .gl_context import require_gl_context
    from .glsl_backend import (
        dispatch_stats,
        shader_cache_stats,
    )

    deployment, _ = build_parametric_mandelbrot_glsl_deployment(
        iterations,
        profiling=profile or verbose_profile,
        verbose_profile=verbose_profile,
        entrypoint="mandelbrot_frame_program",
    )
    batch_size = max(1, int(batch_size))
    if timeline_fps <= 0:
        raise ValueError("timeline_fps must be positive")
    unit_x, unit_y = normalized_plane(width, height)

    print(
        f"program : {deployment.source_node_count} ProcessGraph nodes -> "
        f"{deployment.primitive_count} scheduled nodes in "
        f"{deployment.dispatch_count} deployment regions",
        flush=True,
    )
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode(
        (width, height),
        pygame.OPENGL | pygame.DOUBLEBUF,
        vsync=0,
    )
    pygame.display.set_caption("Parametric AbstractTensor Mandelbrot — GLSL")
    info = require_gl_context()
    print(
        f"gpu     : {info['renderer']} (context: {info['source']})",
        flush=True,
    )

    controller = None
    if detail_network:
        controller, measured_scores = build_detail_controller(
            center,
            span,
            iterations=iterations,
            samples=detail_samples,
            epochs=detail_epochs,
            reaction=reaction,
            zoom_rate=zoom_rate,
        )
        print(
            "detail  : AbstractNN/Adam "
            f"{controller.samples} states x {controller.epochs} epochs | "
            f"loss {controller.initial_loss:.4g}->{controller.final_loss:.4g} | "
            f"holdout r={controller.validation_correlation:.3f} | "
            f"measured {measured_scores.min():.2f}..{measured_scores.max():.2f}",
            flush=True,
        )

    audio = None
    audio_playback_ready = False
    if audio_path is not None:
        pluck = Path(__file__).resolve().parents[5] / "spectral-analyzer"
        if str(pluck) not in sys.path:
            sys.path.insert(0, str(pluck))
        from audio_reactive_controls import AudioReactiveControlStream

        audio = AudioReactiveControlStream(audio_path, gain=audio_gain)
        if play_audio:
            try:
                pygame.mixer.init(
                    frequency=audio.sample_rate,
                    size=-16,
                    channels=2,
                    buffer=2048,
                )
                pygame.mixer.music.load(str(audio.path))
                audio_playback_ready = True
            except pygame.error as error:
                print(
                    f"audio   : playback unavailable ({error}); "
                    "analysis continues",
                    flush=True,
                )
        print(
            f"audio   : {audio.path} | {audio.sample_rate} Hz | "
            "fftfree bass/low-mid/high-mid/treble controls",
            flush=True,
        )

    static_feeds = {
        "unit_x": unit_x,
        "unit_y": unit_y,
        "width": width,
        "height": height,
        "iterations": iterations,
    }
    deployment.compile_process_graph()
    deployment.capture_fused_programs({
        **static_feeds,
        "center_x": np.full(batch_size, center.real, dtype=np.float32),
        "center_y": np.full(batch_size, center.imag, dtype=np.float32),
        "span": np.full(batch_size, span, dtype=np.float32),
        "family_mix": np.zeros(batch_size, dtype=np.float32),
        "julia_x": np.full(batch_size, -0.72, dtype=np.float32),
        "julia_y": np.full(batch_size, 0.24, dtype=np.float32),
        "palette_phase": np.zeros(batch_size, dtype=np.float32),
        "color_drive": np.full(batch_size, 0.52, dtype=np.float32),
    })
    deployment.require_ready()
    print(
        f"deploy  : batch {batch_size} x {width} x {height}; "
        f"{len(deployment.captured_region_programs)} resident "
        "CapturedFusedProgram shaders; ProcessGraph-scheduled GLSL execution",
        flush=True,
    )
    display_program = compileProgram(
        compileShader(
            """#version 430 core
            const vec2 corners[3] = vec2[3](
                vec2(-1,-1), vec2(3,-1), vec2(-1,3));
            void main(){ gl_Position=vec4(corners[gl_VertexID],0,1); }""",
            GL.GL_VERTEX_SHADER,
        ),
        compileShader(
            """#version 430 core
            layout(std430, binding=0) readonly buffer RGBFrame { float rgb[]; };
            uniform uint image_width;
            uniform uint image_height;
            uniform uint frame_index;
            out vec4 color;
            void main(){
                uint x=uint(gl_FragCoord.x);
                uint y=uint(gl_FragCoord.y);
                if(x>=image_width || y>=image_height){ color=vec4(0); return; }
                uint frame_stride=image_width*image_height*3u;
                uint index=frame_index*frame_stride+(y*image_width+x)*3u;
                vec3 pixel=vec3(rgb[index],rgb[index+1u],rgb[index+2u])/255.0;
                color=vec4(clamp(pixel,0.0,1.0),1.0);
            }""",
            GL.GL_FRAGMENT_SHADER,
        ),
    )
    vao = int(GL.glGenVertexArrays(1))
    width_location = GL.glGetUniformLocation(display_program, "image_width")
    height_location = GL.glGetUniformLocation(display_program, "image_height")
    frame_location = GL.glGetUniformLocation(display_program, "frame_index")

    if audio_playback_ready:
        pygame.mixer.music.play(loops=-1)
    started = time.perf_counter()
    travel = 0.0
    timeline_frame = 0
    report_started = started
    report_frames = 0
    frame = 0
    dispatch_baseline = dispatch_stats()
    cache_baseline = shader_cache_stats()
    frame_dispatches = 0
    predicted_detail = 1.0
    profile_rows: list[dict[str, float]] = []
    running = True
    try:
        while running and (max_frames is None or frame < max_frames):
            batch_started = time.perf_counter()
            dispatch_before_batch = dispatch_stats()["calls"]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break

            batched_values = {
                name: []
                for name in (
                    "center_x",
                    "center_y",
                    "span",
                    "family_mix",
                    "julia_x",
                    "julia_y",
                    "palette_phase",
                    "color_drive",
                )
            }
            loudness = bass = low_mid = high_mid = treble = 0.0
            animated_span = span
            family_mix = 0.0
            logical_delta = 1.0 / timeline_fps
            for batch_frame in range(batch_size):
                logical_time = (
                    timeline_frame + batch_frame
                ) / timeline_fps
                if audio is not None:
                    controls = audio.sample(logical_time)
                    loudness = controls.loudness
                    bass = controls.bass
                    low_mid = controls.low_mid
                    high_mid = controls.high_mid
                    treble = controls.treble
                else:
                    loudness = (
                        0.38 + 0.22 * np.sin(logical_time * 1.31)
                    )
                    bass = 0.5 + 0.5 * np.sin(logical_time * 0.83)
                    low_mid = (
                        0.5
                        + 0.5 * np.sin(logical_time * 0.57 + 0.8)
                    )
                    high_mid = (
                        0.5
                        + 0.5 * np.sin(logical_time * 0.73 + 1.7)
                    )
                    treble = (
                        0.5
                        + 0.5 * np.sin(logical_time * 1.17 + 0.3)
                    )
                # The camera's path speed is the integral of loudness. The
                # learned controller changes dwell time, while every resulting
                # state is still submitted as one row of the tensor batch.
                detail_speed = 1.0
                if controller is not None:
                    candidates = travel + np.asarray([0.0, 0.45, 0.9])
                    candidate_features, _ = detail_state_features(
                        center,
                        span,
                        candidates,
                        bass=np.full(3, bass),
                        low_mid=np.full(3, low_mid),
                        high_mid=np.full(3, high_mid),
                        reaction=reaction,
                        zoom_rate=zoom_rate,
                    )
                    predicted = controller.predict(candidate_features)
                    predicted_detail = float(predicted[0])
                    best_ahead = float(np.argmax(predicted)) * 0.45
                    detail_speed = (
                        0.45 + 1.8 * (1.0 - predicted_detail)
                        + 0.35 * best_ahead
                    )
                travel += logical_delta * speed * detail_speed * (
                    0.28 + reaction * 1.35 * loudness
                )
                (
                    animated_center,
                    animated_span,
                    family_mix,
                    julia_constant,
                ) = dream_parameters(
                    center,
                    span,
                    travel,
                    bass=bass,
                    low_mid=low_mid,
                    high_mid=high_mid,
                    reaction=reaction,
                    zoom_rate=zoom_rate,
                )
                palette_phase = float(
                    0.028 * travel
                    + reaction * 0.09 * (treble - 0.5)
                )
                color_drive = float(
                    0.52 + reaction * 0.24 * (high_mid - 0.5)
                )
                batched_values["center_x"].append(animated_center.real)
                batched_values["center_y"].append(animated_center.imag)
                batched_values["span"].append(animated_span)
                batched_values["family_mix"].append(family_mix)
                batched_values["julia_x"].append(julia_constant.real)
                batched_values["julia_y"].append(julia_constant.imag)
                batched_values["palette_phase"].append(palette_phase)
                batched_values["color_drive"].append(color_drive)
            timeline_frame += batch_size
            controls_finished = time.perf_counter()
            tensor_controls = {
                name: np.asarray(values, dtype=np.float32)
                for name, values in batched_values.items()
            }
            uploads_finished = time.perf_counter()
            submit_started = time.perf_counter()
            fused_outputs = deployment.execute_named({
                **static_feeds,
                **tensor_controls,
            })
            submit_finished = time.perf_counter()
            shell_report = deployment.profile_report()
            gpu_ms = sum(
                row["gpu_ms"]
                for row in shell_report["rows"]
                if row["section"] in {"dispatch", "external"}
            )
            compute_finished = time.perf_counter()

            present_started = time.perf_counter()
            GL.glUseProgram(display_program)
            GL.glUniform1ui(width_location, width)
            GL.glUniform1ui(height_location, height)
            GL.glBindBufferBase(
                GL.GL_SHADER_STORAGE_BUFFER,
                0,
                fused_outputs["frames"].data.buffer_id,
            )
            frames_to_present = batch_size
            if max_frames is not None:
                frames_to_present = min(
                    frames_to_present,
                    max_frames - frame,
                )
            for batch_frame in range(frames_to_present):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif (
                        event.type == pygame.KEYDOWN
                        and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                if not running:
                    break
                surface = pygame.display.get_surface()
                draw_width, draw_height = surface.get_size()
                GL.glViewport(0, 0, draw_width, draw_height)
                GL.glDisable(GL.GL_DEPTH_TEST)
                GL.glClearColor(0.008, 0.012, 0.028, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                GL.glUniform1ui(frame_location, batch_frame)
                GL.glBindVertexArray(vao)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
                GL.glBindVertexArray(0)
                pygame.display.flip()
                frame += 1
                report_frames += 1
            present_finished = time.perf_counter()
            frame_dispatches = (
                dispatch_stats()["calls"] - dispatch_before_batch
            )
            if profile:
                divisor = max(frames_to_present, 1)
                profile_rows.append(
                    {
                        "control": (
                            controls_finished - batch_started
                        ) * 1e3 / divisor,
                        "uploads": (
                            uploads_finished - controls_finished
                        ) * 1e3 / divisor,
                        "submit": (
                            submit_finished - submit_started
                        ) * 1e3 / divisor,
                        "compute_wait": (
                            compute_finished - submit_started
                        ) * 1e3 / divisor,
                        "gpu": gpu_ms / divisor,
                        "present": (
                            present_finished - present_started
                        ) * 1e3 / divisor,
                        "total": (
                            present_finished - batch_started
                        ) * 1e3 / divisor,
                    }
                )
            now = time.perf_counter()
            if now - report_started >= 0.5:
                fps = report_frames / (now - report_started)
                hot_rows = [
                    row
                    for row in shell_report["rows"]
                    if row["section"] in {"dispatch", "external"}
                ]
                hot = max(
                    hot_rows,
                    key=lambda row: (
                        row["gpu_ms"],
                        row["cpu_ms"],
                    ),
                    default=None,
                )
                hot_caption = (
                    f" | hot {hot['label']} "
                    f"{max(hot['gpu_ms'], hot['cpu_ms']):.2f} ms"
                    if hot is not None
                    else ""
                )
                pygame.display.set_caption(
                    "Parametric AbstractTensor Mandelbrot — GLSL | "
                    f"{fps:.1f} solve+render fps | span {animated_span:.5g} | "
                    f"family {family_mix:.2f} | detail {predicted_detail:.2f} | "
                    f"loud {loudness:.2f} | batch {batch_size} | "
                    f"GL launches {frame_dispatches}"
                    + hot_caption
                )
                report_started, report_frames = now, 0
        elapsed = time.perf_counter() - started
        sink_finalize_seconds = max(
            0.0,
            elapsed - feed_seconds - shell_seconds - sink_seconds,
        )
        print(
            f"animated: {frame} solve+render frames in {elapsed:.3f}s "
            f"({frame / max(elapsed, 1e-9):.1f} fps)",
            flush=True,
        )
        dispatch_final = dispatch_stats()
        cache_final = shader_cache_stats()
        dispatch_count = dispatch_final["calls"] - dispatch_baseline["calls"]
        print(
            "dispatch: "
            f"{dispatch_count} physical GLSL launches "
            f"({dispatch_count / max(frame, 1):.1f}/frame) | "
            f"shader cache "
            f"{cache_final['hits'] - cache_baseline['hits']} hits / "
            f"{cache_final['misses'] - cache_baseline['misses']} misses",
            flush=True,
        )
        if profile_rows:
            warmup = min(5, max(0, len(profile_rows) - 1))
            steady = profile_rows[warmup:] or profile_rows
            print(
                f"profile : steady frames ({warmup} warmup frames excluded)"
            )
            for name in (
                "control",
                "uploads",
                "submit",
                "compute_wait",
                "gpu",
                "present",
                "total",
            ):
                values = np.asarray(
                    [row[name] for row in steady], dtype=np.float64
                )
                print(
                    f"  {name:12s} mean {values.mean():8.3f} ms | "
                    f"p95 {np.quantile(values, 0.95):8.3f} ms",
                    flush=True,
                )
            for line in deployment.profile_lines(window=60):
                print(line, flush=True)
    finally:
        deployment.release()
        if audio is not None:
            audio.close()
        if audio_playback_ready:
            pygame.mixer.music.stop()
        GL.glDeleteVertexArrays(1, (vao,))
        GL.glDeleteProgram(display_program)
        pygame.quit()


def animate_glsl(
    *,
    width: int,
    height: int,
    iterations: int,
    center: complex,
    span: float,
    speed: float = 1.0,
    zoom_rate: float = 0.0,
    audio_path: str | Path | None = None,
    audio_gain: float = 1.0,
    reaction: float = 0.20,
    detail_network: bool = True,
    detail_samples: int = 60,
    detail_epochs: int = 20,
    max_frames: int | None = None,
    batch_size: int = 8,
    timeline_fps: float = 60.0,
    record_avi: str | Path | None = None,
    record_fps: float = 30.0,
    record_pcm_dtype: str = "s16le",
    record_segment_bytes: int = 1 << 30,
    profile: bool = False,
    verbose_profile: bool = False,
    program_table: bool = False,
    legacy_fused_network: bool = False,
) -> Path:
    """Submit one control batch through the complete AVI-producing root."""

    if record_avi is None:
        raise ValueError("batched animation recording requires record_avi")
    if timeline_fps <= 0 or record_fps <= 0:
        raise ValueError("timeline_fps and record_fps must be positive")

    frame_count = max_frames if max_frames is not None else batch_size
    frame_count = max(1, int(frame_count))
    unit_x, unit_y = normalized_plane(width, height)

    from .gl_context import require_gl_context
    from .glsl_backend import shader_cache_stats

    controller = None
    if detail_network:
        controller, measured_scores = build_detail_controller(
            center,
            span,
            iterations=iterations,
            samples=detail_samples,
            epochs=detail_epochs,
            reaction=reaction,
            zoom_rate=zoom_rate,
        )
        print(
            "detail  : AbstractNN/Adam "
            f"{controller.samples} states x {controller.epochs} epochs | "
            f"loss {controller.initial_loss:.4g}->"
            f"{controller.final_loss:.4g} | "
            f"holdout r={controller.validation_correlation:.3f} | "
            f"measured {measured_scores.min():.2f}.."
            f"{measured_scores.max():.2f}",
            flush=True,
        )

    audio = _open_control_stream(
        audio_path,
        gain=audio_gain,
        duration=frame_count / timeline_fps,
    )
    deployment = None
    profile_reported = False
    try:
        controls_by_name = {
            name: []
            for name in (
                "center_x",
                "center_y",
                "span",
                "family_mix",
                "julia_x",
                "julia_y",
                "palette_phase",
                "color_drive",
            )
        }
        travel = 0.0
        predicted_detail = 1.0
        for frame_index in range(frame_count):
            logical_time = frame_index / timeline_fps
            controls = audio.sample(logical_time)
            detail_speed = 1.0
            if controller is not None:
                candidates = travel + np.asarray([0.0, 0.45, 0.9])
                candidate_features, _ = detail_state_features(
                    center,
                    span,
                    candidates,
                    bass=np.full(3, controls.bass),
                    low_mid=np.full(3, controls.low_mid),
                    high_mid=np.full(3, controls.high_mid),
                    reaction=reaction,
                    zoom_rate=zoom_rate,
                )
                predicted = controller.predict(candidate_features)
                predicted_detail = float(predicted[0])
                best_ahead = float(np.argmax(predicted)) * 0.45
                detail_speed = (
                    0.45 + 1.8 * (1.0 - predicted_detail)
                    + 0.35 * best_ahead
                )
            travel += (1.0 / timeline_fps) * speed * detail_speed * (
                0.28 + reaction * 1.35 * controls.loudness
            )
            (
                animated_center,
                animated_span,
                family_mix,
                julia_constant,
            ) = dream_parameters(
                center,
                span,
                travel,
                bass=controls.bass,
                low_mid=controls.low_mid,
                high_mid=controls.high_mid,
                reaction=reaction,
                zoom_rate=zoom_rate,
            )
            controls_by_name["center_x"].append(animated_center.real)
            controls_by_name["center_y"].append(animated_center.imag)
            controls_by_name["span"].append(animated_span)
            controls_by_name["family_mix"].append(family_mix)
            controls_by_name["julia_x"].append(julia_constant.real)
            controls_by_name["julia_y"].append(julia_constant.imag)
            controls_by_name["palette_phase"].append(
                0.028 * travel
                + reaction * 0.09 * (controls.treble - 0.5)
            )
            controls_by_name["color_drive"].append(
                0.52 + reaction * 0.24 * (controls.high_mid - 0.5)
            )

        tensor_controls = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in controls_by_name.items()
        }
        destination = Path(record_avi)
        static_feeds = {
            "unit_x": unit_x,
            "unit_y": unit_y,
            "width": width,
            "height": height,
            "iterations": iterations,
        }

        def batch_feeds(start, stop):
            return {
                **static_feeds,
                **{
                    name: values[start:stop]
                    for name, values in tensor_controls.items()
                },
            }

        deployment, _ = build_parametric_mandelbrot_glsl_deployment(
            iterations,
            profiling=profile or verbose_profile,
            verbose_profile=verbose_profile,
            legacy_fused_network=legacy_fused_network,
            entrypoint="mandelbrot_recording_program",
        )
        planned_shells = {
            id(shell): shell
            for shell in (
                deployment,
                *deployment.function_shells.values(),
            )
        }.values()
        planned_shells = tuple(planned_shells)
        print(
            f"program : {sum(shell.source_node_count for shell in planned_shells)} "
            "ProcessGraph nodes -> "
            f"{sum(shell.primitive_count for shell in planned_shells)} "
            "scheduled nodes in "
            f"{sum(shell.dispatch_count for shell in planned_shells)} "
            f"deployment regions across {len(planned_shells)} function shells",
            flush=True,
        )
        info = require_gl_context()
        print(
            f"gpu     : {info['renderer']} (context: {info['source']})",
            flush=True,
        )
        print(
            f"batch   : {frame_count} x {width} x {height} controls -> "
            f"{destination}",
            flush=True,
        )

        cache_before = shader_cache_stats()
        phase_started = time.perf_counter()
        deployment.compile_process_graph()
        structural_compile_seconds = time.perf_counter() - phase_started
        compile_stop = min(frame_count, max(1, int(batch_size)))
        phase_started = time.perf_counter()
        deployment.capture_fused_programs(batch_feeds(0, compile_stop))
        capture_install_seconds = time.perf_counter() - phase_started
        cache_after = shader_cache_stats()
        print(
            "prepare : "
            f"structural {structural_compile_seconds:.3f}s | "
            f"capture/install {capture_install_seconds:.3f}s | "
            "shader cache "
            f"{cache_after['persistent_hits'] - cache_before['persistent_hits']} "
            "persistent hit(s), "
            f"{cache_after['persistent_misses'] - cache_before['persistent_misses']} "
            "miss(es)",
            flush=True,
        )
        if program_table:
            for line in deployment.program_table_lines():
                print(line, flush=True)

        from ..abstraction import AbstractTensor
        from ..compression.containers.avi import DoubleBufferedAVISink
        from ..compression.pcm import PCMFormat

        pcm_format = PCMFormat(
            sample_rate=audio.sample_rate,
            channels=1,
            sample_format=record_pcm_dtype,
        )
        started = time.perf_counter()
        outputs = None
        feed_seconds = 0.0
        shell_seconds = 0.0
        sink_seconds = 0.0
        with DoubleBufferedAVISink(
            destination,
            width=width,
            height=height,
            fps=record_fps,
            pcm_format=pcm_format,
            audio=AbstractTensor.tensor(audio.samples),
            opendml=True,
            segment_bytes=record_segment_bytes,
        ) as sink:
            for batch_start in range(0, frame_count, batch_size):
                batch_stop = min(frame_count, batch_start + batch_size)
                phase_started = time.perf_counter()
                feeds = batch_feeds(batch_start, batch_stop)
                feed_seconds += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                outputs = deployment.execute_named(feeds)
                shell_seconds += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                sink.submit(outputs["video_packets"])
                sink_seconds += time.perf_counter() - phase_started
        elapsed = time.perf_counter() - started
        avi_output = destination
        if not avi_output.is_file():
            raise RuntimeError(
                f"complete recording root returned without AVI: {avi_output}"
            )
        print(
            f"recorded: {frame_count} frames | "
            f"{avi_output.stat().st_size:,} bytes | "
            f"{elapsed:.3f}s execution | detail {predicted_detail:.3f}",
            flush=True,
        )
        print(
            "host    : "
            f"feeds {feed_seconds:.3f}s | "
            f"installed shell {shell_seconds:.3f}s | "
            f"AVI submit {sink_seconds:.3f}s | "
            f"AVI wait/finalize {sink_finalize_seconds:.3f}s",
            flush=True,
        )
        shell_tree = {
            id(shell): shell
            for shell in (
                deployment,
                *deployment.function_shells.values(),
            )
        }.values()
        print(
            f"compile : {sum(len(shell.captured_region_programs) for shell in shell_tree)} "
            "CapturedFusedProgram shaders; "
            f"{sum(len(shell.coordinator_region_indices) for shell in shell_tree)} "
            "coordinator regions",
            flush=True,
        )
        if profile or verbose_profile:
            for line in deployment.profile_lines(window=60):
                print(line, flush=True)
            profile_reported = True
        return avi_output
    finally:
        if deployment is not None:
            if (profile or verbose_profile) and not profile_reported:
                print(
                    "profile : incomplete run; reporting events captured "
                    "before failure",
                    flush=True,
                )
                for line in deployment.profile_lines(window=60):
                    print(line, flush=True)
                for line in deployment.exception_lines(limit=12):
                    print(line, flush=True)
            deployment.release()
        audio.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--center", type=complex, default=complex(-0.743643887, 0.131825904))
    ap.add_argument("--span", type=float, default=0.004)
    ap.add_argument("--c-probe", type=int, default=48,
                    help="edge length of the small grid cross-checked on the C backend")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument(
        "--shell",
        choices=("python", "c", "fortran", "glsl"),
        default="python",
        help=(
            "control-shell language; defaults to the CPU Python shell"
        ),
    )
    ap.add_argument(
        "--backend",
        choices=("auto", "pure_python", "numpy", "c", "fortran", "glsl"),
        default="auto",
        help=(
            "interior tensor backend; auto selects numpy for Python, c for "
            "C, and glsl for GLSL"
        ),
    )
    ap.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        help="timed repetitions for static non-GLSL execution",
    )
    ap.add_argument(
        "--only-glsl",
        action="store_true",
        help="render with GLSL only; skip the NumPy/f64 oracles and C probe",
    )
    ap.add_argument(
        "--animate",
        action="store_true",
        help="record a finite batch of animated controls through the AVI root",
    )
    ap.add_argument(
        "--animation-speed",
        type=float,
        default=1.0,
        help="camera travel multiplier across the recording timeline",
    )
    ap.add_argument(
        "--animation-frames",
        type=int,
        default=0,
        help="frames in the single recording batch; 0 uses --animation-batch",
    )
    ap.add_argument(
        "--animation-batch",
        type=int,
        default=8,
        help="default complete recording batch size",
    )
    ap.add_argument(
        "--animation-fps",
        type=float,
        default=60.0,
        help="logical camera/audio timeline rate within a submitted batch",
    )
    ap.add_argument(
        "--zoom-rate",
        type=float,
        default=0.0,
        help="positive continuously tightens the view; negative loosens it",
    )
    ap.add_argument(
        "--audio",
        type=Path,
        help="loop an audio file and drive the complex dream path with fftfree",
    )
    ap.add_argument(
        "--audio-gain",
        type=float,
        default=1.0,
        help="gain before adaptive spectral control normalization",
    )
    ap.add_argument(
        "--reaction",
        type=float,
        default=0.20,
        help="audio modulation depth; 0 is the restrained autonomous path",
    )
    ap.add_argument(
        "--no-detail-network",
        action="store_true",
        help="disable the tiny AbstractNN controller that skips bland states",
    )
    ap.add_argument(
        "--detail-samples",
        type=int,
        default=60,
        help="low-resolution dream states used to train the detail controller",
    )
    ap.add_argument(
        "--detail-epochs",
        type=int,
        default=20,
        help="AbstractNN training epochs for the live detail controller",
    )
    ap.add_argument(
        "--silent-audio",
        action="store_true",
        help="analyze --audio without playing it",
    )
    ap.add_argument(
        "--record-avi",
        type=Path,
        help="record the animated GLSL solve as 4:4:4 MJPEG/OpenDML AVI",
    )
    ap.add_argument("--record-fps", type=float, default=30.0)
    ap.add_argument(
        "--record-pcm-dtype",
        choices=("s16le", "f32le"),
        default="s16le",
    )
    ap.add_argument(
        "--record-segment-bytes",
        type=int,
        default=1 << 30,
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="synchronize GPU timer queries and print per-stage timings",
    )
    ap.add_argument(
        "--profile-verbose",
        action="store_true",
        help=(
            "stream routed tensor statistics, loop-carried state, region "
            "activity, timings, and shell errors; intentionally synchronizes "
            "the GPU frequently"
        ),
    )
    ap.add_argument(
        "--program-table",
        action="store_true",
        help=(
            "print the compiled shell/call-site hierarchy and every shader "
            "or coordinator region as console tables"
        ),
    )
    ap.add_argument(
        "--legacy-fused-network",
        action="store_true",
        help=(
            "explicitly use the retired GLSLFusedProgramNetwork during the "
            "composed-control runtime transition"
        ),
    )
    args = ap.parse_args(argv)

    if args.only_glsl:
        args.shell = "glsl"
        args.backend = "glsl"
    if args.backend == "auto":
        args.backend = {
            "python": "numpy",
            "c": "c",
            "fortran": "fortran",
            "glsl": "glsl",
        }[args.shell]
    compatible = {
        "python": {"pure_python", "numpy", "c"},
        "c": {"c"},
        "fortran": {"fortran"},
        "glsl": {"glsl"},
    }
    if args.backend not in compatible[args.shell]:
        ap.error(
            f"--shell {args.shell} does not support --backend {args.backend}; "
            f"choose from {sorted(compatible[args.shell])}"
        )

    if args.shell != "glsl":
        if args.animate:
            ap.error("CPU shell benchmarking does not support --animate")
        result, timings = benchmark_cpu_mandelbrot(
            shell=args.shell,
            backend=args.backend,
            width=args.width,
            height=args.height,
            iterations=args.iterations,
            center=args.center,
            span=args.span,
            repeats=args.benchmark_repeats,
        )
        median_ms = float(np.median(timings))
        print(f"shell   : {args.shell}")
        print(f"backend : {args.backend}")
        print(
            f"problem : {args.width}x{args.height} x "
            f"{args.iterations} iterations"
        )
        print(
            f"timing  : median {median_ms:.3f} ms | "
            f"{1e3 / max(median_ms, 1e-12):.2f} solves/s | "
            f"samples {[round(value, 3) for value in timings]}"
        )
        if not (args.shell == "python" and args.backend == "numpy"):
            cx, cy = complex_plane(
                args.width,
                args.height,
                args.center,
                args.span,
            )
            reference = run_abstract_numpy(cx, cy, args.iterations)
            exact = float(np.mean(result == reference) * 100.0)
            max_error = float(np.max(np.abs(result - reference)))
            print(
                f"parity  : {exact:.4f}% exact vs numpy | "
                f"max |diff|={max_error:g}"
            )
        return 0

    if args.animate:
        if args.record_avi is None:
            ap.error(
                "--animate executes the complete recording root and requires "
                "--record-avi"
            )
        animate_glsl(
            width=args.width,
            height=args.height,
            iterations=args.iterations,
            center=args.center,
            span=args.span,
            speed=args.animation_speed,
            zoom_rate=args.zoom_rate,
            audio_path=args.audio,
            audio_gain=args.audio_gain,
            reaction=args.reaction,
            detail_network=not args.no_detail_network,
            detail_samples=args.detail_samples,
            detail_epochs=args.detail_epochs,
            max_frames=args.animation_frames or None,
            batch_size=args.animation_batch,
            timeline_fps=args.animation_fps,
            record_avi=args.record_avi,
            record_fps=args.record_fps,
            record_pcm_dtype=args.record_pcm_dtype,
            record_segment_bytes=args.record_segment_bytes,
            profile=args.profile or args.profile_verbose,
            verbose_profile=args.profile_verbose,
            program_table=args.program_table,
            legacy_fused_network=args.legacy_fused_network,
        )
        return 0

    if args.record_avi is None:
        ap.error(
            "the complete AbstractTensor program requires --record-avi"
        )

    elements = args.width * args.height
    print(f"image   : {args.width}x{args.height} = {elements:,} pixels")

    deployment, _ = build_parametric_mandelbrot_glsl_deployment(
        args.iterations,
        profiling=args.profile or args.profile_verbose,
        verbose_profile=args.profile_verbose,
        legacy_fused_network=args.legacy_fused_network,
    )
    print(
        f"program : {deployment.source_node_count} ProcessGraph nodes -> "
        f"{deployment.primitive_count} scheduled nodes in "
        f"{deployment.dispatch_count} deployment regions"
    )
    print(f"runtime : {deployment.control_runtime}")
    # -- GPU ---------------------------------------------------------------
    from .gl_context import require_gl_context
    info = require_gl_context()
    print(f"gpu     : {info['renderer']} (context: {info['source']})")

    unit_x, unit_y = normalized_plane(args.width, args.height)
    audio = _open_control_stream(
        args.audio,
        gain=args.audio_gain,
        duration=max(1.0, 1.0 / args.animation_fps),
    )
    static_scalars = {
        "center_x": args.center.real,
        "center_y": args.center.imag,
        "span": args.span,
        "family_mix": 0.0,
        "julia_x": -0.72,
        "julia_y": 0.24,
        "palette_phase": 0.0,
        "color_drive": 0.52,
    }
    # This non-animated path intentionally supplies only data/configuration.
    # It must never supply a writer, frame collector, encoder callback, or
    # partially completed container.  The ingested root owns those operations.
    feeds = {
        "unit_x": unit_x,
        "unit_y": unit_y,
        **{
            name: np.float32(value)
            for name, value in static_scalars.items()
        },
        "width": args.width,
        "height": args.height,
        "iterations": args.iterations,
        "avi_path": args.record_avi,
        "avi_fps": args.record_fps,
        "avi_opendml": True,
        "avi_segment_bytes": args.record_segment_bytes,
        "audio_samples": audio.samples,
        "audio_sample_rate": audio.sample_rate,
        "audio_channels": 1,
        "audio_sample_format": args.record_pcm_dtype,
    }
    try:
        deployment.compile_process_graph()
        deployment.capture_fused_programs(feeds)
        if args.program_table:
            for line in deployment.program_table_lines():
                print(line, flush=True)
        t0 = time.perf_counter()
        recording_outputs = deployment.execute_named(feeds)
        gpu = recording_outputs["counts"].numpy().copy()
        avi_output = recording_outputs["avi_output"]
        gpu_ms = (time.perf_counter() - t0) * 1e3
    finally:
        audio.close()
    print(
        f"glsl    : {gpu_ms:8.1f} ms  "
        f"({args.iterations} loop iterations x {elements:,} px, "
        f"{deployment.dispatch_count} planned regions)"
    )
    print(
        f"compile : {len(deployment.captured_region_programs)} "
        "CapturedFusedProgram shaders; "
        f"{len(deployment.coordinator_region_indices)} scalar/shape "
        "coordinator regions"
    )
    if args.profile or args.profile_verbose:
        for line in deployment.profile_lines():
            print(line)
    deployment.release()

    if not args.only_glsl:
        cx, cy = complex_plane(
            args.width,
            args.height,
            args.center,
            args.span,
        )
        # -- oracle --------------------------------------------------------
        t0 = time.perf_counter()
        ref = run_abstract_numpy(cx, cy, args.iterations)
        np_ms = (time.perf_counter() - t0) * 1e3
        print(f"numpy   : {np_ms:8.1f} ms  (same AbstractTensor function)")

        max_err = float(np.max(np.abs(gpu - ref)))
        agree = float(np.mean(gpu == ref)) * 100.0
        print(f"agree   : {agree:.4f}% exact vs numpy-f32, max |diff| = {max_err:g}")

        # Escape-time is chaotic: a 1-ULP boundary difference can change the
        # escape iteration. Compare both float32 paths with float64 so precision
        # sensitivity is not mistaken for a lowering defect.
        ref64 = run_abstract_numpy(
            cx.astype(np.float64), cy.astype(np.float64), args.iterations
        )
        gpu_vs64 = float(np.mean(gpu == ref64)) * 100.0
        np_vs64 = float(np.mean(ref == ref64)) * 100.0
        print(f"vs f64  : glsl-f32 {gpu_vs64:.4f}%, numpy-f32 {np_vs64:.4f}% "
              f"-- both f32 paths differ from f64 by a comparable margin")
        if max_err > 0:
            disagree = gpu != ref
            c2 = ref.reshape(args.height, args.width)
            edge = np.zeros_like(c2, dtype=bool)
            edge[1:-1, 1:-1] = (
                (c2[1:-1, 1:-1] != c2[:-2, 1:-1])
                | (c2[1:-1, 1:-1] != c2[2:, 1:-1])
                | (c2[1:-1, 1:-1] != c2[1:-1, :-2])
                | (c2[1:-1, 1:-1] != c2[1:-1, 2:])
            )
            on_edge = float(np.mean(edge.ravel()[disagree])) * 100.0
            print(f"        : {disagree.sum():,} disagreeing px, {on_edge:.1f}% of them sit "
                  f"on an escape-count boundary (chaotic sensitivity, not a lowering bug)")
    else:
        print("verify  : skipped (--only-glsl)")

    # -- C backend, on a small grid ----------------------------------------
    if not args.skip_c and not args.only_glsl:
        print(
            "c probe : skipped; the C backend does not yet lower structured "
            "ProcessGraph loops (no tape fallback)"
        )

    print(f"wrote   : {avi_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
