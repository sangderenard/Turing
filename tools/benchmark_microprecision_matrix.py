"""Profile compiled multi-limb signal cores across the backend matrix.

This is deliberately a LANDSCAPE, not a winner-picking benchmark.  Every
requested backend gets a row.  A destination that cannot preserve a precision
section reports the exact compiler obligation it misses; it is never replaced
by LLVM, NumPy, or ordinary float arithmetic merely so the table has a number.

For an executable row the profiler separates four costs that are otherwise
easy to conflate:

* source lowering, backend emission, and native compilation;
* public-ABI preparation/marshalling;
* first launch after preparation;
* warm operator throughput over several batch sizes.

Accuracy has two independent oracles.  ``Fraction`` evaluates the emitted
polynomial exactly and therefore isolates limb arithmetic.  ``mpmath``
evaluates the mathematical function at high precision and therefore includes
range/core approximation error.  Both score the exact SUM of the returned
limbs in local binary64 ULPs; no width-one baseline participates.

The optional Mandelbrot render uses the repository's real ``Precision``
expansion at four limbs, including four-limb coordinates.  ``--mandlebrot``
is accepted as an alias because that is a pleasant typo to make at a shell.

Examples::

    python -u -m tools.benchmark_microprecision_matrix
    python -u -m tools.benchmark_microprecision_matrix --operators sin exp
    python -u -m tools.benchmark_microprecision_matrix --mandelbrot-only \
        --mandelbrot build/microprecision-profile/mandelbrot.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.tensors import signal_symbolic as _proof  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.ir_identities import (  # noqa: E402
    BACKEND_PRECISION_CAPABILITIES,
    FMA_MANDATORY,
    apply_precision_pipeline,
    precision_backend_shortfalls,
)
from src.compiler.ssa_llvm_backend import (  # noqa: E402
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.transmogrifier.ssa import IRModule  # noqa: E402
from tools import benchmark_precision_cores as _legacy  # noqa: E402


ALL_BACKENDS = ("llvm", "c", "fortran", "wasm", "spirv", "glsl", "webgpu")
ALL_OPERATORS = tuple(sorted(_proof.CORE_RADII))
DEFAULT_SIZES = (1, 8, 64, 512, 4_096, 65_536)


@dataclass(frozen=True)
class BuildProfile:
    lower_ns: int
    emit_ns: int
    native_compile_ns: int
    static_scalar_instructions: int
    static_fma_instructions: int
    operation_histogram: dict[str, int]


@dataclass(frozen=True)
class AccuracyProfile:
    samples: int
    correctly_rounded_percent: float
    polynomial_ulp_median: float
    polynomial_ulp_p99: float
    polynomial_ulp_max: float
    true_ulp_median: float
    true_ulp_p99: float
    true_ulp_max: float
    true_bits_median: float


@dataclass(frozen=True)
class ProfileRow:
    operator: str
    width: int
    backend: str
    batch_size: int
    status: str
    stage: str
    reason: str | None
    lower_ns: int | None
    emit_ns: int | None
    native_compile_ns: int | None
    prepare_ns: int | None
    first_launch_ns: int | None
    warm_min_ns: int | None
    warm_median_ns: int | None
    warm_p95_ns: int | None
    warm_ns_per_element: float | None
    warm_ns_per_static_instruction: float | None
    static_scalar_instructions: int | None
    static_fma_instructions: int | None
    correctly_rounded_percent: float | None
    polynomial_ulp_median: float | None
    polynomial_ulp_p99: float | None
    polynomial_ulp_max: float | None
    true_ulp_median: float | None
    true_ulp_p99: float | None
    true_ulp_max: float | None
    true_bits_median: float | None


@dataclass
class CompiledCore:
    module: Any
    native: Any
    roles: dict[str, Any]
    coefficients: tuple[Fraction, ...]
    build: BuildProfile


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[position]


def _operation_histogram(module) -> dict[str, int]:
    """Count the lowered numerical body, excluding wrapper/control traffic."""

    selected = [
        function
        for name, function in module.functions.items()
        if "planned_region" in str(name)
    ]
    if not selected:
        selected = list(module.functions.values())
    ignored = {
        "Const", "GetElementPtr", "Load", "Store", "Ret", "Return",
        "Branch", "CondBranch", "Call", "Phi",
    }
    histogram: dict[str, int] = {}
    for function in selected:
        for block in function.blocks.values():
            for instruction in block.instrs:
                operation = str(instruction.op)
                if operation in ignored:
                    continue
                histogram[operation] = histogram.get(operation, 0) + 1
    return dict(sorted(histogram.items()))


def _prepare_core(
    name: str, width: int, two_product_flavor: str = "fma",
) -> CompiledCore:
    """Lower authored source through universal identities, no backend selected.

    ``two_product_flavor`` picks the exact-product spelling: ``"fma"`` for
    lanes with a single-rounding fma, ``"split"`` (Dekker) for lanes
    without one. A module is lowered in one flavour, so lanes needing the
    other get their own freshly prepared core.
    """

    source, entry, parameters, degree = _legacy._pack_source(name, width)

    started = time.perf_counter_ns()
    # The source compiler runs the precision pipeline itself at the
    # completed-module seam, so the flavour has to be ambient while the
    # lowering runs; the explicit call after it is the idempotent
    # revalidation that also catches a flavour mismatch loudly.
    from src.compiler.ir_identities import two_product_flavor_scope

    with two_product_flavor_scope(two_product_flavor):
        lowered = lower_ast_source_to_ssa(source, entry)
    module = lowered if isinstance(lowered, IRModule) else lowered[0]
    apply_precision_pipeline(module, two_product_flavor=two_product_flavor)
    lower_ns = time.perf_counter_ns() - started

    wrapper = next(
        function_name
        for function_name in module.functions
        if function_name.endswith("__" + entry)
    )
    roles = _legacy._roles(module.functions, wrapper)

    histogram = _operation_histogram(module)
    coefficients = tuple(_proof.structured_coefficients(name, degree))
    # Assert that source materialisation and the numerical oracle agree about
    # coefficient arity.  A stale catalogue would otherwise produce a valid
    # ABI with silently zero-filled coefficients.
    coefficient_names = tuple(
        parameter for parameter in parameters
        if parameter.startswith("c") and parameter[1:].isdigit()
    )
    if len(coefficient_names) != len(coefficients):
        raise ValueError(
            f"{name}: source has {len(coefficient_names)} coefficients but "
            f"the exact oracle has {len(coefficients)}"
        )
    return CompiledCore(
        module=module,
        native=None,
        roles=roles,
        coefficients=coefficients,
        build=BuildProfile(
            lower_ns=lower_ns,
            emit_ns=0,
            native_compile_ns=0,
            static_scalar_instructions=sum(histogram.values()),
            static_fma_instructions=histogram.get("Fma", 0),
            operation_histogram=histogram,
        ),
    )


def _wrapper_name(name: str, width: int, prepared: CompiledCore) -> str:
    entry = _legacy._pack_source(name, width)[1]
    return next(
        function_name
        for function_name in prepared.module.functions
        if function_name.endswith("__" + entry)
    )


def _compile_c_core(
    name: str, width: int, root: Path, prepared: CompiledCore,
) -> CompiledCore:
    """Apply C module emission/deployment to the shared prepared module.

    Same shape as the LLVM path: emit, refuse on shortfalls, compile to a
    native, and return a core whose ``native`` speaks the shared execution
    face. The C lane keeps the fma-flavoured lowering -- C99 ``fma`` is the
    single rounding by specification -- so the same module serves both.
    """

    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    wrapper = _wrapper_name(name, width, prepared)
    started = time.perf_counter_ns()
    artifact = emit_ssa_module_to_c(prepared.module, wrapper)
    emit_ns = time.perf_counter_ns() - started
    if artifact.shortfalls:
        raise RuntimeError("; ".join(
            f"{item.operation}: {item.reason}" for item in artifact.shortfalls
        ))
    directory = root / "artifacts" / f"{name}_w{width}_c"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    native = artifact.compile(directory)
    native_compile_ns = time.perf_counter_ns() - started
    return CompiledCore(
        module=prepared.module,
        native=native,
        roles=prepared.roles,
        coefficients=prepared.coefficients,
        build=BuildProfile(
            lower_ns=prepared.build.lower_ns,
            emit_ns=emit_ns,
            native_compile_ns=native_compile_ns,
            static_scalar_instructions=prepared.build.static_scalar_instructions,
            static_fma_instructions=prepared.build.static_fma_instructions,
            operation_histogram=prepared.build.operation_histogram,
        ),
    )


from src.compiler.ssa_fortran_backend import (  # noqa: E402
    FortranCoreNative as _FortranCoreNative,
)


def _compile_fortran_core(
    name: str, width: int, root: Path, prepared: CompiledCore,
) -> CompiledCore:
    """Emit and deploy the shared module through the Fortran c-shell lane."""

    from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
    from src.compiler.ssa_fortran_backend import emit_module

    wrapper = _wrapper_name(name, width, prepared)
    started = time.perf_counter_ns()
    fortran_module = emit_module(prepared.module, progress=lambda _line: None)
    emit_ns = time.perf_counter_ns() - started
    if not fortran_module.complete:
        raise RuntimeError("; ".join(
            f"{item.operation}: {item.reason}"
            for item in fortran_module.shortfalls[:8]
        ))
    directory = root / "artifacts" / f"{name}_w{width}_fortran"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    built = compile_fortran_module_c_shell(
        fortran_module, {}, directory, library=True,
        entrypoint=wrapper, name=f"{name}_w{width}_fortran",
    )
    native_compile_ns = time.perf_counter_ns() - started
    entry_record = next(
        entry for entry in fortran_module.api.entry_points
        if str(entry.name) == wrapper
    )
    native = _FortranCoreNative(built.executable_path, entry_record)
    return CompiledCore(
        module=prepared.module,
        native=native,
        roles=prepared.roles,
        coefficients=prepared.coefficients,
        build=BuildProfile(
            lower_ns=prepared.build.lower_ns,
            emit_ns=emit_ns,
            native_compile_ns=native_compile_ns,
            static_scalar_instructions=prepared.build.static_scalar_instructions,
            static_fma_instructions=prepared.build.static_fma_instructions,
            operation_histogram=prepared.build.operation_histogram,
        ),
    )


class _WasmCoreNative:
    """Adapter giving the wasm artifact the shared execution face."""

    def __init__(self, artifact):
        self._artifact = artifact

    def prepare_execution(self, feeds):
        from src.compiler.ssa_wasm_backend import prepare_wasm_core_execution

        return prepare_wasm_core_execution(self._artifact, feeds)


def _compile_wasm_core(
    name: str, width: int, root: Path, prepared: CompiledCore,
) -> CompiledCore:
    """Emit the shared module as one wasm function behind a Node worker.

    The lane always receives the split-flavoured lowering: WebAssembly has
    no fma, and Dekker's spelling carries the identical residual through
    plain mul/add, so the lane conforms by construction and pays in
    instruction count. Emission builds the binary directly, so the native
    compile column is genuinely zero rather than unmeasured.
    """

    from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core

    wrapper = _wrapper_name(name, width, prepared)
    started = time.perf_counter_ns()
    artifact = emit_ssa_module_to_wasm_core(prepared.module, wrapper)
    emit_ns = time.perf_counter_ns() - started
    if artifact.shortfalls:
        raise RuntimeError("; ".join(
            f"{item.operation}: {item.reason}"
            for item in artifact.shortfalls[:8]
        ))
    return CompiledCore(
        module=prepared.module,
        native=_WasmCoreNative(artifact),
        roles=prepared.roles,
        coefficients=prepared.coefficients,
        build=BuildProfile(
            lower_ns=prepared.build.lower_ns,
            emit_ns=emit_ns,
            native_compile_ns=0,
            static_scalar_instructions=prepared.build.static_scalar_instructions,
            static_fma_instructions=prepared.build.static_fma_instructions,
            operation_histogram=prepared.build.operation_histogram,
        ),
    )


def _compile_llvm_core(
    name: str, width: int, root: Path, prepared: CompiledCore,
) -> CompiledCore:
    """Apply LLVM emission/deployment to an already prepared shared module."""

    wrapper = _wrapper_name(name, width, prepared)
    started = time.perf_counter_ns()
    artifact = emit_ssa_function_to_llvm(prepared.module, wrapper)
    emit_ns = time.perf_counter_ns() - started
    if artifact.shortfalls:
        raise RuntimeError("; ".join(item.reason for item in artifact.shortfalls))
    directory = root / "artifacts" / f"{name}_w{width}_llvm"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    native = compile_artifact(artifact, directory=directory)
    native_compile_ns = time.perf_counter_ns() - started
    return CompiledCore(
        module=prepared.module,
        native=native,
        roles=prepared.roles,
        coefficients=prepared.coefficients,
        build=BuildProfile(
            lower_ns=prepared.build.lower_ns,
            emit_ns=emit_ns,
            native_compile_ns=native_compile_ns,
            static_scalar_instructions=prepared.build.static_scalar_instructions,
            static_fma_instructions=prepared.build.static_fma_instructions,
            operation_histogram=prepared.build.operation_histogram,
        ),
    )


def _feeds(core: CompiledCore, width: int, x: np.ndarray, y: np.ndarray, count: int):
    """Bind buffers and coefficients, WITH the coefficients' low limbs.

    ``float(value)`` alone would collapse each exact rational coefficient
    to one double and zero-fill its appended limb formals -- capping every
    width at the seventeen-digit coefficient floor no matter how exact the
    arithmetic. The lowering appends a coefficient's low-limb formals
    grouped per coefficient in source order, so the value list is the high
    limbs in coefficient order followed by each coefficient's remaining
    limbs in that same order.
    """

    highs: list[float] = []
    lows: list[float] = []
    for exact in core.coefficients:
        parts = _proof.limb_decomposition(exact, width)
        highs.append(parts[0])
        lows.extend(parts[1:])
    return _legacy._feeds(
        core.roles,
        width,
        [*highs, *lows],
        x,
        y,
        count,
    )


def _prepare_native_execution(native, feeds):
    """One execution face for every lane.

    A lane's artifact either carries its own ``prepare_execution`` (the C
    module lane) or is an LLVM artifact served by the LLVM helper. Both
    return an object with ``.buffers`` keyed by SSA value id and ``.run()``,
    which is the whole contract the rows below need.
    """

    preparer = getattr(native, "prepare_execution", None)
    if preparer is not None:
        return preparer(feeds)
    return prepare_artifact_execution(native, feeds)


def _run_for_accuracy(
    core: CompiledCore,
    width: int,
    points: list[float],
) -> np.ndarray:
    count = len(points)
    x = np.zeros(count * width, dtype=np.float64)
    x[::width] = points
    y = np.zeros(count * width, dtype=np.float64)
    output_id = int(core.roles["output"].id)
    execution = _prepare_native_execution(
        core.native, _feeds(core, width, x, y, count)
    )
    execution.run()
    return np.asarray(execution.buffers[output_id]).ravel().copy()


def _mp_function(name: str, mpmath):
    direct = {
        "sin": mpmath.sin, "cos": mpmath.cos, "tan": mpmath.tan,
        "asin": mpmath.asin, "asinh": mpmath.asinh,
        "atan": mpmath.atan, "atanh": mpmath.atanh,
        "exp": mpmath.exp, "expm1": mpmath.expm1,
        # log's core IS log1p's series (see signal_symbolic.TRANSCENDENTALS),
        # so its samples are z near 0 and the truth is log(1+z), not log(z).
        "log": mpmath.log1p, "log1p": mpmath.log1p,
        "sinh": mpmath.sinh, "cosh": mpmath.cosh,
        "tanh": mpmath.tanh,
    }
    if name in direct:
        return direct[name]
    if name == "sec":
        return lambda value: 1 / mpmath.cos(value)
    if name == "csc":
        return lambda value: 1 / mpmath.sin(value)
    if name == "cot":
        return lambda value: 1 / mpmath.tan(value)
    if name == "sech":
        return lambda value: 1 / mpmath.cosh(value)
    if name == "sinc":
        return lambda value: mpmath.sin(value) / value if value else mpmath.mpf(1)
    raise KeyError(f"no high-precision oracle for {name!r}")


def _fraction_polynomial(name: str, coefficients: tuple[Fraction, ...], z: float):
    structure = _proof.TRANSCENDENTALS[name]["structure"]
    argument = Fraction.from_float(float(z))
    structural = argument * argument if structure in {"odd", "even"} else argument
    accumulated = Fraction()
    for coefficient in reversed(coefficients):
        accumulated = accumulated * structural + coefficient
    return accumulated * argument if structure == "odd" else accumulated


def _accuracy(
    name: str,
    width: int,
    core: CompiledCore,
    samples: int,
    oracle_dps: int,
) -> AccuracyProfile:
    import mpmath

    radius = float(_proof.CORE_RADII[name])
    generator = random.Random(0x5EED + width)
    points = [generator.uniform(-radius * 0.98, radius * 0.98) for _ in range(samples)]
    produced = _run_for_accuracy(core, width, points)
    polynomial_ulps: list[float] = []
    true_ulps: list[float] = []
    true_bits: list[float] = []
    rounded = 0
    reference = _mp_function(name, mpmath)

    with mpmath.workdps(int(oracle_dps)):
        for index, point in enumerate(points):
            represented = sum(
                (Fraction.from_float(float(produced[index * width + limb]))
                 for limb in range(width)),
                Fraction(),
            )
            polynomial = _fraction_polynomial(name, core.coefficients, point)
            poly_spacing = math.ulp(float(polynomial))
            polynomial_ulps.append(
                float(abs(represented - polynomial) / Fraction.from_float(poly_spacing))
            )

            exact_point = (
                mpmath.mpf(point.as_integer_ratio()[0])
                / point.as_integer_ratio()[1]
            )
            truth = reference(exact_point)
            represented_mp = (
                mpmath.mpf(represented.numerator) / represented.denominator
            )
            collapsed_truth = float(truth)
            spacing = math.ulp(collapsed_truth)
            absolute = abs(represented_mp - truth)
            true_ulps.append(float(absolute / mpmath.mpf(spacing)))
            scale = max(abs(truth), mpmath.mpf(2) ** -1022)
            relative = absolute / scale
            true_bits.append(
                float("inf") if not relative else float(-mpmath.log(relative, 2))
            )
            rounded += int(float(represented) == collapsed_truth)

    return AccuracyProfile(
        samples=samples,
        correctly_rounded_percent=100.0 * rounded / max(samples, 1),
        polynomial_ulp_median=statistics.median(polynomial_ulps),
        polynomial_ulp_p99=_percentile(polynomial_ulps, 0.99),
        polynomial_ulp_max=max(polynomial_ulps, default=float("nan")),
        true_ulp_median=statistics.median(true_ulps),
        true_ulp_p99=_percentile(true_ulps, 0.99),
        true_ulp_max=max(true_ulps, default=float("nan")),
        true_bits_median=statistics.median(true_bits),
    )


def _shortfall_reason(module, backend: str) -> str | None:
    shortfalls = precision_backend_shortfalls(module, backend)
    if not shortfalls:
        return None
    return "; ".join(
        f"{item['function']}: missing {', '.join(item['missing'])}"
        for item in shortfalls
    )


def _unavailable_rows(
    name: str,
    width: int,
    backend: str,
    sizes: Iterable[int],
    core: CompiledCore,
) -> list[ProfileRow]:
    reason = _shortfall_reason(core.module, backend)
    status = "unsupported" if reason else "unavailable"
    stage = "precision_contract" if reason else "backend_pipeline"
    if reason is None:
        reason = _UNAVAILABLE_REASONS.get(backend) or (
            "backend preserves the declared precision obligations, but its "
            "SSA identity/transformation, native emission, and deployment "
            "route is not implemented end to end yet"
        )
    return [
        ProfileRow(
            operator=name, width=width, backend=backend, batch_size=int(size),
            status=status, stage=stage, reason=reason,
            lower_ns=core.build.lower_ns, emit_ns=None, native_compile_ns=None,
            prepare_ns=None, first_launch_ns=None, warm_min_ns=None,
            warm_median_ns=None, warm_p95_ns=None,
            warm_ns_per_element=None, warm_ns_per_static_instruction=None,
            static_scalar_instructions=core.build.static_scalar_instructions,
            static_fma_instructions=core.build.static_fma_instructions,
            correctly_rounded_percent=None, polynomial_ulp_median=None,
            polynomial_ulp_p99=None, polynomial_ulp_max=None,
            true_ulp_median=None, true_ulp_p99=None, true_ulp_max=None,
            true_bits_median=None,
        )
        for size in sizes
    ]


def _profile_native_rows(
    name: str,
    width: int,
    sizes: Iterable[int],
    repeats: int,
    warmups: int,
    core: CompiledCore,
    accuracy: AccuracyProfile,
    backend: str = "llvm",
) -> list[ProfileRow]:
    rows = []
    radius = float(_proof.CORE_RADII[name])
    output_id = int(core.roles["output"].id)
    for size in sizes:
        count = int(size)
        generator = np.random.default_rng(0xC0DE + count + width)
        x = np.zeros(count * width, dtype=np.float64)
        x[::width] = generator.uniform(-radius * 0.98, radius * 0.98, count)
        y = np.zeros(count * width, dtype=np.float64)

        started = time.perf_counter_ns()
        execution = _prepare_native_execution(
            core.native, _feeds(core, width, x, y, count)
        )
        prepare_ns = time.perf_counter_ns() - started
        if output_id not in execution.buffers:
            raise RuntimeError(
                f"{backend} execution omitted output value {output_id}"
            )

        started = time.perf_counter_ns()
        execution.run()
        first_launch_ns = time.perf_counter_ns() - started
        for _index in range(max(0, warmups)):
            execution.run()
        warm = []
        for _index in range(max(1, repeats)):
            started = time.perf_counter_ns()
            execution.run()
            warm.append(time.perf_counter_ns() - started)
        warm_median = int(statistics.median(warm))
        instruction_count = max(core.build.static_scalar_instructions, 1)
        rows.append(ProfileRow(
            operator=name, width=width, backend=backend, batch_size=count,
            status="passed", stage="runtime", reason=None,
            lower_ns=core.build.lower_ns, emit_ns=core.build.emit_ns,
            native_compile_ns=core.build.native_compile_ns,
            prepare_ns=prepare_ns, first_launch_ns=first_launch_ns,
            warm_min_ns=min(warm), warm_median_ns=warm_median,
            warm_p95_ns=int(_percentile(warm, 0.95)),
            warm_ns_per_element=warm_median / max(count, 1),
            warm_ns_per_static_instruction=(
                warm_median / max(count * instruction_count, 1)
            ),
            static_scalar_instructions=core.build.static_scalar_instructions,
            static_fma_instructions=core.build.static_fma_instructions,
            correctly_rounded_percent=accuracy.correctly_rounded_percent,
            polynomial_ulp_median=accuracy.polynomial_ulp_median,
            polynomial_ulp_p99=accuracy.polynomial_ulp_p99,
            polynomial_ulp_max=accuracy.polynomial_ulp_max,
            true_ulp_median=accuracy.true_ulp_median,
            true_ulp_p99=accuracy.true_ulp_p99,
            true_ulp_max=accuracy.true_ulp_max,
            true_bits_median=accuracy.true_bits_median,
        ))
    return rows


def _failed_rows(name, width, backend, sizes, stage, error) -> list[ProfileRow]:
    reason = f"{type(error).__name__}: {error}"
    return [ProfileRow(
        operator=name, width=width, backend=backend, batch_size=int(size),
        status="failed", stage=stage, reason=reason,
        lower_ns=None, emit_ns=None, native_compile_ns=None, prepare_ns=None,
        first_launch_ns=None, warm_min_ns=None, warm_median_ns=None,
        warm_p95_ns=None, warm_ns_per_element=None,
        warm_ns_per_static_instruction=None, static_scalar_instructions=None,
        static_fma_instructions=None, correctly_rounded_percent=None,
        polynomial_ulp_median=None, polynomial_ulp_p99=None,
        polynomial_ulp_max=None, true_ulp_median=None, true_ulp_p99=None,
        true_ulp_max=None, true_bits_median=None,
    ) for size in sizes]


#: Native lanes beyond LLVM: backend -> (compiler, lowering flavour). The
#: flavour is the lane's honest spelling of the exact product -- "fma"
#: where the destination delivers a single rounding, "split" where it
#: cannot and Dekker's form carries the residual instead.
def _compile_glsl_core(
    name: str, width: int, root: Path, prepared: CompiledCore,
) -> CompiledCore:
    """Emit and deploy through the desktop GLSL compute lane (real GPU)."""

    from src.compiler.ssa_glsl_compute_backend import (
        emit_ssa_module_to_glsl_compute,
    )

    wrapper = _wrapper_name(name, width, prepared)
    started = time.perf_counter_ns()
    artifact = emit_ssa_module_to_glsl_compute(prepared.module, wrapper)
    emit_ns = time.perf_counter_ns() - started
    if artifact.shortfalls:
        raise RuntimeError("; ".join(
            f"{item.operation}: {item.reason}"
            for item in artifact.shortfalls[:8]
        ))
    return CompiledCore(
        module=prepared.module,
        native=artifact,
        roles=prepared.roles,
        coefficients=prepared.coefficients,
        build=BuildProfile(
            lower_ns=prepared.build.lower_ns,
            emit_ns=emit_ns,
            native_compile_ns=0,
            static_scalar_instructions=prepared.build.static_scalar_instructions,
            static_fma_instructions=prepared.build.static_fma_instructions,
            operation_histogram=prepared.build.operation_histogram,
        ),
    )


_LANE_COMPILERS: dict[str, tuple[Any, str]] = {
    "c": (_compile_c_core, "fma"),
    "fortran": (_compile_fortran_core, "fma"),
    "wasm": (_compile_wasm_core, "split"),
    # The RTX-class desktop drivers honour `precise` + fp64 fma(), proven
    # bit-identical to the C lane; the fma flavour is therefore the honest
    # spelling here too.
    "glsl": (_compile_glsl_core, "fma"),
}


#: Lanes with no execution route yet, and exactly why -- so the table says
#: what is actually missing instead of a generic shrug.
_UNAVAILABLE_REASONS = {
    "spirv": (
        "emission is conforming (GLSL.std.450 Fma + NoContraction, "
        "spirv-val clean) but modules are linkage library functions with "
        "no OpEntryPoint or descriptor bindings; no dispatch route exists"
    ),
    "webgpu": (
        "WGSL emission only: no headless WebGPU runtime on this host; "
        "the browser benchmark bundle is the only execution surface"
    ),
}


def profile_matrix(
    *, operators, backends, widths, sizes, repeats, warmups,
    accuracy_samples, oracle_dps, scratch,
) -> tuple[list[ProfileRow], dict[str, Any]]:
    rows: list[ProfileRow] = []
    builds: dict[str, Any] = {}
    for name in operators:
        for width in widths:
            label = f"{name} w{width}"
            print(f"[{label}] shared lowering and universal identities", flush=True)
            try:
                core = _prepare_core(name, width)
            except BaseException as error:
                print(f"[{label}] FAILED: {type(error).__name__}: {error}", flush=True)
                for backend in backends:
                    rows.extend(_failed_rows(
                        name, width, backend, sizes, "build", error
                    ))
                continue
            accuracy = None
            llvm_error = None
            split_core = None
            if "llvm" in backends:
                print(f"[{label} llvm] transforming, emitting, and deploying", flush=True)
                try:
                    core = _compile_llvm_core(name, width, scratch, core)
                    accuracy = _accuracy(
                        name, width, core, accuracy_samples, oracle_dps
                    )
                except BaseException as error:
                    llvm_error = error
                    print(
                        f"[{label} llvm] FAILED: {type(error).__name__}: {error}",
                        flush=True,
                    )
            builds[label] = asdict(core.build)
            for backend in backends:
                if backend == "llvm":
                    if core.native is None or accuracy is None:
                        rows.extend(_failed_rows(
                            name, width, backend, sizes, "llvm_pipeline",
                            llvm_error or RuntimeError(
                                "LLVM transformation/emission/deployment failed"
                            ),
                        ))
                        continue
                    print(f"[{label} llvm] profiling {len(tuple(sizes))} batches", flush=True)
                    try:
                        rows.extend(_profile_native_rows(
                            name, width, sizes, repeats, warmups, core, accuracy
                        ))
                    except BaseException as error:
                        rows.extend(_failed_rows(
                            name, width, backend, sizes, "runtime", error
                        ))
                elif backend in _LANE_COMPILERS:
                    # Every native lane earns its row the same way LLVM
                    # does: its own emission, its own native, its own
                    # measured accuracy -- never numbers borrowed from
                    # another lane. Lanes whose destination lacks a
                    # single-rounding fma receive the split-flavoured core.
                    lane_compiler, lane_flavor = _LANE_COMPILERS[backend]
                    lane_source = core
                    if lane_flavor == "split":
                        if split_core is None:
                            split_core = _prepare_core(name, width, "split")
                        lane_source = split_core
                    print(f"[{label} {backend}] emitting and deploying", flush=True)
                    try:
                        lane_core = lane_compiler(
                            name, width, scratch, lane_source
                        )
                        lane_accuracy = _accuracy(
                            name, width, lane_core, accuracy_samples,
                            oracle_dps,
                        )
                    except BaseException as error:
                        rows.extend(_failed_rows(
                            name, width, backend, sizes,
                            f"{backend}_pipeline", error,
                        ))
                        continue
                    print(f"[{label} {backend}] profiling {len(tuple(sizes))} batches", flush=True)
                    try:
                        rows.extend(_profile_native_rows(
                            name, width, sizes, repeats, warmups,
                            lane_core, lane_accuracy, backend=backend,
                        ))
                    except BaseException as error:
                        rows.extend(_failed_rows(
                            name, width, backend, sizes, "runtime", error
                        ))
                else:
                    # Judge each lane against the lowering flavour it would
                    # actually receive: a lane with no single-rounding fma
                    # gets the split (Dekker) flavour, whose sections incur
                    # no fma obligation at all. Reporting the fma-flavoured
                    # shortfall against such a lane would blame it for an
                    # instruction nobody would send it.
                    lane_core = core
                    if FMA_MANDATORY not in BACKEND_PRECISION_CAPABILITIES.get(
                        backend, ()
                    ):
                        if split_core is None:
                            split_core = _prepare_core(name, width, "split")
                        lane_core = split_core
                    rows.extend(_unavailable_rows(
                        name, width, backend, sizes, lane_core
                    ))
    return rows, builds


def _mp_limb_values(values, width: int, mpmath) -> list[np.ndarray]:
    limbs = [np.empty(len(values), dtype=np.float64) for _ in range(width)]
    for position, original in enumerate(values):
        remainder = mpmath.mpf(original)
        for limb_index in range(width):
            part = float(remainder)
            limbs[limb_index][position] = part
            remainder -= mpmath.mpf(part)
    return limbs


def _precision_from_parts(parts: list[np.ndarray]):
    from src.common.tensors.extended_precision import Precision, interleave
    from src.common.tensors.numpy_backend import NumPyTensorOperations

    tensors = [NumPyTensorOperations.tensor(part) for part in parts]
    return Precision(interleave(tensors), len(parts))


def _mandelbrot_recurrence(
    cx,
    cy,
    zero,
    two,
    iterations: int,
    *,
    collapse,
    mask_value,
    progress=None,
) -> np.ndarray:
    """One recurrence source, parameterized only at representation seams."""

    zr = zero
    zi = zero
    count = int(np.asarray(collapse(zero)).size)
    active = np.ones(count, dtype=bool)
    counts = np.zeros(count, dtype=np.int32)
    for iteration in range(iterations):
        zr2 = zr * zr
        zi2 = zi * zi
        magnitude = np.asarray(collapse(zr2 + zi2), dtype=np.float64).reshape(-1)
        active &= magnitude <= 4.0
        counts += active
        next_zr = zr2 - zi2 + cx
        next_zi = (zr * zi) * two + cy
        mask = mask_value(active)
        zr = next_zr * mask
        zi = next_zi * mask
        if progress is not None:
            progress(iteration, active)
        if not active.any():
            break
    return counts


def _mandelbrot_four_limb_tile(payload):
    """Render one exact row interval; safe as a Windows worker entrypoint."""

    import mpmath
    from src.common.tensors.extended_precision import Precision
    from src.common.tensors.numpy_backend import NumPyTensorOperations

    (
        center_x, center_y, field_span_text, width, height,
        row_start, row_stop, iterations, oracle_dps,
    ) = payload
    tile_height = row_stop - row_start
    with mpmath.workdps(max(int(oracle_dps), 80)):
        cx0 = mpmath.mpf(center_x)
        cy0 = mpmath.mpf(center_y)
        field_span = mpmath.mpf(field_span_text)
        aspect = mpmath.mpf(height) / width
        xs = [
            cx0 + field_span * (
                mpmath.mpf(index) / max(width - 1, 1) - mpmath.mpf("0.5")
            )
            for index in range(width)
        ]
        ys = [
            cy0 + field_span * aspect * (
                mpmath.mpf(index) / max(height - 1, 1) - mpmath.mpf("0.5")
            )
            for index in range(row_start, row_stop)
        ]
        x_parts = _mp_limb_values(xs, 4, mpmath)
        y_parts = _mp_limb_values(ys, 4, mpmath)

    cx = _precision_from_parts([
        np.tile(part, tile_height) for part in x_parts
    ])
    cy = _precision_from_parts([
        np.repeat(part, width) for part in y_parts
    ])
    count = width * tile_height
    zero = NumPyTensorOperations.tensor(np.zeros(count, dtype=np.float64))
    two = NumPyTensorOperations.tensor(np.full(count, 2.0, dtype=np.float64))
    result = _mandelbrot_recurrence(
        cx,
        cy,
        Precision.of(zero, 4),
        two,
        iterations,
        collapse=lambda value: value.collapse().tolist(),
        mask_value=lambda active: NumPyTensorOperations.tensor(
            active.astype(np.float64)
        ),
    )
    return row_start, row_stop, result.reshape(tile_height, width)


def render_mandelbrot(
    destination: Path,
    *, width: int = 1280,
    height: int = 720,
    iterations: int = 96,
    center_x: str = "-1.999985881",
    center_y: str = "0.0",
    span: str | None = None,
    resolution_bits: int = 54,
    oracle_dps: int = 100,
    compare: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    """Render a genuine four-limb Mandelbrot field through ``Precision``."""

    import mpmath
    from PIL import Image, ImageDraw
    from src.common.tensors.extended_precision import Precision
    from src.common.tensors.numpy_backend import NumPyTensorOperations

    limb_width = 4
    started = time.perf_counter()
    with mpmath.workdps(max(int(oracle_dps), 80)):
        cx0, cy0 = map(mpmath.mpf, (center_x, center_y))
        canonical_width = mpmath.mpf(3)
        canonical_pixel_pitch = canonical_width / mpmath.power(2, resolution_bits)
        field_span = (
            mpmath.mpf(span)
            if span is not None
            else canonical_pixel_pitch * max(width - 1, 1)
        )
        aspect = mpmath.mpf(height) / width
        xs = [
            cx0 + field_span * (mpmath.mpf(index) / max(width - 1, 1) - mpmath.mpf("0.5"))
            for index in range(width)
        ]
        ys = [
            cy0 + field_span * aspect * (
                mpmath.mpf(index) / max(height - 1, 1) - mpmath.mpf("0.5")
            )
            for index in range(height)
        ]
        x_parts = _mp_limb_values(xs, limb_width, mpmath)
        y_parts = _mp_limb_values(ys, limb_width, mpmath)
        exact_step = field_span / max(width - 1, 1)
        effective_resolution_bits = float(
            mpmath.log(canonical_width / exact_step, 2)
        )

    jobs = max(1, min(int(jobs), int(height)))
    if jobs == 1:
        cx = _precision_from_parts([
            np.tile(part, height) for part in x_parts
        ])
        cy = _precision_from_parts([
            np.repeat(part, width) for part in y_parts
        ])
        zero = NumPyTensorOperations.tensor(
            np.zeros(width * height, dtype=np.float64)
        )
        two = NumPyTensorOperations.tensor(
            np.full(width * height, 2.0, dtype=np.float64)
        )

        def progress(iteration, active):
            if iteration % 8 == 7:
                print(
                    f"[mandelbrot] iteration {iteration + 1}/{iterations}; "
                    f"active={int(active.sum())}",
                    flush=True,
                )

        wide_counts = _mandelbrot_recurrence(
            cx,
            cy,
            Precision.of(zero, limb_width),
            two,
            iterations,
            collapse=lambda value: value.collapse().tolist(),
            mask_value=lambda active: NumPyTensorOperations.tensor(
                active.astype(np.float64)
            ),
            progress=progress,
        ).reshape(height, width)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        boundaries = [round(index * height / jobs) for index in range(jobs + 1)]
        payloads = [
            (
                center_x, center_y, str(field_span), width, height,
                boundaries[index], boundaries[index + 1], iterations,
                oracle_dps,
            )
            for index in range(jobs)
            if boundaries[index] < boundaries[index + 1]
        ]
        wide_counts = np.empty((height, width), dtype=np.int32)
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_mandelbrot_four_limb_tile, p) for p in payloads]
            completed = 0
            for future in as_completed(futures):
                row_start, row_stop, tile = future.result()
                wide_counts[row_start:row_stop] = tile
                completed += 1
                print(
                    f"[mandelbrot] completed tile {completed}/{len(payloads)} "
                    f"(rows {row_start}:{row_stop})",
                    flush=True,
                )

    # The comparison panel intentionally creates its coordinates in binary64.
    # At the default camera its x step is below one local ULP, so neighbouring
    # columns alias before the recurrence even starts. The four-limb panel's
    # coordinate tuples remain unique; this is the precision fact the image
    # is meant to make visible.
    # Both widths start from the SAME high-precision camera list. Width one
    # rounds each coordinate at its representation boundary; width four keeps
    # the four pieces split above. No second coordinate formula exists.
    xs_f64 = np.asarray([float(value) for value in xs], dtype=np.float64)
    ys_f64 = np.asarray([float(value) for value in ys], dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs_f64, ys_f64)
    flat_x, flat_y = grid_x.reshape(-1), grid_y.reshape(-1)
    binary_counts = None
    if compare:
        binary_counts = _mandelbrot_recurrence(
            flat_x,
            flat_y,
            np.zeros_like(flat_x),
            np.full_like(flat_x, 2.0),
            iterations,
            collapse=lambda value: value,
            mask_value=lambda active: active.astype(np.float64),
        ).reshape(height, width)

    def colour(field):
        values = field.astype(np.float64) / max(iterations, 1)
        outside = field < iterations
        red = 9.0 * (1.0 - values) * values ** 3
        green = 15.0 * (1.0 - values) ** 2 * values ** 2
        blue = 8.5 * (1.0 - values) ** 3 * values
        rgb = np.stack((red, green, blue), axis=-1)
        rgb[~outside] = 0.0
        return np.asarray(np.clip(rgb * 255.0, 0, 255), dtype=np.uint8)

    difference = None
    if compare:
        difference = np.abs(wide_counts.astype(np.int64) - binary_counts)
        difference_scale = max(int(difference.max()), 1)
        heat = np.log1p(difference) / math.log1p(difference_scale)
        difference_rgb = np.stack(
            (heat, heat ** 2 * 0.8, np.zeros_like(heat)), axis=-1
        )
        difference_rgb = np.asarray(
            np.clip(difference_rgb * 255.0, 0, 255), dtype=np.uint8
        )
        panels = np.concatenate(
            (colour(binary_counts), colour(wide_counts), difference_rgb), axis=1
        )
        label_height = 20
        image = Image.new("RGB", (width * 3, height + label_height), "#111318")
        image.paste(Image.fromarray(panels, mode="RGB"), (0, label_height))
        labels = ImageDraw.Draw(image)
        labels.text((4, 4), "same source: 1 limb", fill="white")
        labels.text((width + 4, 4), "same source: 4 limbs", fill="white")
        labels.text((width * 2 + 4, 4), "escape-count difference", fill="white")
        output_width, output_height = width * 3, height + label_height
    else:
        image = Image.fromarray(colour(wide_counts), mode="RGB")
        output_width, output_height = width, height
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    four_x_unique = len(set(zip(*(part.tolist() for part in x_parts))))
    four_y_unique = len(set(zip(*(part.tolist() for part in y_parts))))
    binary_ulp = math.ulp(float(center_x))
    import hashlib
    import inspect

    recurrence_source = inspect.getsource(_mandelbrot_recurrence)
    recurrence_sha256 = hashlib.sha256(
        recurrence_source.encode("utf-8")
    ).hexdigest()
    record = {
        "schema": "turing-four-limb-mandelbrot-v2",
        "path": str(destination.resolve()),
        "panel_width": width,
        "panel_height": height,
        "output_width": output_width,
        "output_height": output_height,
        "comparison_panels": bool(compare),
        "iterations": iterations,
        "precision_limbs": limb_width,
        "render_jobs": jobs,
        "same_recurrence_source": True,
        "recurrence_qualified_name": (
            "tools.benchmark_microprecision_matrix._mandelbrot_recurrence"
        ),
        "recurrence_source_sha256": recurrence_sha256,
        "coordinate_source": (
            "one high-precision camera; rounded only at each width boundary"
        ),
        "center_x": center_x,
        "center_y": center_y,
        "span": str(field_span),
        "pixel_step": str(exact_step),
        "canonical_mandelbrot_width": 3.0,
        "binary64_significand_bits": 53,
        "effective_horizontal_resolution_bits": effective_resolution_bits,
        "requested_horizontal_resolution_bits": int(resolution_bits),
        "beyond_binary64_bits": effective_resolution_bits - 53.0,
        "binary64_ulp_at_center_x": binary_ulp,
        "pixel_step_over_binary64_ulp": float(exact_step) / binary_ulp,
        "binary64_unique_x_coordinates": int(np.unique(xs_f64).size),
        "binary64_unique_y_coordinates": int(np.unique(ys_f64).size),
        "four_limb_unique_x_coordinates": four_x_unique,
        "four_limb_unique_y_coordinates": four_y_unique,
        "coordinate_aliases_x": width - int(np.unique(xs_f64).size),
        "coordinate_aliases_y": height - int(np.unique(ys_f64).size),
        "pixel_escape_count_disagreements": (
            int(np.count_nonzero(difference)) if difference is not None else None
        ),
        "maximum_escape_count_difference": (
            int(difference.max()) if difference is not None else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "four_limb_escaped_pixels": int(np.count_nonzero(wide_counts < iterations)),
        "binary64_escaped_pixels": (
            int(np.count_nonzero(binary_counts < iterations))
            if binary_counts is not None else None
        ),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def _write_results(path: Path, rows: list[ProfileRow], builds: dict[str, Any], args):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "turing-microprecision-profile-v1",
        "created_unix_ns": time.time_ns(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "configuration": {
            "operators": list(args.operators),
            "backends": list(args.backends),
            "widths": list(args.widths),
            "sizes": list(args.sizes),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "accuracy_samples": args.accuracy_samples,
            "oracle": "exact Fraction polynomial + mpmath true function",
            "oracle_dps": args.oracle_dps,
        },
        "builds": builds,
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return csv_path


def _print_rows(rows: list[ProfileRow]):
    print(
        "operator width backend batch status       first-us warm-ns/elt "
        "true-p99-ulp rounded reason",
        flush=True,
    )
    for row in rows:
        first = "-" if row.first_launch_ns is None else f"{row.first_launch_ns / 1e3:.1f}"
        warm = "-" if row.warm_ns_per_element is None else f"{row.warm_ns_per_element:.2f}"
        true = "-" if row.true_ulp_p99 is None else f"{row.true_ulp_p99:.3g}"
        rounded = (
            "-" if row.correctly_rounded_percent is None
            else f"{row.correctly_rounded_percent:.1f}%"
        )
        reason = (row.reason or "")[:72]
        print(
            f"{row.operator:8s} {row.width:5d} {row.backend:7s} "
            f"{row.batch_size:6d} {row.status:12s} {first:>8s} "
            f"{warm:>11s} {true:>12s} {rounded:>7s} {reason}",
            flush=True,
        )


def _selection(values, available, label):
    selected = tuple(available) if values == ["all"] else tuple(values)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown {label}: {sorted(unknown)}")
    return selected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", nargs="+", default=["all"])
    parser.add_argument("--backends", nargs="+", default=["all"])
    parser.add_argument("--widths", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--accuracy-samples", type=int, default=256)
    parser.add_argument("--oracle-dps", type=int, default=160)
    parser.add_argument(
        "--output", type=Path,
        default=Path("build/microprecision-profile/profile.json"),
    )
    parser.add_argument(
        "--scratch", type=Path,
        default=Path("build/microprecision-profile"),
    )
    parser.add_argument(
        "--mandelbrot", "--mandlebrot", nargs="?", const=(
            "build/microprecision-profile/mandelbrot-4limb.png"
        ), metavar="PNG",
    )
    parser.add_argument("--mandelbrot-only", action="store_true")
    parser.add_argument("--fractal-width", type=int, default=1280)
    parser.add_argument("--fractal-height", type=int, default=720)
    parser.add_argument("--fractal-iterations", type=int, default=96)
    parser.add_argument("--fractal-jobs", type=int, default=8)
    parser.add_argument("--fractal-compare", action="store_true")
    parser.add_argument("--fractal-center-x", default="-1.999985881")
    parser.add_argument("--fractal-center-y", default="0.0")
    parser.add_argument("--fractal-span", default=None)
    parser.add_argument("--fractal-resolution-bits", type=int, default=54)
    parser.add_argument("--list-operators", action="store_true")
    args = parser.parse_args(argv)

    if args.list_operators:
        print(" ".join(ALL_OPERATORS))
        return 0
    args.operators = _selection(args.operators, ALL_OPERATORS, "operators")
    args.backends = _selection(args.backends, ALL_BACKENDS, "backends")
    if any(width not in {1, 2, 3, 4} for width in args.widths):
        parser.error(
            "--widths accepts 1 (the plain float64 baseline -- no precision "
            "section, so accuracy shows what the ladder buys) and the "
            "implemented multi-limb widths 2, 3, 4"
        )
    if any(size <= 0 for size in args.sizes):
        parser.error("--sizes must be positive")
    if args.accuracy_samples <= 0 or args.oracle_dps < 40:
        parser.error("accuracy samples must be positive and oracle dps at least 40")

    mandelbrot_record = None
    if args.mandelbrot:
        mandelbrot_record = render_mandelbrot(
            Path(args.mandelbrot),
            width=args.fractal_width,
            height=args.fractal_height,
            iterations=args.fractal_iterations,
            center_x=args.fractal_center_x,
            center_y=args.fractal_center_y,
            span=args.fractal_span,
            resolution_bits=args.fractal_resolution_bits,
            oracle_dps=args.oracle_dps,
            compare=args.fractal_compare,
            jobs=args.fractal_jobs,
        )
        print(
            f"[mandelbrot] wrote {mandelbrot_record['path']} in "
            f"{mandelbrot_record['elapsed_seconds']:.3f}s",
            flush=True,
        )
    if args.mandelbrot_only:
        if not args.mandelbrot:
            parser.error("--mandelbrot-only requires --mandelbrot")
        return 0

    rows, builds = profile_matrix(
        operators=args.operators,
        backends=args.backends,
        widths=tuple(args.widths),
        sizes=tuple(args.sizes),
        repeats=args.repeats,
        warmups=args.warmups,
        accuracy_samples=args.accuracy_samples,
        oracle_dps=args.oracle_dps,
        scratch=args.scratch,
    )
    _print_rows(rows)
    csv_path = _write_results(args.output, rows, builds, args)
    print(f"wrote {args.output.resolve()}", flush=True)
    print(f"wrote {csv_path.resolve()}", flush=True)
    return 1 if any(row.status == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
