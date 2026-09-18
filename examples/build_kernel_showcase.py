"""Build the physics-kernel showcase page: sympy -> LLVM IR -> native + wasm.

    python examples/build_kernel_showcase.py
    python examples/build_kernel_showcase.py --destination C:/dev/Powershell

For each problem the update law is derived symbolically with sympy and
composed stepwise into straight-line LLVM IR (one shared emitter, {+,*,
integer powers, /, sqrt} -- sqrt lowers to the native instruction on both
targets, so no libm anywhere).  The SAME .ll is then compiled twice with
zig's bundled clang, ahead of time:

- native shared library (``-shared -O2``)  -> reference marks on this
  machine: numpy, serial, and pooled through the deployment layer's
  probe -> verdict -> selection path.
- wasm32-freestanding (``-O2 --no-entry``) -> embedded in the page so a
  visitor measures serial vs Web-Worker-pooled FRESH on their machine
  (copy-in/copy-out workers, the browser shell's own design).

No fast-math is enabled on either target: every tier (native, wasm, JS,
numpy) evaluates the identical operation order in IEEE double, so results
agree bitwise and the correctness gates are exact -- which matters
doubly for the chaotic kernel, where any reassociation would diverge.

Output: ``examples/kernel_showcase/index.html`` by default, assembled from
``kernel_showcase_template.html`` with the measured data and base64 wasm
modules injected.  With ``--destination``, the page is written instead to
``<destination>/site/demos/kernel-proving-ground/index.html`` -- the live,
committable location documented in ``PUBLISHING_BUNDLES_TO_ROOT.md`` at the
coordination repository's root (``C:\\dev\\Powershell``, not ``turing/``
itself, which the shared ``resolve_publish_root`` helper rejects for the
same reason it rejects it for compiled program bundles: nothing under
``turing/`` is served by GitHub Pages).
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sympy

from src.compiler.deployment_calibration import machine_fingerprint, probe_span
from src.compiler.deployment_host_pool import HostDeploymentPool
from src.compiler.deployment_lowering import select_deployment_strategy

ELEMENTS = 2_000_000
REPEATS = 3


# --- LLVM IR emitter: {+, *, integer pow, /, sqrt} -------------------------

def _double_literal(value: float) -> str:
    return "0x" + struct.pack(">d", float(value)).hex().upper()


class ExpressionEmitter:
    def __init__(self):
        self.lines: list[str] = []
        self.counter = 0
        self.needs_sqrt = False

    def fresh(self) -> str:
        self.counter += 1
        return f"%v{self.counter}"

    def binary(self, op: str, left: str, right: str) -> str:
        name = self.fresh()
        self.lines.append(f"  {name} = {op} double {left}, {right}")
        return name

    def emit(self, expression, bindings: Mapping) -> str:
        if expression in bindings:
            return bindings[expression]
        if expression.is_Number:
            return _double_literal(float(expression))
        if expression.is_Add:
            parts = [self.emit(part, bindings) for part in expression.args]
            result = parts[0]
            for part in parts[1:]:
                result = self.binary("fadd", result, part)
            return result
        if expression.is_Mul:
            parts = [self.emit(part, bindings) for part in expression.args]
            result = parts[0]
            for part in parts[1:]:
                result = self.binary("fmul", result, part)
            return result
        if expression.is_Pow:
            base, exponent = expression.args
            if exponent == sympy.Rational(1, 2):
                self.needs_sqrt = True
                operand = self.emit(base, bindings)
                name = self.fresh()
                self.lines.append(
                    f"  {name} = call double @llvm.sqrt.f64(double {operand})"
                )
                return name
            if exponent == sympy.Rational(-1, 2):
                root = self.emit(sympy.sqrt(base), bindings)
                return self.binary("fdiv", _double_literal(1.0), root)
            if exponent.is_Integer:
                power = int(exponent)
                if power < 0:
                    inverse = self.emit(sympy.Pow(base, -exponent), bindings)
                    return self.binary("fdiv", _double_literal(1.0), inverse)
                operand = self.emit(base, bindings)
                result = operand
                for _ in range(power - 1):
                    result = self.binary("fmul", result, operand)
                return result
        raise NotImplementedError(f"no LLVM spelling for {expression!r}")


class JSEmitter:
    """Mirror of ExpressionEmitter producing JS with the SAME association
    order, so the page's in-browser reference agrees with the wasm kernel
    bitwise -- load-bearing for the chaotic kernel, where any reordering
    amplifies through the iterations."""

    def emit(self, expression, bindings: Mapping) -> str:
        if expression in bindings:
            return bindings[expression]
        if expression.is_Number:
            return repr(float(expression))
        if expression.is_Add:
            parts = [self.emit(part, bindings) for part in expression.args]
            result = parts[0]
            for part in parts[1:]:
                result = f"({result} + {part})"
            return result
        if expression.is_Mul:
            parts = [self.emit(part, bindings) for part in expression.args]
            result = parts[0]
            for part in parts[1:]:
                result = f"({result} * {part})"
            return result
        if expression.is_Pow:
            base, exponent = expression.args
            if exponent == sympy.Rational(1, 2):
                return f"Math.sqrt({self.emit(base, bindings)})"
            if exponent == sympy.Rational(-1, 2):
                return f"(1.0 / Math.sqrt({self.emit(base, bindings)}))"
            if exponent.is_Integer:
                power = int(exponent)
                if power < 0:
                    inverse = self.emit(
                        sympy.Pow(base, -exponent), bindings,
                    )
                    return f"(1.0 / {inverse})"
                operand = self.emit(base, bindings)
                result = operand
                for _ in range(power - 1):
                    result = f"({result} * {operand})"
                return result
        raise NotImplementedError(f"no JS spelling for {expression!r}")


def emit_problem_js(problem: "Problem") -> str:
    """``{ solve: (a,b)=>..., residual: (o,a,b)=>... | null }`` in the
    emitter's own order.  The residual is the page's defense against a
    SHARED algorithmic error: agreement between wasm and the JS mirror
    proves faithful translation, not a correct answer -- plugging the
    root back into the equation proves the answer."""

    emitter = JSEmitter()
    bindings = {problem.inputs[0]: "a", problem.inputs[1]: "b"}
    lines: list[str] = []
    state_names = [f"s{index}" for index in range(len(problem.state))]
    for name, expression in zip(state_names, problem.init):
        lines.append(f"let {name} = {emitter.emit(expression, bindings)};")
    for step_index in range(problem.steps):
        frame = dict(bindings)
        frame.update(zip(problem.state, state_names))
        updates = [
            emitter.emit(expression, frame) for expression in problem.step
        ]
        # Simultaneous update: both new values are computed from the OLD
        # frame before either assignment, matching the IR emitter exactly.
        temporaries = [
            f"t{step_index}_{index}" for index in range(len(updates))
        ]
        for temporary, update in zip(temporaries, updates):
            lines.append(f"const {temporary} = {update};")
        for name, temporary in zip(state_names, temporaries):
            lines.append(f"{name} = {temporary};")
    frame = dict(bindings)
    frame.update(zip(problem.state, state_names))
    lines.append(f"return {emitter.emit(problem.output, frame)};")
    solve = "(a, b) => { " + " ".join(lines) + " }"
    residual = "null"
    if problem.residual is not None:
        output_symbol = sympy.Symbol("o", real=True)
        residual_bindings = {
            output_symbol: "o",
            problem.inputs[0]: "a",
            problem.inputs[1]: "b",
        }
        residual = (
            "(o, a, b) => Math.abs("
            + JSEmitter().emit(problem.residual, residual_bindings)
            + ")"
        )
    return f"{{ solve: {solve}, residual: {residual} }}"


# --- problem definitions ---------------------------------------------------

@dataclass(frozen=True)
class Problem:
    key: str
    title: str
    physics: str
    equation_html: str
    law_label: str
    state: tuple  # sympy symbols for iterated state
    inputs: tuple  # sympy symbols for the two input arrays (in0, in1)
    init: tuple  # initial state expressions in terms of inputs
    step: tuple  # state update expressions (in state + inputs)
    output: object  # output expression (in state + inputs)
    steps: int
    input_ranges: tuple  # ((low, high), (low, high))
    residual_kind: str  # "polynomial" | "convergence" | "reference"
    residual: object | None  # sympy expr in (output symbol o, inputs)


def build_problems() -> tuple[Problem, ...]:
    x, p, q = sympy.symbols("x p q", real=True)
    a, b, k = sympy.symbols("a b k", real=True, positive=True)
    r, x0 = sympy.symbols("r x0", real=True)
    o = sympy.Symbol("o", real=True)

    cubic_f = x**3 + p * x + q
    cubic_step = sympy.simplify(x - cubic_f / sympy.diff(cubic_f, x))
    quintic_f = x**5 + p * x + q
    quintic_step = sympy.simplify(x - quintic_f / sympy.diff(quintic_f, x))

    return (
        Problem(
            key="cubic",
            title="Cardano's cubic",
            physics=(
                "The depressed cubic x³ + px + q = 0 -- the equation "
                "Cardano published a radical formula for in 1545. Six "
                "sympy-derived Newton steps solve it to machine precision."
            ),
            equation_html="x<sup>3</sup> + p x + q = 0",
            law_label=f"x ← {sympy.sstr(cubic_step)}",
            state=(x,),
            inputs=(p, q),
            init=(sympy.Float(0),),
            step=(cubic_step,),
            output=x,
            steps=6,
            input_ranges=((1.0, 3.0), (-2.0, 2.0)),
            residual_kind="polynomial",
            residual=o**3 + p * o + q,
        ),
        Problem(
            key="quintic",
            title="The unsolvable quintic",
            physics=(
                "Abel and Ruffini proved the Bring quintic "
                "x⁵ + px + q = 0 has no solution in radicals. "
                "Newton's method does not care: eight compiled steps "
                "find the real root anyway."
            ),
            equation_html="x<sup>5</sup> + p x + q = 0",
            law_label=f"x ← {sympy.sstr(quintic_step)}",
            state=(x,),
            inputs=(p, q),
            init=(sympy.Float(0),),
            step=(quintic_step,),
            output=x,
            steps=8,
            input_ranges=((1.0, 3.0), (-2.0, 2.0)),
            residual_kind="polynomial",
            residual=o**5 + p * o + q,
        ),
        Problem(
            key="pendulum",
            title="Exact pendulum period",
            physics=(
                "Beyond the small-angle lie: the true period needs the "
                "complete elliptic integral K(k), computed here by the "
                "arithmetic–geometric mean -- Gauss's iteration, "
                "quadratically convergent, six steps. Output is T/T₀, "
                "how much slower a real pendulum swings at amplitude "
                "θ₀ = 2 arcsin(k)."
            ),
            equation_html=(
                "T = T<sub>0</sub> / agm(1, √(1 − k²))"
            ),
            law_label="(a, b) ← ((a+b)/2, √(ab))",
            state=(a, b),
            inputs=(k, sympy.Symbol("unused")),
            init=(sympy.Float(1), sympy.sqrt(1 - k**2)),
            step=((a + b) / 2, sympy.sqrt(a * b)),
            output=1 / a,
            steps=6,
            input_ranges=((0.01, 0.95), (0.0, 1.0)),
            residual_kind="convergence",
            residual=None,  # |a - b| after the loop, checked via reference
        ),
        Problem(
            key="chaos",
            title="Logistic-map chaos orbit",
            physics=(
                "Sixty-four iterations of x ← r x(1−x) in "
                "the chaotic regime. With no fast-math on any target, "
                "native, wasm, and numpy trajectories agree bit-for-bit "
                "-- a reassociating compiler would send each down a "
                "different orbit."
            ),
            equation_html="x ← r x (1 − x)",
            law_label="x ← r x(1−x), 64 iterations",
            state=(x,),
            inputs=(r, x0),
            init=(x0,),
            step=(r * x * (1 - x),),
            output=x,
            steps=64,
            input_ranges=((3.6, 3.99), (0.05, 0.95)),
            residual_kind="reference",
            residual=None,
        ),
    )


# --- IR assembly -----------------------------------------------------------

def emit_problem_ir(problem: Problem) -> str:
    emitter = ExpressionEmitter()
    body = emitter.lines
    body.extend((
        "  %ap = getelementptr double, ptr %in0, i64 %i",
        "  %av = load double, ptr %ap",
        "  %bp = getelementptr double, ptr %in1, i64 %i",
        "  %bv = load double, ptr %bp",
    ))
    bindings = {problem.inputs[0]: "%av", problem.inputs[1]: "%bv"}
    state_values = [
        emitter.emit(expression, bindings) for expression in problem.init
    ]
    for _ in range(problem.steps):
        frame = dict(bindings)
        frame.update(zip(problem.state, state_values))
        state_values = [
            emitter.emit(expression, frame) for expression in problem.step
        ]
    frame = dict(bindings)
    frame.update(zip(problem.state, state_values))
    result = emitter.emit(problem.output, frame)
    body.extend((
        "  %op = getelementptr double, ptr %out, i64 %i",
        f"  store double {result}, ptr %op",
    ))
    declarations = (
        ("declare double @llvm.sqrt.f64(double)",)
        if emitter.needs_sqrt else ()
    )
    return "\n".join((
        f'source_filename = "showcase.{problem.key}"',
        *declarations,
        "",
        "define void @solve_span(ptr %in0, ptr %in1, ptr %out, "
        "i64 %start, i64 %stop) {",
        "entry:",
        "  %enter = icmp slt i64 %start, %stop",
        "  br i1 %enter, label %body, label %done",
        "body:",
        "  %i = phi i64 [ %start, %entry ], [ %inext, %body ]",
        *body,
        "  %inext = add nsw i64 %i, 1",
        "  %continue = icmp slt i64 %inext, %stop",
        "  br i1 %continue, label %body, label %done",
        "done:",
        "  ret void",
        "}",
        "",
    ))


# --- compilation -----------------------------------------------------------

def _zig(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"compile failed ({completed.returncode}):\n"
            + completed.stderr[-2000:]
        )


def compile_native(ir: str, name: str, build_dir: Path) -> ctypes.CDLL:
    source = build_dir / f"{name}.ll"
    source.write_text(ir, encoding="utf-8")
    suffix = ".dll" if sys.platform == "win32" else ".so"
    library = build_dir / f"{name}{suffix}"
    _zig([
        sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
        "-o", str(library), str(source),
    ])
    handle = ctypes.CDLL(str(library))
    span = handle.solve_span
    span.restype = None
    span.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    return handle


def compile_wasm(ir: str, name: str, build_dir: Path) -> bytes:
    source = build_dir / f"{name}.ll"
    source.write_text(ir, encoding="utf-8")
    module = build_dir / f"{name}.wasm"
    _zig([
        sys.executable, "-m", "ziglang", "cc",
        "-target", "wasm32-freestanding", "-nostdlib", "-O2",
        "-Wl,--no-entry", "-Wl,--export=solve_span",
        "-o", str(module), str(source),
    ])
    return module.read_bytes()


# --- numpy reference (same sympy law, stepwise, same order) ---------------

def numpy_reference(problem: Problem) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    substitutions = (*problem.state, *problem.inputs)
    init_fns = [
        sympy.lambdify(problem.inputs, expression, "numpy")
        for expression in problem.init
    ]
    step_fns = [
        sympy.lambdify(substitutions, expression, "numpy")
        for expression in problem.step
    ]
    output_fn = sympy.lambdify(substitutions, problem.output, "numpy")

    def solve(in0: np.ndarray, in1: np.ndarray) -> np.ndarray:
        state = [
            np.broadcast_to(np.asarray(fn(in0, in1), dtype=np.float64),
                            in0.shape).copy()
            for fn in init_fns
        ]
        for _ in range(problem.steps):
            state = [
                np.asarray(fn(*state, in0, in1), dtype=np.float64)
                for fn in step_fns
            ]
        return np.asarray(output_fn(*state, in0, in1), dtype=np.float64)

    return solve


# --- expert numpy: in-place, buffer-reusing implementations -----------------
#
# The lambdify tier is what sympy GENERATES: every arithmetic op allocates a
# fresh full-size temporary, so an N-step law pays ~ops x N full memory round
# trips.  These are what a skilled numpy user would WRITE: preallocated
# buffers, ``out=`` everywhere.  They are hand-written per problem (four
# fixed kernels; a generic array-register allocator would be overkill) but
# drift-proof: each is gated at build time against the lambdify reference,
# and each PRESERVES the emitters' association order -- commutation is exact
# in IEEE, association is not, and the chaotic kernel amplifies any
# association change into divergence.

def _inplace_cubic(in0, in1, steps):
    x = np.zeros_like(in0)
    x2 = np.empty_like(in0)
    numer = np.empty_like(in0)
    denom = np.empty_like(in0)
    for _ in range(steps):
        np.multiply(x, x, out=x2)
        np.multiply(x2, x, out=numer)   # x^3
        numer *= 2.0
        numer -= in1                    # (-q + 2x^3), commuted: exact
        np.multiply(x2, 3.0, out=denom)
        denom += in0                    # (p + 3x^2)
        np.divide(numer, denom, out=x)
    return x


def _inplace_quintic(in0, in1, steps):
    x = np.zeros_like(in0)
    x2 = np.empty_like(in0)
    x4 = np.empty_like(in0)
    numer = np.empty_like(in0)
    denom = np.empty_like(in0)
    for _ in range(steps):
        np.multiply(x, x, out=x2)
        np.multiply(x2, x2, out=x4)
        np.multiply(x4, x, out=numer)   # x^5
        numer *= 4.0
        numer -= in1                    # (-q + 4x^5), commuted: exact
        np.multiply(x4, 5.0, out=denom)
        denom += in0                    # (p + 5x^4)
        np.divide(numer, denom, out=x)
    return x


def _inplace_pendulum(in0, in1, steps):
    a = np.ones_like(in0)
    b = np.empty_like(in0)
    np.multiply(in0, in0, out=b)
    np.subtract(1.0, b, out=b)
    np.sqrt(b, out=b)                   # b0 = sqrt(1 - k^2)
    mean = np.empty_like(in0)
    for _ in range(steps):
        np.add(a, b, out=mean)
        mean *= 0.5                     # (a+b)/2; *0.5 vs /2: exact
        np.multiply(a, b, out=b)
        np.sqrt(b, out=b)               # sqrt(ab)
        a, mean = mean, a
    return np.divide(1.0, a)


def _inplace_chaos(in0, in1, steps):
    x = in1.copy()
    rx = np.empty_like(in0)
    one_minus = np.empty_like(in0)
    for _ in range(steps):
        np.multiply(in0, x, out=rx)         # (r*x)
        np.subtract(1.0, x, out=one_minus)  # (1-x)
        np.multiply(rx, one_minus, out=x)   # (r*x)*(1-x): same association
    return x


INPLACE_REFERENCES = {
    "cubic": _inplace_cubic,
    "quintic": _inplace_quintic,
    "pendulum": _inplace_pendulum,
    "chaos": _inplace_chaos,
}


def best_of(runs: int, thunk) -> float:
    best = float("inf")
    for _ in range(runs):
        started = time.perf_counter()
        thunk()
        best = min(best, time.perf_counter() - started)
    return best


# --- native benchmark ------------------------------------------------------

def benchmark_native(problem: Problem, library: ctypes.CDLL) -> dict:
    rng = np.random.default_rng(11)
    (low0, high0), (low1, high1) = problem.input_ranges
    in0 = low0 + (high0 - low0) * rng.random(ELEMENTS)
    in1 = low1 + (high1 - low1) * rng.random(ELEMENTS)
    out = np.zeros(ELEMENTS)
    pointers = tuple(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        for array in (in0, in1, out)
    )
    span = library.solve_span

    def kernel(start: int, stop: int) -> None:
        span(*pointers, start, stop)

    reference = numpy_reference(problem)
    kernel(0, ELEMENTS)
    expected = reference(in0, in1)
    agreement = float(np.max(np.abs(out - expected)))
    if agreement > 1e-12:
        raise AssertionError(
            f"{problem.key}: native disagrees with numpy by {agreement:.3e}"
        )
    residual = None
    if problem.residual_kind == "polynomial":
        residual_fn = sympy.lambdify(
            (sympy.Symbol("o", real=True), *problem.inputs),
            problem.residual, "numpy",
        )
        residual = float(np.max(np.abs(residual_fn(out, in0, in1))))

    inplace = INPLACE_REFERENCES[problem.key]
    inplace_out = inplace(in0, in1, problem.steps)
    inplace_agreement = float(np.max(np.abs(inplace_out - expected)))
    if inplace_agreement > 1e-12:
        raise AssertionError(
            f"{problem.key}: in-place numpy drifted from lambdify by "
            f"{inplace_agreement:.3e}"
        )

    verdict = probe_span(
        kernel, ELEMENTS, backend="llvm",
        identity=f"showcase-{problem.key}", repeats=REPEATS,
    )
    choice = select_deployment_strategy(
        backend="llvm", execution_class="thread-workers",
        calibration=verdict,
    )
    serial_seconds = best_of(REPEATS, lambda: kernel(0, ELEMENTS))
    numpy_seconds = best_of(REPEATS, lambda: reference(in0, in1))
    numpy_fast_seconds = best_of(
        REPEATS, lambda: inplace(in0, in1, problem.steps),
    )
    if choice.strategy == "pool" and choice.workers:
        with HostDeploymentPool(workers=choice.workers) as pool:
            pool.deploy_span(kernel, ELEMENTS)
            pooled_seconds = best_of(
                REPEATS, lambda: pool.deploy_span(kernel, ELEMENTS),
            )
    else:
        pooled_seconds = serial_seconds
    print(
        f"  native {problem.key}: lambdify {numpy_seconds*1e3:.1f}ms, "
        f"in-place numpy {numpy_fast_seconds*1e3:.1f}ms, "
        f"serial {serial_seconds*1e3:.1f}ms, "
        f"pooled {pooled_seconds*1e3:.1f}ms @ {choice.workers} workers "
        f"(agreement {agreement:.1e}, in-place {inplace_agreement:.1e}"
        + (f", residual {residual:.1e}" if residual is not None else "")
        + ")"
    )
    return {
        "elements": ELEMENTS,
        "numpy_ms": numpy_seconds * 1e3,
        "numpy_fast_ms": numpy_fast_seconds * 1e3,
        "serial_ms": serial_seconds * 1e3,
        "pooled_ms": pooled_seconds * 1e3,
        "workers": choice.workers or 0,
        "probe_speedup": verdict.speedup,
        "agreement": agreement,
        "residual": residual,
    }


# --- page assembly ---------------------------------------------------------

def resolve_output_dir(destination: str | None) -> Path:
    """Local scratch output by default; the live site location when given
    a ``--destination`` (reusing the compiler's own gallery-root guard, so
    ``turing/`` itself is rejected exactly as it is for compiled bundles)."""

    if destination is None:
        return Path(__file__).parent / "kernel_showcase"
    from src.compiler.site_bundle import resolve_publish_root

    return resolve_publish_root(destination) / "site" / "demos" / "kernel-proving-ground"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", default=None,
        help=(
            "publish root (e.g. C:/dev/Powershell); writes to "
            "<destination>/site/demos/kernel-proving-ground/index.html. "
            "Omit to build locally into examples/kernel_showcase/."
        ),
    )
    args = parser.parse_args()

    problems = build_problems()
    build_dir = Path(tempfile.mkdtemp(prefix="kernel_showcase_"))
    data: dict[str, dict] = {
        "elements": ELEMENTS,
        "machine": machine_fingerprint(),
        "problems": {},
    }
    js_references: list[str] = []
    for problem in problems:
        print(f"building {problem.key}...")
        ir = emit_problem_ir(problem)
        library = compile_native(ir, problem.key, build_dir)
        wasm_bytes = compile_wasm(ir, problem.key, build_dir)
        native = benchmark_native(problem, library)
        js_references.append(
            f"  {problem.key}: {emit_problem_js(problem)},"
        )
        data["problems"][problem.key] = {
            "title": problem.title,
            "physics": problem.physics,
            "equation_html": problem.equation_html,
            "law_label": problem.law_label,
            "steps": problem.steps,
            "input_ranges": problem.input_ranges,
            "residual_kind": problem.residual_kind,
            "ir_lines": ir.count("\n"),
            "wasm_bytes": len(wasm_bytes),
            "wasm_base64": base64.b64encode(wasm_bytes).decode("ascii"),
            "native": native,
        }

    template_path = Path(__file__).parent / "kernel_showcase_template.html"
    output_dir = resolve_output_dir(args.destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = template_path.read_text(encoding="utf-8")
    page = template.replace(
        "__SHOWCASE_DATA__", json.dumps(data, separators=(",", ":")),
    ).replace(
        "__JS_REFERENCE__",
        "{\n" + "\n".join(js_references) + "\n}",
    )
    output_path = output_dir / "index.html"
    output_path.write_text(page, encoding="utf-8")
    print(f"\npage written: {output_path} "
          f"({output_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
