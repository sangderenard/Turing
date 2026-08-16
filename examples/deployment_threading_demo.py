"""Standalone demo: a sympy-derived solver, LLVM-compiled, threaded by the
deployment layer -- with numpy as the reference speed.

    python examples/deployment_threading_demo.py [elements]

What it shows, end to end:

1. sympy derives the Newton update law for the depressed cubic
   x^3 + p*x + q = 0 symbolically (x - f/f'), and the demo composes six
   steps of it into straight-line LLVM IR (stepwise SSA, not naive
   substitution -- symbolic substitution would grow the tree ~4^n).
2. The IR is compiled AHEAD OF TIME to a shared library with the same
   toolchain ``ssa_llvm_backend.compile_artifact`` uses (zig's bundled
   clang: ``python -m ziglang cc -shared -O2`` over the ``.ll``), then
   loaded with ctypes -- the exact shape of a shipped artifact.  ctypes
   calls release the GIL, so host threads really overlap.
3. The deployment layer "tests the water": ``probe_span`` measures serial
   against a ladder of pool sizes, produces a calibration verdict, and
   ``select_deployment_strategy`` turns the measurement into the decision
   -- the same machinery every bundle build now runs as a stage.
4. numpy evaluates the identical six sympy steps vectorized, as the
   reference everyone knows.

Correctness is checked two ways before any timing is believed: the LLVM
and numpy roots must agree elementwise, and the roots must actually
satisfy the cubic (residual check) -- a demo that outruns a wrong answer
demonstrates nothing.
"""

from __future__ import annotations

import ctypes
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sympy

from src.compiler.deployment_calibration import probe_span
from src.compiler.deployment_host_pool import HostDeploymentPool
from src.compiler.deployment_lowering import select_deployment_strategy

NEWTON_STEPS = 6


# --- 1. sympy: derive the update law symbolically --------------------------

def derive_newton_step():
    x, p, q = sympy.symbols("x p q", real=True)
    f = x**3 + p * x + q
    step = sympy.simplify(x - f / sympy.diff(f, x))
    return (x, p, q), step


# --- 2. emit the derived expression as LLVM IR, stepwise -------------------

def _double_literal(value: float) -> str:
    # LLVM accepts IEEE-754 bit patterns as 0x<16 hex digits>: exact, no
    # decimal-representability pitfalls.
    return "0x" + struct.pack(">d", float(value)).hex().upper()


class _ExpressionEmitter:
    """Tiny sympy -> LLVM IR emitter for {+, *, integer powers, /}."""

    def __init__(self):
        self.lines: list[str] = []
        self.counter = 0

    def fresh(self) -> str:
        self.counter += 1
        return f"%v{self.counter}"

    def binary(self, op: str, left: str, right: str) -> str:
        name = self.fresh()
        self.lines.append(f"  {name} = {op} double {left}, {right}")
        return name

    def emit(self, expression, bindings: dict) -> str:
        if expression in bindings:
            return bindings[expression]
        if expression.is_Number:
            return _double_literal(float(expression))
        if expression.is_Add:
            terms = [self.emit(term, bindings) for term in expression.args]
            result = terms[0]
            for term in terms[1:]:
                result = self.binary("fadd", result, term)
            return result
        if expression.is_Mul:
            factors = [self.emit(factor, bindings) for factor in expression.args]
            result = factors[0]
            for factor in factors[1:]:
                result = self.binary("fmul", result, factor)
            return result
        if expression.is_Pow:
            base, exponent = expression.args
            if exponent.is_Integer:
                power = int(exponent)
                if power < 0:
                    inverse = self.emit(sympy.Pow(base, -exponent), bindings)
                    return self.binary(
                        "fdiv", _double_literal(1.0), inverse,
                    )
                emitted = self.emit(base, bindings)
                result = emitted
                for _ in range(power - 1):
                    result = self.binary("fmul", result, emitted)
                return result
        raise NotImplementedError(f"no LLVM spelling for {expression!r}")


def emit_span_kernel(symbols, step, *, steps: int) -> str:
    x, p, q = symbols
    emitter = _ExpressionEmitter()
    body = emitter.lines
    body.extend((
        "  %pp = getelementptr double, ptr %p, i64 %i",
        "  %pv = load double, ptr %pp",
        "  %qp = getelementptr double, ptr %q, i64 %i",
        "  %qv = load double, ptr %qp",
    ))
    current = _double_literal(0.0)  # x0 = 0: f' = p >= 1 keeps it safe
    for _ in range(steps):
        current = emitter.emit(
            step, {x: current, p: "%pv", q: "%qv"},
        )
    body.extend((
        f"  %op = getelementptr double, ptr %out, i64 %i",
        f"  store double {current}, ptr %op",
    ))
    return "\n".join((
        'source_filename = "deployment.threading.demo"',
        "",
        "define void @solve_span(ptr %p, ptr %q, ptr %out, "
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


# --- 3. AOT compile: the exact toolchain compile_artifact uses -------------

def aot_compile(llvm_ir: str, *, name: str) -> ctypes.CDLL:
    """LLVM IR -> shared library, ahead of time, via zig's bundled clang.

    Mirrors ``ssa_llvm_backend.compile_artifact`` line for line: write the
    ``.ll``, run ``python -m ziglang cc -shared -O2``, load the library.
    No JIT anywhere -- this is the shipped-artifact shape.
    """

    build_dir = Path(tempfile.mkdtemp(prefix=f"aot_{name}_"))
    source = build_dir / f"{name}.ll"
    source.write_text(llvm_ir, encoding="utf-8")
    suffix = ".dll" if sys.platform == "win32" else ".so"
    library = build_dir / f"{name}{suffix}"
    command = [
        sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
        "-o", str(library), str(source),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0 or not library.is_file():
        raise RuntimeError(
            f"AOT compile failed ({completed.returncode}):\n"
            + completed.stderr[-2000:]
        )
    print(f"AOT: {source.name} -> {library.name} (ziglang clang, -O2)")
    return ctypes.CDLL(str(library))


# --- timing helper ---------------------------------------------------------

def best_of(runs: int, thunk) -> float:
    best = float("inf")
    for _ in range(runs):
        started = time.perf_counter()
        thunk()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> int:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    repeats = 3

    symbols, step = derive_newton_step()
    print("sympy update law:", step)
    print(f"composed {NEWTON_STEPS} Newton steps into straight-line SSA")

    ir = emit_span_kernel(symbols, step, steps=NEWTON_STEPS)
    library = aot_compile(ir, name="cubic_newton")
    span = library.solve_span
    span.restype = None
    span.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
    ]

    rng = np.random.default_rng(7)
    p_values = 1.0 + 2.0 * rng.random(total)
    q_values = -2.0 + 4.0 * rng.random(total)
    llvm_roots = np.zeros(total)
    pointers = tuple(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        for array in (p_values, q_values, llvm_roots)
    )

    def kernel(start: int, stop: int) -> None:
        span(*pointers, start, stop)

    # --- correctness before speed ---
    step_numpy = sympy.lambdify(symbols, step, "numpy")

    def numpy_solve() -> np.ndarray:
        roots = np.zeros(total)
        for _ in range(NEWTON_STEPS):
            roots = step_numpy(roots, p_values, q_values)
        return roots

    kernel(0, total)
    numpy_roots = numpy_solve()
    agreement = float(np.max(np.abs(llvm_roots - numpy_roots)))
    residual = float(np.max(np.abs(
        llvm_roots**3 + p_values * llvm_roots + q_values
    )))
    print(f"llvm vs numpy agreement: {agreement:.3e}; "
          f"cubic residual: {residual:.3e}")
    if agreement > 1e-9 or residual > 1e-6:
        print("CORRECTNESS FAILED -- refusing to time a wrong answer")
        return 1

    # --- test the water: probe -> verdict -> selection ---
    print(f"\nprobing {total:,} elements (serial vs worker ladder)...")
    verdict = probe_span(
        kernel, total,
        backend="llvm", identity="cubic-newton-6",
        repeats=repeats,
    )
    choice = select_deployment_strategy(
        backend="llvm", execution_class="thread-workers",
        calibration=verdict,
    )
    print(f"verdict: {verdict.best_strategy} "
          f"({verdict.speedup:.2f}x at {verdict.best_workers} workers)")
    print(f"selection: {choice.strategy}"
          + (f" @ {choice.workers} workers" if choice.workers else ""))
    for reason in choice.reasons:
        print("  -", reason)

    # --- the three tiers, measured on equal terms ---
    serial_seconds = best_of(repeats, lambda: kernel(0, total))
    numpy_seconds = best_of(repeats, numpy_solve)
    if choice.strategy == "pool" and choice.workers:
        with HostDeploymentPool(workers=choice.workers) as pool:
            pool.deploy_span(kernel, total)  # warmup
            pooled_seconds = best_of(
                repeats, lambda: pool.deploy_span(kernel, total),
            )
    else:
        pooled_seconds = serial_seconds

    def row(label: str, seconds: float) -> str:
        rate = total / seconds / 1e6
        return (f"  {label:<26} {seconds * 1e3:9.2f} ms   "
                f"{rate:8.1f} Melem/s   {numpy_seconds / seconds:5.2f}x numpy")

    print(f"\nresults ({total:,} cubics solved, best of {repeats}):")
    print(row("numpy (vectorized sympy)", numpy_seconds))
    print(row("llvm serial", serial_seconds))
    print(row(
        f"llvm + pool ({choice.workers or 0} workers)", pooled_seconds,
    ))
    print(f"\n  threading impact on llvm: "
          f"{serial_seconds / pooled_seconds:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
