"""Self-contained LLVM signal math with selectable error tolerance.

Two independent circular-trigonometry implementations are emitted:

``lut``
    Period reduction followed by lookup and linear interpolation in an
    immutable sine table.  Cosine is a phase shift and tangent is a ratio.

``continuous``
    Period/quadrant reduction followed by a tolerance-selected odd polynomial.
    Cosine and tangent share the same continuous sine implementation.

The optional ``epsilon`` passed to :func:`build_llvm_signal_math` selects both
the LUT interval count and continuous polynomial degree from conservative
error bounds.  The functions are scalar ``f64`` legalization targets; tensor
iteration belongs to the C-kernel/tape translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import struct


@dataclass(frozen=True)
class LLVMSignalMath:
    llvm_ir: str
    epsilon: float
    lut_intervals: int
    lut_error_bound: float
    continuous_degree: int
    continuous_error_bound: float


class LLVMTrigSolver(str, Enum):
    LUT = "lut"
    CONTINUOUS = "continuous"


def _llvm_f64(value: float) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{bits:016X}"


def _power_of_two_ceiling(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _lut_intervals(epsilon: float) -> tuple[int, float]:
    # Linear interpolation error is bounded by M*h^2/8.  For sin, M <= 1.
    maximum_step = math.sqrt(8.0 * epsilon)
    required = max(4, math.ceil(math.tau / maximum_step))
    intervals = _power_of_two_ceiling(required)
    return intervals, (math.tau / intervals) ** 2 / 8.0


def _continuous_terms(epsilon: float) -> tuple[int, float]:
    # After quadrant reduction |x| <= pi/2.  The first omitted sine-series
    # term bounds the alternating-series remainder on this interval.
    radius = math.pi / 2.0
    terms = 1
    while True:
        omitted_power = 2 * terms + 1
        bound = radius ** omitted_power / math.factorial(omitted_power)
        if bound <= epsilon:
            return terms, bound
        terms += 1
        if terms > 10:
            # f64 rounding dominates beyond this point.
            return terms, bound


def _continuous_polynomial(terms: int) -> tuple[list[str], str]:
    coefficients = [
        ((-1.0) ** index) / math.factorial(2 * index + 1)
        for index in range(terms)
    ]
    lines = ["  %y2 = fmul double %y, %y"]
    accumulator = _llvm_f64(coefficients[-1])
    for index, coefficient in enumerate(reversed(coefficients[:-1]), start=1):
        product = f"%poly.product.{index}"
        total = f"%poly.sum.{index}"
        lines.append(f"  {product} = fmul double {accumulator}, %y2")
        lines.append(
            f"  {total} = fadd double {product}, {_llvm_f64(coefficient)}"
        )
        accumulator = total
    lines.append(f"  %result = fmul double %y, {accumulator}")
    return lines, "%result"


def build_llvm_signal_math(epsilon: float | None = None) -> LLVMSignalMath:
    """Build LUT and continuous LLVM trig solvers for ``epsilon``.

    ``epsilon`` is an absolute error target for sine/cosine on finite inputs
    for which f64 period reduction remains meaningful.  Tangent inherits and
    amplifies component error near its poles.
    """

    epsilon = 1.0e-6 if epsilon is None else float(epsilon)
    if not math.isfinite(epsilon) or not 1.0e-15 <= epsilon <= 1.0e-2:
        raise ValueError("trig epsilon must be finite and between 1e-15 and 1e-2")
    intervals, lut_bound = _lut_intervals(epsilon)
    terms, continuous_bound = _continuous_terms(epsilon)
    table = [math.sin(math.tau * index / intervals) for index in range(intervals + 1)]
    table_values = ", ".join(
        f"double {_llvm_f64(value)}" for value in table
    )
    polynomial_lines, polynomial_result = _continuous_polynomial(terms)
    tau = _llvm_f64(math.tau)
    pi = _llvm_f64(math.pi)
    half_pi = _llvm_f64(math.pi / 2.0)
    intervals_f64 = _llvm_f64(float(intervals))
    interval_max = intervals - 1

    llvm_ir = f"""
source_filename = "turing.llvm-signal-math"

@turing.sine.lut = private unnamed_addr constant [{intervals + 1} x double] [{table_values}], align 8

declare double @llvm.floor.f64(double)

define internal double @turing_wrap_tau_f64(double %x) {{
entry:
  %turns = fdiv double %x, {tau}
  %whole = call double @llvm.floor.f64(double %turns)
  %periods = fmul double %whole, {tau}
  %wrapped = fsub double %x, %periods
  ret double %wrapped
}}

define internal double @turing_lut_linear_f64(
    ptr %table, i64 %intervals, double %x) {{
entry:
  %intervals.f64 = uitofp i64 %intervals to double
  %normalized = fdiv double %x, {tau}
  %scaled = fmul double %normalized, %intervals.f64
  %base.f64 = call double @llvm.floor.f64(double %scaled)
  %base.raw = fptoui double %base.f64 to i64
  %past.last = icmp uge i64 %base.raw, %intervals
  %last = sub i64 %intervals, 1
  %base = select i1 %past.last, i64 %last, i64 %base.raw
  %next = add i64 %base, 1
  %base.as.f64 = uitofp i64 %base to double
  %weight = fsub double %scaled, %base.as.f64
  %left.ptr = getelementptr inbounds double, ptr %table, i64 %base
  %right.ptr = getelementptr inbounds double, ptr %table, i64 %next
  %left = load double, ptr %left.ptr, align 8
  %right = load double, ptr %right.ptr, align 8
  %delta = fsub double %right, %left
  %weighted = fmul double %weight, %delta
  %result = fadd double %left, %weighted
  ret double %result
}}

define double @turing_lut_sin_f64(double %x) {{
entry:
  %wrapped = call double @turing_wrap_tau_f64(double %x)
  %result = call double @turing_lut_linear_f64(
      ptr @turing.sine.lut, i64 {intervals}, double %wrapped)
  ret double %result
}}

define double @turing_lut_cos_f64(double %x) {{
entry:
  %shifted = fadd double %x, {half_pi}
  %result = call double @turing_lut_sin_f64(double %shifted)
  ret double %result
}}

define double @turing_lut_tan_f64(double %x) {{
entry:
  %sine = call double @turing_lut_sin_f64(double %x)
  %cosine = call double @turing_lut_cos_f64(double %x)
  %result = fdiv double %sine, %cosine
  ret double %result
}}

define double @turing_continuous_sin_f64(double %x) {{
entry:
  %shifted = fadd double %x, {pi}
  %turns = fdiv double %shifted, {tau}
  %whole = call double @llvm.floor.f64(double %turns)
  %periods = fmul double %whole, {tau}
  %positive = fsub double %shifted, %periods
  %reduced = fsub double %positive, {pi}
  %above = fcmp ogt double %reduced, {half_pi}
  %upper.reflected = fsub double {pi}, %reduced
  %upper = select i1 %above, double %upper.reflected, double %reduced
  %below = fcmp olt double %upper, {_llvm_f64(-math.pi / 2.0)}
  %lower.reflected = fsub double {_llvm_f64(-math.pi)}, %upper
  %y = select i1 %below, double %lower.reflected, double %upper
{chr(10).join(polynomial_lines)}
  ret double {polynomial_result}
}}

define double @turing_continuous_cos_f64(double %x) {{
entry:
  %shifted = fadd double %x, {half_pi}
  %result = call double @turing_continuous_sin_f64(double %shifted)
  ret double %result
}}

define double @turing_continuous_tan_f64(double %x) {{
entry:
  %sine = call double @turing_continuous_sin_f64(double %x)
  %cosine = call double @turing_continuous_cos_f64(double %x)
  %result = fdiv double %sine, %cosine
  ret double %result
}}
"""
    # Keep these literals available in the emitted text for inspection even
    # though the generic lookup obtains its interval count as an argument.
    llvm_ir += (
        f"\n; selected_epsilon={epsilon:.17g}"
        f"\n; lut_intervals={intervals}"
        f"\n; lut_scale={intervals_f64}"
        f"\n; lut_last_interval={interval_max}"
        f"\n; continuous_degree={2 * terms - 1}\n"
    )
    from llvmlite import binding as llvm

    module = llvm.parse_assembly(llvm_ir)
    module.verify()
    return LLVMSignalMath(
        llvm_ir=llvm_ir,
        epsilon=epsilon,
        lut_intervals=intervals,
        lut_error_bound=lut_bound,
        continuous_degree=2 * terms - 1,
        continuous_error_bound=continuous_bound,
    )


def link_llvm_trig_solver(
    llvm_ir: str,
    solver: LLVMTrigSolver | str,
    *,
    epsilon: float | None = None,
) -> tuple[str, LLVMSignalMath | None]:
    """Resolve C/libm trig calls to a selected LLVM implementation."""

    solver = LLVMTrigSolver(solver)
    signal = build_llvm_signal_math(epsilon)
    prefix = (
        "turing_lut"
        if solver is LLVMTrigSolver.LUT
        else "turing_continuous"
    )
    rewritten = llvm_ir
    selected_symbols: list[str] = []
    for operation in ("sin", "cos", "tan"):
        selected_symbol = f"{prefix}_{operation}_f64"
        selected_symbols.append(selected_symbol)
        rewritten = rewritten.replace(
            f"call double @{operation}(",
            f"call double @{selected_symbol}(",
        )

    from llvmlite import binding as llvm

    # LLVM parses each module independently before linking, so every selected
    # definition used by the consumer must first have an external declaration
    # in that consumer.  Linking then resolves these declarations to the exact
    # implementations emitted by ``build_llvm_signal_math``.
    declarations = "\n".join(
        f"declare double @{symbol}(double)"
        for symbol in selected_symbols
        if f"@{symbol}(" in rewritten
        and f"declare double @{symbol}(" not in rewritten
        and f"define double @{symbol}(" not in rewritten
    )
    if declarations:
        rewritten = f"{rewritten.rstrip()}\n\n{declarations}\n"

    destination = llvm.parse_assembly(rewritten)
    source = llvm.parse_assembly(signal.llvm_ir)
    llvm.link_modules(destination, source)
    destination.verify()
    return str(destination), signal


__all__ = [
    "LLVMSignalMath",
    "LLVMTrigSolver",
    "build_llvm_signal_math",
    "link_llvm_trig_solver",
]
