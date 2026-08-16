"""Selectable mathematical constants with explicit approximation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import struct


class PiSolver(str, Enum):
    """How a backend must materialize the repository ``Pi`` operation."""

    LITERAL = "literal"
    MACHIN = "machin"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class PiMaterialization:
    solver: PiSolver
    value: float | None
    absolute_error_bound: float | None
    terms_atan_1_5: int = 0
    terms_atan_1_239: int = 0
    llvm_function: str = ""
    llvm_symbol: str | None = None
    requested_epsilon: float | None = None

    def contract(self) -> dict[str, object]:
        return {
            "constant_identity": "pi",
            "constant_solver": self.solver.value,
            "absolute_error_bound": self.absolute_error_bound,
            "requested_epsilon": self.requested_epsilon,
            "terms_atan_1_5": self.terms_atan_1_5,
            "terms_atan_1_239": self.terms_atan_1_239,
        }


def _double_literal(value: float) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{bits:016X}"


def _atan_sum(x: float, terms: int) -> float:
    return sum(
        (-1.0) ** index * x ** (2 * index + 1) / (2 * index + 1)
        for index in range(terms)
    )


def _atan_remainder(x: float, terms: int) -> float:
    return x ** (2 * terms + 1) / (2 * terms + 1)


def _machin_terms(epsilon: float) -> tuple[int, int, float]:
    terms_5 = terms_239 = 1
    while True:
        bound_5 = 16.0 * _atan_remainder(1.0 / 5.0, terms_5)
        bound_239 = 4.0 * _atan_remainder(1.0 / 239.0, terms_239)
        if bound_5 + bound_239 <= epsilon:
            return terms_5, terms_239, bound_5 + bound_239
        if bound_5 >= bound_239:
            terms_5 += 1
        else:
            terms_239 += 1


def _llvm_atan_series(prefix: str, x: float, terms: int) -> tuple[list[str], str]:
    lines: list[str] = []
    accumulator = "0x0000000000000000"
    for index in range(terms):
        magnitude = x ** (2 * index + 1) / (2 * index + 1)
        register = f"%{prefix}.{index}"
        operation = "fadd" if index % 2 == 0 else "fsub"
        lines.append(
            f"  {register} = {operation} double {accumulator}, "
            f"{_double_literal(magnitude)}"
        )
        accumulator = register
    return lines, accumulator


def materialize_pi(
    solver: PiSolver | str = PiSolver.LITERAL,
    epsilon: float | None = None,
) -> PiMaterialization:
    """Return a literal or a Machin-series π implementation and its bound."""

    solver = PiSolver(solver)
    if solver is PiSolver.REJECT:
        return PiMaterialization(solver, None, None)
    if solver is PiSolver.LITERAL:
        # Correctly-rounded f64 literal: half an ulp bounds representation
        # error relative to the exact real constant.
        return PiMaterialization(
            solver,
            math.pi,
            math.ulp(math.pi) * 0.5,
        )
    epsilon = 1.0e-12 if epsilon is None else float(epsilon)
    if not math.isfinite(epsilon) or not 1.0e-15 <= epsilon <= 1.0e-2:
        raise ValueError("pi epsilon must be finite and between 1e-15 and 1e-2")
    terms_5, terms_239, bound = _machin_terms(epsilon)
    atan_5 = _atan_sum(1.0 / 5.0, terms_5)
    atan_239 = _atan_sum(1.0 / 239.0, terms_239)
    value = 16.0 * atan_5 - 4.0 * atan_239
    lines_5, result_5 = _llvm_atan_series("atan5", 1.0 / 5.0, terms_5)
    lines_239, result_239 = _llvm_atan_series(
        "atan239", 1.0 / 239.0, terms_239
    )
    symbol = "turing_machin_pi_f64"
    llvm_function = "\n".join((
        f"define internal double @{symbol}() {{",
        "entry:",
        *lines_5,
        *lines_239,
        f"  %scaled5 = fmul double {result_5}, {_double_literal(16.0)}",
        f"  %scaled239 = fmul double {result_239}, {_double_literal(4.0)}",
        "  %pi = fsub double %scaled5, %scaled239",
        "  ret double %pi",
        "}",
    ))
    return PiMaterialization(
        solver,
        value,
        bound,
        terms_5,
        terms_239,
        llvm_function,
        symbol,
        epsilon,
    )


#: Odd Taylor coefficients for sin(r), outermost first, for Horner in r**2:
#:     sin(r) = r * (((...)*r2 + c) ... )
#: Used after reducing the argument onto [-pi/2, pi/2], where truncating here
#: leaves (pi/2)**15 / 15! ~ 7e-10. One definition, rendered by each backend in
#: its own syntax, so the lanes cannot drift into different series.
SIN_SERIES_COEFFICIENTS: tuple[float, ...] = (
    1.0 / 6227020800.0, -1.0 / 39916800.0, 1.0 / 362880.0,
    -1.0 / 5040.0, 1.0 / 120.0, -1.0 / 6.0, 1.0,
)

#: The truncation bound the coefficients above deliver over the reduced range.
SIN_SERIES_ERROR_BOUND = 7.0e-10


def sin_series_terms() -> tuple[tuple[float, ...], float, float]:
    """(coefficients, pi, error bound) for the reduced-argument sine series.

    A backend renders the reduction -- k = nearest(x/pi), r = x - k*pi, and a
    sign flip on odd k -- in its own syntax, but the constant and the series
    it evaluates come from here rather than being restated per lane.
    """

    return SIN_SERIES_COEFFICIENTS, float(materialize_pi("literal").value),         SIN_SERIES_ERROR_BOUND


__all__ = [
    "PiMaterialization", "PiSolver", "materialize_pi",
    "SIN_SERIES_COEFFICIENTS", "SIN_SERIES_ERROR_BOUND", "sin_series_terms",
]
