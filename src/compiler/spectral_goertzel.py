"""Goertzel single-bin DFT, authored as a straight-line SymPy law.

The Goertzel recurrence for N samples and bin k is

    s[-2] = s[-1] = 0
    s[n] = x[n] + 2*cos(w)*s[n-1] - s[n-2],   w = 2*pi*k/N

with the bin's real/imaginary parts read from the last two states.  N and k
are compile-time (the window a law is authored for, exactly like a tire law
is authored for one fixed mesh), so the recurrence is unrolled into one
SymPy expression per sample -- there is no loop in the authored law, only
the same kind of long CSE-shared expression the membrane law already is.
That keeps this on the sanctioned SymPy -> AbstractTensor -> native path:
lower_ast_source_to_ssa with a batch ExtractionContract, the same route
used for every law in native_law_kernels.py.
"""

from __future__ import annotations

import math

import sympy

from .symbolic_equation_compiler import (
    SymbolicEquationCompilation, SymbolicPublication, compile_sympy_equations)


def symbolic_goertzel_bin_equations(
    window: int, bin_index: int,
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Unrolled Goertzel recurrence for one DFT bin of a ``window``-sample block."""

    if window < 2:
        raise ValueError("goertzel window must be at least 2 samples")
    samples = {
        f"sample_{index}": sympy.Symbol(f"sample_{index}", real=True)
        for index in range(window)
    }
    omega = 2.0 * math.pi * float(bin_index) / float(window)
    coefficient = sympy.Float(2.0 * math.cos(omega))
    previous_two, previous_one = sympy.Integer(0), sympy.Integer(0)
    for index in range(window):
        current = samples[f"sample_{index}"] + coefficient * previous_one - previous_two
        previous_two, previous_one = previous_one, current
    # X[k] = s[N-1]*e^{iw} - s[N-2] against numpy's forward convention
    # (X[k] = sum x[n] e^{-2 pi i k n/N}), verified numerically against
    # np.fft.fft bit-for-bit before landing this formula.
    real = previous_one * sympy.Float(math.cos(omega)) - previous_two
    imaginary = previous_one * sympy.Float(math.sin(omega))
    power = (
        previous_one ** 2 + previous_two ** 2
        - coefficient * previous_one * previous_two
    )
    equations = (
        sympy.Eq(sympy.Symbol("goertzel_real", real=True), real, evaluate=False),
        sympy.Eq(sympy.Symbol("goertzel_imag", real=True), imaginary, evaluate=False),
        sympy.Eq(sympy.Symbol("goertzel_power", real=True), power, evaluate=False),
        sympy.Eq(
            sympy.Symbol("goertzel_magnitude", real=True),
            sympy.sqrt(power), evaluate=False,
        ),
    )
    return equations, samples


def compile_goertzel_bin_ssa(window: int, bin_index: int) -> SymbolicEquationCompilation:
    """Compile the Goertzel law for one (window, bin) pair to repository SSA."""

    equations, _ = symbolic_goertzel_bin_equations(window, bin_index)
    return compile_sympy_equations(
        equations,
        name=f"goertzel_bin_{bin_index}_of_{window}",
        publications=tuple(
            SymbolicPublication(str(equation.lhs), f"spectral.goertzel.{equation.lhs}")
            for equation in equations
        ),
        dtype="float64",
    )


__all__ = ["symbolic_goertzel_bin_equations", "compile_goertzel_bin_ssa"]
