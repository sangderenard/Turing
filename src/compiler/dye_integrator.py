"""The integrator dye flow runs through, with its convergence stated.

Transport deposits dye and moves it along edges. What a reader actually
wants is the *accumulated* result -- everything that ever arrives at a
location, over every path, of every length. That is an integral over
paths, and this module is the thing that performs it, reports whether it
converged, and says what it converged to.

The mathematics
---------------
Let ``A`` be the transport operator: ``A[i, j]`` is the fraction of j's
dye that crosses to i in one step. Let ``s`` be the injection at the
sources. Total accumulation satisfies

    x = A x + s        so        x = (I - A)^-1 s = sum_k A^k s

the Neumann series -- literally the sum over all path lengths, which is
why this is an integrator rather than a smoother. It converges precisely
when the spectral radius rho(A) < 1, and then

    ||x - x_k|| <= rho^k / (1 - rho) * ||s||

so the iteration count needed for a tolerance is known in advance rather
than discovered by running out of patience.

Why convergence is guaranteed here, not hoped for
-------------------------------------------------
Divide-transport makes each column of ``A`` sum to at most one: a node
hands out its dye among its successors and never manufactures more. A
column-substochastic non-negative matrix has ``rho(A) <= 1``, and strictly
below one as soon as any dye leaves the system -- which it does at every
sink, and at every back edge through ``decay < 1``. So the series
converges on any graph, cyclic or not, with no trip count, no unrolling
policy and no guess.

That is also what makes a dead path provably dead. Under a convergent
operator the residual bounds the tail exactly, so a location still at zero
after saturation is at zero because nothing reaches it -- not because the
iteration stopped early.

Reported, not assumed
---------------------
``integrate`` returns the residual at every step and an estimate of
``rho`` from the ratio of successive residuals. If the residuals are not
contracting, that shows up in the record instead of being hidden behind a
fixed iteration count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import sympy

#: Symbols for the convergence statement, so the bound can be manipulated
#: rather than only quoted in a docstring.
RHO = sympy.Symbol("rho", nonnegative=True)
STEP = sympy.Symbol("k", integer=True, nonnegative=True)
SOURCE_NORM = sympy.Symbol("s_norm", nonnegative=True)

#: ||x - x_k|| <= rho^k / (1 - rho) * ||s||
TAIL_BOUND = RHO ** STEP / (1 - RHO) * SOURCE_NORM


def steps_for_tolerance(rho: float, tolerance: float, source_norm: float = 1.0):
    """Smallest k with the tail bound under `tolerance`, solved symbolically.

    Returns ``None`` when rho >= 1, where the series does not converge and
    no iteration count is enough -- which is the honest answer rather than
    a large number.
    """
    if not (0.0 <= float(rho) < 1.0):
        return None
    if float(rho) == 0.0:
        return 0
    # Closed form, not a solver call. Inverting rho^k/(1-rho)*s <= tol is
    # one logarithm; handing it to sympy.solve invited it to search, and
    # on the log form it did -- the first version of this function hung
    # rather than returning, which is a poor way for a bound to behave.
    ratio = float(tolerance) * (1.0 - float(rho)) / max(float(source_norm), 1e-300)
    if ratio <= 0.0:
        return None
    return int(sympy.ceiling(
        sympy.log(sympy.Float(ratio)) / sympy.log(sympy.Float(rho))
    ))


@dataclass(frozen=True, slots=True)
class Integration:
    """What the integrator did, so the result can be trusted or rejected."""

    accumulated: Any
    residuals: tuple[float, ...]
    converged: bool
    tolerance: float

    @property
    def steps(self) -> int:
        return len(self.residuals)

    @property
    def contraction(self) -> float:
        """Observed rho, from the ratio of the last two residuals.

        Measured rather than assumed. A value at or above one means the
        operator did not contract on this graph and the accumulation is
        not a fixed point, whatever the residual happens to read.
        """
        if len(self.residuals) < 2:
            return 0.0
        previous, last = self.residuals[-2], self.residuals[-1]
        return 0.0 if previous <= 0.0 else last / previous

    def report(self) -> str:
        state = "converged" if self.converged else "DID NOT CONVERGE"
        return (
            f"{state} in {self.steps} steps, "
            f"residual {self.residuals[-1]:.3e} <= {self.tolerance:.3e}, "
            f"observed contraction {self.contraction:.4f}"
        )


def integrate(
    transport: Mapping[int, Sequence[tuple[int, float]]],
    injection: Mapping[int, float],
    size: int,
    *,
    tolerance: float = 1e-9,
    max_steps: int = 10_000,
) -> Integration:
    """Sum A^k s until the increment falls under `tolerance`.

    ``transport[j]`` lists ``(i, weight)``: the share of j's dye that
    crosses to i. Accumulation is by repeated application rather than by
    inverting ``(I - A)``, because the operator is sparse and an inverse
    would be dense -- and because the partial sums ARE the answer at every
    step, which makes the result anytime.
    """
    from ..common.tensors.abstraction import AbstractTensor

    accumulated = AbstractTensor.zeros((size,), dtype=float)
    frontier = AbstractTensor.zeros((size,), dtype=float)
    values = [0.0] * size
    for index, amount in injection.items():
        values[int(index)] = float(amount)
    frontier = AbstractTensor.tensor(values)
    accumulated = accumulated + frontier

    residuals: list[float] = []
    converged = False
    diverged = False
    for _step in range(int(max_steps)):
        carried = [0.0] * size
        current = frontier.tolist()
        for origin, edges in transport.items():
            held = current[int(origin)]
            if held == 0.0:
                continue
            for target, share in edges:
                carried[int(target)] += held * float(share)
        frontier = AbstractTensor.tensor(carried)
        accumulated = accumulated + frontier
        residual = float(frontier.abs().sum().item())
        # Sanity breaks. A contracting operator is the premise, not an
        # observation, so the loop verifies it instead of grinding to
        # max_steps accumulating garbage. Each of these stops immediately
        # and is visible in the record as a failure to converge.
        if not (residual == residual) or residual in (
            float("inf"), float("-inf"),
        ):
            diverged = True
            break
        residuals.append(residual)
        if residual <= tolerance:
            converged = True
            break
        if len(residuals) >= 2 and residuals[-1] > residuals[-2]:
            # Growing residual means rho >= 1 on this graph: the series
            # does not converge and no further step improves it.
            diverged = True
            break

    return Integration(
        accumulated=accumulated,
        residuals=tuple(residuals),
        converged=converged and not diverged,
        tolerance=float(tolerance),
    )


def column_masses(
    transport: Mapping[int, Sequence[tuple[int, float]]],
) -> dict[int, float]:
    """Outgoing share per node: the substochasticity check, per column.

    A column above one manufactures dye and breaks the convergence
    guarantee, so this is worth asserting rather than trusting. Copy
    transport does exactly that at a fan-out, which is why divide is the
    default on cyclic graphs.
    """
    return {
        int(origin): sum(float(share) for _target, share in edges)
        for origin, edges in transport.items()
    }


__all__ = [
    "RHO", "STEP", "SOURCE_NORM", "TAIL_BOUND", "Integration",
    "steps_for_tolerance", "integrate", "column_masses",
]
