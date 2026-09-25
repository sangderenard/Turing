"""Lagrange in three senses, written in AbstractTensor so the compiler takes it as-is.

* **Interpolation and interpolatory quadrature.** Through ``K`` nodes the
  Lagrange basis ``l_k`` reproduces every polynomial of degree ``< K``, so
  ``sum_k w_k f(x_k)`` with ``w_k = integral of l_k`` integrates such a
  polynomial exactly.  The weights come from the moment system
  ``sum_k w_k x_k**j = integral of x**j`` -- one small linear solve, valid for
  ANY node set: equispaced (Newton-Cotes), Chebyshev, Legendre, a spline
  patch's points, or nodes supplied at run time.
* **Lagrangian mechanics.** The action ``S = integral of L(q, dq/dt, t) dt``
  is resolved on a path given by its values at the nodes: ``dq/dt`` through
  the basis's differentiation matrix, the integral through the weights
  above.  ``dS/dq`` at the free nodes is the discrete Euler-Lagrange
  residual; a stationary path makes it vanish.
* **Lagrange multipliers.** The stationary point of a quadratic
  ``x.A.x/2 - b.x`` under linear constraints ``C x = d`` is one KKT solve,
  returning ``x`` and the multipliers.

Everything is ordinary tensor arithmetic plus ``linalg.solve``; nothing here
reaches past the AbstractTensor surface.
"""

from __future__ import annotations

from typing import Any, Callable

from .abstraction import AbstractTensor
from . import linalg


def _tensor(value: Any) -> AbstractTensor:
    return value if isinstance(value, AbstractTensor) else AbstractTensor.get_tensor(value)


def _off_diagonal_mask(count: int, like: AbstractTensor) -> AbstractTensor:
    """``(K, K)`` with 1 off the diagonal and 0 on it."""

    identity = linalg.eye(count)
    return identity * 0.0 + 1.0 - identity


def lagrange_basis(nodes: Any, points: Any) -> AbstractTensor:
    """``l_k(x)`` for every node ``k`` and every point: shape ``(K, *points.shape)``.

    ``l_k(x) = prod_{m != k} (x - x_m) / (x_k - x_m)``, evaluated as one
    masked product: the ``m == k`` factor is replaced by 1.
    """

    nodes = _tensor(nodes).reshape(-1)
    points = _tensor(points)
    count = int(nodes.shape[0])
    flat = points.reshape(-1)
    mask = _off_diagonal_mask(count, nodes)                      # (K, K)
    # numerator factors (x - x_m) for every (k, m, point); m == k -> 1
    shift = flat.reshape(1, 1, -1) - nodes.reshape(1, count, 1)  # (1, K, P)
    numerator = (shift * mask.reshape(count, count, 1)
                 + (1.0 - mask.reshape(count, count, 1))).prod(1)   # (K, P)
    spacing = nodes.reshape(count, 1) - nodes.reshape(1, count)    # (K, K)
    denominator = (spacing * mask + (1.0 - mask)).prod(1)          # (K,)
    basis = numerator / denominator.reshape(count, 1)
    return basis.reshape(count, *tuple(points.shape))


def lagrange_interpolate(nodes: Any, values: Any, points: Any) -> AbstractTensor:
    """``p(x) = sum_k values_k l_k(x)``; ``values`` has the nodes on axis 0."""

    basis = lagrange_basis(nodes, points)
    values = _tensor(values)
    count = int(basis.shape[0])
    trailing = (1,) * (len(tuple(basis.shape)) - 1)
    return (values.reshape(count, *trailing) * basis).sum(0)


def lagrange_weights(nodes: Any, lower: Any, upper: Any) -> AbstractTensor:
    """Interpolatory quadrature weights on ``[lower, upper]`` for these nodes.

    Solves ``sum_k w_k x_k**j = (upper**(j+1) - lower**(j+1)) / (j+1)`` for
    ``j < K``: exact for every polynomial of degree ``< K``.  Keep the nodes
    inside the interval and moderate in number (the Vandermonde system is the
    usual interpolation conditioning); Chebyshev or Legendre nodes behave.
    """

    nodes = _tensor(nodes).reshape(-1)
    count = int(nodes.shape[0])
    powers = AbstractTensor.arange(count) * 1.0                       # (K,)
    vandermonde = nodes.reshape(1, count) ** powers.reshape(count, 1)  # (j, k)
    lower = _tensor(lower) * 1.0
    upper = _tensor(upper) * 1.0
    moments = (upper ** (powers + 1.0) - lower ** (powers + 1.0)) / (powers + 1.0)
    return linalg.solve(vandermonde, moments.reshape(count, 1)).reshape(count)


def lagrange_integrate(function: Callable[[AbstractTensor], AbstractTensor],
                       nodes: Any, lower: Any, upper: Any) -> AbstractTensor:
    """``sum_k w_k f(x_k)`` -- the integral of the interpolant of ``f``."""

    nodes = _tensor(nodes).reshape(-1)
    weights = lagrange_weights(nodes, lower, upper)
    values = _tensor(function(nodes))
    count = int(nodes.shape[0])
    trailing = (1,) * (len(tuple(values.shape)) - 1)
    return (weights.reshape(count, *trailing) * values).sum(0)


def differentiation_matrix(nodes: Any) -> AbstractTensor:
    """``D[i, k] = l_k'(x_i)``: ``D @ q`` is the derivative of the interpolant at the nodes.

    Barycentric form: with ``c_k = prod_{m != k} (x_k - x_m)``,
    ``D[i, k] = (c_i / c_k) / (x_i - x_k)`` off the diagonal and each row
    sums to zero (the derivative of a constant).
    """

    nodes = _tensor(nodes).reshape(-1)
    count = int(nodes.shape[0])
    mask = _off_diagonal_mask(count, nodes)
    spacing = nodes.reshape(count, 1) - nodes.reshape(1, count)
    products = (spacing * mask + (1.0 - mask)).prod(1)             # c_k
    safe = spacing * mask + (1.0 - mask)                            # 1 on diagonal
    off = (products.reshape(count, 1) / products.reshape(1, count)) / safe * mask
    diagonal = off.sum(1)
    return off - linalg.eye(count) * diagonal.reshape(count, 1)


def action(lagrangian: Callable[[AbstractTensor, AbstractTensor, AbstractTensor], AbstractTensor],
           path: Any, times: Any, lower: Any = None, upper: Any = None) -> AbstractTensor:
    """``S = integral of L(q, dq/dt, t) dt`` for the path through ``path`` at ``times``.

    ``path`` has the nodes on axis 0 (extra trailing axes are coordinates).
    The integral runs over ``[times.min(), times.max()]`` unless bounds are
    given.
    """

    times = _tensor(times).reshape(-1)
    path = _tensor(path)
    count = int(times.shape[0])
    lower = times.min() if lower is None else lower
    upper = times.max() if upper is None else upper
    weights = lagrange_weights(times, lower, upper)
    trailing = tuple(path.shape)[1:]
    flat = path.reshape(count, -1)
    velocity = (differentiation_matrix(times) @ flat).reshape(count, *trailing)
    shaped_times = times.reshape(count, *((1,) * len(trailing)))
    density = _tensor(lagrangian(path, velocity, shaped_times))
    density = density.reshape(count, -1).sum(1)
    return (weights * density).sum()


def euler_lagrange_residual(lagrangian, path: AbstractTensor, times: Any,
                            lower: Any = None, upper: Any = None) -> AbstractTensor:
    """``dS/dq`` at every node: zero at the free nodes of a stationary path.

    The end nodes carry the boundary values, so their entries are the
    boundary momenta rather than a residual; a caller holding the ends fixed
    reads ``residual[1:-1]``.
    """

    if not getattr(path, "requires_grad", False):
        path.requires_grad_(True)
    value = action(lagrangian, path, times, lower, upper)
    (gradient,) = AbstractTensor.autograd.grad(value, [path], retain_graph=True)
    return gradient


def lagrange_multipliers(hessian: Any, gradient: Any, constraints: Any,
                         targets: Any) -> tuple[AbstractTensor, AbstractTensor]:
    """Stationary point of ``x.A.x/2 - b.x`` subject to ``C x = d``.

    Solves the KKT system ``[[A, C^T], [C, 0]] [x; lam] = [b; d]`` and
    returns ``(x, lam)``; ``lam`` are the Lagrange multipliers, one per
    constraint row.
    """

    A = _tensor(hessian) * 1.0
    b = _tensor(gradient).reshape(-1) * 1.0
    C = _tensor(constraints) * 1.0
    d = _tensor(targets).reshape(-1) * 1.0
    n = int(A.shape[0])
    m = int(C.shape[0])
    zeros = C @ C.transpose(0, 1) * 0.0                              # (m, m)
    top = AbstractTensor.concat([A, C.transpose(0, 1)], dim=1)
    bottom = AbstractTensor.concat([C, zeros], dim=1)
    system = AbstractTensor.concat([top, bottom], dim=0)
    rhs = AbstractTensor.concat([b, d], dim=0).reshape(n + m, 1)
    solution = linalg.solve(system, rhs).reshape(n + m)
    return solution[:n], solution[n:]


__all__ = (
    "action",
    "differentiation_matrix",
    "euler_lagrange_residual",
    "lagrange_basis",
    "lagrange_integrate",
    "lagrange_interpolate",
    "lagrange_multipliers",
    "lagrange_weights",
)
