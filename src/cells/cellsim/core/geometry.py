"""Geometry helpers with optional NumPy broadcasting.

Previously these helpers accepted only scalar floats and used ``math`` for
their implementation.  To support vectorised operations across many volumes at
once we now rely on :mod:`numpy` and broadcast over the input.  When a scalar
is provided the return value remains a Python ``float`` to preserve backward
compatibility with existing callers.
"""

from __future__ import annotations

import numpy as np


def _to_array(V):
    """Return ``V`` as a ``float`` or ``ndarray`` for broadcasting."""

    arr = np.asarray(V, dtype=float)
    return arr


def _maybe_scalar(x):
    """Return ``x`` as a Python float if it is a scalar array."""

    return float(x) if np.ndim(x) == 0 else x


def sphere_radius_from_volume(V: float | np.ndarray) -> float | np.ndarray:
    """Return the radius of a sphere given its volume ``V``.

    ``V`` may be a scalar or any array-like object.  Negative volumes are
    clamped to zero before evaluation to avoid ``nan`` results when taking the
    cube-root.
    """

    V_arr = _to_array(V)
    R = np.cbrt(3.0 * np.maximum(V_arr, 0.0) / (4.0 * np.pi))
    return _maybe_scalar(R)


def sphere_area_from_volume(V: float | np.ndarray):
    """Return ``(area, radius)`` for a sphere of volume ``V``.

    The computation is fully vectorised and supports broadcasting.  The return
    types mirror the input: scalars in, scalars out; arrays in, arrays out.
    """

    R = sphere_radius_from_volume(V)
    A = 4.0 * np.pi * R * R
    return _maybe_scalar(A), R
