"""Curvature helpers for discrete spring faces.

These utilities approximate the curvature contribution of a face attached to
an edge.  The current implementation assumes each edge belongs to an implied
regular hexagon.  Only the two edge endpoints are provided; the remaining
vertices of the hexagon are virtual and fixed at 60° increments.  The
resulting scalar can modulate spring forces in a non-linear fashion.

The model is intentionally lightweight and serves as a placeholder until the
full discrete exterior calculus operators are available.
"""
from __future__ import annotations

from ...tensors.abstraction import AbstractTensor


def hex_face_curvature(p0: AbstractTensor, p1: AbstractTensor) -> AbstractTensor:
    """Approximate face curvature for an edge ``p0`` → ``p1``.

    The edge is treated as one side of a regular hexagon.  Curvature is the
    signed deviation between the edge direction and the implied hexagonal
    normal.  For 3‑D positions only the first two components are considered,
    matching the 2‑D stencil assumption.

    Parameters
    ----------
    p0, p1:
        Endpoint positions.  They must be coercible to ``AbstractTensor``
        and contain at least two components.

    Returns
    -------
    AbstractTensor
        Scalar curvature value; zero means flat.
    """
    v0 = AbstractTensor.get_tensor(p0)[:2]
    v1 = AbstractTensor.get_tensor(p1)[:2]
    edge = v1 - v0
    if AbstractTensor.linalg.norm(edge) == 0:
        return AbstractTensor.tensor(0.0)
    normal = AbstractTensor.tensor([-edge[1], edge[0]])
    num = AbstractTensor.dot(edge, normal)
    denom = AbstractTensor.linalg.norm(edge) * AbstractTensor.linalg.norm(normal) + 1e-12
    return num / denom
