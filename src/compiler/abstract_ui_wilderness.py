"""Where a map stands in the wilderness: a point on the unit hypersphere in n dimensions.

Owner's design (2026-11-10/12). The wilderness is a flat index inside an n-dimensional hypersphere. A walker sees a
flat area projected as far as the eye can see, terrain always generated just out of sight and solidifying as they
approach; every world map has a place in it, and you only run into one by coincidence, when your local position's
cosine similarity to the map's is close. This module gives every map that place.

The embedding is BLANK on purpose (owner, 2026-11-12: "we're going to evolve it into doing anything"): it carries no
meaning yet, only a well-spread, stable position per identity. It is the generalized golden-ratio (Fibonacci)
lattice: the Kronecker sequence of the d-dimensional golden ratio (``phi_d``, the positive root of
x^(d+1) = x + 1; phi_1 is the golden ratio, the classic Fibonacci-sphere spread) gives a low-discrepancy point in the
unit cube for any whole index, and the inverse normal CDF per coordinate followed by normalisation carries it onto
the sphere S^(n-1) (a normal vector normalised is uniform on the sphere). The index is the identity's SHA-256, and the
cube coordinates are computed in exact 64-bit fixed point, so the same identity always lands on the same point and
two identities are spread as evenly as the lattice spreads consecutive indices.

    position(identity, dimensions=DIMENSIONS) -> tuple of floats, unit length
    cosine(a, b)                               -> the similarity a walker's chance of meeting a map follows
"""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist

DIMENSIONS = 8
"""The default n. A parameter of the blank embedding, not a meaning; any n >= 2 is valid."""

_SCALE = 1 << 64
_NORMAL = NormalDist()


def golden_ratio(d: int) -> float:
    """phi_d: the positive root of x**(d+1) = x + 1 (Newton's method from 2)."""

    x = 2.0
    for _ in range(64):
        x = x - (x ** (d + 1) - x - 1) / ((d + 1) * x ** d - 1)
    return x


def _multipliers(count: int) -> tuple[int, ...]:
    phi = golden_ratio(count)
    return tuple(int((1.0 / phi ** (k + 1)) * _SCALE) % _SCALE for k in range(count))


def index_of(identity: str) -> int:
    """The lattice index of an identity: its SHA-256 as a whole number."""

    if not isinstance(identity, str) or not identity:
        raise ValueError("an identity is a nonempty string")
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big")


def position(identity: str, dimensions: int = DIMENSIONS) -> tuple[float, ...]:
    """The identity's place on the unit sphere in ``dimensions`` dimensions."""

    if not isinstance(dimensions, int) or dimensions < 2:
        raise ValueError("the wilderness has at least two dimensions")
    index = index_of(identity)
    coordinates = []
    for multiplier in _multipliers(dimensions):
        fraction = ((index * multiplier) % _SCALE + 0.5) / _SCALE      # exact frac(i * alpha_k), kept off 0 and 1
        coordinates.append(_NORMAL.inv_cdf(fraction))
    length = math.sqrt(sum(c * c for c in coordinates))
    return tuple(c / length for c in coordinates)


def cosine(a, b) -> float:
    if len(a) != len(b):
        raise ValueError("positions in different dimensions")
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


__all__ = ["DIMENSIONS", "cosine", "golden_ratio", "index_of", "position"]
