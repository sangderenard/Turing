"""AbstractTensor geometry groundwork for native Studio contacts, not a product.

Coordinates must already be in the particle law's frame and units. Triangle
vertices are the exact shared render vertices, not bounding boxes. Triangles
are two-sided sheets; this does NOT infer the inside of a closed obstacle.
The room instead uses six inward half-spaces, retaining containment even after
a centre crosses a wall. The caller must replace the legacy slab walls.

convex_part_contact reduces triangles only within one explicitly authored
closed convex solid or convex planar sheet. Independent parts and room planes
remain independent contacts. No arbitrary nonconvex mesh reduction or browser
integrator is provided. These helpers are not yet a deployed fountain product.
"""
from __future__ import annotations

from src.common.tensors import AbstractTensor
from meowdoku.entities import as_float
from meowdoku.fountain3d import _contact, _cross, _dot, _cap


TRIANGLE_FIELDS = ("ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz")


def triangle_input_names(slot: int) -> tuple[str, ...]:
    """Scalar names for the existing named-input SSA trace/manifest mechanism.

    Material identity is deliberately host-side metadata keyed by this slot;
    optical materials are not contact coefficients. This is not an extension
    of the currently deployed fountain ARGUMENTS or its compiled ABI.
    """
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        raise ValueError("triangle slot must be a nonnegative integer")
    return tuple(f"studio_t{slot}_{field}" for field in TRIANGLE_FIELDS)


def _sub(a, b):
    return tuple(a[k] - b[k] for k in range(3))


def _select(mask, a, b):
    return tuple(AbstractTensor.where(mask, a[k], b[k]) for k in range(3))


def _safe_positive(value):
    # Only an exactly zero divisor is replaced. No epsilon moves the surface
    # or erases a small, representable triangle. All eagerly evaluated arms
    # remain finite for coincident vertices and zero-length edges.
    return AbstractTensor.where(value > 0.0, value, value * 0.0 + 1.0)


def _segment_point(p, a, b):
    edge = _sub(b, a)
    t = (_dot(_sub(p, a), edge) / _safe_positive(_dot(edge, edge))).clamp(0.0, 1.0)
    return tuple(a[k] + t * edge[k] for k in range(3))


def closest_triangle(p, a, b, c):
    """Return (distance, normal, closest point, valid) for particle span p.

    The projected face is used only inside all three edges. Otherwise the
    closest of the three bounded segments gives the true edge/vertex point.
    Nonzero distance normals point from that point toward p, on either side.
    At exact coincidence a valid face uses its winding normal; a collapsed
    triangle has validity zero and zero normal (its closest point remains
    well-defined). Validity must gate contacts. Inputs must be finite.
    """
    ab, ac = _sub(b, a), _sub(c, a)
    face = _cross(ab, ac)
    area2 = _dot(face, face)
    valid = as_float(area2 > 0.0)
    offset = _dot(_sub(p, a), face) / _safe_positive(area2)
    projected = tuple(p[k] - offset * face[k] for k in range(3))
    inside = area2 > 0.0
    for u, v in ((a, b), (b, c), (c, a)):
        inside = inside & (_dot(_cross(_sub(v, u), _sub(projected, u)), face) >= 0.0)
    point = _segment_point(p, a, b)
    d2 = _dot(_sub(p, point), _sub(p, point))
    for u, v in ((b, c), (c, a)):
        candidate = _segment_point(p, u, v)
        candidate_d2 = _dot(_sub(p, candidate), _sub(p, candidate))
        closer = candidate_d2 < d2
        point = _select(closer, candidate, point)
        d2 = AbstractTensor.where(closer, candidate_d2, d2)
    point = _select(inside, projected, point)
    delta = _sub(p, point)
    distance = _dot(delta, delta).sqrt()
    face_length = _safe_positive(area2).sqrt()
    # SSA where broadcasts unlike-shaped branches through broadcast_double,
    # which the page's elementwise translator explicitly cannot emit. Lift
    # the scalar face normal into the particle span using ordinary arithmetic,
    # as fountain3d's wall normals do, before the existing select primitive.
    normal = tuple(valid * AbstractTensor.where(
        distance > 0.0, delta[k] / _safe_positive(distance), distance * 0.0 + face[k] / face_length
    ) for k in range(3))
    return distance, normal, point, valid


def room_planes(p, low, high):
    """Six (signed inward distance, inward normal) pairs for an exact AABB room.

    Bounds are in the particle frame; callers validate low < high. No nearest
    face reduction: an interior corner can contact three independent walls.
    Negative distance outside preserves restoring, rather than two-sided,
    containment. Degenerate room bounds are not silently repaired.
    """
    zero = p[0] * 0.0
    result = []
    for k in range(3):
        inward = tuple(zero + (1.0 if j == k else 0.0) for j in range(3))
        result.append((p[k] - low[k], inward))
        result.append((high[k] - p[k], tuple(-v for v in inward)))
    return tuple(result)


def convex_part_contact(p, triangles, *, closed):
    """One exact contact for one authored convex solid or convex planar sheet.

    ``triangles`` is a nonempty compile-time sequence of runtime vertex tuples.
    The producer must certify a closed outward-wound convex boundary when
    closed=True, or a filled convex coplanar sheet otherwise. This function
    does NOT infer topology, repair winding, weld coordinates or accept a
    whole compound object in place of its independent convex parts.

    Return (signed distance, normal, closest point, validity, triangle index).
    Source index preserves the producer's triangle/material identity through
    reduction; ties retain the first authored triangle. No optical coefficient
    is interpreted. Any degenerate face invalidates the part, rather than
    silently opening a solid or dropping an authored surface.

    The closest point on a convex set is unique outside it: selecting its
    nearest boundary triangle removes triangulation seams without losing a
    second physical contact on this part. Inside a closed solid, the nearest
    boundary gives the shortest exit; negate distance and reverse its normal.
    An exact boundary coincidence retains the authored outward face normal.
    Distinct parts and room walls MUST NOT undergo this reduction together.

    This removes triangulation seams WITHIN a part, not seams between touching
    convex pieces of one authored compound body. The producer must retain
    compound/body identity. Adjacent course strips cannot simply be summed:
    their common coplanar boundary doubles stiffness and their neighbouring
    edges add spurious side forces. Such compound unions remain unsupported
    by this helper and must be rejected or resolved by an authored union
    manifold before activation, never silently relabelled as separate bodies.
    """
    if not triangles:
        raise ValueError("a convex part requires at least one triangle")
    if not isinstance(closed, bool):
        raise TypeError("closed is an authored topology flag, not a tensor")
    distance, normal, point, valid = closest_triangle(p, *triangles[0])
    source = p[0] * 0.0
    all_valid = p[0] * 0.0 + valid
    inside = (p[0] * 0.0) == 0.0
    for index, (a, b, c) in enumerate(triangles):
        if index:
            candidate_d, candidate_n, candidate_q, candidate_valid = closest_triangle(p, a, b, c)
            closer = candidate_d < distance
            distance = AbstractTensor.where(closer, candidate_d, distance)
            normal = _select(closer, candidate_n, normal)
            point = _select(closer, candidate_q, point)
            source = AbstractTensor.where(closer, source * 0.0 + float(index), source)
            all_valid = all_valid * candidate_valid
        if closed:
            outward = _cross(_sub(b, a), _sub(c, a))
            inside = inside & (_dot(_sub(p, a), outward) <= 0.0)
    if closed:
        reverse = as_float(inside & (distance > 0.0))
        normal = tuple(v * (1.0 - 2.0 * reverse) for v in normal)
        distance = distance * (1.0 - 2.0 * as_float(inside))
    return distance, normal, point, all_valid, source


def planar_surface_contact(p, triangles, *, planar, convex_footprint):
    """Reduce one authored planar exterior surface, reporting unsupported edges.

    Topology flags are compile-time certificates from construction, never
    inferred using a coordinate tolerance. All triangles belong to the same
    body AND surface; floor and rail surfaces must remain separate. A folded
    group is refused rather than approximated by its first triangle's plane.

    A nonconvex planar footprint (for example the gentle S running surface)
    still has one unique closest face point when the particle's perpendicular
    projection is in the union of its triangles. Only that domain is supported.
    Outside it, multiple concave boundary features can contact the same sphere;
    nearest-triangle selection is not a complete contact manifold. The last
    return value explicitly reports support per particle, independently of
    geometric validity. Callers MUST refuse unsupported potentially touching
    particles, not silently gate their contacts to zero.

    Return convex_part_contact's five values followed by support. Geometry is
    two-sided sheet geometry; a solid compound still needs inside/exit handling
    and adjoining surface ownership before this is a complete native product.
    """
    if not isinstance(planar, bool) or not isinstance(convex_footprint, bool):
        raise TypeError("surface certificates must be authored Boolean flags")
    if not planar:
        raise ValueError("folded surface groups require an explicit contact manifold")
    result = convex_part_contact(p, triangles, closed=False)
    zero = p[0] * 0.0
    if convex_footprint:
        return (*result, zero + 1.0)
    covered = zero > 0.0
    for a, b, c in triangles:
        face = _cross(_sub(b, a), _sub(c, a))
        area2 = _dot(face, face)
        offset = _dot(_sub(p, a), face) / _safe_positive(area2)
        projection = tuple(p[k] - offset * face[k] for k in range(3))
        inside = area2 > 0.0
        for u, v in ((a, b), (b, c), (c, a)):
            inside = inside & (_dot(_cross(_sub(v, u), _sub(projection, u)), face) >= 0.0)
        covered = covered | inside
    return (*result, as_float(covered))


def stationary_contact(distance, normal, valid, velocity, spin, radius,
                       kn, mass, mu, zeta, roll, hertz, coh):
    """The existing infinite-mass wall contact law, with explicit mechanics.

    Matches fountain3d.make_law's wall call and bounded penetration. Optical
    material identity never determines friction, restitution or rolling drag.
    """
    return _contact(_cap(radius - distance, 6.0 * radius), valid, normal,
                    velocity, _cross(tuple(radius * w for w in spin), normal),
                    spin, kn, mass, mu, zeta, roll, radius, radius, radius,
                    hertz, coh)
