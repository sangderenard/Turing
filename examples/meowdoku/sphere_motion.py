"""Bodies on a sphere: great-circle flight, turning, and the patch a player sees.

Every body of a spherical entity game carries a unit POSITION ``u`` (the
direction from the sphere's centre), a unit HEADING ``h`` tangent to the
sphere at ``u``, and a ground speed.  Flight is exact along great circles;
turning rotates the heading about the local vertical.  The screen is a PATCH:
the azimuthal-equidistant view about one body (the player), its heading up
the screen -- distances from the centre are true arc lengths, so what is near
the player is undistorted and the whole sphere stays one state.

Vectors are tuples of three spans (or scalars that broadcast); the functions
use only arithmetic, ``sqrt``, ``sin``, ``cos`` and ``atan2``, so the same text
runs inside a dt-system law on AbstractTensor spans and on numpy arrays
wrapped as AbstractTensor (the autopilot's views).
"""

from __future__ import annotations

from src.common.tensors import AbstractTensor

where = AbstractTensor.where
EPS = 1e-12


def vwhere(mask, a, b):
    """Per-lane choice between two vectors."""
    return (where(mask, a[0], b[0]), where(mask, a[1], b[1]), where(mask, a[2], b[2]))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def normalized(a):
    n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() + EPS
    return (a[0] / n, a[1] / n, a[2] / n)


def tangent(u, v):
    """``v`` with its component along ``u`` removed, normalized: a heading at ``u``."""
    return normalized(add(v, scale(u, -dot(u, v))))


def right_of(u, h):
    """The body's right: heading cross vertical (``h x u``)."""
    return cross(h, u)


def advance(u, h, distance, radius):
    """Fly ``distance`` along the great circle through ``u`` with heading ``h``."""
    a = distance / radius
    c, s = a.cos(), a.sin()
    u1 = normalized(add(scale(u, c), scale(h, s)))
    h1 = tangent(u1, add(scale(h, c), scale(u, -s)))
    return u1, h1


def turn(u, h, angle):
    """Rotate the heading about the local vertical; positive ``angle`` turns right."""
    c, s = angle.cos(), angle.sin()
    return tangent(u, add(scale(h, c), scale(right_of(u, h), s)))


def bearing(u, h, g):
    """Signed angle from heading ``h`` to tangent direction ``g`` at ``u`` (right positive)."""
    return dot(g, right_of(u, h)).atan2(dot(g, h))


def toward(u, target):
    """The heading at ``u`` that points along the great circle to ``target``."""
    return tangent(u, target)


def arc(u, v, radius):
    """Great-circle distance between unit positions ``u`` and ``v``."""
    c = cross(u, v)
    return radius * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(dot(u, v))


def offset(u, h, ahead, right, radius):
    """The position ``ahead`` along the heading and ``right`` to its side
    (small offsets: a muzzle, a lateral gun), with the heading carried there."""
    u1 = normalized(add(u, add(scale(h, ahead / radius), scale(right_of(u, h), right / radius))))
    return u1, tangent(u1, h)


def place(u, h, arc_angle, azimuth):
    """The position ``arc_angle`` radians of arc from ``u`` in the direction
    ``azimuth`` (0 dead ahead, right positive)."""
    d = add(scale(h, azimuth.cos()), scale(right_of(u, h), azimuth.sin()))
    return normalized(add(scale(u, arc_angle.cos()), scale(d, arc_angle.sin())))


def project(u, centre, heading, radius):
    """Patch coordinates of ``u`` about ``centre`` facing ``heading``: (right,
    ahead) in units, true arc length from the centre."""
    r = right_of(centre, heading)
    e, n, c = dot(u, r), dot(u, heading), dot(u, centre)
    rho = (e * e + n * n).sqrt()
    k = radius * rho.atan2(c) / (rho + EPS)
    return e * k, n * k


def patch_velocity(h, speed, X, Y, centre_h, centre_speed, centre, yaw_rate):
    """Apparent velocity in the patch (right, ahead per second) of a body
    flying ``speed`` along ``h`` at patch point (X, Y), seen from a centre
    flying ``centre_speed`` along ``centre_h`` and turning ``yaw_rate`` (right
    positive): relative flight in the tangent plane plus the frame's turn."""
    r = right_of(centre, centre_h)
    rel = add(scale(h, speed), scale(centre_h, -centre_speed))
    return dot(rel, r) - yaw_rate * Y, dot(rel, centre_h) + yaw_rate * X
