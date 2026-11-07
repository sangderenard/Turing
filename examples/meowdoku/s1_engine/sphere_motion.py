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
    """Fly ``distance`` along the great circle through ``u`` with heading ``h``
    (entity_motion.sphere_advance: the law the page compiles too)."""
    from meowdoku.entity_motion import sphere_advance
    r = sphere_advance(u[0], u[1], u[2], h[0], h[1], h[2], distance, radius)
    return (r[0], r[1], r[2]), (r[3], r[4], r[5])


def turn(u, h, angle):
    """Rotate the heading about the local vertical; positive ``angle`` turns
    right (entity_motion.sphere_turn: the law the page compiles too)."""
    from meowdoku.entity_motion import sphere_turn
    return sphere_turn(u[0], u[1], u[2], h[0], h[1], h[2], angle)


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
    ahead) in units, true arc length from the centre (entity_motion.sphere_project:
    the one text the page's core is lowered from too)."""
    from meowdoku.entity_motion import sphere_project
    return sphere_project(u[0], u[1], u[2], centre[0], centre[1], centre[2],
                          heading[0], heading[1], heading[2], radius)


def patch_velocity(h, speed, X, Y, centre_h, centre_speed, centre, yaw_rate):
    """Apparent velocity in the patch (right, ahead per second) of a body
    flying ``speed`` along ``h`` at patch point (X, Y), seen from a centre
    flying ``centre_speed`` along ``centre_h`` and turning ``yaw_rate`` (right
    positive): relative flight in the tangent plane plus the frame's turn."""
    r = right_of(centre, centre_h)
    rel = add(scale(h, speed), scale(centre_h, -centre_speed))
    return dot(rel, r) - yaw_rate * Y, dot(rel, centre_h) + yaw_rate * X


def carry(u, h, spd, turn, z, vz, flight, dt, radius, air, ballistic, flights):
    """Carry bodies on for ``dt`` the way ``scroller_step`` flies them, from
    the state the page was last sent: the heading turned by ``turn * dt``
    (``entity_motion.sphere_turn``), the body flown ``spd * dt`` along its
    great circle (``sphere_advance``; a craft's by the thin air at its height,
    ``air_gain``) and climbed ``vz * dt`` (``climb``).  A missile or a bomb
    (``flight`` by ``flights``) flies engine_toy's trajectory law instead --
    drag and gravity -- whose horizontal advance is flown along the heading,
    its speed, height and climb taken from the law.  The page runs these same
    laws (as WebAssembly cores) on each body it was sent; the server's
    snapshot encoder runs this text on spans to know what the page will have
    drawn, and sends a body again only when that has drifted.

    ``air`` is (fall, gain); ``ballistic`` the trajectory coefficients (mass,
    air density, gravity and the (diameter, drag) of a missile, a bomb, a
    canopy); ``flights`` the flight codes.  Returns u, h, spd, z, vz."""
    from meowdoku.entity_motion import air_gain, climb, sphere_advance, sphere_turn
    from src.common.tensors import AbstractTensor
    where = AbstractTensor.where
    zero = spd * 0.0
    h1 = sphere_turn(u[0], u[1], u[2], h[0], h[1], h[2], turn * dt)
    thin = air_gain(z, zero + air[0], zero + air[1])[0]
    distance = spd * dt * where(flight == flights["craft"], thin, 1.0)
    missile, bomb = flight == flights["missile"], flight == flights["bomb"]
    shot = missile | bomb | (flight == flights["canopy"])
    from meowdoku.scroller import trajectory_step
    diameter = where(missile, zero + ballistic["missile"][0], where(bomb, zero + ballistic["bomb"][0], zero + ballistic["canopy"][0]))
    drag = where(missile, zero + ballistic["missile"][1], where(bomb, zero + ballistic["bomb"][1], zero + ballistic["canopy"][1]))
    mx, mz, _my, mvx, mvz, _mvy = trajectory_step(zero, z, zero, spd, vz, zero, dt, ballistic["mass"], diameter, drag,
                                                  ballistic["air_density"], ballistic["gravity"], 0.0, 0.0, 0.0)[:6]
    distance = where(shot, mx, distance)
    px, py, pz, qx, qy, qz = sphere_advance(u[0], u[1], u[2], h1[0], h1[1], h1[2], distance, radius)
    return ((px, py, pz), (qx, qy, qz), where(shot, mvx, spd), where(shot, mz, climb(z, vz, dt)[0]), where(shot, mvz, vz))
