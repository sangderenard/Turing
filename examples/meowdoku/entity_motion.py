"""Player motion for entity games: one law, run by the server and the page.

A player ship is integrated, never placed.  Per axis its velocity relaxes
toward a commanded velocity with damping time ``tau`` (implicit Euler, so a
step of any length is stable -- the server steps at the dt system's pace, a
browser at its frame rate), and a held pointer commands the velocity that
would close the gap to the pointer in ``reach`` seconds; keys command
``axis * speed``.  Walls stop the ship and zero the velocity into them.

The function uses only arithmetic and ``abs`` -- no branch, no library -- so
the SAME text runs on AbstractTensor spans inside a dt-system law and on
plain numbers, and is what the source compiler lowers to the page's
predictor: a direct WebAssembly core (``build_predictor``), the route and
ABI of ``examples/mobile_reference_app`` (its ``turing-wasm-shell.js`` loads
it).  The float annotations are for that lowering; the server passes spans.  The page runs it for its own ship on every
frame and reconciles against the server's state (see ``entities.py``).
"""

from __future__ import annotations

import hashlib
import math
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("x", "v", "axis", "target", "lo", "hi", "speed", "tau", "reach", "dt")
OUTPUTS = ("x_next", "v_next")


def damped_axis(x: float, v: float, axis: float, target: float, lo: float, hi: float,
                speed: float, tau: float, reach: float, dt: float):
    """One axis of a damped ship over ``dt``.  ``target < 0``: no pointer."""
    steering = (target >= 0.0) * 1.0
    gap = (target - x) / reach
    pointer = (abs(gap + speed) - abs(gap - speed)) * 0.5
    keys = (abs(axis + 1.0) - abs(axis - 1.0)) * 0.5 * speed
    command = steering * pointer + (1.0 - steering) * keys
    moved = v + (command - v) * (dt / (tau + dt))
    free = x + moved * dt
    x_next = (abs(free - lo) - abs(free - hi) + lo + hi) * 0.5
    v_next = (x_next - x) / dt
    return x_next, v_next


def ground_travel(u: float, v: float, gx: float, gy: float, turn: float, speed: float, dt: float):
    """The land under a spherical game's ship, as one law for server and page:
    the ground frame's heading (unit ``gx, gy`` on the tileable map) turns by
    ``turn`` radians -- an exact rotation by the rational (Cayley) form of the
    half-angle, so a unit heading stays unit with no transcendental -- and the
    map point (``u, v``) travels ``speed * dt`` along it.  Arithmetic only:
    the server's NumPy and the page's WebAssembly compute the same IEEE values."""
    h = turn * 0.5
    d = 1.0 + h * h
    c = (1.0 - h * h) / d
    s = (2.0 * h) / d
    gx_next = gx * c - gy * s
    gy_next = gx * s + gy * c
    u_next = u + gx_next * speed * dt
    v_next = v + gy_next * speed * dt
    return u_next, v_next, gx_next, gy_next


def follow_view(view: float, x: float, follow: float, width: float):
    """A scrolling view that follows a runner: never back, and forward by as
    much as the runner is past ``follow`` of the ``width`` (``max`` written
    with ``abs``, as ``damped_axis`` is)."""
    gap = x - view - follow * width
    view_next = view + (gap + abs(gap)) * 0.5
    return (view_next,)


def bank_turn(yaw: float, bank: float, level: float, gain: float, dt: float):
    """A banked craft turns: its yaw moves by ``gain`` times its bank (the
    bank axis is held offset by ``level``, so ``bank - level`` is the angle)
    over ``dt`` -- the coordinated turn of a rolled wing, linear in the bank."""
    yaw_next = yaw + gain * (bank - level) * dt
    return (yaw_next,)


def sphere_turn(ux: float, uy: float, uz: float, hx: float, hy: float, hz: float, angle: float):
    """Turn a heading ``h`` about the vertical ``u`` by ``angle`` (right
    positive): an exact rotation by the rational (Cayley) form of the
    half-angle -- arithmetic only -- then the heading kept tangent and unit
    (one Newton step: the rotation keeps length, so this only clears
    rounding).  The flight of every body of a spherical game."""
    t = angle * 0.5
    d = 1.0 + t * t
    c = (1.0 - t * t) / d
    s = (2.0 * t) / d
    rx = hy * uz - hz * uy
    ry = hz * ux - hx * uz
    rz = hx * uy - hy * ux
    nx = hx * c + rx * s
    ny = hy * c + ry * s
    nz = hz * c + rz * s
    k = nx * ux + ny * uy + nz * uz
    nx = nx - k * ux
    ny = ny - k * uy
    nz = nz - k * uz
    inv = 1.5 - 0.5 * (nx * nx + ny * ny + nz * nz)
    return nx * inv, ny * inv, nz * inv


def sphere_advance(ux: float, uy: float, uz: float, hx: float, hy: float, hz: float, distance: float, radius: float):
    """Fly ``distance`` along the great circle through ``u`` heading ``h``:
    the same rational rotation by the arc, the position kept unit and the
    heading tangent and unit (Newton steps, as ``sphere_turn``)."""
    t = distance / radius * 0.5
    d = 1.0 + t * t
    c = (1.0 - t * t) / d
    s = (2.0 * t) / d
    px = ux * c + hx * s
    py = uy * c + hy * s
    pz = uz * c + hz * s
    qx = hx * c - ux * s
    qy = hy * c - uy * s
    qz = hz * c - uz * s
    inv = 1.5 - 0.5 * (px * px + py * py + pz * pz)
    px = px * inv
    py = py * inv
    pz = pz * inv
    k = qx * px + qy * py + qz * pz
    qx = qx - k * px
    qy = qy - k * py
    qz = qz - k * pz
    inv2 = 1.5 - 0.5 * (qx * qx + qy * qy + qz * qz)
    return px, py, pz, qx * inv2, qy * inv2, qz * inv2


def air_gain(z: float, fall: float, gain: float):
    """Air thins with height (density 1 / (1 + fall * z)); a craft meets
    less resistance there and its throttle carries it faster, by up to
    ``gain``: the factor on its speed."""
    rho = 1.0 / (1.0 + fall * z)
    return (1.0 + gain * (1.0 - rho),)


def terrain_height(x: float, y: float, z: float):
    """The land of a spherical game, one law for server and page: five
    octaves of value noise over the unit-sphere point (x, y, z), lattice
    corners hashed without a sine (Hoskins), trilinear with smoothstep
    weights; then sea level and the relief's shaping.  Returns (height in
    board units, raw noise 0..1).  Arithmetic and math.floor only."""
    total = 0.0
    weight = 0.0
    # octave 0: frequency 3.0, amplitude 1.0
    px = x * 3.0 + 100.0
    py = y * 3.0 + 100.0
    pz = z * 3.0 + 100.0
    ix = math.floor(px) * 1.0
    iy = math.floor(py) * 1.0
    iz = math.floor(pz) * 1.0
    fx = px - ix
    fy = py - iy
    fz = pz - iz
    wx = fx * fx * (3.0 - 2.0 * fx)
    wy = fy * fy * (3.0 - 2.0 * fy)
    wz = fz * fz * (3.0 - 2.0 * fz)
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h0 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h1 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h2 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h3 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h4 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h5 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h6 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h7 = hv - math.floor(hv) * 1.0
    l00 = h0 + (h1 - h0) * wx
    l10 = h2 + (h3 - h2) * wx
    l01 = h4 + (h5 - h4) * wx
    l11 = h6 + (h7 - h6) * wx
    m0 = l00 + (l10 - l00) * wy
    m1 = l01 + (l11 - l01) * wy
    total = total + (m0 + (m1 - m0) * wz) * 1.0
    weight = weight + 1.0
    # octave 1: frequency 6.0, amplitude 0.5
    px = x * 6.0 + 117.0
    py = y * 6.0 + 129.0
    pz = z * 6.0 + 141.0
    ix = math.floor(px) * 1.0
    iy = math.floor(py) * 1.0
    iz = math.floor(pz) * 1.0
    fx = px - ix
    fy = py - iy
    fz = pz - iz
    wx = fx * fx * (3.0 - 2.0 * fx)
    wy = fy * fy * (3.0 - 2.0 * fy)
    wz = fz * fz * (3.0 - 2.0 * fz)
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h0 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h1 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h2 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h3 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h4 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h5 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h6 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h7 = hv - math.floor(hv) * 1.0
    l00 = h0 + (h1 - h0) * wx
    l10 = h2 + (h3 - h2) * wx
    l01 = h4 + (h5 - h4) * wx
    l11 = h6 + (h7 - h6) * wx
    m0 = l00 + (l10 - l00) * wy
    m1 = l01 + (l11 - l01) * wy
    total = total + (m0 + (m1 - m0) * wz) * 0.5
    weight = weight + 0.5
    # octave 2: frequency 12.0, amplitude 0.25
    px = x * 12.0 + 134.0
    py = y * 12.0 + 158.0
    pz = z * 12.0 + 182.0
    ix = math.floor(px) * 1.0
    iy = math.floor(py) * 1.0
    iz = math.floor(pz) * 1.0
    fx = px - ix
    fy = py - iy
    fz = pz - iz
    wx = fx * fx * (3.0 - 2.0 * fx)
    wy = fy * fy * (3.0 - 2.0 * fy)
    wz = fz * fz * (3.0 - 2.0 * fz)
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h0 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h1 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h2 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h3 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h4 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h5 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h6 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h7 = hv - math.floor(hv) * 1.0
    l00 = h0 + (h1 - h0) * wx
    l10 = h2 + (h3 - h2) * wx
    l01 = h4 + (h5 - h4) * wx
    l11 = h6 + (h7 - h6) * wx
    m0 = l00 + (l10 - l00) * wy
    m1 = l01 + (l11 - l01) * wy
    total = total + (m0 + (m1 - m0) * wz) * 0.25
    weight = weight + 0.25
    # octave 3: frequency 24.0, amplitude 0.125
    px = x * 24.0 + 151.0
    py = y * 24.0 + 187.0
    pz = z * 24.0 + 223.0
    ix = math.floor(px) * 1.0
    iy = math.floor(py) * 1.0
    iz = math.floor(pz) * 1.0
    fx = px - ix
    fy = py - iy
    fz = pz - iz
    wx = fx * fx * (3.0 - 2.0 * fx)
    wy = fy * fy * (3.0 - 2.0 * fy)
    wz = fz * fz * (3.0 - 2.0 * fz)
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h0 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h1 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h2 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h3 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h4 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h5 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h6 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h7 = hv - math.floor(hv) * 1.0
    l00 = h0 + (h1 - h0) * wx
    l10 = h2 + (h3 - h2) * wx
    l01 = h4 + (h5 - h4) * wx
    l11 = h6 + (h7 - h6) * wx
    m0 = l00 + (l10 - l00) * wy
    m1 = l01 + (l11 - l01) * wy
    total = total + (m0 + (m1 - m0) * wz) * 0.125
    weight = weight + 0.125
    # octave 4: frequency 48.0, amplitude 0.0625
    px = x * 48.0 + 168.0
    py = y * 48.0 + 216.0
    pz = z * 48.0 + 264.0
    ix = math.floor(px) * 1.0
    iy = math.floor(py) * 1.0
    iz = math.floor(pz) * 1.0
    fx = px - ix
    fy = py - iy
    fz = pz - iz
    wx = fx * fx * (3.0 - 2.0 * fx)
    wy = fy * fy * (3.0 - 2.0 * fy)
    wz = fz * fz * (3.0 - 2.0 * fz)
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h0 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h1 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h2 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 0.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h3 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h4 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 0.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h5 = hv - math.floor(hv) * 1.0
    ax = (ix + 0.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h6 = hv - math.floor(hv) * 1.0
    ax = (ix + 1.0) * 0.1031
    ay = (iy + 1.0) * 0.1030
    az = (iz + 1.0) * 0.0973
    ax = ax - math.floor(ax) * 1.0
    ay = ay - math.floor(ay) * 1.0
    az = az - math.floor(az) * 1.0
    dd = ax * (az + 33.33) + ay * (ay + 33.33) + az * (ax + 33.33)
    ax = ax + dd
    ay = ay + dd
    az = az + dd
    hv = (ax + ay) * az
    h7 = hv - math.floor(hv) * 1.0
    l00 = h0 + (h1 - h0) * wx
    l10 = h2 + (h3 - h2) * wx
    l01 = h4 + (h5 - h4) * wx
    l11 = h6 + (h7 - h6) * wx
    m0 = l00 + (l10 - l00) * wy
    m1 = l01 + (l11 - l01) * wy
    total = total + (m0 + (m1 - m0) * wz) * 0.0625
    weight = weight + 0.0625
    raw = total / weight
    r = (raw - 0.48) / (1.0 - 0.48)
    r = (abs(r) + r) * 0.5
    r = (r + 1.0 - abs(r - 1.0)) * 0.5
    height = r * r * (1.5 - 0.5 * r) * 0.75
    return height, raw


#: The engine laws the page runs, each lowered to its own WebAssembly core:
#: (function, input names, output names, file stem).
PREDICTORS = (
    (damped_axis, NAMES, OUTPUTS, "motion"),
    (ground_travel, ("u", "v", "gx", "gy", "turn", "speed", "dt"), ("u_next", "v_next", "gx_next", "gy_next"), "travel"),
    (follow_view, ("view", "x", "follow", "width"), ("view_next",), "follow"),
    (bank_turn, ("yaw", "bank", "level", "gain", "dt"), ("yaw_next",), "bank"),
    (sphere_turn, ("ux", "uy", "uz", "hx", "hy", "hz", "angle"), ("hx_next", "hy_next", "hz_next"), "turn"),
    (sphere_advance, ("ux", "uy", "uz", "hx", "hy", "hz", "distance", "radius"),
     ("ux_next", "uy_next", "uz_next", "hx_next", "hy_next", "hz_next"), "advance"),
    (air_gain, ("z", "fall", "gain"), ("factor",), "air"),
    (terrain_height, ("x", "y", "z"), ("height", "raw"), "terrain"),
)


def build_predictor(out_dir: Path) -> dict:
    """Lower every law in PREDICTORS to a WebAssembly core in ``out_dir``
    (``<stem>.wasm`` + ``<stem>.abi.json``), each rebuilt only when its
    source changes; returns the contracts by stem."""
    return {stem: _build_one(out_dir, fn, names, outputs, stem) for fn, names, outputs, stem in PREDICTORS}


def _build_one(out_dir: Path, fn, names, outputs_names, stem: str) -> dict:
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core

    out_dir = Path(out_dir)
    source = "import math\n" + inspect.getsource(fn)       # the module's import, which the laws read
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    abi_path, wasm_path = out_dir / f"{stem}.abi.json", out_dir / f"{stem}.wasm"
    if abi_path.exists() and wasm_path.exists():
        contract = json.loads(abi_path.read_text(encoding="utf-8"))
        if contract.get("source_digest") == digest:
            return contract
    module, outputs, exports = lower_ast_source_to_ssa(
        source, fn.__name__, name="entity_motion",
        extraction_contract=ExtractionContract(ROOT / "extraction_contracts/program_extraction.yaml"),
        progress_bars=False)
    entry = exports[0]
    wasm = emit_ssa_module_to_wasm_core(module, entry, entry_name=entry)
    if not wasm.complete:
        raise RuntimeError(f"{fn.__name__} WebAssembly emission shortfalls: {wasm.shortfalls}")
    argument_ids = tuple(int(value.id) for value in module.functions[entry].args)
    output_ids = tuple(int(value.id) for value in outputs[entry])
    input_names = dict(zip(argument_ids, names))
    output_names = dict(zip(output_ids, outputs_names))
    descriptors, offset = [], 8
    for kind, value_id in wasm.parameters:
        descriptor = {"kind": kind, "value_id": value_id,
                      "role": "output" if value_id in output_names else "input",
                      "name": output_names.get(value_id, input_names.get(value_id, f"value_{value_id}"))}
        if kind == "buffer":
            descriptor["offset"] = offset
            descriptor["dtype"] = dict(wasm.output_dtypes).get(value_id, "float64")
            offset += 8
        descriptors.append(descriptor)
    contract = {"schema": "turing.mobile-wasm-abi.v1", "entrypoint": entry, "inputs": list(names),
                "outputs": list(outputs_names), "parameters": descriptors, "source_digest": digest}
    out_dir.mkdir(parents=True, exist_ok=True)
    wasm_path.write_bytes(wasm.binary)
    abi_path.write_text(json.dumps(contract, indent=1), encoding="utf-8")
    return contract
