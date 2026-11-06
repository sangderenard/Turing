"""Fountain: a soft-sphere (DEM) granular fountain that fills a basin, takes an image, fills it again.

ONE batched per-step law (``fountain_step``), authored once over spans of
particle ids in the repository's AbstractTensor style, compiled by the
repository to native, wasm and (generated) WebGPU compute.

NO PER-ID TABLES.  Everything an id is -- its emit time, where it leaves the
nozzle, its launch velocity, its species and so radius, mass, friction and
drag -- is DERIVED IN THE LAW from the id itself by an integer hash done in
exact float arithmetic (every intermediate is a whole number below 2**24, so
f64 and f32 agree to the bit and native, wasm and WGSL cannot disagree).  The
only per-id columns are the state (x y vx vy w age), the iota ``pid`` (which
the generated shader reads from its own invocation index, never from memory)
and the image colours ``ir ig ib``.  So the id count can run to millions
without shipping a byte of layout.

PHYSICS
-------
* Disks of four SPECIES that differ in radius, density (so mass), friction
  and drag.  Contacts are SOFT SPHERES: a normal spring-dashpot (stiffness
  proportional to the pair's reduced mass, damping from a damping ratio) plus
  tangential Coulomb friction (regularised with a tanh of the contact's slip
  speed) acting on the slip of the contact point, spin included, so disks
  roll.  The basin is a floor and two side walls that are contacts of the
  same law (a wall is a disk of infinite mass).  Gravity and linear air drag
  act on every particle.
* Contacts WRITE FORCES (and torques); the integrator (semi-implicit Euler,
  fixed ``DT``) is the only thing that moves anything.  Both sides of a pair
  evaluate the same symmetric expressions, so they are equal and opposite
  without any per-pair bookkeeping, impulse or position correction.
* The FOUNTAIN is an emitter: id ``k`` leaves at ``k / n * EMIT_SPAN``, at a
  nozzle position a low-discrepancy step along the bar from the previous id,
  with a hashed launch angle and speed.  Until its age reaches its emit time
  a particle is parked, takes no part in contacts and is not drawn.  The
  pile forms and settles by contact forces, so regular and irregular packing
  emerge from the species mix.
* The basin and nozzle scale with ``sqrt(n)`` so the pile keeps its shape.

NEIGHBOUR SEARCH (in the law, from tensor ops; deterministic)
-------------------------------------------------------------
A uniform grid of cells of the largest diameter.  Each step the particles are
sorted by the pair ``(cell, id)`` with a BITONIC SORT written as gathers and
compare-exchanges over two carried arrays (``log2(P)(log2(P)+1)/2`` stages;
pairs are unique, so the order is unique and nothing about thread scheduling
can change it).  ``P`` is the id count rounded up to a power of two; the
extra ids are parked forever.  Each of the three rows of the 3x3 block is a
run of the sorted cells found by two binary searches (gathers); the contact
loop walks the first ``K_RUN`` of each run.  No atomics and no scatter.

COLOUR
------
Phase 1: hue from the ANGLE of the id's launch vector, as a trig ramp
(``mode 0``: ``atan2``) or as a rotation of the chroma vector about the grey
axis with cos/sin read off the vector (``mode 1``, no ``atan2``).  The image
step samples the picked image at every id's ACTUAL rest position into per-id
colours ``ir ig ib``; phase 2 replays the identical simulation with
``use_image = 1`` and the law emits ``hue * (1 - use) + image * use``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from meowdoku.entities import LawPiece, as_float
from meowdoku.games import Game
from src.common.tensors import AbstractTensor
from src.common.tensors import numpy_backend  # noqa: F401  (registers the NumPy backend the eager law runs on)

# ----------------------------------------------------------------- the world
N_BASE = 4096                          # the basin's reference count (its width is WIDTH_BASE)
WIDTH_BASE = 2.0
FLOOR_Y = 0.0
EMIT_SPAN = 6.0                        # seconds over which ids are emitted
SETTLE_SECONDS = 14.0                  # sim time by which the pile has come to rest
DT = 1.0 / 360.0
G = 1.0
OMEGA = 150.0                          # contact stiffness: kn = m * OMEGA^2 (OMEGA * DT stays well under 1: a grain feels several contacts)
ZETA = 0.7                             # normal damping ratio
VREF = 0.02                            # tangential slip speed over which friction saturates
SPIN_DRAG = 3.0                        # 1/s
K_RUN = 10                             # candidates walked per row run
BAR_HEIGHT = 0.12

#: species: name, radius, density, friction, linear drag (1/s)
SPECIES = (
    ("grain", 0.0090, 1.0, 0.35, 0.10),
    ("bead",  0.0105, 1.6, 0.20, 0.06),
    ("dust",  0.0120, 0.7, 0.55, 0.25),
    ("pearl", 0.0140, 2.2, 0.10, 0.04),
)
MONO_WEIGHT = (0.6, 0.4, 0.0, 0.0)       # mix 0: near-monodisperse grains and beads (a crystal wants to form)
POLY_WEIGHT = (0.15, 0.15, 0.35, 0.35)   # mix 1: all four species, widely polydisperse
CELL = 2.0 * max(s[1] for s in SPECIES)

STATE = ("x", "y", "vx", "vy", "w", "age")
CONSTANTS = ("pid", "ir", "ig", "ib")
UNIFORMS = ("mode", "use_image", "mix")
IOTA = ("pid",)                        # columns the generated shader derives from its invocation index
OUTPUTS = ("x_next", "y_next", "vx_next", "vy_next", "w_next", "age_next", "cr", "cg", "cb", "rad")
ARGUMENTS = STATE + CONSTANTS + UNIFORMS

TAU = 2.0 * math.pi


def pad_count(n: int) -> int:
    """The sort length: ``n`` rounded up to a power of two."""
    return 1 << max(1, (int(n) - 1).bit_length())


def basin(n: int) -> dict[str, float]:
    """Geometry for ``n`` active ids: basin and nozzle scale with sqrt(n / N_BASE)."""
    s = math.sqrt(n / N_BASE)
    width = WIDTH_BASE * s
    nozzle_y = 0.6 * s + 0.7
    cols = int(math.ceil(width / CELL))
    rows = int(math.ceil((nozzle_y + 0.1) / CELL))
    return {"scale": s, "width": width, "half": width / 2.0, "nozzle_y": nozzle_y,
            "cols": cols, "rows": rows, "cells": cols * rows,
            "view": (-1.05 * s, 1.05 * s, -0.05 * s, nozzle_y + 1.35)}


# --------------------------------------------------------------- id hashing
# Exact in f32 and f64: every intermediate is a whole number below 2**24.
_PRIME = 4093.0


def _mod(x, m):
    return x - (x / m).floor() * m


def _lin(pid, a):
    """``pid * a mod 4093`` for a pid up to 2**24, without ever forming the product."""
    hi = (pid * (1.0 / 4096.0)).floor()
    lo = pid - hi * 4096.0
    return _mod(_mod(lo * a, _PRIME) + _mod(hi * float((4096 * int(a)) % 4093), _PRIME), _PRIME)


def _mix(pid, a, c):
    """A hashed uniform in [0, 1) of the id (24 bits from two independent 12-bit hashes)."""
    r1 = _mod(_lin(pid, a) + c, _PRIME)
    r1 = _mod(r1 * r1 + 3.0 * r1 + 11.0, _PRIME)
    r2 = _mod(_lin(pid, a + 6.0) + 2.0 * c + 1.0, _PRIME)
    r2 = _mod(r2 * r2 + 5.0 * r2 + 13.0, _PRIME)
    return (r1 + r2 * (1.0 / _PRIME)) * (1.0 / _PRIME)


# ------------------------------------------------------------------ the law
def _ramp(h, offset):
    """One channel of the HSV ramp at hue ``h`` in [0, 1] (full saturation, value)."""
    f = h + offset
    f = f - f.floor()
    return ((f * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0)


def _contact(delta, nx, ny, vrx, vry, spin_slip, kn, m_eff, mu):
    """Force and torque on the particle from one contact (``delta`` > 0 overlap,
    ``n`` the unit normal pointing at THIS particle, ``vr`` its velocity relative
    to the other body, ``spin_slip`` the spin part of the contact slip).  Returns
    (fx, fy, tangential force); torque = -radius * the third value."""
    vn = vrx * nx + vry * ny
    cn = 2.0 * ZETA * (kn * m_eff).sqrt()
    fn = (kn * delta - cn * vn).clamp(0.0, 1.0e9)
    tx, ty = -ny, nx
    vt = vrx * tx + vry * ty - spin_slip
    ft = -(mu * fn) * (vt * (1.0 / VREF)).tanh()
    return fn * nx + ft * tx, fn * ny + ft * ty, ft


def make_law(n: int, active: int | None = None):
    """The fountain law over ``n`` ids (a power of two); ``active`` of them (default all)
    are ever emitted, the rest stay parked (a count that is not a power of two)."""
    active = int(n if active is None else active)
    geo = basin(active)
    s, half, cols, rows, cells = geo["scale"], geo["half"], geo["cols"], geo["rows"], geo["cells"]
    nozzle_y, bar_half = geo["nozzle_y"], 0.8 * geo["scale"]
    stages = []
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            stages.append((k, j))
            j //= 2
        k *= 2
    search_steps = int(math.ceil(math.log2(n + 1))) + 1
    emit_rate = EMIT_SPAN / float(active)

    def fountain_step(x, y, vx, vy, w, age, pid, ir, ig, ib, mode, use_image, mix):
        """One fixed step of every particle (see the module docstring)."""
        # ---- the id's emit time: derived from the id, nothing shipped
        t_emit = AbstractTensor.where(pid < float(active), pid * emit_rate, pid * 0.0 + 1.0e9)
        act = as_float(age >= t_emit)
        # ---- grid cell; only particles inside the grid take part (parked ones, and the dilute
        # spray above the grid's top, get the cell past the end: sorted last, in nobody's block)
        cx = ((x + half) * (1.0 / CELL)).floor().clamp(0.0, cols - 1.0)
        cy = ((y - FLOOR_Y) * (1.0 / CELL)).floor().clamp(0.0, rows - 1.0)
        gridded = act * as_float(y < rows * CELL + FLOOR_Y)
        kc = AbstractTensor.where(gridded > 0.5, cy * float(cols) + cx, cy * 0.0 + float(cells))
        ki = pid
        # ---- stable bitonic sort of (cell, id) pairs: gathers and compare-exchanges
        for (kk, jj) in stages:
            bit = (pid * (1.0 / jj)).floor() - 2.0 * (pid * (1.0 / (2 * jj))).floor()
            up = 1.0 - ((pid * (1.0 / kk)).floor() - 2.0 * (pid * (1.0 / (2 * kk))).floor())
            partner = pid + jj * (1.0 - 2.0 * bit)
            oc = kc.index_select(0, partner)
            oi = ki.index_select(0, partner)
            less = as_float(kc < oc) + as_float((kc - oc).abs() < 0.5) * as_float(ki < oi)
            keep_min = AbstractTensor.where(bit < 0.5, up, 1.0 - up)
            keep_self = AbstractTensor.where(keep_min > 0.5, less, 1.0 - less)
            kc = AbstractTensor.where(keep_self > 0.5, kc, oc)
            ki = AbstractTensor.where(keep_self > 0.5, ki, oi)
        # ---- the run of sorted cells covering each of the three rows of the 3x3 block
        fx = x * 0.0
        fy = fx
        tq = fx
        top = float(n - 1)
        runs = []
        for dy in (-1.0, 0.0, 1.0):
            row = cy + dy
            ok = as_float(row >= 0.0) * as_float(row <= rows - 1.0)
            rowc = row.clamp(0.0, rows - 1.0)
            c_lo = rowc * float(cols) + (cx - 1.0).clamp(0.0, cols - 1.0)
            c_hi = rowc * float(cols) + (cx + 1.0).clamp(0.0, cols - 1.0)
            ends = []
            for target in (c_lo, c_hi + 1.0):
                lo = cx * 0.0
                hi = cx * 0.0 + float(n)
                for _ in range(search_steps):
                    mid = ((lo + hi) * 0.5).floor()
                    probe = kc.index_select(0, mid.clamp(0.0, top))
                    live = as_float(lo < hi)
                    below = as_float(probe < target)
                    lo = AbstractTensor.where(live * below > 0.5, mid + 1.0, lo)
                    hi = AbstractTensor.where(live * (1.0 - below) > 0.5, mid, hi)
                ends.append(lo)
            runs.append((ends[0], ends[1], ok))
        t_emit = AbstractTensor.where(pid < float(active), pid * emit_rate, pid * 0.0 + 1.0e9)
        # ---- the rest of what this id is, derived again HERE (not before the sort: a value that
        # crosses the sort's dispatches would hold an arena slot for all of them)
        bar_x = _mod(_lin(pid, 2531.0), _PRIME) * (1.0 / _PRIME)
        bar_y = _mod(_lin(pid, 3187.0), _PRIME) * (1.0 / _PRIME)
        theta = (_mix(pid, 1759.0, 17.0) * 2.0 - 1.0) * 0.7
        speed = 0.9 + _mix(pid, 2203.0, 29.0) * 0.7
        vx0 = speed * theta.sin()
        vy0 = speed * theta.cos()
        x0 = (bar_x - 0.5) * (2.0 * bar_half)
        y0 = nozzle_y + (bar_y - 0.5) * BAR_HEIGHT
        u = _mix(pid, 1237.0, 5.0)
        w0 = (1.0 - mix) * MONO_WEIGHT[0] + mix * POLY_WEIGHT[0]
        w1 = (1.0 - mix) * MONO_WEIGHT[1] + mix * POLY_WEIGHT[1]
        w2 = (1.0 - mix) * MONO_WEIGHT[2] + mix * POLY_WEIGHT[2]
        c0, c1, c2 = w0, w0 + w1, w0 + w1 + w2
        sp = as_float(u >= c0) + as_float(u >= c1) + as_float(u >= c2)
        is1, is2, is3 = (as_float((sp - float(k)).abs() < 0.5) for k in (1, 2, 3))
        radius = SPECIES[0][1] + (SPECIES[1][1] - SPECIES[0][1]) * is1 + (SPECIES[2][1] - SPECIES[0][1]) * is2 \
            + (SPECIES[3][1] - SPECIES[0][1]) * is3
        density = SPECIES[0][2] + (SPECIES[1][2] - SPECIES[0][2]) * is1 + (SPECIES[2][2] - SPECIES[0][2]) * is2 \
            + (SPECIES[3][2] - SPECIES[0][2]) * is3
        mu = SPECIES[0][3] + (SPECIES[1][3] - SPECIES[0][3]) * is1 + (SPECIES[2][3] - SPECIES[0][3]) * is2 \
            + (SPECIES[3][3] - SPECIES[0][3]) * is3
        drag = SPECIES[0][4] + (SPECIES[1][4] - SPECIES[0][4]) * is1 + (SPECIES[2][4] - SPECIES[0][4]) * is2 \
            + (SPECIES[3][4] - SPECIES[0][4]) * is3
        mass = density * math.pi * radius * radius
        just = act * as_float((age - DT) < t_emit)
        # ---- particle-particle contacts: forces from the neighbours' state
        def neighbour_props(j):
            uj = _mix(j, 1237.0, 5.0)
            spj = as_float(uj >= c0) + as_float(uj >= c1) + as_float(uj >= c2)
            j1, j2, j3 = (as_float((spj - float(k)).abs() < 0.5) for k in (1, 2, 3))
            rj = SPECIES[0][1] + (SPECIES[1][1] - SPECIES[0][1]) * j1 + (SPECIES[2][1] - SPECIES[0][1]) * j2 \
                + (SPECIES[3][1] - SPECIES[0][1]) * j3
            dj = SPECIES[0][2] + (SPECIES[1][2] - SPECIES[0][2]) * j1 + (SPECIES[2][2] - SPECIES[0][2]) * j2 \
                + (SPECIES[3][2] - SPECIES[0][2]) * j3
            muj = SPECIES[0][3] + (SPECIES[1][3] - SPECIES[0][3]) * j1 + (SPECIES[2][3] - SPECIES[0][3]) * j2 \
                + (SPECIES[3][3] - SPECIES[0][3]) * j3
            return rj, dj * math.pi * rj * rj, muj

        for lo, hi, ok in runs:
            for slot in range(K_RUN):
                idx = lo + float(slot)
                valid = as_float(idx < hi) * ok * gridded
                j = ki.index_select(0, idx.clamp(0.0, top))
                valid = valid * as_float((j - pid).abs() > 0.5)
                xj = x.index_select(0, j)
                yj = y.index_select(0, j)
                vxj = vx.index_select(0, j)
                vyj = vy.index_select(0, j)
                wj = w.index_select(0, j)
                rj, mj, muj = neighbour_props(j)
                dx = x - xj
                dyy = y - yj
                dist = (dx * dx + dyy * dyy + 1.0e-12).sqrt()
                delta = (radius + rj - dist) * valid
                touch = as_float(delta > 0.0)
                nx = dx / dist
                ny = dyy / dist
                m_eff = mass * mj / (mass + mj)
                kn = OMEGA * OMEGA * m_eff
                gx, gy, gt = _contact(delta, nx, ny, vx - vxj, vy - vyj, radius * w + rj * wj, kn, m_eff,
                                      (mu * muj).sqrt())
                fx = fx + gx * touch
                fy = fy + gy * touch
                tq = tq - radius * gt * touch
        # ---- walls and floor: the same contact against a body of infinite mass
        kw = OMEGA * OMEGA * mass
        floor_delta = (radius - (y - FLOOR_Y)).clamp(0.0, 1.0e9) * act
        gx, gy, gt = _contact(floor_delta, y * 0.0, y * 0.0 + 1.0, vx, vy, radius * w, kw, mass, mu)
        touch = as_float(floor_delta > 0.0)
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        left_delta = (radius - (x + half)).clamp(0.0, 1.0e9) * act
        gx, gy, gt = _contact(left_delta, y * 0.0 + 1.0, y * 0.0, vx, vy, radius * w, kw, mass, mu)
        touch = as_float(left_delta > 0.0)
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        right_delta = (radius - (half - x)).clamp(0.0, 1.0e9) * act
        gx, gy, gt = _contact(right_delta, y * 0.0 - 1.0, y * 0.0, vx, vy, radius * w, kw, mass, mu)
        touch = as_float(right_delta > 0.0)
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        # ---- the integrator: gravity, drag, the forces above; emission places the id at the nozzle
        inertia = 0.5 * mass * radius * radius
        vxn = AbstractTensor.where(just > 0.5, vx0, vx + act * (fx / mass - drag * vx) * DT)
        vyn = AbstractTensor.where(just > 0.5, vy0, vy + act * (fy / mass - G - drag * vy) * DT)
        wn = (w + act * (tq / inertia - SPIN_DRAG * w) * DT) * (1.0 - just)
        xn = AbstractTensor.where(just > 0.5, x0, x + act * vxn * DT)
        yn = AbstractTensor.where(just > 0.5, y0, y + act * vyn * DT)
        age_next = age + DT
        # ---- colour from the angle of the launch vector.  theta is the launch angle from the
        # vertical (+-0.7 rad).  The hue turns four times as fast as theta so the fan spans most of
        # the wheel: trig = atan2, rotation = the double-angle formulas applied twice to cos/sin
        # read off the vector (no atan2).
        cth = vy0 / speed
        sth = vx0 / speed
        c2 = cth * cth - sth * sth
        s2 = 2.0 * cth * sth
        c = c2 * c2 - s2 * s2
        sn = 2.0 * c2 * s2
        hue = vx0.atan2(vy0) * (4.0 / TAU) + 0.5
        root3 = 0.5773502691896258
        rot_r = (c * (2.0 / 3.0)) * 0.85 + 0.5
        rot_g = (c * (-1.0 / 3.0) + sn * root3) * 0.85 + 0.5
        rot_b = (c * (-1.0 / 3.0) - sn * root3) * 0.85 + 0.5
        m = as_float(mode > 0.5)
        hue_r = _ramp(hue, 0.0)
        hue_g = _ramp(hue, 2.0 / 3.0)
        hue_b = _ramp(hue, 1.0 / 3.0)
        base_r = hue_r + m * (rot_r.clamp(0.0, 1.0) - hue_r)
        base_g = hue_g + m * (rot_g.clamp(0.0, 1.0) - hue_g)
        base_b = hue_b + m * (rot_b.clamp(0.0, 1.0) - hue_b)
        keep = 1.0 - use_image
        cr = base_r * keep + ir * use_image
        cg = base_g * keep + ig * use_image
        cb = base_b * keep + ib * use_image
        return xn, yn, vxn, vyn, wn, age_next, cr, cg, cb, radius

    return fountain_step


# ------------------------------------------------------------------- columns
def _f32(a) -> np.ndarray:
    """Rounded through float32: the page's buffers and the reference see one value."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def uniforms(*, mode: int = 0, use_image: float = 0.0, mix: float = 0.5) -> dict[str, np.ndarray]:
    """The law's scalar columns (shape ()) as float32-representable values."""
    return {"mode": _f32(np.float64(mode)), "use_image": _f32(np.float64(use_image)), "mix": _f32(np.float64(mix))}


def initial_columns(n: int, *, mix: float = 0.5, mode: int = 0, use_image: float = 0.0) -> dict[str, np.ndarray]:
    """Round zero for ``n`` ids (a power of two): the state is zero, ``pid`` is the iota and the
    image colours are zero.  Nothing else is per-id."""
    zeros = np.zeros(n)
    columns = {name: zeros.copy() for name in STATE + ("ir", "ig", "ib")}
    columns["pid"] = np.arange(n, dtype=np.float64)
    columns.update(uniforms(mode=mode, use_image=use_image, mix=mix))
    return columns


def fit_extent(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rest positions -> image coordinates ``(u across, v down)``: the pile's own
    bounding box is the picture frame.  The page's runtime applies the same rule."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    u = (x - x.min()) / max(float(x.max() - x.min()), 1e-12)
    v = 1.0 - (y - y.min()) / max(float(y.max() - y.min()), 1e-12)
    return _f32(u), _f32(v)


def sample_raster(raster: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-pixel sample of an ``(h, w, 3)`` uint8 raster at frame fractions
    (``u`` across, ``v`` down), as float64 in [0, 1] (``byte / 255``, float32
    rounded).  The page's ``sampleRasterAt`` is the same formula."""
    h, w = raster.shape[:2]
    px = np.clip((np.asarray(u) * w).astype(np.int64), 0, w - 1)
    py = np.clip((np.asarray(v) * h).astype(np.int64), 0, h - 1)
    return _f32(raster[py, px, :3] / 255.0)


# ------------------------------------------------------------------- piece
class FountainPiece(LawPiece):
    """The fountain law as a ``LawPiece`` (the repository's law-product machinery)."""

    entry = "fountain_step"
    argument_names = ARGUMENTS
    output_names = OUTPUTS

    def __init__(self, batch: int, active: int | None = None):
        super().__init__(batch)
        # ``LawPiece`` files its native product under entry + source digest + argument list;
        # the law and its shapes depend on the id count and the active count.
        self.entry = f"fountain_step_{int(batch)}_{int(active or batch)}"


def make_piece(n: int, active: int | None = None) -> FountainPiece:
    """A ``FountainPiece`` whose law is ``make_law(n, active)``."""
    class Sized(FountainPiece):
        law = staticmethod(make_law(n, active))
    return Sized(n, active)


def run_frames(piece: LawPiece, columns: dict[str, np.ndarray], steps: int) -> dict[str, np.ndarray]:
    """Advance ``steps`` fixed steps through ``piece`` (its compiled product
    when compiled, otherwise the AbstractTensor law); returns the final state,
    the colour columns and ``rad``."""
    state = {name: np.array(columns[name], dtype=np.float64) for name in STATE}
    extra = {}
    for _ in range(int(steps)):
        outputs = piece(*[state[n] if n in state else columns[n] for n in ARGUMENTS])
        produced = dict(zip(OUTPUTS, outputs))
        for name in STATE:
            state[name] = produced[name + "_next"]
        extra = {name: produced[name] for name in ("cr", "cg", "cb", "rad")}
    return {**state, **extra}


# -------------------------------------------------------------------- game
#: size tag -> (active ids, sort length).  ``small`` is the wasm demonstration; the rest are the
#: ladder the page offers (4096 .. 10M).  The sort length is the next power of two.
SIZES = {
    "small": (256, 256), "4k": (4096, 4096), "16k": (16384, 16384), "64k": (65536, 65536),
    "256k": (262144, 262144), "1M": (1048576, 1048576), "4M": (4194304, 4194304),
    "10M": (10_000_000, pad_count(10_000_000)),
}
DEFAULT_SIZE = "4k"
N_FULL = SIZES[DEFAULT_SIZE][0]
N_SMALL = SIZES["small"][0]
#: The free board the game occupies in the main scene (board units); the world window of the
#: default size is mapped onto it, and the page maps the other sizes' windows the same way.
BOARD = (10.5, 10.5 * (basin(N_FULL)["view"][3] - basin(N_FULL)["view"][2]) / (basin(N_FULL)["view"][1] - basin(N_FULL)["view"][0]))


class FountainGame(Game):
    """Server side of the fountain: phase, mix, size, the image hand-off and
    verification by the law's native product.  It does not stream particles:
    the main page's generic compute runtime runs the compiled law (WebGPU
    compute or the wasm product, as the compiler's verdict and the browser
    allow) and draws the particles in the shared scene.

    Actions (``/api/action``): ``fountain_start`` (phase 1; optional ``mix``,
    ``mode``), ``fountain_mode``, ``fountain_speed``, ``fountain_size``,
    ``fountain_image`` (per-id colours as base64 bytes ``rgb``, three per id,
    or just ``count`` for sizes too large to ship), ``fountain_replay``
    (phase 2) and ``fountain_verify`` (run the law ``steps`` steps from round
    zero natively, small sizes only, and report a digest)."""

    game = "fountain"
    input = "compute"
    title = "Fountain"

    def __init__(self, board_id: str = "", *, size: str = DEFAULT_SIZE):
        super().__init__(board_id)
        self.size = size
        self.mix, self.mode, self.speed = 0.5, 0, 1
        self.phase = "fill"
        self.epoch = 1                       # bumped whenever the page must restart the fountain
        self.image: np.ndarray | None = None
        self.has_image = False
        self.piece: FountainPiece | None = None
        self.verified: dict[str, Any] | None = None
        self.status = "fountain"

    @property
    def n(self) -> int:
        return SIZES[self.size][0]

    @property
    def size_tag(self) -> str:
        return self.size

    def columns(self, *, image: bool = False) -> dict[str, np.ndarray]:
        pad = SIZES[self.size][1]
        cols = initial_columns(pad, mix=self.mix, mode=self.mode, use_image=1.0 if image else 0.0)
        if image and self.image is not None:
            cols["ir"][: self.n], cols["ig"][: self.n], cols["ib"][: self.n] = (
                _f32(self.image[:, c] / 255.0) for c in range(3))
        return cols

    def reset(self) -> None:
        self.phase, self.verified = "fill", None
        self.epoch += 1

    def do(self, payload: dict) -> None:
        kind = str(payload.get("type"))
        if kind == "fountain_start":
            self.mix = float(payload.get("mix", self.mix))
            self.mode = int(payload.get("mode", self.mode))
            self.reset()
        elif kind == "fountain_mode":
            self.mode = int(payload["mode"])
        elif kind == "fountain_speed":
            self.speed = int(payload["speed"])
        elif kind == "fountain_size":
            tag = str(payload["size"])
            if tag not in SIZES:
                raise ValueError(f"fountain_size: unknown size {tag!r}")
            self.size = tag
            self.image, self.has_image, self.piece = None, False, None
            self.reset()
        elif kind == "fountain_image":
            if "rgb" in payload:
                import base64
                raw = np.frombuffer(base64.b64decode(str(payload["rgb"])), dtype=np.uint8)
                if raw.size != self.n * 3:
                    raise ValueError(f"fountain_image: expected {self.n * 3} bytes, got {raw.size}")
                self.image = raw.reshape(self.n, 3).copy()
            elif int(payload.get("count", -1)) != self.n:
                raise ValueError("fountain_image: neither rgb bytes nor a matching count")
            self.has_image = True
        elif kind == "fountain_replay":
            if not self.has_image:
                raise ValueError("fountain_replay: supply an image first")
            self.phase, self.verified = "replay", None
            self.epoch += 1
        elif kind == "fountain_verify":
            self.verify(int(payload.get("steps", 600)), image=bool(payload.get("image", False)))
        else:
            raise ValueError(f"unknown fountain action {kind!r}")

    def verify(self, steps: int, *, image: bool = False) -> dict[str, Any]:
        active, pad = SIZES[self.size]
        if pad > 4096:
            raise ValueError("fountain_verify: the native lane is built for sizes up to 4096 ids")
        columns = self.columns(image=image)
        if self.piece is None:
            self.piece = make_piece(pad, active)
            self.piece.compile(columns)
        final = run_frames(self.piece, columns, steps)
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(final[n], dtype=np.float32).tobytes()
                                         for n in STATE + ("cr", "cg", "cb"))).hexdigest()[:16]
        emitted = int((columns["pid"][:active] * (EMIT_SPAN / active) <= steps * DT).sum())
        self.verified = {"steps": steps, "lane": self.piece.lane, "digest": digest,
                         "emitted": emitted, "top": float(final["y"].max())}
        return self.verified

    def controls_ui(self) -> list[dict[str, Any]]:
        sample = {"fit": "extent", "read": ["x", "y"], "into": ["ir", "ig", "ib"], "raster": [96, 72],
                  "post": {"action": "fountain_image", "field": "rgb", "max_ids": 65536}}
        sizes = [[tag, f"{count:,}"] for tag, (count, _pad) in SIZES.items()]
        return [
            {"id": "start", "kind": "button", "label": "Start", "action": "fountain_start"},
            {"id": "image", "kind": "file", "label": "Image", "sample": sample},
            {"id": "camera", "kind": "file", "label": "Camera", "capture": "environment", "sample": sample},
            {"id": "replay", "kind": "button", "label": "Replay", "action": "fountain_replay",
             "enabled": self.has_image},
            {"id": "mix", "kind": "range", "label": "poly mix", "min": 0, "max": 1, "step": 0.05,
             "value": self.mix, "action": "fountain_start", "field": "mix"},
            {"id": "mode", "kind": "select", "label": "colour", "value": self.mode, "action": "fountain_mode",
             "field": "mode", "options": [[0, "trig (atan2)"], [1, "rotation"]]},
            {"id": "size", "kind": "select", "label": "ids", "value": self.size, "action": "fountain_size",
             "field": "size", "options": sizes},
            {"id": "speed", "kind": "select", "label": "speed", "value": self.speed, "action": "fountain_speed",
             "field": "speed", "options": [[1, "1x"], [2, "2x"], [4, "4x"]]},
            {"id": "verify", "kind": "button", "label": "Verify", "action": "fountain_verify"},
        ]

    def state(self) -> dict[str, Any]:
        width, height = BOARD
        base = self.base_state(int(round(height)), int(round(width)))
        verified = self.verified
        text = {"fill": "phase 1: the fountain fills the basin by contact forces; hue from the launch angle",
                "replay": "phase 2: the identical simulation, coloured by your image at each id's rest position"}[self.phase]
        base.update({
            "board": "free", "width": width, "height": height, "input": self.input,
            "palette": {"floor": [0.10, 0.11, 0.16]}, "cells": ["floor"], "marks": [], "hints": [],
            "pieces": [], "entities": [],
            "status": f"{text} · {self.n:,} ids" + (f" · server {verified['lane']}: {verified['emitted']} emitted, "
                                                    f"top {verified['top']:.3f}, digest {verified['digest']}" if verified else ""),
            "n": self.n, "phase": self.phase, "mix": self.mix, "mode": self.mode,
            "has_image": self.has_image, "verified": verified,
            "controls_ui": self.controls_ui(),
            "compute": {
                "manifest": "fountain.manifest.json", "size": self.size, "epoch": self.epoch,
                "uniforms": {"mode": float(self.mode), "use_image": 1.0 if self.phase == "replay" else 0.0,
                             "mix": float(self.mix)},
                "speed": self.speed, "named": ["ir", "ig", "ib"],
            },
        })
        return base

    def result(self):
        return None
