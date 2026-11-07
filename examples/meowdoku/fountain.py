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
  a particle is parked, takes no part in contacts and is not drawn.  At its
  time it is ARMED at its nozzle slot and is emitted at the first step no live
  grain is within contact distance of the slot (plus what a pair can close in
  one step): a grain never spawns inside another, so the contact springs
  inject no energy at the nozzle.  The pile forms and settles by contact
  forces, so regular and irregular packing emerge from the species mix.
* The basin and nozzle scale with ``sqrt(n)`` so the pile keeps its shape, and
  take the screen's aspect (``basin``) at the same pile area; the basin is a
  set of RUNTIME uniforms, so one program serves every aspect.
* GRAVITY is a vector, a RUNTIME INPUT (``grav_x``, ``grav_y``, default ``(0, -G)``), as are the basin's numbers: a phone's
  tilt turns it moment by moment with no rebuild.  ``lid`` (a ceiling at that height, off by default) keeps grains in the
  basin when gravity points up.  The inputs enter the law as plain additions, so the default is bit for bit the old law.
* DIAGNOSTICS: the law reduces, by a fixed tree of additions, the kinetic +
  rotational energy, the largest speed, and the number of ids that are not
  finite / outside the walls / parked / live, plus the contacts the neighbour
  walk dropped.  The page ends phase 1 on the energy (``SETTLE``).

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
#: The pile is AT REST (phase 1 ends, at step S; phase 2 stops at exactly S) when everything is emitted and the law's
#: kinetic + rotational energy has stayed under BOTH ``ke_per_id`` per id and ``relative`` of its peak, with the largest speed
#: under ``speed``, for ``hold_samples`` consecutive samples taken every ``sample`` steps.  ``max_seconds`` caps a pile that
#: never quiets.  ``web/compute.js`` ``SettleDetector`` is this rule.
SETTLE = {"sample": 30, "hold_samples": 12, "relative": 0.01, "ke_per_id": 1.0e-7, "speed": 0.25, "max_seconds": 40.0}
DT = 1.0 / 360.0
G = 1.0
OMEGA = 150.0                          # contact stiffness: kn = m * OMEGA^2 (OMEGA * DT stays well under 1: a grain feels several contacts)
ZETA = 0.7                             # normal damping ratio
VREF = 0.02                            # tangential slip speed over which friction saturates
SPIN_DRAG = 3.0                        # 1/s
STRATA = 1.0                           # default hue cycles over the emission in mode 2 (time strata); a runtime uniform ``strata``
#: The time-strata rates the page offers (hue cycles over one emission): slower keeps wider bands of one colour in place.
STRATA_CHOICES = (0.5, 1.0, 2.0, 4.0, 8.0)
# the page's render looks for the particle field (web/particle_styles.js STYLES -- a test holds the two lists equal)
PARTICLE_STYLES = (("points", "points"), ("liquid", "liquid"), ("sand", "sand"), ("glow", "glow"), ("heat", "heat"),
                   ("stamp", "stamp (memory)"))
# the page's noise-filter replacement modes (web/noise_filter.js NOISE_MODES -- a test holds the two lists equal)
NOISE_MODES = (("mean", "mean colour"), ("spectral", "spectral class"))
K_RUN = 24                             # candidates walked per row run (a hex pack of the smallest grains is ~9; 24 covers a 2.5x compressed one)
ARMED = -1.0e9                         # the vy of an id waiting at its nozzle slot (see ``fountain_step``)
EMIT_MARGIN = 1.0e-3                   # an id is emitted only where no live grain, at its NEXT-step position, is within contact distance + this
#: The share of a contact's slip that ONE step's friction may remove.  Coulomb friction is regularised with a tanh whose slope
#: mu fn / VREF is stiff, and an explicit step of a force that stiff reverses the slip it acts on (and, for the several contacts
#: of a grain, pumps energy: a settled pile kept spinning and rattling for ever).  Friction is therefore limited to
#: ``FRICTION_SHARE`` of the slip per step, in the contact's own slip mass (m_eff / 3 for a disk's slip, spin included); a grain
#: has up to six neighbours and a wall, so the share is about 1/7.
FRICTION_SHARE = 0.15
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

#: The species table is a set of RUNTIME TUNABLES (uniforms, no rebuild), each a MULTIPLE of the default above so that the
#: default is the old law bit for bit: radius ``rscale``, friction ``fscale``, density ``dscale``, damping ratio ``zscale``
#: (restitution), and the ORBEEZ pair ``swell`` (growth rate, fraction of the initial radius per second, 0 = does not swell)
#: and ``swmax`` (largest radius, as a multiple of the initial one).  ``cell_scale`` multiplies CELL (the grid cell, 2 x the
#: largest radius a grain can reach), ``solvent`` in [0, 1] is the fluid present for swelling (air 0, water 1).
RSCALE_RANGE = (0.1, 10.0)             # radius: 0.1x to 10x each species' default (log scale on the page)
MU_RANGE = (0.0, 2.0)                  # sliding friction: frictionless to very sticky
RESTITUTION_RANGE = (0.02, 0.95)
DENSITY_RANGE = (0.2, 5.0)
SWELL_RATE_MAX = 0.5                   # 1/s
SWELL_MAX_RATIO = 4.0
FULL_COORDINATION = 6.0                # contacts of a grain with no open surface (a hex pack); exposure = 1 - contacts / this
SOLVENTS = {"air": 0.0, "water": 1.0, "glycerine": 0.5}
SPECIES_TUNABLES = ("rscale", "fscale", "dscale", "zscale", "swell", "swmax")
STATE = ("x", "y", "vx", "vy", "w", "age", "rs")          # rs: the grain's CURRENT radius (it grows for a swelling species)
CONSTANTS = ("pid", "ir", "ig", "ib")
UNIFORMS = ("mode", "use_image", "mix", "half", "nozzle_y", "bar_half", "cols", "rows", "grav_x", "grav_y", "lid", "strata",   # the run, its basin, the RUNTIME INPUTS, then the colour rate
            "cell_scale", "solvent", "case_omega", "case_domega", "pivot_x", "pivot_y") + tuple(f"{k}{s}" for k in SPECIES_TUNABLES for s in range(4))
#: The uniforms the PAGE drives moment by moment (no rebuild, no reset): the gravity vector (the phone's tilt), ``lid``, a
#: ceiling that is far away (``LID_OFF``) until the page, driving gravity from a sensor, closes it so a grain cannot fall
#: out of the open top when gravity points up, and ``rows``, the grid's height, which it raises with the lid so grains under
#: the lid still meet each other.  The server sends their defaults (and the closed values as ``tilt_inputs``); the page's
#: runtime owns them after that.
RUNTIME_INPUTS = ("grav_x", "grav_y", "lid", "rows", "solvent", "case_omega", "case_domega") + tuple(
    f"{k}{s}" for k in ("fscale", "dscale", "zscale", "swell", "swmax") for s in range(4))      # the radius (and so the grid) is fixed when a run starts
LID_OFF = 1.0e9
LID_ABOVE = 1.45                       # the lid the page closes while a sensor drives gravity: this far above the nozzle (the jet's apex is ~1.34)
IOTA = ("pid",)                        # columns the generated shader derives from its invocation index
#: ``ke``..``n_drop`` are the law's DIAGNOSTICS: reduced by a fixed-order tree, so element 0 holds the total over all ids
#: (kinetic + rotational energy, the largest speed squared, and the counts of the ids that are not finite, outside the
#: walls, parked, live, plus the contacts the neighbour walk dropped).  The four classes partition the ids.
DIAGNOSTICS = ("ke", "vmax2", "n_nan", "n_out", "n_parked", "n_live", "n_drop")
OUTPUTS = ("x_next", "y_next", "vx_next", "vy_next", "w_next", "age_next", "rs_next", "cr", "cg", "cb", "rad") + DIAGNOSTICS
ARGUMENTS = STATE + CONSTANTS + UNIFORMS

TAU = 2.0 * math.pi


def merge_exchange_stages(n: int) -> list[tuple[int, int, int]]:
    """Knuth 5.2.2 Algorithm M (merge exchange) for any ``n``: the sorting network as stages
    ``(p, d, r)`` -- compare-exchange ``(i, i + d)`` for every ``i`` with ``bit_p(i) == r`` (r in {0, 1})
    and ``i + d < n``.  The pairs of a stage are disjoint, so a stage is one dispatch."""
    stages: list[tuple[int, int, int]] = []
    if n < 2:
        return stages
    t = (n - 1).bit_length()
    p = 1 << (t - 1)
    while p > 0:
        q, r, d = 1 << (t - 1), 0, p
        while d > 0:
            stages.append((p, d, 1 if r else 0))
            d, q, r = q - p, q >> 1, p
        p >>= 1
    return stages


def reduction_shifts(n: int) -> list[int]:
    """The shifts 1, 2, 4 .. of the law's fixed-order reduction (each is a gather, so a dispatch): after them element 0
    of a reduced column holds the total over all ``n`` ids."""
    return [1 << k for k in range((n - 1).bit_length())] if n > 1 else []


def pad_count(n: int) -> int:
    """The sort length: ``n`` rounded up to a power of two."""
    return 1 << max(1, (int(n) - 1).bit_length())


ASPECT_MIN, ASPECT_MAX = 0.35, 3.0     # the viewport aspects (width / height) a basin will stretch to
VIEW_BOTTOM, VIEW_TOP, VIEW_SIDE = 0.05, 1.35, 1.02


def fit_scale(s: float, aspect: float) -> float:
    """The width factor ``k`` of a basin whose VIEW (side walls to the top of the plume) has width:height ``aspect``.
    The pile keeps its area: width goes as ``s k`` and the nozzle height as ``0.6 s / k + 0.7`` (a narrower basin
    heaps the pile higher), so the view is ``2 VIEW_SIDE s k`` wide and ``VIEW_BOTTOM s + 0.7 + VIEW_TOP + 0.6 s / k`` tall;
    equating the ratio to ``aspect`` is a quadratic in ``k``."""
    a = min(max(float(aspect), ASPECT_MIN), ASPECT_MAX)
    b = a * (0.7 + VIEW_TOP + VIEW_BOTTOM * s)
    w = 2.0 * VIEW_SIDE * s
    return (b + math.sqrt(b * b + 4.0 * w * 0.6 * a * s)) / (2.0 * w)


def basin(n: int, aspect: float | None = None, cell: float = CELL) -> dict[str, float]:
    """Geometry for ``n`` active ids: basin and nozzle scale with sqrt(n / N_BASE).  ``aspect`` (the screen's
    width / height, clamped) stretches it to fill that screen at the same pile area; ``None`` is the reference basin.
    Every number the law needs is a RUNTIME uniform (see ``uniforms``): one program serves every aspect."""
    s = math.sqrt(n / N_BASE)
    k = 1.0 if not aspect else fit_scale(s, aspect)
    width = WIDTH_BASE * s * k
    nozzle_y = 0.6 * s / k + 0.7
    cols = int(math.ceil(width / cell))
    rows = int(math.ceil((nozzle_y + 0.1) / cell))
    return {"scale": s, "stretch": k, "width": width, "half": width / 2.0, "nozzle_y": nozzle_y, "bar_half": 0.8 * s * k,
            "cols": cols, "rows": rows, "cells": cols * rows,
            "view": (-VIEW_SIDE * s * k, VIEW_SIDE * s * k, -VIEW_BOTTOM * s, nozzle_y + VIEW_TOP)}


def tilt_inputs(n: int, aspect: float | None = None, cell: float = CELL) -> dict[str, float]:
    """What the page writes to the runtime inputs ``lid`` and ``rows`` while a sensor drives gravity: a ceiling above the
    jet's reach and a grid tall enough to cover everything under it."""
    geo = basin(n, aspect, cell)
    lid = geo["nozzle_y"] + LID_ABOVE
    return {"lid": float(_f32(lid)), "rows": float(math.ceil(lid / cell))}


def board_size(view) -> tuple[float, float]:
    """The free board (scene units) a basin view maps onto: the view's aspect, the reference basin's area."""
    a = (view[1] - view[0]) / (view[3] - view[2])
    ref = basin(N_BASE)["view"]
    a0 = (ref[1] - ref[0]) / (ref[3] - ref[2])
    return (10.5, 10.5 / a) if a >= a0 else (13.5 * a, 13.5)


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


def _contact(delta, nx, ny, vrx, vry, spin_slip, kn, m_eff, mu, zeta=ZETA):
    """Force and torque on the particle from one contact (``delta`` > 0 overlap,
    ``n`` the unit normal pointing at THIS particle, ``vr`` its velocity relative
    to the other body, ``spin_slip`` the spin part of the contact slip).  Returns
    (fx, fy, tangential force); torque = -radius * the third value."""
    vn = vrx * nx + vry * ny
    cn = 2.0 * zeta * (kn * m_eff).sqrt()
    fn = (kn * delta - cn * vn).clamp(0.0, 1.0e9)
    tx, ty = -ny, nx
    vt = vrx * tx + vry * ty - spin_slip
    # tanh of the slip speed over VREF.  An emitted grain meets the plume at ~1.5 m/s, so the argument
    # reaches ~75; a tanh built from exp(2x) (Metal/Apple shaders, fast-math) is inf/inf = NaN past
    # |x| ~ 44 in f32, and one NaN force removes the grain and every grain it touches.  tanh(20) is 1
    # to 1e-17, so clamping the argument changes no f64 result and keeps every f32 lane finite.
    tanh = (vt * (1.0 / VREF)).clamp(-20.0, 20.0).tanh()
    coulomb = mu * fn * tanh.abs()
    stick = (FRICTION_SHARE / (3.0 * DT)) * m_eff * vt.abs()          # the most that removes FRICTION_SHARE of the slip in one step
    ft = -(coulomb - (coulomb - stick).clamp(0.0, 1.0e30)) * (vt / (vt.abs() + 1.0e-30))
    return fn * nx + ft * tx, fn * ny + ft * ty, ft


def _cap(v, hi):
    """``min(v, hi)`` for a RUNTIME bound ``hi`` (a scalar uniform): clamp bounds are compile-time numbers."""
    return v - (v - hi).clamp(0.0, 1.0e9)


def _nozzle(pid, bar_half, nozzle_y):
    """Where id ``pid`` leaves the nozzle: a low-discrepancy step along the bar (the golden-ratio step 2531/4093, so ids
    emitted close together in time are far apart along the bar) and a hashed height within it."""
    bar_x = _mod(_lin(pid, 2531.0), _PRIME) * (1.0 / _PRIME)
    bar_y = _mod(_lin(pid, 3187.0), _PRIME) * (1.0 / _PRIME)
    return (bar_x - 0.5) * (2.0 * bar_half), nozzle_y + (bar_y - 0.5) * BAR_HEIGHT


def make_law(n: int, active: int | None = None):
    """The fountain law over ``n`` ids; ``active`` of them (default all) are ever emitted, the rest stay parked.
    The basin (``half``, ``nozzle_y``, ``bar_half``, ``cols``, ``rows``) is a runtime uniform: see ``basin``."""
    active = int(n if active is None else active)
    stages = merge_exchange_stages(n)
    search_steps = int(math.ceil(math.log2(n + 1))) + 1
    emit_rate = EMIT_SPAN / float(active)
    scan_shifts = reduction_shifts(n)

    def fountain_step(*args):
        """One fixed step of every particle (see the module docstring)."""
        a = dict(zip(ARGUMENTS, args))
        x, y, vx, vy, w, age, rs = (a[k] for k in STATE)
        pid, ir, ig, ib = (a[k] for k in CONSTANTS)
        mode, use_image, mix, half, nozzle_y, bar_half, cols, rows, grav_x, grav_y, lid, strata = (
            a[k] for k in ("mode", "use_image", "mix", "half", "nozzle_y", "bar_half", "cols", "rows", "grav_x", "grav_y", "lid", "strata"))
        solvent = a["solvent"]
        # the CASE FRAME: the walls are static in the frame the law runs in, and the case turns at ``case_omega`` (rad/s, about
        # the pivot, counterclockwise positive) with angular acceleration ``case_domega``.  A grain feels, besides gravity
        # (which the page rotates), the fictitious accelerations  -2 w x v  (Coriolis: applied as an exact rotation of the
        # velocity, so any rate is stable),  w^2 (r - p)  (centrifugal) and  -dw/dt x (r - p)  (Euler); its own spin, measured
        # in this frame, gains -dw/dt.  Defaults (0, and the box centre as pivot) add exact zeros: the old law bit for bit.
        case_omega, case_domega = a["case_omega"], a["case_domega"]
        cell = CELL * a["cell_scale"]
        cells = cols * rows

        def tab(name):
            return [a[f"{name}{s}"] for s in range(4)]

        def pick4(vals, m1, m2, m3):
            return vals[0] + (vals[1] - vals[0]) * m1 + (vals[2] - vals[0]) * m2 + (vals[3] - vals[0]) * m3

        def frame(search):
            """emit time and the state of each id, from the id and its state.  Evaluated again after the sort
            rather than carried across its ~300 dispatches (a value that crosses them holds an arena slot, 4 bytes
            per id each, for all of them).

            An id is in one of three kinds of state.  PARKED is the exact zero state (x = y = 0: no real grain rests
            at the origin).  ARMED (``vy`` = ``ARMED``) is an id whose time has come: it waits AT the nozzle slot it
            will leave from (its stored x, y), takes part in no contact and does not move, and is emitted at the first
            step its slot is clear (no live grain, where it will be NEXT step, within contact distance).  Anything else is LIVE.  With ``search`` an armed id looks up its neighbours at
            its slot, as a live id does at its place."""
            t = AbstractTensor.where(pid < float(active), pid * emit_rate, pid * 0.0 + 1.0e9)
            a = as_float(age >= t)
            parked = as_float((x.abs() + y.abs()) < 1.0e-30)
            armed = as_float(vy < ARMED * 0.5)
            live = a * (1.0 - parked) * (1.0 - armed)
            pending = a * armed
            arming = a * parked
            # only particles inside the grid take part (parked ones and the dilute spray above the grid's top
            # get the cell past the end: sorted last, in nobody's block)
            ix = _cap(((x + half) * (1.0 / cell)).floor().clamp(0.0, 1.0e9), cols - 1.0)
            iy = _cap(((y - FLOOR_Y) * (1.0 / cell)).floor().clamp(0.0, 1.0e9), rows - 1.0)
            inside = as_float(y < rows * cell + FLOOR_Y)
            # armed ids are sorted into the grid too (at their slots): they see each other, and live grains see them
            return t, a, arming, live, pending, ix, iy, (live + pending * float(search)) * inside, (live + pending) * inside
        t_emit, act, arming, live, pending, cx, cy, searching, gridded = frame(False)
        kc = AbstractTensor.where(gridded > 0.5, cy * cols + cx, cy * 0.0 + cells)
        ki = pid
        # ---- sort of (cell, id) pairs by Knuth's merge exchange (any length, no padding): each stage
        # is a gather of the partner and a compare-exchange; the pairs of one stage are disjoint
        top_i = float(n - 1)
        for (pp, dd, rr) in stages:
            b_self = (pid * (1.0 / pp)).floor() - 2.0 * (pid * (1.0 / (2 * pp))).floor()
            pj = (pid - float(dd)).clamp(0.0, top_i)
            b_other = (pj * (1.0 / pp)).floor() - 2.0 * (pj * (1.0 / (2 * pp))).floor()
            lower = as_float((b_self - float(rr)).abs() < 0.5) * as_float(pid + float(dd) < float(n))
            upper = as_float((b_other - float(rr)).abs() < 0.5) * as_float(pid >= float(dd)) * (1.0 - lower)
            partner = pid + float(dd) * (lower - upper)
            oc = kc.index_select(0, partner)
            oi = ki.index_select(0, partner)
            less = as_float(kc < oc) + as_float((kc - oc).abs() < 0.5) * as_float(ki < oi)
            keep_self = (1.0 - lower - upper) + lower * less + upper * (1.0 - less)
            kc = AbstractTensor.where(keep_self > 0.5, kc, oc)
            ki = AbstractTensor.where(keep_self > 0.5, ki, oi)
        # ---- the run of sorted cells covering each of the three rows of the 3x3 block
        t_emit, act, arming, live, pending, cx, cy, searching, gridded = frame(True)
        fx = x * 0.0
        fy = fx
        tq = fx
        blocked = fx
        ncont = fx                                            # the grains (and walls) this grain touches: its closed surface
        dropped = fx
        top = float(n - 1)
        runs = []
        for dy in (-1.0, 0.0, 1.0):
            row = cy + dy
            ok = as_float(row >= 0.0) * as_float(row <= rows - 1.0)
            rowc = _cap(row.clamp(0.0, 1.0e9), rows - 1.0)
            c_lo = rowc * cols + _cap((cx - 1.0).clamp(0.0, 1.0e9), cols - 1.0)
            c_hi = rowc * cols + _cap((cx + 1.0).clamp(0.0, 1.0e9), cols - 1.0)
            ends = []
            for target in (c_lo, c_hi + 1.0):
                lo = cx * 0.0
                hi = cx * 0.0 + float(n)
                for _ in range(search_steps):
                    mid = ((lo + hi) * 0.5).floor()
                    probe = kc.index_select(0, mid.clamp(0.0, top))
                    run_live = as_float(lo < hi)
                    below = as_float(probe < target)
                    lo = AbstractTensor.where(run_live * below > 0.5, mid + 1.0, lo)
                    hi = AbstractTensor.where(run_live * (1.0 - below) > 0.5, mid, hi)
                ends.append(lo)
            runs.append((ends[0], ends[1], ok))
            # candidates beyond the walk's reach are DROPPED contacts: counted (the law's dropped output), never silent
            dropped = dropped + (ends[1] - ends[0] - float(K_RUN)).clamp(0.0, 1.0e9) * ok * searching
        t_emit = AbstractTensor.where(pid < float(active), pid * emit_rate, pid * 0.0 + 1.0e9)
        # ---- the rest of what this id is, derived again HERE (not before the sort: a value that crosses
        # the sort's dispatches would hold an arena slot for all of them)
        x0, y0 = _nozzle(pid, bar_half, nozzle_y)
        theta = (_mix(pid, 1759.0, 17.0) * 2.0 - 1.0) * 0.7
        speed = 0.9 + _mix(pid, 2203.0, 29.0) * 0.7
        vx0 = speed * theta.sin()
        vy0 = speed * theta.cos()
        u = _mix(pid, 1237.0, 5.0)
        w0 = (1.0 - mix) * MONO_WEIGHT[0] + mix * POLY_WEIGHT[0]
        w1 = (1.0 - mix) * MONO_WEIGHT[1] + mix * POLY_WEIGHT[1]
        w2 = (1.0 - mix) * MONO_WEIGHT[2] + mix * POLY_WEIGHT[2]
        c0, c1, c2 = w0, w0 + w1, w0 + w1 + w2
        sp = as_float(u >= c0) + as_float(u >= c1) + as_float(u >= c2)
        is1, is2, is3 = (as_float((sp - float(k)).abs() < 0.5) for k in (1, 2, 3))
        # the species table: the default (compile-time) value times the runtime multiple (exactly 1 by default)
        radius0 = (SPECIES[0][1] + (SPECIES[1][1] - SPECIES[0][1]) * is1 + (SPECIES[2][1] - SPECIES[0][1]) * is2
                   + (SPECIES[3][1] - SPECIES[0][1]) * is3) * pick4(tab("rscale"), is1, is2, is3)
        density = (SPECIES[0][2] + (SPECIES[1][2] - SPECIES[0][2]) * is1 + (SPECIES[2][2] - SPECIES[0][2]) * is2
                   + (SPECIES[3][2] - SPECIES[0][2]) * is3) * pick4(tab("dscale"), is1, is2, is3)
        mu = (SPECIES[0][3] + (SPECIES[1][3] - SPECIES[0][3]) * is1 + (SPECIES[2][3] - SPECIES[0][3]) * is2
              + (SPECIES[3][3] - SPECIES[0][3]) * is3) * pick4(tab("fscale"), is1, is2, is3)
        zsc = pick4(tab("zscale"), is1, is2, is3)
        swell = pick4(tab("swell"), is1, is2, is3)
        swmax = pick4(tab("swmax"), is1, is2, is3)
        # the grain's CURRENT radius is state (it grows for a swelling species); an id not yet placed has its species radius
        radius = AbstractTensor.where(rs > 0.0, rs, radius0)
        drag = SPECIES[0][4] + (SPECIES[1][4] - SPECIES[0][4]) * is1 + (SPECIES[2][4] - SPECIES[0][4]) * is2 \
            + (SPECIES[3][4] - SPECIES[0][4]) * is3
        mass = density * math.pi * radius * radius

        # ---- particle-particle contacts: forces from the neighbours' state
        def neighbour_props(j):
            uj = _mix(j, 1237.0, 5.0)
            spj = as_float(uj >= c0) + as_float(uj >= c1) + as_float(uj >= c2)
            j1, j2, j3 = (as_float((spj - float(k)).abs() < 0.5) for k in (1, 2, 3))
            rj0 = (SPECIES[0][1] + (SPECIES[1][1] - SPECIES[0][1]) * j1 + (SPECIES[2][1] - SPECIES[0][1]) * j2
                   + (SPECIES[3][1] - SPECIES[0][1]) * j3) * pick4(tab("rscale"), j1, j2, j3)
            dj = (SPECIES[0][2] + (SPECIES[1][2] - SPECIES[0][2]) * j1 + (SPECIES[2][2] - SPECIES[0][2]) * j2
                  + (SPECIES[3][2] - SPECIES[0][2]) * j3) * pick4(tab("dscale"), j1, j2, j3)
            muj = (SPECIES[0][3] + (SPECIES[1][3] - SPECIES[0][3]) * j1 + (SPECIES[2][3] - SPECIES[0][3]) * j2
                   + (SPECIES[3][3] - SPECIES[0][3]) * j3) * pick4(tab("fscale"), j1, j2, j3)
            rsj = rs.index_select(0, j)
            rj = AbstractTensor.where(rsj > 0.0, rsj, rj0)           # the neighbour's current radius
            return rj, dj * math.pi * rj * rj, muj, pick4(tab("zscale"), j1, j2, j3)

        for lo, hi, ok in runs:
            for slot in range(K_RUN):
                idx = lo + float(slot)
                valid = as_float(idx < hi) * ok * searching
                j = ki.index_select(0, idx.clamp(0.0, top))
                valid = valid * as_float((j - pid).abs() > 0.5)
                armed_j = as_float(vy.index_select(0, j) < ARMED * 0.5)         # an armed neighbour is a slot, not a body
                xj = x.index_select(0, j)
                yj = y.index_select(0, j)
                vxj = vx.index_select(0, j)
                vyj = vy.index_select(0, j)
                wj = w.index_select(0, j)
                rj, mj, muj, zscj = neighbour_props(j)
                # coincident centres (two ids at one point) have no contact normal; they are pushed apart
                # along a direction fixed by the ids (antisymmetric, so equal and opposite) instead of
                # dividing a zero separation
                same = as_float((x - xj).abs() + (y - yj).abs() < 1.0e-9)
                side = as_float(pid > j) - as_float(pid < j)
                dx = x - xj + same * side * 1.0e-4
                dyy = y - yj
                dist = (dx * dx + dyy * dyy + 1.0e-12).sqrt()
                delta = (radius + rj - dist) * valid * (1.0 - armed_j)
                touch = as_float(delta > 0.0)
                # an armed id is emitted only into a CLEAR slot: no live neighbour, at the position it moves to this
                # step, within the contact distance (plus EMIT_MARGIN): the grain is placed overlapping nothing
                nxt_x = x - (xj + vxj * DT)
                nxt_y = y - (yj + vyj * DT)
                blocked = blocked + pending * valid * (1.0 - armed_j) * as_float(
                    radius + rj + EMIT_MARGIN - (nxt_x * nxt_x + nxt_y * nxt_y + 1.0e-12).sqrt() > 0.0)
                # ... and not into the slot of a LOWER armed id that is within contact distance of this slot (the lower
                # id goes first; both are stationary, so the slots' own distance decides)
                blocked = blocked + pending * valid * armed_j * as_float(j < pid) * as_float(radius + rj + EMIT_MARGIN - dist > 0.0)
                nx = dx / dist
                ny = dyy / dist
                m_eff = mass * mj / (mass + mj)
                kn = OMEGA * OMEGA * m_eff
                gx, gy, gt = _contact(delta, nx, ny, vx - vxj, vy - vyj, radius * w + rj * wj, kn, m_eff,
                                      (mu * muj).sqrt(), ZETA * (zsc * zscj).sqrt())
                ncont = ncont + touch
                fx = fx + gx * touch
                fy = fy + gy * touch
                tq = tq - radius * gt * touch
        # ---- walls and floor: the same contact against a body of infinite mass
        kw = OMEGA * OMEGA * mass
        floor_delta = (radius - (y - FLOOR_Y)).clamp(0.0, 1.0e9) * live
        gx, gy, gt = _contact(floor_delta, y * 0.0, y * 0.0 + 1.0, vx, vy, radius * w, kw, mass, mu, ZETA * zsc)
        touch = as_float(floor_delta > 0.0)
        ncont = ncont + touch
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        left_delta = (radius - (x + half)).clamp(0.0, 1.0e9) * live
        gx, gy, gt = _contact(left_delta, y * 0.0 + 1.0, y * 0.0, vx, vy, radius * w, kw, mass, mu, ZETA * zsc)
        touch = as_float(left_delta > 0.0)
        ncont = ncont + touch
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        right_delta = (radius - (half - x)).clamp(0.0, 1.0e9) * live
        gx, gy, gt = _contact(right_delta, y * 0.0 - 1.0, y * 0.0, vx, vy, radius * w, kw, mass, mu, ZETA * zsc)
        touch = as_float(right_delta > 0.0)
        ncont = ncont + touch
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        # the lid: the same contact against a ceiling at the runtime height ``lid`` (a body of infinite mass; far away unless the page closes it)
        lid_delta = (radius - (lid - y)).clamp(0.0, 1.0e9) * live
        gx, gy, gt = _contact(lid_delta, y * 0.0, y * 0.0 - 1.0, vx, vy, radius * w, kw, mass, mu, ZETA * zsc)
        touch = as_float(lid_delta > 0.0)
        ncont = ncont + touch
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        # ---- the integrator: gravity (the runtime vector ``grav_x, grav_y``), drag, the forces above.  A pending id with a clear slot is placed at the
        # nozzle with its launch velocity; one with a blocked slot waits, parked, for the next step.
        just = pending * as_float(blocked < 0.5)
        place = just + arming
        inertia = 0.5 * mass * radius * radius
        # The turning case is capped at what the walls can hold: the centrifugal load of a grain at the far wall (w^2 R) must be met
        # by the contact spring (OMEGA^2 x penetration), so w is limited to OMEGA sqrt(0.5 r / R) for the grain's radius r and the
        # chamber's reach R (its half-diagonal), and the angular acceleration to OMEGA^2 0.5 r / R (the Euler load); the planes
        # are sided, so a grain that does outrun the spring is still pushed back.
        half_height = 0.5 * rows * cell
        reach = (half * half + half_height * half_height).sqrt() + 1.0e-6
        w_cap = OMEGA * (0.5 * radius * (1.0 / reach)).sqrt()
        a_cap = OMEGA * OMEGA * 0.5 * radius * (1.0 / reach)
        case_omega = -_cap(-_cap(case_omega, w_cap), w_cap)
        case_domega = -_cap(-_cap(case_domega, a_cap), a_cap)
        rel_x, rel_y = x - a["pivot_x"], y - a["pivot_y"]
        fict_x = case_omega * case_omega * rel_x + case_domega * rel_y
        fict_y = case_omega * case_omega * rel_y - case_domega * rel_x
        vx_a = vx + live * (fx / mass + grav_x - drag * vx + fict_x) * DT
        vy_a = vy + live * (fy / mass + grav_y - drag * vy + fict_y) * DT
        turn = (case_omega * DT) * -2.0                               # Coriolis: the velocity turns by -2 w DT (clockwise for w > 0)
        c_turn, s_turn = turn.cos(), turn.sin()
        vx_c = AbstractTensor.where(live > 0.5, vx_a * c_turn - vy_a * s_turn, vx_a)      # only a live grain's velocity (an armed id's vy is a sentinel)
        vy_c = AbstractTensor.where(live > 0.5, vx_a * s_turn + vy_a * c_turn, vy_a)
        vxn = AbstractTensor.where(just > 0.5, vx0, vx_c)
        vyn = AbstractTensor.where(just > 0.5, vy0, AbstractTensor.where(arming > 0.5, vy * 0.0 + ARMED, vy_c))
        wn = (w + live * (tq / inertia - SPIN_DRAG * w - case_domega) * DT) * (1.0 - just)
        xn = AbstractTensor.where(place > 0.5, x0, x + live * vxn * DT)
        yn = AbstractTensor.where(place > 0.5, y0, y + live * vyn * DT)
        age_next = age + DT
        # ---- ORBEEZ: a swelling species takes up solvent through its OPEN surface.  A grain's surface is closed by what it
        # touches (the grains and walls in contact, ``FULL_COORDINATION`` closing it entirely), so its radius grows at
        # ``swell`` x its initial radius per second times the exposed fraction times the solvent present (0 in air), up to
        # ``swmax`` x the initial radius.  The mass follows the area (the density stays constant).  A pure function of the
        # state: deterministic.  With swell = 0 (the default) every term is an exact zero.
        exposure = (1.0 - ncont * (1.0 / FULL_COORDINATION)).clamp(0.0, 1.0)
        grown = _cap(rs + live * (swell * radius0 * exposure * solvent) * DT, radius0 * swmax)
        rsn = AbstractTensor.where(place > 0.5, radius0, grown)
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
        # mode 2: TIME STRATA -- hue turns ``strata`` times over the emission (a runtime uniform: the page's colour-rate
        # option), so the settled pile is layered by the order its grains arrived (ids are emitted in order)
        m2 = as_float(mode > 1.5)
        hue = hue + m2 * (pid * (strata * (1.0 / float(active))) - hue)
        m = as_float(mode > 0.5) * (1.0 - m2)
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
        # ---- CONSERVATION COUNTERS and ENERGY of the state just computed.  Every id is in exactly one class:
        # nan (not finite), parked (the zero state), outside (past a wall or under the floor) or live.
        finite = as_float(xn.abs() < 1.0e30) * as_float(yn.abs() < 1.0e30) * as_float(vxn.abs() < 1.0e30) \
            * as_float(vyn.abs() < 1.0e30) * as_float(wn.abs() < 1.0e30)
        placed = finite * as_float((xn.abs() + yn.abs()) >= 1.0e-30) * as_float(vyn >= ARMED * 0.5)
        outside = placed * (1.0 - (1.0 - as_float(xn.abs() > half + radius)) * (1.0 - as_float(yn < -radius)))
        c_nan = 1.0 - finite
        c_parked = finite * (1.0 - placed)
        c_out = outside
        c_live = placed - outside
        v2 = AbstractTensor.where(placed > 0.5, vxn * vxn + vyn * vyn, vxn * 0.0)
        ke = 0.5 * mass * v2 + AbstractTensor.where(placed > 0.5, 0.5 * inertia * wn * wn, vxn * 0.0)
        c_drop = dropped
        # ---- fixed-order reduction (Hillis-Steele): after the shifts 1, 2, 4 .. element 0 holds the total,
        # always as the same tree of additions, so native, wasm and WebGPU add in one order
        for shift in scan_shifts:
            gate = as_float(pid + float(shift) < float(n))
            partner = (pid + float(shift)).clamp(0.0, top_i)
            ke = ke + ke.index_select(0, partner) * gate
            c_nan = c_nan + c_nan.index_select(0, partner) * gate
            c_parked = c_parked + c_parked.index_select(0, partner) * gate
            c_out = c_out + c_out.index_select(0, partner) * gate
            c_live = c_live + c_live.index_select(0, partner) * gate
            c_drop = c_drop + c_drop.index_select(0, partner) * gate
            v2 = v2 + (v2.index_select(0, partner) * gate - v2).clamp(0.0, 1.0e30)
        return xn, yn, vxn, vyn, wn, age_next, rsn, cr, cg, cb, radius, ke, v2, c_nan, c_out, c_parked, c_live, c_drop

    return fountain_step


# ------------------------------------------------------------------- columns
def _f32(a) -> np.ndarray:
    """Rounded through float32: the page's buffers and the reference see one value."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def zeta_of(restitution: float) -> float:
    """Damping ratio of a spring-dashpot contact that returns ``restitution`` of the approach speed."""
    ln = math.log(min(max(float(restitution), 1.0e-4), 0.9999))
    return -ln / math.sqrt(math.pi ** 2 + ln * ln)


def restitution_of(zeta: float) -> float:
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


def species_params(params: dict | None = None) -> dict[str, float]:
    """The page's species table in its own units: ``r{s}`` radius, ``mu{s}`` friction (0-2), ``rho{s}`` density, ``e{s}``
    restitution, ``swell{s}`` rate and ``swmax{s}`` ratio, ``solvent`` -- defaults, then the overrides clamped to their ranges."""
    p: dict[str, float] = {"solvent": 0.0}
    for s, (_name, r, rho, mu, _drag) in enumerate(SPECIES):
        p.update({f"r{s}": r, f"rho{s}": rho, f"mu{s}": mu, f"e{s}": restitution_of(ZETA), f"swell{s}": 0.0, f"swmax{s}": 1.0})
    ranges = {"r": (RSCALE_RANGE[0], RSCALE_RANGE[1], True), "rho": DENSITY_RANGE + (False,), "mu": MU_RANGE + (False,),
              "e": RESTITUTION_RANGE + (False,), "swell": (0.0, SWELL_RATE_MAX, False), "swmax": (1.0, SWELL_MAX_RATIO, False)}
    for k, v in (params or {}).items():
        if k == "solvent":
            p[k] = min(max(float(v), 0.0), 1.0)
            continue
        key = k.rstrip("0123456789")
        if key in ranges and k[len(key):].isdigit() and int(k[len(key):]) < len(SPECIES):
            lo, hi, relative = ranges[key]
            base = SPECIES[int(k[len(key):])][1] if relative else 1.0
            p[k] = min(max(float(v), lo * base), hi * base)
    return p


def cell_needed(p: dict[str, float]) -> float:
    """The grid cell the species table needs: twice the largest radius any grain can reach (a swelling species grows to
    ``swmax`` x its radius)."""
    return 2.0 * max(p[f"r{s}"] * (p[f"swmax{s}"] if p[f"swell{s}"] > 0.0 else 1.0) for s in range(len(SPECIES)))


def uniforms(*, n: int, mode: int = 0, use_image: float = 0.0, mix: float = 0.5, aspect: float | None = None,
             gravity: tuple[float, float] = (0.0, -G), lid: float = LID_OFF, strata: float = STRATA,
             params: dict | None = None) -> dict[str, np.ndarray]:
    """The law's scalar columns (shape ()) as float32-representable values: the run (``mode``, ``use_image``, ``mix``),
    its basin for ``n`` active ids at the screen ``aspect`` (``None``: the reference basin) and the RUNTIME INPUTS
    (``gravity`` vector, ``lid`` height), whose defaults are the plain downward pull of ``G`` and no ceiling."""
    p = species_params(params)
    cell = cell_needed(p)
    geo = basin(n, aspect, cell)
    out = {"mode": mode, "use_image": use_image, "mix": mix}
    out.update({k: geo[k] for k in ("half", "nozzle_y", "bar_half", "cols", "rows")})
    out.update({"grav_x": gravity[0], "grav_y": gravity[1], "lid": lid, "strata": strata})
    out.update({"cell_scale": cell / CELL, "solvent": p["solvent"], "case_omega": 0.0, "case_domega": 0.0,
                "pivot_x": 0.5 * (geo["view"][0] + geo["view"][1]), "pivot_y": 0.5 * (geo["view"][2] + geo["view"][3])})
    for s, (_name, r, rho, mu, _drag) in enumerate(SPECIES):                 # multiples of the defaults: exactly 1.0 when untouched
        out.update({f"rscale{s}": p[f"r{s}"] / r, f"fscale{s}": p[f"mu{s}"] / mu if mu else 1.0, f"dscale{s}": p[f"rho{s}"] / rho,
                    f"zscale{s}": zeta_of(p[f"e{s}"]) / ZETA, f"swell{s}": p[f"swell{s}"], f"swmax{s}": p[f"swmax{s}"]})
    return {k: _f32(np.float64(v)) for k, v in out.items()}


def initial_columns(n: int, *, mix: float = 0.5, mode: int = 0, use_image: float = 0.0, active: int | None = None,
                    aspect: float | None = None, strata: float = STRATA, params: dict | None = None) -> dict[str, np.ndarray]:
    """Round zero for ``n`` ids: the state is zero, ``pid`` is the iota and the image colours are zero.
    Nothing else is per-id.  The basin is that of ``active`` ids (default ``n``) at ``aspect``."""
    zeros = np.zeros(n)
    columns = {name: zeros.copy() for name in STATE + ("ir", "ig", "ib")}
    columns["pid"] = np.arange(n, dtype=np.float64)
    columns.update(uniforms(n=int(active or n), mode=mode, use_image=use_image, mix=mix, aspect=aspect, strata=strata, params=params))
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


class SettleDetector:
    """The settle rule over a series of diagnostics sampled every ``SETTLE['sample']`` steps: ``push`` returns the step S
    that completes the hold, else ``None``."""

    def __init__(self, active: int, pad: int, params: dict | None = None):
        self.p, self.active, self.pad, self.peak, self.quiet = params or SETTLE, int(active), int(pad), 0.0, 0

    def push(self, step: int, ke: float, vmax2: float, n_parked: float) -> int | None:
        self.peak = max(self.peak, ke)
        emitted = n_parked <= self.pad - self.active
        quiet = (emitted and ke <= self.p["ke_per_id"] * self.active and ke <= self.p["relative"] * self.peak
                 and math.sqrt(vmax2) <= self.p["speed"])
        self.quiet = self.quiet + 1 if quiet else 0
        return step if self.quiet >= self.p["hold_samples"] else None


def settle_step(series: dict[str, np.ndarray], active: int, pad: int, params: dict | None = None) -> int | None:
    """S for a per-step series of the diagnostics (``series[name][k]`` after step ``k + 1``)."""
    p = params or SETTLE
    detector = SettleDetector(active, pad, p)
    for step in range(p["sample"], len(series["ke"]) + 1, p["sample"]):
        found = detector.push(step, float(series["ke"][step - 1]), float(series["vmax2"][step - 1]), float(series["n_parked"][step - 1]))
        if found is not None:
            return found
    return None


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
        extra = {name: produced[name] for name in ("cr", "cg", "cb", "rad") + DIAGNOSTICS}
    return {**state, **extra}


# -------------------------------------------------------------------- game
#: size tag -> (active ids, sort length).  ``small`` is the wasm demonstration; the rest are the
#: ladder the page offers (4096 .. 10M).  The sort length is the next power of two.
SIZES = {
    "small": (256, 256), "4k": (4096, 4096), "16k": (16384, 16384), "64k": (65536, 65536),
    "256k": (262144, 262144), "1M": (1048576, 1048576), "4M": (4194304, 4194304),
    "10M": (10_000_000, 10_000_000),
}
DEFAULT_SIZE = "4k"
N_FULL = SIZES[DEFAULT_SIZE][0]
N_SMALL = SIZES["small"][0]
#: The free board the game occupies in the main scene (board units); the world window of the
#: default size is mapped onto it, and the page maps the other sizes' windows the same way.
BOARD = board_size(basin(N_FULL)["view"])


class FountainGame(Game):
    """Server side of the fountain: phase, mix, size, the image hand-off and
    verification by the law's native product.  It does not stream particles:
    the main page's generic compute runtime runs the compiled law (WebGPU
    compute or the wasm product, as the compiler's verdict and the browser
    allow) and draws the particles in the shared scene.

    Actions (``/api/action``): ``fountain_start`` (phase 1; optional ``mix``,
    ``mode``), ``fountain_mode``, ``fountain_speed`` (0 follows the newest frame), ``fountain_run`` (``continuous`` or ``settle``), ``fountain_size``,
    ``fountain_image`` (per-id colours as base64 bytes ``rgb``, three per id,
    or just ``count`` for sizes too large to ship), ``fountain_replay``
    (phase 2) and ``fountain_verify`` (run the law ``steps`` steps from round
    zero natively, small sizes only, and report a digest)."""

    game = "fountain"
    input = "compute"
    title = "Fountain"

    def __init__(self, board_id: str = "", *, size: str = DEFAULT_SIZE, aspect: float | None = None):
        super().__init__(board_id)
        self.size = size
        self.mix, self.mode, self.speed = 0.5, 2, 1
        self.strata = STRATA
        self.params: dict[str, float] = {}       # the page's species-table overrides (see ``species_params``)
        self.run = "continuous"              # "continuous": never stops; "settle": stops at rest (the image replay needs it)
        self.aspect: float | None = aspect       # the screen's width / height when the run started; part of the recorded run
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
        cols = initial_columns(pad, mix=self.mix, mode=self.mode, use_image=1.0 if image else 0.0,
                               active=self.n, aspect=self.aspect, strata=self.strata, params=self.params)
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
            # a run's basin is fixed when it starts (and kept by its replay): a screen that rotates later changes nothing
            if payload.get("aspect"):
                self.aspect = float(payload["aspect"])
            self.reset()
        elif kind == "fountain_mode":
            self.mode = int(payload["mode"])
        elif kind == "fountain_strata":
            self.strata = float(payload["strata"])
        elif kind == "fountain_param":
            name = str(payload["name"])
            known = species_params()
            if name not in known:
                raise ValueError(f"fountain_param: unknown tunable {name!r}")
            self.params[name] = species_params({name: payload["value"]})[name]
            if name.rstrip("0123456789") == "r":          # the radius sets the grid: a new run (the others are runtime inputs)
                if payload.get("aspect"):
                    self.aspect = float(payload["aspect"])
                self.reset()
        elif kind == "fountain_choice":
            name, value = str(payload["name"]), str(payload["value"])
            if name != "solvent" or value not in SOLVENTS:
                raise ValueError(f"fountain_choice: unknown choice {name!r} = {value!r}")
            self.params["solvent"] = SOLVENTS[value]
        elif kind == "fountain_speed":
            self.speed = int(payload["speed"])
        elif kind == "fountain_run":
            run = str(payload["run"])
            if run not in ("continuous", "settle"):
                raise ValueError(f"fountain_run: unknown run {run!r}")
            self.run = run
            self.reset()
        elif kind == "fountain_size":
            tag = str(payload["size"])
            if tag not in SIZES:
                raise ValueError(f"fountain_size: unknown size {tag!r}")
            self.size = tag
            if payload.get("aspect"):
                self.aspect = float(payload["aspect"])
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

    def solvent_name(self) -> str:
        v = species_params(self.params)["solvent"]
        return next((k for k, x in SOLVENTS.items() if abs(x - v) < 1e-9), "custom")

    def _species_controls(self) -> list[dict[str, Any]]:
        """The species table as page controls: grouped, every one a ``fountain_param`` range.  The radius is a log slider
        (0.1x to 10x the default; a restart, since it sets the grid), friction 0 (frictionless) to 2 (very sticky)."""
        p = species_params(self.params)
        out: list[dict[str, Any]] = []
        for s, (name, r, rho, mu, _drag) in enumerate(SPECIES):
            def ctl(key, label, lo, hi, step, extra=None):
                spec = {"id": f"{key}{s}", "kind": "range", "label": f"{name} {label}", "min": lo, "max": hi, "step": step,
                        "value": p[f"{key}{s}"], "action": "fountain_param", "name": f"{key}{s}", "field": "value", "group": "Grains",
                        "species": s}
                spec.update(extra or {})
                return spec
            out.append(ctl("r", "radius", r * RSCALE_RANGE[0], r * RSCALE_RANGE[1], r / 20.0,
                           {"scale": "log", "log_min": r * RSCALE_RANGE[0], "log_floor": True, "aspect": True}))
            out.append(ctl("rho", "density", DENSITY_RANGE[0], DENSITY_RANGE[1], 0.1))
            out.append(ctl("mu", "friction", MU_RANGE[0], MU_RANGE[1], 0.05))
            out.append(ctl("e", "restitution", RESTITUTION_RANGE[0], RESTITUTION_RANGE[1], 0.01))
            out.append(ctl("swell", "swell rate (orbeez, 1/s)", 0.0, SWELL_RATE_MAX, 0.01))
            out.append(ctl("swmax", "max swell (x radius)", 1.0, SWELL_MAX_RATIO, 0.1))
        out.append({"id": "solvent", "kind": "select", "label": "solvent", "value": self.solvent_name(), "action": "fountain_choice",
                    "name": "solvent", "field": "value", "group": "Grains",
                    "options": [[k, k] for k in SOLVENTS] + ([["custom", "custom"]] if self.solvent_name() == "custom" else [])})
        return out

    def controls_ui(self) -> list[dict[str, Any]]:
        sample = {"fit": "extent", "read": ["x", "y"], "into": ["ir", "ig", "ib"], "raster": [96, 72],
                  "post": {"action": "fountain_image", "field": "rgb", "max_ids": 65536}}
        sizes = [[tag, f"{count:,}"] for tag, (count, _pad) in SIZES.items()]
        return [
            {"id": "start", "kind": "button", "label": "Start", "action": "fountain_start", "aspect": True},
            {"id": "image", "kind": "file", "label": "Image", "sample": sample},
            {"id": "camera", "kind": "file", "label": "Camera", "capture": "environment", "sample": sample},
            {"id": "replay", "kind": "button", "label": "Image replay", "action": "fountain_replay",
             "enabled": self.has_image},
            {"id": "run", "kind": "select", "label": "run", "value": self.run, "action": "fountain_run",
             "field": "run", "options": [["continuous", "continuous"], ["settle", "stop at rest"]]},
            # the run modes are the page's own (``local``): the compute runtime's recording, not a server action
            {"id": "rebirth", "kind": "button", "label": "Rebirth", "local": "rebirth"},
            {"id": "rec_replay", "kind": "button", "label": "Replay", "local": "replay"},
            {"id": "bounce", "kind": "button", "label": "Bounce", "local": "bounce"},
            {"id": "continue", "kind": "button", "label": "Continue", "local": "continue"},
            {"id": "tilt", "kind": "toggle", "label": "phone gravity", "local": "tilt", "value": True},
            # the particle field's render look is the page's own too (web/particle_styles.js STYLES: same ids, same order); render only
            {"id": "pstyle", "kind": "select", "label": "look", "local": "particle_style", "value": "points",
             "options": [[tag, label] for tag, label in PARTICLE_STYLES]},
            # the spectral noise filter over the look (web/noise_filter.js): speckle becomes a solid colour; also the page's own, render only
            {"id": "nf", "kind": "toggle", "label": "noise filter", "local": "noise_on", "value": False},
            {"id": "nfmode", "kind": "select", "label": "replace with", "local": "noise_mode", "value": "mean",
             "options": [[tag, label] for tag, label in NOISE_MODES]},
            {"id": "nfstrength", "kind": "range", "label": "strength", "min": 0, "max": 1, "step": 0.05, "local": "noise_strength", "value": 0.85},
            {"id": "nfthr", "kind": "range", "label": "threshold", "min": 0.2, "max": 0.9, "step": 0.05, "local": "noise_threshold", "value": 0.5},
            {"id": "mix", "kind": "range", "label": "poly mix", "min": 0, "max": 1, "step": 0.05,
             "value": self.mix, "action": "fountain_start", "field": "mix", "aspect": True},
            {"id": "mode", "kind": "select", "label": "colour", "value": self.mode, "action": "fountain_mode",
             "field": "mode", "options": [[2, "time strata"], [0, "trig (atan2)"], [1, "rotation"]]},
            {"id": "strata", "kind": "select", "label": "colour rate", "value": self.strata, "action": "fountain_strata",
             "field": "strata", "options": [[s, f"{s:g} cycle" + ("" if s == 1 else "s")] for s in STRATA_CHOICES]},
            {"id": "size", "kind": "select", "label": "ids", "value": self.size, "action": "fountain_size",
             "field": "size", "options": sizes, "aspect": True},
            {"id": "speed", "kind": "select", "label": "speed", "value": self.speed, "action": "fountain_speed",
             "field": "speed", "options": [[1, "1x"], [2, "2x"], [4, "4x"], [0, "max"]]},
            {"id": "verify", "kind": "button", "label": "Verify", "action": "fountain_verify"},
        ] + self._species_controls()

    def basin(self) -> dict[str, Any]:
        """The running basin: ``n`` ids at the aspect the run started with."""
        return basin(self.n, self.aspect, cell_needed(species_params(self.params)))

    def state(self) -> dict[str, Any]:
        geo = self.basin()
        width, height = board_size(geo["view"])
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
            "camera_gestures": {"orbit": "one", "zoom": True, "pan": True, "tilt": True},   # camera-first: the page's scene-camera layer
            "compute": {
                "manifest": "fountain.manifest.json", "size": self.size, "epoch": self.epoch,
                "uniforms": {k: float(v) for k, v in uniforms(n=self.n, mode=self.mode, mix=self.mix, aspect=self.aspect, strata=self.strata,
                                                              use_image=1.0 if self.phase == "replay" else 0.0, params=self.params).items()},
                "view": list(geo["view"]), "aspect": self.aspect,
                "speed": self.speed, "named": ["ir", "ig", "ib"], "run": self.run,
                "tilt": tilt_inputs(self.n, self.aspect, cell_needed(species_params(self.params))),
            },
        })
        return base

    def result(self):
        return None
