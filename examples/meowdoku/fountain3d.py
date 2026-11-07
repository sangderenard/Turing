"""Fountain: a soft-sphere (DEM) granular fountain in a 3-D slab chamber that fills a basin, takes an image, fills it again.

ONE batched per-step law (``fountain_step``), authored once over spans of
particle ids in the repository's AbstractTensor style, compiled by the
repository to native, wasm and (generated) WebGPU compute.

NO PER-ID TABLES.  Everything an id is -- its emit time, where it leaves the
inlet, its launch velocity, its species and so radius, mass and friction -- is
DERIVED IN THE LAW from the id itself by an integer hash done in exact float
arithmetic (every intermediate is a whole number below 2**24, so f64 and f32
agree to the bit and native, wasm and WGSL cannot disagree).  The only per-id
columns are the state (x y z vx vy vz wx wy wz age), the iota ``pid`` (which
the generated shader reads from its own invocation index, never from memory)
and the image colours ``ir ig ib``.

THE CHAMBER (3-D)
-----------------
A SLAB between two plexiglass walls ``z = -depth/2`` and ``z = +depth/2`` (``dh`` = depth/2, a runtime uniform; a thin slab,
about one grain diameter, is the 2-D fountain of before).  The basin outline in the x-y plane is a POLYLINE of up to
``NV`` vertices (uniforms ``px0..py7``; unused vertices repeat the last one), extruded through the depth: each segment is a
wall that is a contact of the same law as a grain (a body of infinite mass).  The default is the old basin (a floor and two
side walls, open top).  The INLET is a tube: a source line/rectangle of half-widths ``bar_half`` x ``bar_z`` about a runtime
position, perpendicular to a runtime direction ``(in_dx, in_dy, in_dz)``, launching grains along it within a cone of
half-angle ``spread`` (0 = a colinear stream).  GRAVITY is a full 3-vector runtime input.

PHYSICS
-------
* SPHERES of four SPECIES, each with a radius, density (mass = rho 4/3 pi r^3), sliding friction, restitution (the
  contact's damping ratio), ROLLING resistance (roughness) and shape IRREGULARITY.  Irregularity is modelled as a
  deterministic per-id radius jitter: grain ``k`` of a species has radius ``r (1 + irr h(k))`` with ``h`` in [-1, 1) hashed
  from its id, so a species is a population of different sizes.  (Rolling resistance is the separate roughness term.)
* Contacts are SOFT SPHERES: a normal spring-dashpot (stiffness ``omega^2`` times the pair's reduced mass), blended
  from linear toward HERTZ (force ~ overlap^1.5) by ``hertz``, with an adhesion (COHESION ``coh`` weights, within a
  short range), tangential Coulomb friction regularised with a tanh of the slip speed, acting on the slip of the contact
  point, spin included, so spheres roll, and a constant-torque rolling resistance on the relative spin.
* An INTERSTITIAL FLUID of density ``fl_rho`` and viscosity ``fl_mu`` acts on every grain: buoyancy (gravity times
  1 - fl_rho / rho), Stokes drag (6 pi mu r v) and quadratic drag, integrated IMPLICITLY so a stiff setting damps and
  never explodes.  Air, water and glycerine are presets of the page.
* TEMPERATURE: a Langevin thermostat.  ``temp`` (kT, a runtime uniform; 0 = off, exactly) adds to every live grain a
  random force and torque whose variance follows from the drag it feels (fluctuation-dissipation: 2 gamma m kT / DT), so
  small, light grains in a viscous fluid jitter most, like dust in water.  The noise is a hash of (id, step, component).
  The measured temperature is ``ke / (3 n_live)`` of the diagnostics (translational + rotational energy over the six
  degrees of freedom: each holds kT / 2).
* Contacts WRITE FORCES (and torques); the integrator (semi-implicit Euler, fixed ``DT``) is the only thing that moves
  anything.  Both sides of a pair evaluate the same symmetric expressions, so they are equal and opposite without any
  per-pair bookkeeping.
* STABILITY: the explicit contact spring needs ``omega * DT * sqrt(contacts)`` well under 2.  ``uniforms`` clamps the
  stiffness to ``OMEGA_MAX / sqrt(1 + 2 hertz)`` (Hertz stiffens the spring as the overlap grows); damping and drag are
  implicit; the friction and rolling forces are limited to a share of the slip per step.
* The inlet EMITS: id ``k`` leaves at ``k * emit_rate`` (runtime), armed at its slot until no live grain is within contact
  distance of it (see ``fountain_step``), so a grain never spawns inside another.
* DIAGNOSTICS: the law reduces, by a fixed tree of additions, the kinetic + rotational energy, the largest speed, and the
  number of ids that are not finite / outside the chamber / parked / live, plus the contacts the neighbour walk dropped.

NEIGHBOUR SEARCH (in the law, from tensor ops; deterministic)
-------------------------------------------------------------
A uniform grid of columns of ``cell`` (the largest diameter) in x and y, running through the WHOLE depth: the slab is
thin, so a column's z cells are few and the 27-cell neighbourhood is the three rows of 3 columns.  Each step the
particles are sorted by the pair ``(cell, id)`` with Knuth's merge exchange; each of the three rows is a run of the
sorted cells found by two binary searches; the contact loop walks the first ``k_run`` (a RUNTIME uniform up to
``K_RUN_MAX``: the loop is fixed-trip, the cap masks the candidates past it, so a lower cap saves no time on the device
-- it trades accuracy for nothing but the dropped-contact count; the dropped contacts are counted in ``n_drop``).

COLOUR
------
Phase 1: hue from the ANGLE of the id's launch vector (mode 0 atan2, mode 1 rotation) or from the id's emission order
(mode 2, time strata, ``strata`` cycles).  The image step samples the picked image at every id's rest position into per-id
colours; phase 2 replays the identical simulation with ``use_image = 1``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from meowdoku.entities import LawPiece, as_float
from src.common.tensors import AbstractTensor
from src.common.tensors import numpy_backend  # noqa: F401  (registers the NumPy backend the eager law runs on)

# ----------------------------------------------------------------- the world
N_BASE = 4096                          # the basin's reference count (its width is WIDTH_BASE)
WIDTH_BASE = 2.0
FLOOR_Y = 0.0
EMIT_SPAN = 6.0                        # seconds over which ids are emitted (at nozzle rate 1)
#: The pile is AT REST (phase 1 ends, at step S; phase 2 stops at exactly S) when everything is emitted and the law's
#: kinetic + rotational energy has stayed under BOTH ``ke_per_id`` per id and ``relative`` of its peak, with the largest speed
#: under ``speed``, for ``hold_samples`` consecutive samples taken every ``sample`` steps.  ``max_seconds`` caps a pile that
#: never quiets.  ``web/compute.js`` ``SettleDetector`` is this rule.  (``ke_per_id`` is per SPHERE: a sphere's mass is ~80x
#: less than the disk it replaced, so the floor is 80x lower.)
SETTLE = {"sample": 30, "hold_samples": 12, "relative": 0.01, "ke_per_id": 1.0e-9, "speed": 0.25, "max_seconds": 40.0}
DT = 1.0 / 360.0
G = 1.0
OMEGA = 150.0                          # default contact stiffness: kn = m_eff * omega^2
OMEGA_MAX = 200.0                      # the CFL bound at one contact blend: omega * DT * sqrt(contacts) stays under 2
ZETA = 0.7                             # default normal damping ratio (restitution ~0.046)
VREF = 0.02                            # tangential slip speed over which friction saturates
WREF = 2.0                             # relative spin (rad/s) over which rolling resistance saturates
SPIN_DRAG = 3.0                        # 1/s
TEMP_MAX = 3.0e-6                      # the hottest kT the page offers (a grain of mass 3e-6 then moves at ~1.7 rms; the slider is logarithmic)
TEMP_MIN_LOG = 1.0e-10                 # the coldest non-zero kT on the page's logarithmic slider
HERTZ_REF = 0.04                       # the overlap (in mean radii) at which the Hertz and linear springs agree
COH_RANGE = 0.1                        # adhesion acts to this gap (in mean radii)
STRATA = 1.0                           # default hue cycles over the emission in mode 2 (time strata); a runtime uniform ``strata``
# the page's render looks for the particle field (web/particle_styles.js STYLES -- a test holds the two lists equal)
PARTICLE_STYLES = (("points", "points"), ("liquid", "liquid"), ("sand", "sand"), ("glow", "glow"), ("heat", "heat"),
                   ("stamp", "stamp (memory)"))
K_RUN_MAX = 24                         # candidates walked per row run: the compile-time maximum of the runtime uniform ``k_run``
K_RUN = 24                             # its default (a hex pack of the smallest grains in a thin slab is ~9-15 per run)
ARMED = -1.0e9                         # the vy of an id waiting at its inlet slot (see ``fountain_step``)
EMIT_MARGIN = 1.0e-3                   # an id is emitted only where no live grain, at its NEXT-step position, is within contact distance + this
#: The share of a contact's slip that ONE step's friction may remove (see the 2-D history of this law): friction is limited to
#: ``FRICTION_SHARE`` of the slip per step, in the contact's own slip mass (m_eff / 3.5 for a sphere's slip, spin included).
FRICTION_SHARE = 0.15
BAR_HEIGHT = 0.12                      # the source's extent ALONG the stream
NV = 8                                 # polygon vertices (uniforms px0..px7, py0..py7)
DEPTH_MAX = 0.05                       # the deepest slab: ~2 grain layers, the most the neighbour walk covers (``n_drop`` says when not)
DEPTH_DEFAULT = 0.03                   # thin: a bit more than the widest diameter, so the pile is the 2-D one
LID_OFF = 1.0e9
LID_ABOVE = 1.45                       # the lid the page closes while a sensor drives gravity: this far above the nozzle (the jet's apex is ~1.34)

#: species: name, radius, density, sliding friction, rolling resistance, irregularity (the DEFAULTS of the page's table)
SPECIES = (
    ("grain", 0.0090, 1.0, 0.35, 0.0, 0.0),
    ("bead",  0.0105, 1.6, 0.20, 0.0, 0.0),
    ("dust",  0.0120, 0.7, 0.55, 0.0, 0.0),
    ("pearl", 0.0140, 2.2, 0.10, 0.0, 0.0),
)
MONO_WEIGHT = (0.6, 0.4, 0.0, 0.0)       # mix 0: near-monodisperse grains and beads (a crystal wants to form)
POLY_WEIGHT = (0.15, 0.15, 0.35, 0.35)   # mix 1: all four species, widely polydisperse
CELL = 2.0 * max(s[1] for s in SPECIES)  # the reference grid cell; the running one is ``cell_of(params)``

#: interstitial fluids: name -> (density relative to a grain of density 1, viscosity in the sim's units).  Sim units: grain
#: density 1, g = 1, grain radius ~0.01.  Stokes damping rate of the grain = 4.5 mu / (rho r^2) (1/s): air 0.1, water 1.5,
#: glycerine 11 (a grain falls ~0.05 per second in it).
FLUIDS = {"air": (0.0005, 3.0e-6), "water": (0.38, 2.5e-5), "glycerine": (0.48, 2.0e-4)}
DEFAULT_FLUID = "air"

SHAPES = ("box", "funnel", "hexagon", "ramp")
INLETS = ("fountain", "jet", "pour", "side")


def zeta_of(restitution: float) -> float:
    """Damping ratio of a spring-dashpot contact that returns ``restitution`` of the approach speed."""
    ln = math.log(min(max(float(restitution), 1.0e-4), 0.9999))
    return -ln / math.sqrt(math.pi ** 2 + ln * ln)


def restitution_of(zeta: float) -> float:
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


#: The tunables the page sets (no rebuild).  name -> (label, lo, hi, step, default, restart, group).  ``restart`` ones change
#: what an id IS at emission or the grid itself (radius, irregularity, mix weights, the chamber, the inlet, the rate): a
#: new run.  The others are RUNTIME INPUTS of the law, applied at a step boundary and logged with their step.
def _params() -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for s, (name, r, rho, mu, roll, irr) in enumerate(SPECIES):
        out[f"r{s}"] = (f"{name} radius", 0.005, 0.020, 0.0005, r, True, "Grains")
        out[f"rho{s}"] = (f"{name} density", 0.2, 5.0, 0.1, rho, False, "Grains")
        out[f"mu{s}"] = (f"{name} friction", 0.0, 1.0, 0.05, mu, False, "Grains")
        out[f"e{s}"] = (f"{name} restitution", 0.02, 0.95, 0.01, restitution_of(ZETA), False, "Grains")
        out[f"roll{s}"] = (f"{name} roughness", 0.0, 0.3, 0.01, roll, False, "Grains")
        out[f"irr{s}"] = (f"{name} irregularity", 0.0, 0.3, 0.01, irr, True, "Grains")
        out[f"wt{s}"] = (f"{name} share", 0.0, 1.0, 0.05, MONO_WEIGHT[s], True, "Grains")
    out.update({
        "omega": ("stiffness (compressibility)", 40.0, OMEGA_MAX, 5.0, OMEGA, False, "Physics"),
        "hertz": ("Hertz blend", 0.0, 1.0, 0.05, 0.0, False, "Physics"),
        "coh": ("cohesion", 0.0, 3.0, 0.05, 0.0, False, "Physics"),
        "spin_drag": ("spin drag", 0.0, 20.0, 0.5, SPIN_DRAG, False, "Physics"),
        "temp": ("temperature kT", 0.0, TEMP_MAX, 1.0e-9, 0.0, False, "Physics"),
        "k_run": ("neighbour cap", 4.0, float(K_RUN_MAX), 1.0, float(K_RUN), False, "Physics"),
        "nozzle_speed": ("nozzle speed", 0.3, 2.0, 0.05, 1.0, False, "Physics"),
        "rate": ("nozzle rate", 0.25, 4.0, 0.05, 1.0, True, "Physics"),
        "fl_rho": ("fluid density", 0.0, 1.2, 0.01, FLUIDS[DEFAULT_FLUID][0], False, "Fluid"),
        "fl_mu": ("fluid viscosity", 0.0, 5.0e-4, 1.0e-5, FLUIDS[DEFAULT_FLUID][1], False, "Fluid"),
        "lognormal": ("size distribution (0 uniform, 1 lognormal)", 0.0, 1.0, 1.0, 0.0, True, "Grains"),
        "bi_ratio": ("bidisperse size ratio (large / small)", 1.0, 3.0, 0.05, 1.0, True, "Grains"),
        "bi_frac": ("bidisperse number fraction large", 0.0, 1.0, 0.05, 0.5, True, "Grains"),
        "depth": ("depth", 0.02, DEPTH_MAX, 0.002, DEPTH_DEFAULT, True, "Chamber"),
        "spread": ("inlet spread (rad)", 0.0, 0.9, 0.05, 0.7, True, "Chamber"),
    })
    return out


PARAMS = _params()


def resolve_params(mix: float = 0.5, params: dict | None = None) -> dict[str, float]:
    """Every tunable's value: the defaults, the mix weights from ``mix`` (poly mix is the macro that sets the four shares),
    then the caller's overrides, each clamped to its range."""
    p = {k: float(v[4]) for k, v in PARAMS.items()}
    for s in range(len(SPECIES)):
        p[f"wt{s}"] = (1.0 - mix) * MONO_WEIGHT[s] + mix * POLY_WEIGHT[s]
    for k, v in (params or {}).items():
        if k in PARAMS:
            p[k] = min(max(float(v), PARAMS[k][1]), PARAMS[k][2])
    if sum(p[f"wt{s}"] for s in range(len(SPECIES))) < 1.0e-6:
        p["wt0"] = 1.0
    return p


def size_factor_max(p: dict[str, float], s: int) -> float:
    """The largest multiple of species ``s``'s mean radius any of its grains can have: its uniform width, or the lognormal's
    tail (the law clamps the normal at 3 sigma), times the bidisperse mode's large-size factor."""
    spread = p[f"irr{s}"]
    wide = math.exp(3.0 * spread) if p["lognormal"] > 0.5 else 1.0 + spread
    return wide * math.sqrt(max(p["bi_ratio"], 1.0))


def rmax_of(p: dict[str, float]) -> float:
    return max(p[f"r{s}"] * size_factor_max(p, s) for s in range(len(SPECIES)))


def cell_of(p: dict[str, float]) -> float:
    """The grid cell: the largest diameter any grain can have (species radius with its size spread and bidisperse mode)."""
    return 2.0 * rmax_of(p)


def stiffness_of(p: dict[str, float]) -> float:
    """The contact stiffness the law is given: the page's ``omega`` clamped to the CFL bound (Hertz stiffens the spring up
    to 1 + 2 hertz times its linear tangent, so the bound drops with it)."""
    return min(p["omega"], OMEGA_MAX / math.sqrt(1.0 + 2.0 * p["hertz"]))


STATE = ("x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "age")
CONSTANTS = ("pid", "ir", "ig", "ib")
_SPECIES_UNIFORMS = tuple(f"{k}{s}" for k in ("r", "rho", "mu", "zeta", "roll", "irr") for s in range(len(SPECIES)))
POLYGON_UNIFORMS = tuple(f"{c}{k}" for c in ("px", "py") for k in range(NV))
UNIFORMS = (("mode", "use_image", "half", "nozzle_y", "bar_half", "cols", "rows", "cell", "dh", "bar_z", "k_run",
             "nozzle_x", "nozzle_z", "in_dx", "in_dy", "in_dz", "e1x", "e1y", "e1z", "e2x", "e2y", "e2z",
             "spread", "nozzle_speed", "emit_rate")
            + POLYGON_UNIFORMS
            + ("grav_x", "grav_y", "grav_z", "lid", "strata", "omega", "hertz", "coh", "spin_drag", "temp", "fl_rho", "fl_mu", "lognormal", "bi_ratio", "bi_frac",
               "wt0", "wt1", "wt2", "wt3", "case_omega", "case_domega", "pivot_x", "pivot_y")
            + _SPECIES_UNIFORMS)
#: The uniforms the PAGE drives moment by moment (no rebuild, no reset): the gravity vector (the phone's tilt), ``lid`` and
#: ``rows`` (a ceiling and the grid's height, which the page closes while a sensor drives gravity), and the live tunables.  The
#: server sends their defaults; the page's runtime owns them after that and LOGS each change with its step, so a re-run
#: (Continue) reproduces it exactly.
LIVE_PARAMS = tuple(k for k, v in PARAMS.items() if not v[5])
RUNTIME_INPUTS = ("grav_x", "grav_y", "grav_z", "lid", "rows", "strata", "case_omega", "case_domega", "omega", "hertz", "coh", "spin_drag", "temp", "k_run",
                  "nozzle_speed", "fl_rho", "fl_mu") \
    + tuple(f"{k}{s}" for k in ("rho", "mu", "zeta", "roll") for s in range(len(SPECIES)))
IOTA = ("pid",)                        # columns the generated shader derives from its invocation index
#: ``ke``..``n_drop`` are the law's DIAGNOSTICS: reduced by a fixed-order tree, so element 0 holds the total over all ids
#: (kinetic + rotational energy, the largest speed squared, and the counts of the ids that are not finite, outside the
#: chamber, parked, live, plus the contacts the neighbour walk dropped).  The four classes partition the ids.
DIAGNOSTICS = ("ke", "vmax2", "n_nan", "n_out", "n_parked", "n_live", "n_drop")
OUTPUTS = tuple(f"{s}_next" for s in STATE) + ("cr", "cg", "cb", "rad", "shown") + DIAGNOSTICS
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
    ``cell`` is the grid cell (``cell_of``).  Every number the law needs is a RUNTIME uniform (see ``uniforms``)."""
    s = math.sqrt(n / N_BASE)
    k = 1.0 if not aspect else fit_scale(s, aspect)
    width = WIDTH_BASE * s * k
    nozzle_y = 0.6 * s / k + 0.7
    cols = int(math.ceil(width / cell))
    rows = int(math.ceil((nozzle_y + 0.1) / cell))
    return {"scale": s, "stretch": k, "width": width, "half": width / 2.0, "nozzle_y": nozzle_y, "bar_half": 0.8 * s * k,
            "cols": cols, "rows": rows, "cells": cols * rows, "cell": cell,
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


def shape_vertices(shape: str, geo: dict[str, float]) -> list[tuple[float, float]]:
    """The basin outline as a polyline (at most ``NV`` vertices; the law pads it by repeating the last), the chamber on its
    LEFT as it runs (counterclockwise).  ``box`` is the old basin: a floor and two side walls, open top, tall (it keeps the
    splash).  Every other shape is a CLOSED polygon (its first vertex repeated, so its top is its own ceiling) INSCRIBED in the
    visible sim window in BOTH axes, centred: |x| <= the window's half-width less a margin of three grain diameters, from the
    floor (y = 0) to the window's top less that margin, as large as fits.  ``hexagon``; ``funnel`` a vessel with a V bottom
    that narrows to its throat; ``ramp`` a vessel with a sloping floor."""
    h, ny = geo["half"], geo["nozzle_y"]
    if shape == "box":
        top = ny + 2.5
        return [(-h, top), (-h, 0.0), (h, 0.0), (h, top)]
    margin = 3.0 * geo.get("cell", CELL)
    w = 0.5 * (geo["view"][1] - geo["view"][0]) - margin          # the window is centred on x = 0
    top = geo["view"][3] - margin
    mid = 0.5 * top
    if shape == "funnel":
        return [(-w, top), (-w, 0.55 * top), (-0.2 * w, 0.0), (0.2 * w, 0.0), (w, 0.55 * top), (w, top), (-w, top)]
    if shape == "hexagon":
        return [(-w, mid), (-0.5 * w, 0.0), (0.5 * w, 0.0), (w, mid), (0.5 * w, top), (-0.5 * w, top), (-w, mid)]
    if shape == "ramp":
        return [(-w, top), (-w, 0.3 * top), (w, 0.0), (w, top), (-w, top)]
    raise ValueError(f"unknown chamber shape {shape!r}")


def chord(verts: list[tuple[float, float]], y: float) -> tuple[float, float]:
    """The x extent of the chamber at height ``y``: the leftmost and rightmost crossings of the outline."""
    xs = []
    for (ax, ay), (bx, by) in zip(verts[:-1], verts[1:]):
        if (ay - y) * (by - y) <= 0.0 and ay != by:
            xs.append(ax + (bx - ax) * (y - ay) / (by - ay))
    return (min(xs), max(xs)) if xs else (-1.0e9, 1.0e9)


def inlet_spec(inlet: str, geo: dict[str, float], p: dict[str, float], verts: list[tuple[float, float]] | None = None) -> dict[str, float]:
    """The inlet (a tube: position, direction, cross-section) for a preset, as the law's uniforms.  ``fountain`` is the old
    bar with a fan; the others are colinear tubes (``spread`` 0 unless the page raised it)."""
    h, ny = geo["half"], geo["nozzle_y"]
    if inlet == "fountain":
        pos, d, half_w = (0.0, ny, 0.0), (0.0, 1.0, 0.0), geo["bar_half"]
    elif inlet == "jet":
        pos, d, half_w = (0.0, 0.25 * ny, 0.0), (0.0, 1.0, 0.0), 0.04
    elif inlet == "pour":
        pos, d, half_w = (0.0, ny, 0.0), (0.0, -1.0, 0.0), 0.04
    elif inlet == "side":
        pos, d, half_w = (-h + 0.06, 0.55 * ny, 0.0), (1.0, 0.25, 0.0), 0.04
    else:
        raise ValueError(f"unknown inlet {inlet!r}")
    if verts is not None:
        # INSIDE the shape: the source's centre goes to within the chamber's chord at its height (and below the top), and the
        # bar is narrowed to 80% of the chord's half-width there
        margin = 3.0 * geo.get("cell", CELL)
        top = max(v[1] for v in verts) - margin
        y = min(pos[1], top)
        xl, xr = chord(verts, y)
        cx = min(max(pos[0], xl + margin), xr - margin) if xr - xl > 2 * margin else 0.5 * (xl + xr)
        half_w = min(half_w, 0.8 * 0.5 * (xr - xl))
        pos = (cx, y, pos[2])
    norm = math.sqrt(sum(c * c for c in d))
    d = tuple(c / norm for c in d)
    ref = (0.0, 0.0, 1.0) if abs(d[2]) <= 0.9 else (1.0, 0.0, 0.0)          # e1 = d x ref, e2 = d x e1
    e1 = (d[1] * ref[2] - d[2] * ref[1], d[2] * ref[0] - d[0] * ref[2], d[0] * ref[1] - d[1] * ref[0])
    n1 = math.sqrt(sum(c * c for c in e1))
    e1 = tuple(c / n1 for c in e1)
    e2 = (d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0])
    return {"nozzle_x": pos[0], "nozzle_y": pos[1], "nozzle_z": pos[2], "bar_half": half_w,
            "in_dx": d[0], "in_dy": d[1], "in_dz": d[2], "e1x": e1[0], "e1y": e1[1], "e1z": e1[2],
            "e2x": e2[0], "e2y": e2[1], "e2z": e2[2]}


def inlet_default_spread(inlet: str) -> float:
    return PARAMS["spread"][4] if inlet == "fountain" else 0.0


def uniforms(*, n: int, mode: int = 0, use_image: float = 0.0, mix: float = 0.5, aspect: float | None = None,
             gravity: tuple[float, float, float] = (0.0, -G, 0.0), lid: float = LID_OFF, strata: float = STRATA,
             params: dict | None = None, shape: str = "box", inlet: str = "fountain") -> dict[str, np.ndarray]:
    """The law's scalar columns (shape ()) as float32-representable values: the run (``mode``, ``use_image``), the chamber
    for ``n`` active ids at the screen ``aspect`` (``None``: the reference basin) with its ``shape`` outline, depth and
    ``inlet``, the RUNTIME INPUTS (``gravity`` vector, ``lid``, the live tunables of ``params``) and the species table.
    ``mix`` sets the four species shares unless ``params`` overrides them.  The stiffness is clamped to the CFL bound."""
    p = resolve_params(mix, params)
    if "spread" not in (params or {}):
        p["spread"] = inlet_default_spread(inlet)
    cell = cell_of(p)
    geo = basin(n, aspect, cell)
    out: dict[str, float] = {"mode": mode, "use_image": use_image}
    out.update({k: geo[k] for k in ("half", "nozzle_y", "bar_half", "cols", "rows")})
    verts = shape_vertices(shape, geo)
    out.update(inlet_spec(inlet, geo, p, None if shape == "box" else verts))
    rmax = rmax_of(p)
    dh = 0.5 * p["depth"]
    out.update({"cell": cell, "dh": dh, "bar_z": max(0.0, dh - 1.1 * rmax), "k_run": min(max(round(p["k_run"]), 1), K_RUN_MAX),
                "spread": p["spread"], "nozzle_speed": p["nozzle_speed"], "emit_rate": EMIT_SPAN / float(n) / p["rate"]})
    if len(verts) > NV:
        raise ValueError(f"a chamber outline has at most {NV} vertices")
    verts = verts + [verts[-1]] * (NV - len(verts))
    out.update({f"px{k}": verts[k][0] for k in range(NV)})
    out.update({f"py{k}": verts[k][1] for k in range(NV)})
    out.update({"grav_x": gravity[0], "grav_y": gravity[1], "grav_z": gravity[2], "lid": lid, "strata": strata,
                "case_omega": 0.0, "case_domega": 0.0, "pivot_x": 0.0, "pivot_y": 0.5 * (geo["view"][2] + geo["view"][3]),
                "omega": stiffness_of(p), "hertz": p["hertz"], "coh": p["coh"], "spin_drag": p["spin_drag"], "temp": p["temp"],
                "lognormal": 1.0 if p["lognormal"] > 0.5 else 0.0, "bi_ratio": p["bi_ratio"], "bi_frac": p["bi_frac"],
                "fl_rho": p["fl_rho"], "fl_mu": p["fl_mu"]})
    for s in range(len(SPECIES)):
        out.update({f"wt{s}": p[f"wt{s}"], f"r{s}": p[f"r{s}"], f"rho{s}": p[f"rho{s}"], f"mu{s}": p[f"mu{s}"],
                    f"zeta{s}": zeta_of(p[f"e{s}"]), f"roll{s}": p[f"roll{s}"], f"irr{s}": p[f"irr{s}"]})
    return {k: _f32(np.float64(v)) for k, v in out.items()}


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


def _lin_t(pid, a):
    """``pid * a mod 4093`` for a pid up to 2**24 and a RUNTIME multiplier ``a`` < 4093 (4096 = 3 mod 4093)."""
    hi = (pid * (1.0 / 4096.0)).floor()
    lo = pid - hi * 4096.0
    return _mod(_mod(lo * a, _PRIME) + _mod(hi * _mod(a * 3.0, _PRIME), _PRIME), _PRIME)


def _mix_t(pid, a, c):
    """``_mix`` with a runtime multiplier ``a`` and offset ``c`` (exact in f32 and f64 like ``_mix``)."""
    r1 = _mod(_lin_t(pid, a) + c, _PRIME)
    r1 = _mod(r1 * r1 + 3.0 * r1 + 11.0, _PRIME)
    r2 = _mod(_lin_t(pid, a + 6.0) + 2.0 * c + 1.0, _PRIME)
    r2 = _mod(r2 * r2 + 5.0 * r2 + 13.0, _PRIME)
    return (r1 + r2 * (1.0 / _PRIME)) * (1.0 / _PRIME)


def _noise(pid, step, comp):
    """A zero-mean, unit-variance random number that is a PURE FUNCTION of (id, step, component): the sum of two hashed
    uniforms (triangular, variance 1/6, scaled).  No generator state: a replay from any step repeats it bit for bit.  ``step``
    is a whole number below 2**24 (the run's step count), ``comp`` a Python integer."""
    hi = (step * (1.0 / 4093.0)).floor()
    lo = step - hi * 4093.0
    a1 = 1.0 + 2.0 * _mod(hi * 7.0 + comp * 131.0 + 17.0, 1999.0)
    c1 = _mod(lo * 31.0 + comp * 977.0 + 5.0, _PRIME)
    a2 = 1.0 + 2.0 * _mod(hi * 11.0 + comp * 173.0 + 29.0, 1999.0)
    c2 = _mod(lo * 37.0 + comp * 811.0 + 3.0, _PRIME)
    return (_mix_t(pid, a1, c1) + _mix_t(pid, a2, c2) - 1.0) * 2.449489742783178


# ------------------------------------------------------------------ the law
def _ramp(h, offset):
    """One channel of the HSV ramp at hue ``h`` in [0, 1] (full saturation, value)."""
    f = h + offset
    f = f - f.floor()
    return ((f * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _contact(dl, valid, n, vr, spin_slip, wrel, kn, m_eff, mu, zeta, roll, radius, reff, rbar, hertz, coh):
    """Force (3-vector) and torque (3-vector) on the particle from one contact.  ``dl`` is the signed overlap (> 0
    touching), ``valid`` the weight of the contact (0 or 1 for a pair; a wall's share), ``n`` the unit normal pointing at
    THIS particle, ``vr`` its velocity relative to the other body, ``spin_slip`` = (r_i w_i + r_j w_j) x n, ``wrel`` =
    w_i - w_j, ``reff`` the effective radius of the rolling contact, ``rbar`` the mean radius.

    Normal: spring blended from linear to Hertz by ``hertz`` (the Hertz force equals the linear one at HERTZ_REF mean
    radii of overlap; past 4x that the blend stays linear), dashpot of damping ratio ``zeta``, minus an adhesion of
    ``coh`` weights of the pair within COH_RANGE.  Tangential: Coulomb friction limited to a share of the slip per step.
    Rolling: a constant torque against the relative spin, limited the same way."""
    vn = _dot(vr, n)
    cn = 2.0 * zeta * (kn * m_eff).sqrt()
    over = dl.clamp(0.0, 1.0e9)
    xr = (over * (1.0 / (HERTZ_REF * rbar + 1.0e-12))).clamp(0.0, 4.0)
    elastic = kn * over * ((1.0 - hertz) + hertz * xr.sqrt())
    rep = (elastic - cn * vn).clamp(0.0, 1.0e9) * as_float(dl > 0.0)
    near = as_float(dl > -COH_RANGE * rbar)
    fn = (rep - coh * (m_eff * G) * near) * valid
    rep = rep * valid
    # slip of the contact point (tangential): relative velocity less its normal part, less the spin's
    vt = (vr[0] - vn * n[0] - spin_slip[0], vr[1] - vn * n[1] - spin_slip[1], vr[2] - vn * n[2] - spin_slip[2])
    vtn = (_dot(vt, vt) + 1.0e-20).sqrt()
    # tanh of the slip speed over VREF, argument clamped (a tanh built from exp(2x) is inf/inf = NaN past ~44 in f32)
    tanh = (vtn * (1.0 / VREF)).clamp(0.0, 20.0).tanh()
    coulomb = mu * rep * tanh
    stick = (FRICTION_SHARE / (3.5 * DT)) * m_eff * vtn          # the most that removes FRICTION_SHARE of the slip in one step
    ft_mag = coulomb - (coulomb - stick).clamp(0.0, 1.0e30)
    ft = (-(ft_mag / vtn) * vt[0], -(ft_mag / vtn) * vt[1], -(ft_mag / vtn) * vt[2])
    tq_f = _cross(n, ft)
    # rolling resistance
    wn = _dot(wrel, n)
    wt = (wrel[0] - wn * n[0], wrel[1] - wn * n[1], wrel[2] - wn * n[2])
    wtn = (_dot(wt, wt) + 1.0e-20).sqrt()
    tanh_r = (wtn * (1.0 / WREF)).clamp(0.0, 20.0).tanh()
    r_max = roll * rep * reff * tanh_r
    r_stick = (FRICTION_SHARE / DT) * (0.4 * m_eff * rbar * rbar) * wtn
    r_mag = r_max - (r_max - r_stick).clamp(0.0, 1.0e30)
    force = (fn * n[0] + ft[0], fn * n[1] + ft[1], fn * n[2] + ft[2])
    torque = (-radius * tq_f[0] - (r_mag / wtn) * wt[0], -radius * tq_f[1] - (r_mag / wtn) * wt[1],
              -radius * tq_f[2] - (r_mag / wtn) * wt[2])
    return force, torque


def _cap(v, hi):
    """``min(v, hi)`` for a RUNTIME bound ``hi`` (a scalar uniform): clamp bounds are compile-time numbers."""
    return v - (v - hi).clamp(0.0, 1.0e9)


def _nozzle(pid, centre, e1, e2, d, bar_half, bar_z):
    """Where id ``pid`` leaves the inlet tube: a low-discrepancy step along the source line (the golden-ratio step
    2531/4093, so ids emitted close together in time are far apart along it), a hashed place across the depth and a hashed
    height along the stream.  ``centre``, ``e1``, ``e2``, ``d`` are 3-tuples of runtime scalars."""
    u1 = _mod(_lin(pid, 2531.0), _PRIME) * (1.0 / _PRIME)
    u2 = _mod(_lin(pid, 3187.0), _PRIME) * (1.0 / _PRIME)
    u3 = _mod(_lin(pid, 1613.0), _PRIME) * (1.0 / _PRIME)
    a = (u1 - 0.5) * (2.0 * bar_half)
    b = (u3 - 0.5) * (2.0 * bar_z)
    c = (u2 - 0.5) * BAR_HEIGHT
    return tuple(centre[k] + a * e1[k] + b * e2[k] + c * d[k] for k in range(3))


def make_law(n: int, active: int | None = None):
    """The fountain law over ``n`` ids; ``active`` of them (default all) are ever emitted, the rest stay parked.
    The chamber, inlet, species table and every tunable are runtime uniforms: see ``uniforms``."""
    active = int(n if active is None else active)
    stages = merge_exchange_stages(n)
    search_steps = int(math.ceil(math.log2(n + 1))) + 1
    scan_shifts = reduction_shifts(n)
    nsp = len(SPECIES)

    def fountain_step(*args):
        """One fixed step of every particle (see the module docstring)."""
        a = dict(zip(ARGUMENTS, args))
        x, y, z, vx, vy, vz, wx, wy, wz, age = (a[k] for k in STATE)
        pid, ir, ig, ib = (a[k] for k in CONSTANTS)
        half, nozzle_y, bar_half, cols, rows, cell, dh, bar_z = (a[k] for k in ("half", "nozzle_y", "bar_half", "cols", "rows", "cell", "dh", "bar_z"))
        emit_rate, k_run, spread, nozzle_speed = a["emit_rate"], a["k_run"], a["spread"], a["nozzle_speed"]
        grav = (a["grav_x"], a["grav_y"], a["grav_z"])
        lid, strata, mode, use_image = a["lid"], a["strata"], a["mode"], a["use_image"]
        omega, hertz, coh, spin_drag, fl_rho, fl_mu, temp = (a[k] for k in ("omega", "hertz", "coh", "spin_drag", "fl_rho", "fl_mu", "temp"))
        lognormal, bi_ratio, bi_frac = a["lognormal"], a["bi_ratio"], a["bi_frac"]
        centre = (a["nozzle_x"], nozzle_y, a["nozzle_z"])
        d_in = (a["in_dx"], a["in_dy"], a["in_dz"])
        e1 = (a["e1x"], a["e1y"], a["e1z"])
        e2 = (a["e2x"], a["e2y"], a["e2z"])
        cells = cols * rows
        weights = [a[f"wt{s}"] for s in range(nsp)]
        total_w = weights[0] + weights[1] + weights[2] + weights[3] + 1.0e-9
        c0 = weights[0] / total_w
        c1 = (weights[0] + weights[1]) / total_w
        c2 = (weights[0] + weights[1] + weights[2]) / total_w

        def table(name):
            return [a[f"{name}{s}"] for s in range(nsp)]

        def pick(vals, m1, m2, m3):
            return vals[0] + (vals[1] - vals[0]) * m1 + (vals[2] - vals[0]) * m2 + (vals[3] - vals[0]) * m3

        def masks(u):
            sp = as_float(u >= c0) + as_float(u >= c1) + as_float(u >= c2)
            return tuple(as_float((sp - float(k)).abs() < 0.5) for k in (1, 2, 3))

        def species(ident):
            """radius (with its irregularity jitter), mass, friction, damping ratio, rolling coefficient of an id."""
            m1, m2, m3 = masks(_mix(ident, 1237.0, 5.0))
            # the SIZE DISTRIBUTION, a hash of the id: the species' mean radius times a spread factor (uniform: 1 + irr u,
            # u in [-1, 1); lognormal: exp(irr z - irr^2 / 2), z a unit normal clamped at 3 sigma, so the mean stays 1) times the
            # bidisperse mode's factor (a ``bi_frac`` share of the grains are sqrt(bi_ratio) larger, the rest as much smaller;
            # ratio 1 is exactly 1).  Each is an exact 1.0 when off.
            u1 = _mix(ident, 2999.0, 61.0)
            z = ((u1 + _mix(ident, 3511.0, 43.0) - 1.0) * 2.449489742783178).clamp(-3.0, 3.0)
            spread = pick(table("irr"), m1, m2, m3)
            size_f = (1.0 - lognormal) * (1.0 + spread * (u1 * 2.0 - 1.0)) + lognormal * (spread * z - 0.5 * spread * spread).exp()
            big = as_float(_mix(ident, 3733.0, 71.0) < bi_frac)
            sq = bi_ratio.sqrt()
            bi_f = 1.0 + big * (sq - 1.0) + (1.0 - big) * (1.0 / sq - 1.0)
            rr = pick(table("r"), m1, m2, m3) * size_f * bi_f
            rho = pick(table("rho"), m1, m2, m3)
            return (rr, rho, rho * ((4.0 / 3.0) * math.pi) * rr * rr * rr, pick(table("mu"), m1, m2, m3),
                    pick(table("zeta"), m1, m2, m3), pick(table("roll"), m1, m2, m3))

        def frame(search):
            """emit time and the state of each id, from the id and its state.  Evaluated again after the sort
            rather than carried across its ~300 dispatches (a value that crosses them holds an arena slot, 4 bytes
            per id each, for all of them).

            An id is in one of three kinds of state.  PARKED is the exact zero state (x = y = z = 0: no real grain rests
            at the origin).  ARMED (``vy`` = ``ARMED``) is an id whose time has come: it waits AT the inlet slot it
            will leave from (its stored x, y, z), takes part in no contact and does not move, and is emitted at the first
            step its slot is clear (no live grain, where it will be NEXT step, within contact distance).  Anything else is
            LIVE.  With ``search`` an armed id looks up its neighbours at its slot, as a live id does at its place."""
            t = AbstractTensor.where(pid < float(active), pid * emit_rate, pid * 0.0 + 1.0e9)
            a_ = as_float(age >= t)
            parked = as_float((x.abs() + y.abs() + z.abs()) < 1.0e-30)
            armed = as_float(vy < ARMED * 0.5)
            live = a_ * (1.0 - parked) * (1.0 - armed)
            pending = a_ * armed
            arming = a_ * parked
            # only particles inside the grid take part (parked ones and the dilute spray above the grid's top
            # get the cell past the end: sorted last, in nobody's block).  A column runs through the whole depth.
            ix = _cap(((x + half) / cell).floor().clamp(0.0, 1.0e9), cols - 1.0)
            iy = _cap(((y - FLOOR_Y) / cell).floor().clamp(0.0, 1.0e9), rows - 1.0)
            inside = as_float(y < rows * cell + FLOOR_Y)
            return t, a_, arming, live, pending, ix, iy, (live + pending * float(search)) * inside, (live + pending) * inside
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
        zero = x * 0.0
        f_ = [zero, zero, zero]
        q_ = [zero, zero, zero]
        blocked = zero
        dropped = zero
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
            # candidates beyond the walk's reach (``k_run``) are DROPPED contacts: counted (the law's dropped output), never silent
            dropped = dropped + (ends[1] - ends[0] - k_run).clamp(0.0, 1.0e9) * ok * searching
        t_emit = AbstractTensor.where(pid < float(active), pid * emit_rate, pid * 0.0 + 1.0e9)
        # ---- the rest of what this id is, derived again HERE (not before the sort: a value that crosses
        # the sort's dispatches would hold an arena slot for all of them)
        x0, y0, z0 = _nozzle(pid, centre, e1, e2, d_in, bar_half, bar_z)
        u_theta = _mix(pid, 1759.0, 17.0) * 2.0 - 1.0
        theta = u_theta * spread
        speed = (0.9 + _mix(pid, 2203.0, 29.0) * 0.7) * nozzle_speed
        ct, st = theta.cos(), theta.sin()
        vx0 = speed * (d_in[0] * ct + e1[0] * st)
        vy0 = speed * (d_in[1] * ct + e1[1] * st)
        vz0 = speed * (d_in[2] * ct + e1[2] * st)
        radius, density, mass, mu, zeta, roll = species(pid)
        inertia = 0.4 * mass * radius * radius
        w_self = (wx, wy, wz)

        # ---- particle-particle contacts: forces from the neighbours' state
        for lo, hi, ok in runs:
            for slot in range(K_RUN_MAX):
                idx = lo + float(slot)
                valid = as_float(idx < hi) * ok * searching * as_float(k_run > float(slot) + 0.5)
                j = ki.index_select(0, idx.clamp(0.0, top))
                valid = valid * as_float((j - pid).abs() > 0.5)
                vyj = vy.index_select(0, j)
                armed_j = as_float(vyj < ARMED * 0.5)                           # an armed neighbour is a slot, not a body
                xj = x.index_select(0, j)
                yj = y.index_select(0, j)
                zj = z.index_select(0, j)
                vxj = vx.index_select(0, j)
                vzj = vz.index_select(0, j)
                wj = (wx.index_select(0, j), wy.index_select(0, j), wz.index_select(0, j))
                rj, _dj, mj, muj, zetaj, rollj = species(j)
                # coincident centres (two ids at one point) have no contact normal; they are pushed apart
                # along a direction fixed by the ids (antisymmetric, so equal and opposite) instead of
                # dividing a zero separation
                same = as_float((x - xj).abs() + (y - yj).abs() + (z - zj).abs() < 1.0e-9)
                side = as_float(pid > j) - as_float(pid < j)
                dx = x - xj + same * side * 1.0e-4
                dyy = y - yj
                dzz = z - zj
                dist = (dx * dx + dyy * dyy + dzz * dzz + 1.0e-12).sqrt()
                # an armed id is emitted only into a CLEAR slot: no live neighbour, at the position it moves to this
                # step, within the contact distance (plus EMIT_MARGIN): the grain is placed overlapping nothing
                nxt_x = x - (xj + vxj * DT)
                nxt_y = y - (yj + vyj * DT)
                nxt_z = z - (zj + vzj * DT)
                blocked = blocked + pending * valid * (1.0 - armed_j) * as_float(
                    radius + rj + EMIT_MARGIN - (nxt_x * nxt_x + nxt_y * nxt_y + nxt_z * nxt_z + 1.0e-12).sqrt() > 0.0)
                # ... and not into the slot of a LOWER armed id that is within contact distance of this slot (the lower
                # id goes first; both are stationary, so the slots' own distance decides)
                blocked = blocked + pending * valid * armed_j * as_float(j < pid) * as_float(radius + rj + EMIT_MARGIN - dist > 0.0)
                nrm = (dx / dist, dyy / dist, dzz / dist)
                m_eff = mass * mj / (mass + mj)
                kn = omega * omega * m_eff
                vr = (vx - vxj, vy - vyj, vz - vzj)
                sp_sum = (radius * wx + rj * wj[0], radius * wy + rj * wj[1], radius * wz + rj * wj[2])
                spin_slip = _cross(sp_sum, nrm)
                wrel = (wx - wj[0], wy - wj[1], wz - wj[2])
                force, torque = _contact(radius + rj - dist, valid * (1.0 - armed_j), nrm, vr, spin_slip, wrel, kn, m_eff,
                                         (mu * muj).sqrt(), (zeta * zetaj).sqrt(), (roll * rollj).sqrt(), radius,
                                         radius * rj / (radius + rj), 0.5 * (radius + rj), hertz, coh)
                for k in range(3):
                    f_[k] = f_[k] + force[k]
                    q_[k] = q_[k] + torque[k]

        # ---- walls: the same contact against a body of infinite mass.  The basin outline is a polyline (each segment a wall
        # through the depth: the contact is with the nearest point of the segment, in the x-y plane); a segment's two
        # ends carry half a wall each where it meets the next segment (or is a free end), so a vertex is one wall
        kw = omega * omega * mass

        def wall(dl, valid, n_vec):
            force, torque = _contact(dl, valid, n_vec, (vx, vy, vz), _cross((radius * wx, radius * wy, radius * wz), n_vec),
                                     w_self, kw, mass, mu, zeta, roll, radius, radius, radius, hertz, coh)
            for k in range(3):
                f_[k] = f_[k] + force[k]
                q_[k] = q_[k] + torque[k]
        # THICK, SIDED, OVERLAPPING walls.  Each segment is a half-space bounded by its line, whose normal is the segment's LEFT
        # normal (the polyline runs with the chamber on its left): a grain on the wrong side of the line, however deep, is
        # pushed back toward the chamber by the same contact law (never a clamp or a teleport).  The overlap is bounded by a
        # THICKNESS of 6 radii (the penetration depth the spring feels saturates there: a bounded, stable force) and the
        # segment extends past each end by ``ext`` (a third of the half-width), its weight tapering linearly to 0 over that
        # margin, so neighbouring walls overlap at every vertex: a grain that outruns a corner is still met by one of the walls
        # and there is no half-weight seam.  A grain inside the chamber near a concave corner meets both walls, as it does
        # against a floor and a side wall.
        ext = 0.3 * half + 1.0e-6
        for k in range(NV - 1):
            ax, ay, bx, by = a[f"px{k}"], a[f"py{k}"], a[f"px{k + 1}"], a[f"py{k + 1}"]
            ex, ey = bx - ax, by - ay
            len2 = ex * ex + ey * ey
            seg_len = (len2 + 1.0e-12).sqrt()
            inv_len = 1.0 / seg_len
            nlx, nly = -ey * inv_len, ex * inv_len
            side_s = (x - ax) * nlx + (y - ay) * nly
            along = ((x - ax) * ex + (y - ay) * ey) * inv_len                 # distance along the segment from its start
            w_start = ((along + ext) * (1.0 / ext)).clamp(0.0, 1.0)
            w_end = ((seg_len + ext - along) * (1.0 / ext)).clamp(0.0, 1.0)
            share = as_float(len2 > 1.0e-8) * w_start * w_end
            wall(_cap(radius - side_s, 6.0 * radius), live * share, (zero + nlx, zero + nly, zero))
        wall(radius - (dh - z), live, (zero, zero, zero - 1.0))             # the front plexiglass (+z), its normal points at the grain: -z
        wall(radius - (z + dh), live, (zero, zero, zero + 1.0))             # the back plexiglass (-z)
        wall(radius - (lid - y), live, (zero, zero - 1.0, zero))            # the lid: a ceiling at the runtime height ``lid`` (far unless the page closes it)
        # ---- the integrator: gravity (the runtime vector), the interstitial fluid, the forces above.  A pending id with a clear
        # slot is placed at the inlet with its launch velocity; one with a blocked slot waits for the next step.  The fluid's
        # drag (Stokes 6 pi mu r v and quadratic) is implicit, so any viscosity damps and none explodes.
        just = pending * as_float(blocked < 0.5)
        place = just + arming
        buoy = 1.0 - fl_rho / density
        speed_now = (vx * vx + vy * vy + vz * vz).sqrt()
        kdrag = 4.5 * fl_mu / (density * radius * radius) + 0.176 * fl_rho / (density * radius) * speed_now
        damp = 1.0 / (1.0 + live * kdrag * DT)
        # THERMOSTAT (Langevin): a random force per component of variance 2 gamma m kT / DT, with gamma the drag rate just
        # added (fluctuation-dissipation), so the grain's velocity relaxes to a Maxwell distribution of temperature kT; the
        # factor sqrt(1 + gamma DT / 2) makes the implicit drag's stationary variance exactly kT / m.  The same for the spin
        # (its rate: the spin drag plus the fluid's rotational Stokes drag).  The noise is a pure hash of (id, step, component),
        # so a replay repeats it; at kT = 0 every term is an exact zero.
        gamma_r = spin_drag + 10.0 * fl_mu / (density * radius * radius)
        step_now = (age * (1.0 / DT) + 0.5).floor()
        f_amp = (2.0 * kdrag * mass * temp * (1.0 / DT)).sqrt() * (1.0 + 0.5 * kdrag * DT).sqrt()
        q_amp = (2.0 * gamma_r * inertia * temp * (1.0 / DT)).sqrt() * (1.0 + 0.5 * gamma_r * DT).sqrt()
        for k in range(3):
            f_[k] = f_[k] + f_amp * _noise(pid, step_now, k)
            q_[k] = q_[k] + q_amp * _noise(pid, step_now, 3 + k)
        # the CASE FRAME (the walls are static here; the case turns about the z axis through (pivot_x, pivot_y) at ``case_omega``
        # rad/s, counterclockwise positive, with ``case_domega``): besides the rotated gravity a grain feels the centrifugal
        # w^2 (r - p) and Euler -dw x (r - p) accelerations, and its velocity turns by -2 w DT (Coriolis, an exact rotation, so any
        # rate is stable); its own z-spin gains -dw/dt.  The defaults (0) add exact zeros.
        # The turning case is capped at what the walls can hold: the centrifugal load of a grain at the far wall (w^2 R) must be
        # met by the contact spring (omega^2 x penetration), so w is limited to omega sqrt(0.5 r / R) for the grain's radius r and
        # the chamber's reach R (its half-diagonal), and the angular acceleration to omega^2 0.5 r / R (the Euler load).  Past
        # that the spring would let a grain through by its own explicit step; the page's rate is honoured up to the cap.
        half_height = 0.5 * rows * cell
        reach = (half * half + half_height * half_height).sqrt() + 1.0e-6
        w_cap = omega * (0.5 * radius * (1.0 / reach)).sqrt()
        a_cap = omega * omega * 0.5 * radius * (1.0 / reach)
        case_omega = -_cap(-_cap(a["case_omega"], w_cap), w_cap)
        case_domega = -_cap(-_cap(a["case_domega"], a_cap), a_cap)
        rel_x, rel_y = x - a["pivot_x"], y - a["pivot_y"]
        fict_x = case_omega * case_omega * rel_x + case_domega * rel_y
        fict_y = case_omega * case_omega * rel_y - case_domega * rel_x
        vx_a = (vx + live * (f_[0] / mass + grav[0] * buoy + fict_x) * DT) * damp
        vy_a = (vy + live * (f_[1] / mass + grav[1] * buoy + fict_y) * DT) * damp
        turn = (case_omega * DT) * -2.0
        c_turn, s_turn = turn.cos(), turn.sin()
        vx_c = AbstractTensor.where(live > 0.5, vx_a * c_turn - vy_a * s_turn, vx_a)
        vy_c = AbstractTensor.where(live > 0.5, vx_a * s_turn + vy_a * c_turn, vy_a)
        vxn = AbstractTensor.where(just > 0.5, vx0, vx_c)
        vyn = AbstractTensor.where(just > 0.5, vy0, AbstractTensor.where(arming > 0.5, vy * 0.0 + ARMED, vy_c))
        vzn = AbstractTensor.where(just > 0.5, vz0, (vz + live * (f_[2] / mass + grav[2] * buoy) * DT) * damp)
        sdamp = (1.0 - just) / (1.0 + live * gamma_r * DT)
        wxn = (wx + live * (q_[0] / inertia) * DT) * sdamp
        wyn = (wy + live * (q_[1] / inertia) * DT) * sdamp
        wzn = (wz + live * (q_[2] / inertia - case_domega) * DT) * sdamp
        xn = AbstractTensor.where(place > 0.5, x0, x + live * vxn * DT)
        yn = AbstractTensor.where(place > 0.5, y0, y + live * vyn * DT)
        zn = AbstractTensor.where(place > 0.5, z0, z + live * vzn * DT)
        age_next = age + DT
        # ---- colour from the angle of the launch vector.  theta_c is a hashed launch angle (+-0.7 rad, whatever the cone):
        # the hue turns four times as fast as it so the fan spans most of the wheel: trig = atan2, rotation = the
        # double-angle formulas applied twice to cos/sin (no atan2).
        theta_c = u_theta * 0.7
        cth = theta_c.cos()
        sth = theta_c.sin()
        c2 = cth * cth - sth * sth
        s2 = 2.0 * cth * sth
        c = c2 * c2 - s2 * s2
        sn = 2.0 * c2 * s2
        hue = sth.atan2(cth) * (4.0 / TAU) + 0.5
        root3 = 0.5773502691896258
        rot_r = (c * (2.0 / 3.0)) * 0.85 + 0.5
        rot_g = (c * (-1.0 / 3.0) + sn * root3) * 0.85 + 0.5
        rot_b = (c * (-1.0 / 3.0) - sn * root3) * 0.85 + 0.5
        # mode 2: TIME STRATA -- hue turns ``strata`` times over the emission (a runtime uniform), so the settled pile is
        # layered by the order its grains arrived (ids are emitted in order)
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
        # nan (not finite), parked (the zero state), outside (past a wall, the plexiglass or under the floor) or live.
        finite = as_float(xn.abs() < 1.0e30) * as_float(yn.abs() < 1.0e30) * as_float(zn.abs() < 1.0e30) \
            * as_float(vxn.abs() < 1.0e30) * as_float(vyn.abs() < 1.0e30) * as_float(vzn.abs() < 1.0e30) \
            * as_float(wxn.abs() < 1.0e30) * as_float(wyn.abs() < 1.0e30) * as_float(wzn.abs() < 1.0e30)
        placed = finite * as_float((xn.abs() + yn.abs() + zn.abs()) >= 1.0e-30) * as_float(vyn >= ARMED * 0.5)
        inside_box = (1.0 - as_float(xn.abs() > half + radius)) * (1.0 - as_float(yn < -radius)) \
            * (1.0 - as_float(zn.abs() > dh + radius))
        outside = placed * (1.0 - inside_box)
        c_nan = 1.0 - finite
        c_parked = finite * (1.0 - placed)
        c_out = outside
        c_live = placed - outside
        v2 = AbstractTensor.where(placed > 0.5, vxn * vxn + vyn * vyn + vzn * vzn, vxn * 0.0)
        ke = 0.5 * mass * v2 + AbstractTensor.where(placed > 0.5, 0.5 * inertia * (wxn * wxn + wyn * wyn + wzn * wzn), vxn * 0.0)
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
        return (xn, yn, zn, vxn, vyn, vzn, wxn, wyn, wzn, age_next, cr, cg, cb, radius, act,
                ke, v2, c_nan, c_out, c_parked, c_live, c_drop)

    return fountain_step


# ------------------------------------------------------------------- columns
def _f32(a) -> np.ndarray:
    """Rounded through float32: the page's buffers and the reference see one value."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def initial_columns(n: int, *, mix: float = 0.5, mode: int = 0, use_image: float = 0.0, active: int | None = None,
                    aspect: float | None = None, strata: float = STRATA, params: dict | None = None,
                    shape: str = "box", inlet: str = "fountain") -> dict[str, np.ndarray]:
    """Round zero for ``n`` ids: the state is zero, ``pid`` is the iota and the image colours are zero.
    Nothing else is per-id.  The chamber is that of ``active`` ids (default ``n``) at ``aspect``."""
    zeros = np.zeros(n)
    columns = {name: zeros.copy() for name in STATE + ("ir", "ig", "ib")}
    columns["pid"] = np.arange(n, dtype=np.float64)
    columns.update(uniforms(n=int(active or n), mode=mode, use_image=use_image, mix=mix, aspect=aspect, strata=strata,
                            params=params, shape=shape, inlet=inlet))
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
        extra = {name: produced[name] for name in ("cr", "cg", "cb", "rad", "shown") + DIAGNOSTICS}
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


def measured_temperature(diag) -> float:
    """kT measured from the law's diagnostics: the kinetic + rotational energy over the live grains' six degrees of freedom
    (each holds kT / 2), so ``ke = 3 n_live kT``."""
    return float(diag["ke"]) / (3.0 * max(float(diag["n_live"]), 1.0))


def __getattr__(name):
    """The server side (``FountainGame``, the page's option lists) lives in ``fountain_game``, which the compiled products'
    digest does not include: an edit to the page's controls never re-keys the law.  Names the law does not define are
    looked up there, so ``fountain.FountainGame`` keeps working."""
    import importlib
    try:
        return getattr(importlib.import_module("meowdoku.fountain_game"), name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
