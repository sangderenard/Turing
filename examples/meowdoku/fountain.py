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
* The 3-D law's TUNABLES, implemented as it implements them (uniforms; hashed noise and hashed sizes, no per-id column; every one
  exactly neutral at its default): contact stiffness ``omega`` and the Hertz blend, cohesion, spin drag, a Langevin THERMOSTAT ``temp``
  (hashed deterministic noise; the measured T = ke / (1.5 n_live)), nozzle speed and rate, the neighbour cap ``k_run`` (with the
  ``n_drop`` readout), the per-species rolling roughness and size irregularity, and the lognormal / bidisperse size distributions.
  ONE FLUID (``fl_rho``, ``fl_mu``; ``FLUIDS``: air / water / glycerine / custom) drives the drag, the buoyancy AND the orbeez uptake.
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
loop walks the first ``k_run`` (at most ``K_RUN_MAX``) of each run.  No atomics and no scatter.

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
OMEGA = 150.0                          # default contact stiffness: kn = m_eff * omega^2 (omega * DT stays well under 1: a grain feels several contacts)
OMEGA_MAX = 200.0                      # the CFL bound at one contact blend: omega * DT * sqrt(contacts) stays under 2
ZETA = 0.7                             # normal damping ratio
VREF = 0.02                            # tangential slip speed over which friction saturates
WREF = 2.0                             # relative spin (rad/s) over which rolling resistance saturates
SPIN_DRAG = 3.0                        # default spin drag, 1/s
TEMP_MAX = 3.0e-4                      # the hottest kT the page offers (a disk of mass 2.5e-4 then moves at ~1.5 rms; the slider is logarithmic)
TEMP_MIN_LOG = 1.0e-8                  # the coldest non-zero kT on the page's logarithmic slider
HERTZ_REF = 0.04                       # the overlap (in mean radii) at which the Hertz and linear springs agree
COH_RANGE = 0.1                        # adhesion acts to this gap (in mean radii)
STRATA = 1.0                           # default hue cycles over the emission in mode 2 (time strata); a runtime uniform ``strata``
#: The time-strata rates the page offers (hue cycles over one emission): slower keeps wider bands of one colour in place.
STRATA_CHOICES = (0.5, 1.0, 2.0, 4.0, 8.0)
# the page's render looks for the particle field (web/particle_styles.js STYLES -- a test holds the two lists equal)
PARTICLE_STYLES = (("points", "points"), ("liquid", "liquid"), ("sand", "sand"), ("glow", "glow"), ("heat", "heat"),
                   ("stamp", "stamp (memory)"))
# the page's noise-filter replacement modes (web/noise_filter.js NOISE_MODES -- a test holds the two lists equal)
NOISE_MODES = (("mean", "mean colour"), ("spectral", "spectral class"))
K_RUN_MAX = 24                         # candidates walked per row run (a hex pack of the smallest grains is ~9; 24 covers a 2.5x compressed one): the compile-time maximum of the runtime uniform ``k_run``
K_RUN = 24                             # its default
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
#: largest radius a grain can reach).  The FLUID (``fl_rho``, ``fl_mu``) drives the drag, the buoyancy AND the swelling (``FLUIDS``).
RSCALE_RANGE = (0.1, 10.0)             # radius: 0.1x to 10x each species' default (log scale on the page)
MU_RANGE = (0.0, 2.0)                  # sliding friction: frictionless to very sticky
RESTITUTION_RANGE = (0.02, 0.95)
DENSITY_RANGE = (0.2, 5.0)
SWELL_RATE_MAX = 0.5                   # 1/s
SWELL_MAX_RATIO = 4.0
FULL_COORDINATION = 6.0                # contacts of a grain with no open surface (a hex pack); exposure = 1 - contacts / this
#: Interstitial fluids: name -> (density relative to a grain of density 1, viscosity in the sim's units) -- the 3-D law's table.  Sim
#: units: grain density 1, g = 1, grain radius ~0.01 (Stokes damping rate of a grain = 4.5 mu / (rho r^2): water 1.4 /s, glycerine 11 /s).
#: AIR is (0, 0) here, not the 3-D law's (0.0005, 3e-6): the 2-D species' own linear drag (``SPECIES``, 0.04-0.25 /s) already IS the
#: air, so "air" adds nothing and the default is the old law bit for bit; water and glycerine add the fluid's drag and buoyancy on top.
#: The SAME fluid sets the ORBEEZ uptake (the law's ``solvent``): a liquid soaks a swelling grain (WET_RHO), a viscous one more slowly (WET_MU).
FLUIDS = {"air": (0.0, 0.0), "water": (0.38, 2.5e-5), "glycerine": (0.48, 2.0e-4)}
DEFAULT_FLUID = "air"
WET_RHO = 0.38                         # a fluid at least this dense (water) fills a swelling grain's pores entirely; a thinner one proportionally less (air 0)
WET_MU = 1.0e-4                        # up to this viscosity the uptake is not rate-limited; above it the uptake falls as WET_MU / mu (glycerine: 1/2)
SPECIES_TUNABLES = ("rscale", "fscale", "dscale", "zscale", "swell", "swmax", "roll", "irr")
DOF = 3                                # a disk's degrees of freedom (x, y, spin), kT / 2 each: the measured temperature's divisor (the 3-D sphere has 6)
NV = 8                                 # polygon vertices (uniforms px0..px7, py0..py7)
SHAPES = ("box", "funnel", "hexagon", "ramp")       # box: the old basin (the three planes); the others add the polygon walls
POLYGON_UNIFORMS = tuple(f"{c}{k}" for c in ("px", "py") for k in range(NV))
#: FAST chambers: REGULAR (equilateral) n-gons on the polyline path (n + 1 <= NV vertices), closed, one edge flat at the bottom.
NGON_SHAPES = {"ngon3": 3, "ngon4": 4, "ngon5": 5, "ngon6": 6, "ngon7": 7}
#: SDF chambers (the analytic path): the 2-D sections of the solids (sphere -> circle, cube -> square, tetrahedron -> triangle,
#: toroidal shell -> annulus), the hourglass, and the oil & glycerine toy.
SDF_SHAPES = ("sdf_circle", "sdf_square", "sdf_triangle", "sdf_annulus", "sdf_hourglass", "sdf_toy")
#: The chamber-shape select's options (key, label); the SDF ones are marked.
CHAMBER_CHOICES = (("box", "box (open basin)"), ("funnel", "funnel"), ("hexagon", "hexagon"), ("ramp", "ramp"),
                   ("ngon3", "triangle (fast, polyline)"), ("ngon4", "square (fast, polyline)"), ("ngon5", "pentagon (fast, polyline)"),
                   ("ngon6", "hexagon, regular (fast, polyline)"), ("ngon7", "heptagon (fast, polyline)"),
                   ("sdf_circle", "sphere / circle (SDF)"), ("sdf_square", "cube / square (SDF)"), ("sdf_triangle", "tetrahedron / triangle (SDF)"),
                   ("sdf_annulus", "toroidal shell / annulus (SDF)"), ("sdf_hourglass", "hourglass (SDF)"),
                   ("sdf_toy", "oil & glycerine toy: ramps, hourglass, pegs (SDF)"),
                   ("waterwheel", "waterwheel: a nozzle falls on a hinged paddle wheel (box)"))
#: ANALYTIC SDF CHAMBERS (runtime uniforms, no recompile when the chamber changes).  ``SDF_SLOTS`` primitive slots, each ``q<k><field>``:
#: ``kind`` (0 unused), ``role`` (0 FLUID: the chamber's inside; 1 SOLID: an obstacle inside it), the centre ``cx cy``, the rotation ``rot``
#: and the parameters ``p0..p5`` (see ``sdf_field``).  ``sdf_on`` (0 / 1) picks the SDF walls; 0 is the polyline walls and the old law bit for bit.
SDF_SLOTS = 12
SDF_FIELDS = ("kind", "role", "cx", "cy", "rot", "p0", "p1", "p2", "p3", "p4", "p5")
SDF_KINDS = {"box": 1, "circle": 2, "capsule": 3, "annulus": 4, "ngon": 5, "hourglass": 6, "pegs": 7}
SDF_FAR = 1.0e9
#: TYPED slots: the kind of each slot is fixed at compile time (3 boxes, 2 circles, 3 capsules, an annulus, an n-gon, an hourglass, a
#: peg array), so a slot evaluates only its own SDF; ``kind`` is just its used flag (0: masked out) and a preset fills the slots it needs.
SDF_LAYOUT = ("box", "box", "box", "circle", "circle", "capsule", "capsule", "capsule", "annulus", "ngon", "hourglass", "pegs")
assert len(SDF_LAYOUT) == SDF_SLOTS
SDF_UNIFORMS = ("sdf_on",) + tuple(f"q{k}{f}" for k in range(SDF_SLOTS) for f in SDF_FIELDS)
STATE = ("x", "y", "vx", "vy", "w", "age", "rs")          # rs: the grain's CURRENT radius (it grows for a swelling species)
CONSTANTS = ("pid", "ir", "ig", "ib")
UNIFORMS = ("mode", "use_image", "mix", "half", "nozzle_y", "bar_half", "cols", "rows", "grav_x", "grav_y", "lid", "strata",   # the run, its basin, the RUNTIME INPUTS, then the colour rate
            "cell_scale", "fl_rho", "fl_mu", "omega", "hertz", "coh", "spin_drag", "temp", "k_run", "nozzle_speed", "emit_scale",
            "lognormal", "bi_ratio", "bi_frac", "case_omega", "case_domega", "pivot_x", "pivot_y") + POLYGON_UNIFORMS + tuple(f"{k}{s}" for k in SPECIES_TUNABLES for s in range(4)) + SDF_UNIFORMS
#: The uniforms the PAGE drives moment by moment (no rebuild, no reset): the gravity vector (the phone's tilt), ``lid``, a
#: ceiling that is far away (``LID_OFF``) until the page, driving gravity from a sensor, closes it so a grain cannot fall
#: out of the open top when gravity points up, and ``rows``, the grid's height, which it raises with the lid so grains under
#: the lid still meet each other.  The server sends their defaults (and the closed values as ``tilt_inputs``); the page's
#: runtime owns them after that.
RUNTIME_INPUTS = ("grav_x", "grav_y", "lid", "rows", "fl_rho", "fl_mu", "omega", "hertz", "coh", "spin_drag", "temp", "k_run", "nozzle_speed",
                  "case_omega", "case_domega") + tuple(
    f"{k}{s}" for k in ("fscale", "dscale", "zscale", "swell", "swmax", "roll") for s in range(4))      # the radius, its spread and the emission rate (so the grid) are fixed when a run starts
LID_OFF = 1.0e9
LID_ABOVE = 1.45                       # the lid the page closes while a sensor drives gravity: this far above the nozzle (the jet's apex is ~1.34)
#: The HINGED BODY (the waterwheel): present only in the BODY VARIANT of the law (``make_law(.., body=True)``), whose argument list adds the
#: two replicated state columns ``bth`` (angle, counterclockwise) and ``bom`` (angular velocity) and the uniforms below; the default law
#: has none of it (its signature, products and the 10M size are unchanged).  See the module's WATERWHEEL section.
BODY_STATE = ("bth", "bom")
BODY_UNIFORMS = ("body_x", "body_y", "body_rh", "body_pl", "body_ht", "body_np", "body_i", "body_gam", "body_alpha", "body_tau")
BODY_RUNTIME_INPUTS = ("body_i", "body_gam", "body_alpha", "body_tau")
BODY_DIAGNOSTICS = ("body_th", "body_om", "body_tq")
BODY_SHAPES = ("waterwheel",)
IOTA = ("pid",)                        # columns the generated shader derives from its invocation index
#: ``ke``..``n_drop`` are the law's DIAGNOSTICS: reduced by a fixed-order tree, so element 0 holds the total over all ids
#: (kinetic + rotational energy, the largest speed squared, and the counts of the ids that are not finite, outside the
#: walls, parked, live, plus the contacts the neighbour walk dropped).  The four classes partition the ids.
DIAGNOSTICS = ("ke", "vmax2", "n_nan", "n_out", "n_parked", "n_live", "n_drop")
OUTPUTS = ("x_next", "y_next", "vx_next", "vy_next", "w_next", "age_next", "rs_next", "cr", "cg", "cb", "rad") + DIAGNOSTICS
BODY_OUTPUTS = ("x_next", "y_next", "vx_next", "vy_next", "w_next", "age_next", "rs_next", "bth_next", "bom_next", "cr", "cg", "cb",
                "rad") + DIAGNOSTICS + BODY_DIAGNOSTICS
ARGUMENTS = STATE + CONSTANTS + UNIFORMS
BODY_ARGUMENTS = STATE + BODY_STATE + CONSTANTS + UNIFORMS + BODY_UNIFORMS

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
            "cols": cols, "rows": rows, "cells": cols * rows, "cell": cell,
            "view": (-VIEW_SIDE * s * k, VIEW_SIDE * s * k, -VIEW_BOTTOM * s, nozzle_y + VIEW_TOP)}


def shape_vertices(shape: str, geo: dict[str, float]) -> list[tuple[float, float]]:
    """The basin outline as a polyline (at most ``NV`` vertices; the law pads it by repeating the last), the chamber on its
    LEFT as it runs (counterclockwise).  (The box needs none: it is the three planes.)  Every non-box shape is a CLOSED polygon (its first vertex repeated, so its top is its own ceiling) INSCRIBED in the
    visible sim window in BOTH axes, centred: |x| <= the window's half-width less a margin of three grain diameters, from the
    floor (y = 0) to the window's top less that margin, as large as fits.  ``hexagon``; ``funnel`` a vessel with a V bottom
    that narrows to its throat; ``ramp`` a vessel with a sloping floor.  (The same shapes as the 3-D fountain's; the planes of the
    old basin stay, outside them.)"""
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
    if shape in NGON_SHAPES:
        return ngon_vertices(NGON_SHAPES[shape], w, top)
    raise ValueError(f"unknown chamber shape {shape!r}")


def ngon_fit(count: int, w: float, top: float) -> tuple[float, float]:
    """The circumradius and the centre height of the REGULAR (equilateral) ``count``-gon, one edge flat at the bottom, as large as
    fits inside ``|x| <= w``, ``0 <= y <= top`` (uniform scale: both axes inscribed, whichever binds), centred in both."""
    ang = [-0.5 * math.pi - math.pi / count + TAU * i / count for i in range(count)]
    xs, ys = [math.cos(t) for t in ang], [math.sin(t) for t in ang]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    r = min(2.0 * w / width, top / height)
    return r, 0.5 * top - 0.5 * r * (max(ys) + min(ys))


def ngon_vertices(count: int, w: float, top: float) -> list[tuple[float, float]]:
    """A closed regular ``count``-gon polyline (first vertex repeated), counterclockwise, inscribed in the window (see ``ngon_fit``)."""
    r, cy = ngon_fit(count, w, top)
    pts = [(r * math.cos(-0.5 * math.pi - math.pi / count + TAU * i / count),
            cy + r * math.sin(-0.5 * math.pi - math.pi / count + TAU * i / count)) for i in range(count)]
    xs = [p[0] for p in pts]
    mid = 0.5 * (max(xs) + min(xs))
    pts = [(p[0] - mid, p[1]) for p in pts]
    return pts + [pts[0]]


def chord(verts: list[tuple[float, float]], y: float) -> tuple[float, float]:
    """The x extent of the chamber at height ``y``: the leftmost and rightmost crossings of the outline."""
    xs = []
    for (ax, ay), (bx, by) in zip(verts[:-1], verts[1:]):
        if (ay - y) * (by - y) <= 0.0 and ay != by:
            xs.append(ax + (bx - ax) * (y - ay) / (by - ay))
    return (min(xs), max(xs)) if xs else (-1.0e9, 1.0e9)


def tilt_inputs(n: int, aspect: float | None = None, cell: float = CELL, shape: str = "box") -> dict[str, float]:
    """What the page writes to the runtime inputs ``lid`` and ``rows`` while a sensor drives gravity: a ceiling above the
    jet's reach and a grid tall enough to cover everything under it."""
    geo = chamber_basin(shape, n, aspect, cell)
    lid = max(geo["nozzle_y"] + LID_ABOVE, geo.get("top", 0.0) + cell)             # (a designed SDF chamber may be taller than the jet's reach)
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
    is a whole number below 2**24 (the run's step count), ``comp`` a Python integer.  (The 3-D law's, verbatim.)"""
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


def _contact(delta, weight, gap, nx, ny, vrx, vry, spin_slip, wrel, kn, m_eff, mu, zeta, roll, reff, rbar, hertz, coh):
    """Force and torque on the particle from one contact (the 3-D law's contact, in the plane).  ``delta`` > 0 is the overlap
    (already weighted: 0 for a contact that does not exist), ``weight`` the contact's weight (0 or 1 for a pair; a wall's share),
    ``gap`` the UNWEIGHTED overlap (< 0 apart: the adhesion reaches to ``COH_RANGE`` mean radii), ``n`` the unit normal pointing at
    THIS particle, ``vr`` its velocity relative to the other body, ``spin_slip`` the spin part of the contact slip, ``wrel`` the
    relative spin w_i - w_j, ``reff`` the effective radius of the rolling contact, ``rbar`` the mean radius.  Returns
    (fx, fy, tangential force, rolling torque); the friction torque = -radius * the third value.

    Normal: a spring blended from linear to Hertz by ``hertz`` (the Hertz force equals the linear one at HERTZ_REF mean radii of
    overlap; past 4x that the blend stays linear), a dashpot of damping ratio ``zeta``, minus an adhesion of ``coh`` weights of the
    pair within COH_RANGE (acts apart as well as touching).  Tangential: Coulomb friction limited to a share of the slip per step.
    Rolling: a constant torque against the relative spin, limited the same way (a disk's 1/2 m r^2)."""
    vn = vrx * nx + vry * ny
    cn = 2.0 * zeta * (kn * m_eff).sqrt()
    over = delta.clamp(0.0, 1.0e9)
    xr = (over * (1.0 / (HERTZ_REF * rbar + 1.0e-12))).clamp(0.0, 4.0)
    elastic = kn * over * ((1.0 - hertz) + hertz * xr.sqrt())
    rep = (elastic - cn * vn).clamp(0.0, 1.0e9) * as_float(delta > 0.0)
    near = as_float(gap > -COH_RANGE * rbar)
    fn = rep - coh * (m_eff * G) * near * weight
    tx, ty = -ny, nx
    vt = vrx * tx + vry * ty - spin_slip
    # tanh of the slip speed over VREF.  An emitted grain meets the plume at ~1.5 m/s, so the argument
    # reaches ~75; a tanh built from exp(2x) (Metal/Apple shaders, fast-math) is inf/inf = NaN past
    # |x| ~ 44 in f32, and one NaN force removes the grain and every grain it touches.  tanh(20) is 1
    # to 1e-17, so clamping the argument changes no f64 result and keeps every f32 lane finite.
    tanh = (vt * (1.0 / VREF)).clamp(-20.0, 20.0).tanh()
    coulomb = mu * rep * tanh.abs()
    stick = (FRICTION_SHARE / (3.0 * DT)) * m_eff * vt.abs()          # the most that removes FRICTION_SHARE of the slip in one step
    ft = -(coulomb - (coulomb - stick).clamp(0.0, 1.0e30)) * (vt / (vt.abs() + 1.0e-30))
    # rolling resistance: against the relative spin, at most ``roll`` x the repulsion x the effective radius, and no more than removes
    # FRICTION_SHARE of the relative spin in one step (in the contact's spin mass 1/2 m_eff rbar^2).  roll = 0: exact zeros.
    wtn = (wrel * wrel + 1.0e-20).sqrt()
    tanh_r = (wtn * (1.0 / WREF)).clamp(0.0, 20.0).tanh()
    r_max = roll * rep * reff * tanh_r
    r_stick = (FRICTION_SHARE / DT) * (0.5 * m_eff * rbar * rbar) * wtn
    r_mag = r_max - (r_max - r_stick).clamp(0.0, 1.0e30)
    return fn * nx + ft * tx, fn * ny + ft * ty, ft, -(r_mag / wtn) * wrel


def _cap(v, hi):
    """``min(v, hi)`` for a RUNTIME bound ``hi`` (a scalar uniform): clamp bounds are compile-time numbers."""
    return v - (v - hi).clamp(0.0, 1.0e9)


def _nozzle(pid, bar_half, nozzle_y):
    """Where id ``pid`` leaves the nozzle: a low-discrepancy step along the bar (the golden-ratio step 2531/4093, so ids
    emitted close together in time are far apart along the bar) and a hashed height within it."""
    bar_x = _mod(_lin(pid, 2531.0), _PRIME) * (1.0 / _PRIME)
    bar_y = _mod(_lin(pid, 3187.0), _PRIME) * (1.0 / _PRIME)
    return (bar_x - 0.5) * (2.0 * bar_half), nozzle_y + (bar_y - 0.5) * BAR_HEIGHT


# ----------------------------------------------------- analytic SDF chambers
def _sgn(v):
    """+1 where ``v >= 0`` else -1 (never 0: a gradient on an axis of symmetry picks a side)."""
    return as_float(v >= 0.0) * 2.0 - 1.0


def _max(a, b):
    return b + (a - b).clamp(0.0, 1.0e30)


def _min(a, b):
    return a - (a - b).clamp(0.0, 1.0e30)


def _sdf_box(lx, ly, hw, hh, rc):
    """Rounded box: half extents ``hw, hh``, corner radius ``rc``.  Returns (f, gx, gy): f < 0 inside, the gradient of f."""
    sx, sy = _sgn(lx), _sgn(ly)
    qx, qy = lx.abs() - (hw - rc), ly.abs() - (hh - rc)
    ox, oy = qx.clamp(0.0, 1.0e9), qy.clamp(0.0, 1.0e9)
    ol = (ox * ox + oy * oy + 1.0e-12).sqrt()
    qm = _max(qx, qy)
    f = ol + _min(qm, 0.0) - rc
    out = as_float((ox + oy) > 0.0)
    xdom = as_float(qx > qy)
    return f, (out * (ox / ol) + (1.0 - out) * xdom) * sx, (out * (oy / ol) + (1.0 - out) * (1.0 - xdom)) * sy


def _sdf_circle(lx, ly, rad):
    r = (lx * lx + ly * ly + 1.0e-12).sqrt()
    return r - rad, lx / r, ly / r


def _sdf_capsule(lx, ly, half_len, rad):
    """A segment along the local x axis from ``-half_len`` to ``half_len``, thickened by ``rad``."""
    m1 = lx - (lx - half_len).clamp(0.0, 1.0e30)
    t = m1 + (-half_len - m1).clamp(0.0, 1.0e30)
    ex = lx - t
    er = (ex * ex + ly * ly + 1.0e-12).sqrt()
    return er - rad, ex / er, ly / er


def _sdf_annulus(lx, ly, big_r, half_t):
    r = (lx * lx + ly * ly + 1.0e-12).sqrt()
    u = r - big_r
    s = _sgn(u)
    return u.abs() - half_t, s * lx / r, s * ly / r


def _sdf_ngon(lx, ly, rad, count):
    """Regular ``count``-gon of circumradius ``rad`` with an edge normal along +x: the plane distance to the nearest edge
    (exact inside; a bound outside), found by folding the angle into the nearest edge's sector."""
    nn = count.clamp(3.0, 64.0)
    sector = TAU * (1.0 / nn)
    k = (ly.atan2(lx) * (1.0 / sector) + 0.5).floor()
    ca, sa = (k * sector).cos(), (k * sector).sin()
    return lx * ca + ly * sa - rad * (sector * 0.5).cos(), ca, sa


def _sdf_hourglass(lx, ly, top_w, throat_w, half_h, slope):
    """|x| <= min(top_w, throat_w + slope |y|), |y| <= half_h, each side Lipschitz-normalised (slope 0: the full taper from the
    throat to ``top_w`` at the caps)."""
    hh = half_h.clamp(1.0e-6, 1.0e9)
    sx, sy = _sgn(lx), _sgn(ly)
    ax, ay = lx.abs(), ly.abs()
    kdef = (top_w - throat_w) * (1.0 / hh)
    has = as_float(slope > 1.0e-9)
    kk = kdef + has * (slope - kdef)
    inv = 1.0 / (1.0 + kk * kk).sqrt()
    fa, gax, gay = (ax - throat_w - kk * ay) * inv, sx * inv, -kk * sy * inv      # the tapering wall
    fb = ax - top_w                                                                  # the vertical wall (gradient (sx, 0))
    fc = ay - hh                                                                     # the cap (gradient (0, sy))
    pb = as_float(fa >= fb)
    f1 = _max(fa, fb)
    g1x, g1y = pb * gax + (1.0 - pb) * sx, pb * gay
    pc = as_float(f1 >= fc)
    return _max(f1, fc), pc * g1x, pc * g1y + (1.0 - pc) * sy


def _sdf_pegs(lx, ly, pitch_x, pitch_y, peg_r, cols, rows, stagger):
    """A PEG ARRAY by domain repetition with clamped indices: the nearest peg in each of the three rows around the point's own.
    Returns (f, gx, gy) of the nearest peg."""
    px = pitch_x.clamp(1.0e-6, 1.0e9)
    py = pitch_y.clamp(1.0e-6, 1.0e9)
    rm = (rows - 1.0).clamp(0.0, 1.0e9)
    cm = (cols - 1.0).clamp(0.0, 1.0e9)
    j0 = (ly * (1.0 / py) + 0.5 * rm + 0.5).floor()
    best, bgx, bgy = None, None, None
    for dj in (-1.0, 0.0, 1.0):
        j = _min((j0 + dj).clamp(0.0, 1.0e9), rm)
        odd = j - 2.0 * (j * 0.5).floor()
        shift = stagger * 0.5 * px * odd
        i = _min(((lx - shift) * (1.0 / px) + 0.5 * cm + 0.5).floor().clamp(0.0, 1.0e9), cm)
        ex = lx - ((i - 0.5 * cm) * px + shift)
        ey = ly - (j - 0.5 * rm) * py
        er = (ex * ex + ey * ey + 1.0e-12).sqrt()
        f = er - peg_r
        if best is None:
            best, bgx, bgy = f, ex / er, ey / er
        else:
            take = as_float(f < best)
            best = best + take * (f - best)
            bgx = bgx + take * (ex / er - bgx)
            bgy = bgy + take * (ey / er - bgy)
    return best, bgx, bgy


def sdf_field(x, y, q):
    """The composed chamber SDF at ``(x, y)`` from the slot uniforms (``q(k, field)`` gives slot ``k``'s scalar).
    d = max over FLUID slots of their inside distance, then min with the signed distance to each SOLID: ``d > 0`` is inside the
    fluid, the distance to the nearest wall.  Returns ``(d, nx, ny)``: ``n`` is the ANALYTIC gradient of the active primitive
    (unit, pointing into the fluid), chosen by masks, no finite differences."""
    zero = x * 0.0
    d_f, f_nx, f_ny = zero - SDF_FAR, zero, zero
    d_s, s_nx, s_ny = zero + SDF_FAR, zero, zero
    for k in range(SDF_SLOTS):
        kind, role, cx, cy, rot = (q(k, f) for f in ("kind", "role", "cx", "cy", "rot"))
        p0, p1, p2, p3, p4, p5 = (q(k, f) for f in ("p0", "p1", "p2", "p3", "p4", "p5"))
        c, s = rot.cos(), rot.sin()
        dx, dy = x - cx, y - cy
        lx, ly = c * dx + s * dy, c * dy - s * dx

        fns = {"box": lambda: _sdf_box(lx, ly, p0, p1, p2), "circle": lambda: _sdf_circle(lx, ly, p0),
               "capsule": lambda: _sdf_capsule(lx, ly, p0, p1), "annulus": lambda: _sdf_annulus(lx, ly, p0, p1),
               "ngon": lambda: _sdf_ngon(lx, ly, p0, p1), "hourglass": lambda: _sdf_hourglass(lx, ly, p0, p1, p2, p3),
               "pegs": lambda: _sdf_pegs(lx, ly, p0, p1, p2, p3, p4, p5)}
        f, gx, gy = fns[SDF_LAYOUT[k]]()                          # TYPED slot: only its own kind's SDF is evaluated
        gwx, gwy = c * gx - s * gy, s * gx + c * gy
        used = as_float(kind > 0.5)
        fluid = used * as_float(role < 0.5)
        solid = used * (1.0 - fluid)
        # SELECT, never blend: with the SDF_FAR sentinel, ``d + take * (c - d)`` is (-1e9) + (1e9 + c), which float32 (the GPU)
        # rounds to 0 -- the composed distance was 0 everywhere on the GPU while the float64 eager lane hid it.
        cf = fluid * (-f) + (1.0 - fluid) * (-SDF_FAR)             # exact: the role is 0 or 1 (a scalar uniform: no where(), it would broadcast)
        take = cf > d_f
        d_f = AbstractTensor.where(take, cf, d_f)
        f_nx = AbstractTensor.where(take, zero - gwx, f_nx)
        f_ny = AbstractTensor.where(take, zero - gwy, f_ny)
        cs = solid * f + (1.0 - solid) * SDF_FAR
        take = cs < d_s
        d_s = AbstractTensor.where(take, cs, d_s)
        s_nx = AbstractTensor.where(take, gwx, s_nx)
        s_ny = AbstractTensor.where(take, gwy, s_ny)
    pick = d_s < d_f
    return AbstractTensor.where(pick, d_s, d_f), AbstractTensor.where(pick, s_nx, f_nx), AbstractTensor.where(pick, s_ny, f_ny)


def make_law(n: int, active: int | None = None, sdf: bool = True, body: bool = False):
    """The fountain law over ``n`` ids; ``active`` of them (default all) are ever emitted, the rest stay parked.
    The basin (``half``, ``nozzle_y``, ``bar_half``, ``cols``, ``rows``) is a runtime uniform: see ``basin``."""
    active = int(n if active is None else active)
    stages = merge_exchange_stages(n)
    search_steps = int(math.ceil(math.log2(n + 1))) + 1
    emit_period = EMIT_SPAN / float(active)           # at nozzle rate 1; the runtime uniform ``emit_scale`` (1 / rate) stretches it
    scan_shifts = reduction_shifts(n)

    def fountain_step(*args):
        """One fixed step of every particle (see the module docstring)."""
        a = dict(zip(BODY_ARGUMENTS if body else ARGUMENTS, args))
        # <purrline>
        if _PURRLINE["on"]:
            a = dict(zip(PURRLINE_ARGUMENTS, args))
        # </purrline>
        x, y, vx, vy, w, age, rs = (a[k] for k in STATE)
        pid, ir, ig, ib = (a[k] for k in CONSTANTS)
        mode, use_image, mix, half, nozzle_y, bar_half, cols, rows, grav_x, grav_y, lid, strata = (
            a[k] for k in ("mode", "use_image", "mix", "half", "nozzle_y", "bar_half", "cols", "rows", "grav_x", "grav_y", "lid", "strata"))
        # the tunables the 3-D law has, implemented as it implements them (hashed noise, hashed sizes, no per-id column): all are
        # uniforms, and at their neutral values (hertz = coh = temp = fl_rho = fl_mu = roll = irr = 0, lognormal = 0, bi_ratio = 1,
        # nozzle_speed = emit_scale = 1, omega = 150, spin_drag = 3, k_run = 24) every term below is an exact 1 or 0: the old law bit for bit
        omega, hertz, coh, spin_drag, temp = (a[k] for k in ("omega", "hertz", "coh", "spin_drag", "temp"))
        fl_rho, fl_mu, k_run, nozzle_speed = (a[k] for k in ("fl_rho", "fl_mu", "k_run", "nozzle_speed"))
        lognormal, bi_ratio, bi_frac = a["lognormal"], a["bi_ratio"], a["bi_frac"]
        # the ORBEEZ solvent IS the fluid: the uptake is the fraction of a swelling grain's pores the fluid fills (its density over
        # WET_RHO, 1 at most: air 0, water 1) times the rate limit of its viscosity (1 up to WET_MU, then WET_MU / mu: glycerine 1/2)
        solvent = (fl_rho * (1.0 / WET_RHO)).clamp(0.0, 1.0) * (WET_MU / (fl_mu + 1.0e-12)).clamp(0.0, 1.0)
        # the CASE FRAME: the walls are static in the frame the law runs in, and the case turns at ``case_omega`` (rad/s, about
        # the pivot, counterclockwise positive) with angular acceleration ``case_domega``.  A grain feels, besides gravity
        # (which the page rotates), the fictitious accelerations  -2 w x v  (Coriolis: applied as an exact rotation of the
        # velocity, so any rate is stable),  w^2 (r - p)  (centrifugal) and  -dw/dt x (r - p)  (Euler); its own spin, measured
        # in this frame, gains -dw/dt.  Defaults (0, and the box centre as pivot) add exact zeros: the old law bit for bit.
        case_omega, case_domega = a["case_omega"], a["case_domega"]
        cell = CELL * a["cell_scale"]
        cells = cols * rows
        emit_rate = emit_period * a["emit_scale"]

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
            dropped = dropped + (ends[1] - ends[0] - k_run).clamp(0.0, 1.0e9) * ok * searching
        t_emit = AbstractTensor.where(pid < float(active), pid * emit_rate, pid * 0.0 + 1.0e9)
        # ---- the rest of what this id is, derived again HERE (not before the sort: a value that crosses
        # the sort's dispatches would hold an arena slot for all of them)
        x0, y0 = _nozzle(pid, bar_half, nozzle_y)
        theta = (_mix(pid, 1759.0, 17.0) * 2.0 - 1.0) * 0.7
        speed = (0.9 + _mix(pid, 2203.0, 29.0) * 0.7) * nozzle_speed
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
        # the SIZE DISTRIBUTION, a hash of the id (the 3-D law's, verbatim): the species' mean radius times a spread factor (uniform:
        # 1 + irr u, u in [-1, 1); lognormal: exp(irr z - irr^2 / 2), z a unit normal clamped at 3 sigma, so the mean stays 1) times the
        # bidisperse mode's factor (a ``bi_frac`` share of the grains are sqrt(bi_ratio) larger, the rest as much smaller; ratio 1 is
        # exactly 1).  Each is an exact 1.0 when off.  A grain's radius is fixed at its placement (``rs``), so the neighbours read it from the state.
        size_u = _mix(pid, 2999.0, 61.0)
        size_z = ((size_u + _mix(pid, 3511.0, 43.0) - 1.0) * 2.449489742783178).clamp(-3.0, 3.0)
        spread = pick4(tab("irr"), is1, is2, is3)
        size_f = (1.0 - lognormal) * (1.0 + spread * (size_u * 2.0 - 1.0)) + lognormal * (spread * size_z - 0.5 * spread * spread).exp()
        big = as_float(_mix(pid, 3733.0, 71.0) < bi_frac)
        bi_sq = bi_ratio.sqrt()
        bi_f = 1.0 + big * (bi_sq - 1.0) + (1.0 - big) * (1.0 / bi_sq - 1.0)
        radius0 = radius0 * size_f * bi_f
        roll = pick4(tab("roll"), is1, is2, is3)
        density =(SPECIES[0][2] + (SPECIES[1][2] - SPECIES[0][2]) * is1 + (SPECIES[2][2] - SPECIES[0][2]) * is2
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
            rj = AbstractTensor.where(rsj > 0.0, rsj, rj0)           # the neighbour's current radius (every gridded id is placed: rs > 0)
            return rj, dj * math.pi * rj * rj, muj, pick4(tab("zscale"), j1, j2, j3), pick4(tab("roll"), j1, j2, j3)

        for lo, hi, ok in runs:
            for slot in range(K_RUN_MAX):
                idx = lo + float(slot)
                valid = as_float(idx < hi) * ok * searching * as_float(k_run > float(slot) + 0.5)
                j = ki.index_select(0, idx.clamp(0.0, top))
                valid = valid * as_float((j - pid).abs() > 0.5)
                armed_j = as_float(vy.index_select(0, j) < ARMED * 0.5)         # an armed neighbour is a slot, not a body
                xj = x.index_select(0, j)
                yj = y.index_select(0, j)
                vxj = vx.index_select(0, j)
                vyj = vy.index_select(0, j)
                wj = w.index_select(0, j)
                rj, mj, muj, zscj, rollj = neighbour_props(j)
                # coincident centres (two ids at one point) have no contact normal; they are pushed apart
                # along a direction fixed by the ids (antisymmetric, so equal and opposite) instead of
                # dividing a zero separation
                same = as_float((x - xj).abs() + (y - yj).abs() < 1.0e-9)
                side = as_float(pid > j) - as_float(pid < j)
                dx = x - xj + same * side * 1.0e-4
                dyy = y - yj
                dist = (dx * dx + dyy * dyy + 1.0e-12).sqrt()
                weight = valid * (1.0 - armed_j)
                gap = radius + rj - dist
                delta = gap * valid * (1.0 - armed_j)
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
                kn = omega * omega * m_eff
                gx, gy, gt, gr = _contact(delta, weight, gap, nx, ny, vx - vxj, vy - vyj, radius * w + rj * wj, w - wj, kn, m_eff,
                                          (mu * muj).sqrt(), ZETA * (zsc * zscj).sqrt(), (roll * rollj).sqrt(),
                                          radius * rj / (radius + rj), 0.5 * (radius + rj), hertz, coh)
                ncont = ncont + touch
                fx = fx + gx          # a contact that does not exist writes an exact zero (its repulsion is masked); adhesion acts apart too
                fy = fy + gy
                tq = tq - radius * gt * touch + gr
        # ---- walls and floor: the same contact against a body of infinite mass
        kw = omega * omega * mass
        walls = [fx, fy, tq, ncont]

        def wall(delta, weight, gap, nx_, ny_, vrx_=None, vry_=None, wrel_=None):
            """The contact of this grain with a wall (a body of infinite mass at rest; the grain's own radius is the mean and effective
            radius), accumulated into ``fx, fy, tq, ncont``: the repulsion exists only where the grain overlaps (``delta`` > 0), the
            adhesion (``coh``) also within COH_RANGE of it."""
            gx_, gy_, gt_, gr_ = _contact(delta, weight, gap, nx_, ny_, vx if vrx_ is None else vrx_, vy if vry_ is None else vry_,
                                          radius * w, w if wrel_ is None else wrel_, kw, mass, mu, ZETA * zsc, roll,
                                          radius, radius, hertz, coh)
            touch_ = as_float(delta > 0.0)
            walls[3] = walls[3] + touch_
            walls[0] = walls[0] + gx_
            walls[1] = walls[1] + gy_
            walls[2] = walls[2] - radius * gt_ * touch_ + gr_
            return gx_, gy_, gr_, touch_

        gap = radius - (y - FLOOR_Y)
        wall(gap.clamp(0.0, 1.0e9) * live, live, gap, y * 0.0, y * 0.0 + 1.0)
        gap = radius - (x + half)
        wall(gap.clamp(0.0, 1.0e9) * live, live, gap, y * 0.0 + 1.0, y * 0.0)
        gap = radius - (half - x)
        wall(gap.clamp(0.0, 1.0e9) * live, live, gap, y * 0.0 - 1.0, y * 0.0)
        # the lid: the same contact against a ceiling at the runtime height ``lid`` (a body of infinite mass; far away unless the page closes it)
        gap = radius - (lid - y)
        wall(gap.clamp(0.0, 1.0e9) * live, live, gap, y * 0.0, y * 0.0 - 1.0)
        # the POLYGON walls (a shape other than the box): THICK, SIDED, OVERLAPPING half-spaces, the same contact law.  Each
        # segment's normal is its LEFT normal (the chamber on the left as the polyline runs), so a grain on the wrong side of a
        # line is pushed back, never out; the penetration the spring feels saturates at 6 radii (a bounded force); the segment
        # extends 0.3 x half past each end with its weight tapering to 0, so neighbouring walls overlap at every vertex.  The
        # default (all vertices 0) has zero-length segments: weight 0, exact zeros.
        ext = 0.3 * half + 1.0e-6
        for k in range(NV - 1):
            ax, ay, bx, by = a[f"px{k}"], a[f"py{k}"], a[f"px{k + 1}"], a[f"py{k + 1}"]
            ex, ey = bx - ax, by - ay
            len2 = ex * ex + ey * ey
            seg_len = (len2 + 1.0e-12).sqrt()
            inv_len = 1.0 / seg_len
            nlx, nly = -ey * inv_len, ex * inv_len
            side_s = (x - ax) * nlx + (y - ay) * nly
            along = ((x - ax) * ex + (y - ay) * ey) * inv_len
            w_start = ((along + ext) * (1.0 / ext)).clamp(0.0, 1.0)
            w_end = ((seg_len + ext - along) * (1.0 / ext)).clamp(0.0, 1.0)
            share = as_float(len2 > 1.0e-8) * w_start * w_end
            gap = _cap(radius - side_s, 6.0 * radius)
            wall(gap.clamp(0.0, 1.0e9) * live * share, live * share, gap, y * 0.0 + nlx, y * 0.0 + nly)
        # the SDF walls (``sdf_on`` = 1): the SAME thick sided saturating contact against the composed analytic chamber SDF.  Contact when
        # d < radius, the penetration (radius - d) saturating at 6 radii, the force along the SDF's analytic gradient (into the fluid): a
        # grain on the wrong side of a wall is pushed back by a bounded force, never clamped or moved.  sdf_on = 0: exact zeros.
        sdf_on = a["sdf_on"]
        if sdf:
            d_sdf, sdf_nx, sdf_ny = sdf_field(x, y, lambda k, field: a[f"q{k}{field}"])
            gap = _cap(radius - d_sdf, 6.0 * radius)
            wall(gap.clamp(0.0, 1.0e9) * live * sdf_on, live * sdf_on, gap, sdf_nx, sdf_ny)
            sdf_gone = live * sdf_on * as_float(d_sdf < -radius)       # the grain centre is more than a radius on the wrong side
        else:
            sdf_gone = x * 0.0
        # <purrline>
        if _PURRLINE["on"]:
            # ---- PURRLINE SAND: the occupancy grid (static solid cells) and the moving boxes (a contact against a moving surface), both the wall
            # contact law with the penetration saturating at 6 radii.  A grain pressed into the grid by a box leaves along the grid's free face (the
            # box's normal is replaced by the grid's), never into an occupied cell.  ``pl_near`` marks the grains in touch of a solid (for the speed bound
            # below) and ``pl_sx, pl_sy`` the surface velocity they are bounded against.
            og_on = a["og_on"]
            d_g, g_nx, g_ny = _sdf_occgrid(x - a["og_cx"], y - a["og_cy"], a["og_cell"], a["og_cols"], a["og_rows"], a["og_w0"], a["og_w1"], a["og_w2"])
            gap = _cap(radius - d_g, 6.0 * radius)
            wall(gap.clamp(0.0, 1.0e9) * live * og_on, live * og_on, gap, g_nx, g_ny)
            d_mv, mv_nx, mv_ny, mv_vx, mv_vy = moving_field(x, y, lambda k, f: a[f"mb{k}_{f}"], x * 0.0)
            squeeze = as_float(d_g < 2.0 * radius) * og_on * as_float(mv_nx * g_nx + mv_ny * g_ny < 0.0) * as_float(d_mv < radius)
            mv_nx = AbstractTensor.where(squeeze > 0.5, g_nx, mv_nx)
            mv_ny = AbstractTensor.where(squeeze > 0.5, g_ny, mv_ny)
            gap = _cap(radius - d_mv, PURRLINE_PUSH * radius)             # a box's spring saturates at one radius: a grain it squeezes against a wall or a cell sits ON the surface, not through it
            wall(gap.clamp(0.0, 1.0e9) * live, live, gap, mv_nx, mv_ny, vx - mv_vx, vy - mv_vy)
            pl_box = as_float(d_mv < 2.0 * radius)
            pl_near = live * (pl_box + (1.0 - pl_box) * as_float(d_g < 2.0 * radius) * og_on)
            pl_sx, pl_sy = mv_vx * pl_box, mv_vy * pl_box
        # </purrline>
        tb = x * 0.0
        if body:
            # ---- the HINGED BODY (waterwheel): a hub circle and ``body_np`` capsule paddles in the body frame (hinge ``body_x, body_y``,
            # angle ``bth``).  The N paddles are ONE evaluation: the grain's angle in the body frame picks the nearest paddle axis
            # (``round(phi N / 2 pi)``) and the capsule is evaluated in that paddle's frame; its analytic gradient is rotated back.  The
            # contact is the wall contact (the same ``_contact``) against a surface moving at ``omega x r`` (spin relative to ``bom``);
            # the body takes the equal and opposite force at the contact point ``centre - n radius`` (the point this grain's own friction
            # torque arm uses, so angular momentum is conserved pair by pair) and the reaction of the rolling couple.
            bth, bom = a["bth"], a["bom"]
            bnp = _max(a["body_np"], a["body_np"] * 0.0 + 1.0)
            dxb, dyb = x - a["body_x"], y - a["body_y"]
            rr = (dxb * dxb + dyb * dyb + 1.0e-12).sqrt()
            d_hub = rr - a["body_rh"]
            cb0, sb0 = bth.cos(), bth.sin()
            phi = (cb0 * dyb - sb0 * dxb).atan2(cb0 * dxb + sb0 * dyb + 1.0e-9)
            sector = (phi * (bnp * (1.0 / TAU)) + 0.5).floor()
            alpha = bth + sector * (TAU * (1.0 / bnp))
            ca, sa = alpha.cos(), alpha.sin()
            px_, py_ = ca * dxb + sa * dyb, ca * dyb - sa * dxb
            half_p = 0.5 * a["body_pl"]
            d_pad, pgx, pgy = _sdf_box(px_ - half_p, py_, half_p, a["body_ht"], a["body_ht"])
            near_pad = d_pad < d_hub
            d_b = AbstractTensor.where(near_pad, d_pad, d_hub)
            nbx = AbstractTensor.where(near_pad, ca * pgx - sa * pgy, dxb / rr)
            nby = AbstractTensor.where(near_pad, sa * pgx + ca * pgy, dyb / rr)
            gap_b = _cap(radius - d_b, 6.0 * radius)
            gbx, gby, grb, _touch = wall(gap_b.clamp(0.0, 1.0e9) * live, live, gap_b, nbx, nby, vx + bom * dyb, vy - bom * dxb, w - bom)
            arm_x, arm_y = dxb - nbx * radius, dyb - nby * radius
            tb = ((arm_y * gbx - arm_x * gby) - grb) * live
        fx, fy, tq, ncont = walls
        # ---- the integrator: gravity (the runtime vector ``grav_x, grav_y``), drag, the forces above.  A pending id with a clear slot is placed at the
        # nozzle with its launch velocity; one with a blocked slot waits, parked, for the next step.  The FLUID's drag (Stokes 6 pi mu r v and
        # quadratic, as the 3-D law) is implicit, so any viscosity damps and none explodes, and it buoys the grain (gravity x (1 - fluid / grain
        # density)); the species' own linear ``drag`` (the air) stays explicit as it was.
        just = pending * as_float(blocked < 0.5)
        place = just + arming
        inertia = 0.5 * mass * radius * radius
        buoy = 1.0 - fl_rho / density
        speed_now = (vx * vx + vy * vy).sqrt()
        kdrag = 4.5 * fl_mu / (density * radius * radius) + 0.176 * fl_rho / (density * radius) * speed_now
        damp = 1.0 / (1.0 + live * kdrag * DT)
        # THERMOSTAT (Langevin), the 3-D law's: a random force per component of variance 2 gamma m kT / DT, with gamma the drag rate
        # (the species' air plus the fluid's, just added: fluctuation-dissipation), so the grain's velocity relaxes to a Maxwell
        # distribution of temperature kT; the factor sqrt(1 + gamma DT / 2) makes the implicit drag's stationary variance exactly kT / m.
        # The same for the spin (its rate: the spin drag plus the fluid's rotational Stokes drag).  The noise is a pure hash of (id, step,
        # component), so a replay repeats it; at kT = 0 every term is an exact zero.  A disk has three degrees of freedom (x, y, spin).
        gamma_t = drag + kdrag
        gamma_f = 10.0 * fl_mu / (density * radius * radius)
        gamma_r = spin_drag + gamma_f
        step_now = (age * (1.0 / DT) + 0.5).floor()
        f_amp = (2.0 * gamma_t * mass * temp * (1.0 / DT)).sqrt() * (1.0 + 0.5 * gamma_t * DT).sqrt()
        q_amp = (2.0 * gamma_r * inertia * temp * (1.0 / DT)).sqrt() * (1.0 + 0.5 * gamma_r * DT).sqrt()
        fx = fx + f_amp * _noise(pid, step_now, 0)
        fy = fy + f_amp * _noise(pid, step_now, 1)
        tq = tq + q_amp * _noise(pid, step_now, 5)
        # The turning case is capped at what the walls can hold: the centrifugal load of a grain at the far wall (w^2 R) must be met
        # by the contact spring (OMEGA^2 x penetration), so w is limited to OMEGA sqrt(0.5 r / R) for the grain's radius r and the
        # chamber's reach R (its half-diagonal), and the angular acceleration to OMEGA^2 0.5 r / R (the Euler load); the planes
        # are sided, so a grain that does outrun the spring is still pushed back.
        half_height = 0.5 * rows * cell
        reach = (half * half + half_height * half_height).sqrt() + 1.0e-6
        w_cap = omega * (0.5 * radius * (1.0 / reach)).sqrt()
        a_cap = omega * omega * 0.5 * radius * (1.0 / reach)
        case_omega = -_cap(-_cap(case_omega, w_cap), w_cap)
        case_domega = -_cap(-_cap(case_domega, a_cap), a_cap)
        rel_x, rel_y = x - a["pivot_x"], y - a["pivot_y"]
        fict_x = case_omega * case_omega * rel_x + case_domega * rel_y
        fict_y = case_omega * case_omega * rel_y - case_domega * rel_x
        vx_a = (vx + live * (fx / mass + grav_x * buoy - drag * vx + fict_x) * DT) * damp
        vy_a = (vy + live * (fy / mass + grav_y * buoy - drag * vy + fict_y) * DT) * damp
        turn = (case_omega * DT) * -2.0                               # Coriolis: the velocity turns by -2 w DT (clockwise for w > 0)
        c_turn, s_turn = turn.cos(), turn.sin()
        vx_c = AbstractTensor.where(live > 0.5, vx_a * c_turn - vy_a * s_turn, vx_a)      # only a live grain's velocity (an armed id's vy is a sentinel)
        vy_c = AbstractTensor.where(live > 0.5, vx_a * s_turn + vy_a * c_turn, vy_a)
        vxn = AbstractTensor.where(just > 0.5, vx0, vx_c)
        vyn = AbstractTensor.where(just > 0.5, vy0, AbstractTensor.where(arming > 0.5, vy * 0.0 + ARMED, vy_c))
        # <purrline>
        if _PURRLINE["on"]:
            # the velocity a solid can impart is BOUNDED: relative to the surface it touches, at most the contact's own saturation speed (omega x 6 radii)
            vcap = a["omega"] * 6.0 * radius
            rvx, rvy = vxn - pl_sx * pl_near, vyn - pl_sy * pl_near
            keep = 1.0 - pl_near * (1.0 - (vcap * (1.0 / (rvx * rvx + rvy * rvy + 1.0e-12).sqrt())).clamp(0.0, 1.0))
            vxn = pl_sx * pl_near + rvx * keep
            vyn = pl_sy * pl_near + rvy * keep
        # </purrline>
        wn = (w + live * (tq / inertia - spin_drag * w - case_domega) * DT) * (1.0 - just) * (1.0 / (1.0 + live * gamma_f * DT))
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
        outside = placed * (1.0 - (1.0 - as_float(xn.abs() > half + radius)) * (1.0 - as_float(yn < -radius)) * (1.0 - sdf_gone))
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
            if body:
                tb = tb + tb.index_select(0, partner) * gate
            c_nan = c_nan + c_nan.index_select(0, partner) * gate
            c_parked = c_parked + c_parked.index_select(0, partner) * gate
            c_out = c_out + c_out.index_select(0, partner) * gate
            c_live = c_live + c_live.index_select(0, partner) * gate
            c_drop = c_drop + c_drop.index_select(0, partner) * gate
            v2 = v2 + (v2.index_select(0, partner) * gate - v2).clamp(0.0, 1.0e30)
        if body:
            # ---- the BODY's step, in engine_toy's order (DrivetrainSolver): the applied torques first (the grains' reaction, from the
            # reduction's element 0; ``body_tau``, an absolute torque a machine-sim coupling writes; ``body_alpha`` x I, the motor as an
            # acceleration), ``omega += torque / I dt``; then the hinge, a ``rotational-bearing`` edge to the frame (omega = 0), solved
            # as ``rotational_friction.friction_impulse`` does: the viscous drag ``gam I |omega|`` capped at sticking (it never reverses omega).
            inertia_b = a["body_i"]
            tau_c = tb.index_select(0, pid * 0.0)
            om1 = a["bom"] + (tau_c + a["body_tau"] + a["body_alpha"] * inertia_b) * (1.0 / inertia_b) * DT
            mag = _min(a["body_gam"] * inertia_b * om1.abs() * DT, om1.abs() * inertia_b)
            om2 = om1 - _sgn(om1) * mag * (1.0 / inertia_b)
            th1 = a["bth"] + om2 * DT
            th2 = th1 - TAU * (th1 * (1.0 / TAU) + 0.5).floor()
            return (xn, yn, vxn, vyn, wn, age_next, rsn, th2, om2, cr, cg, cb, radius, ke, v2, c_nan, c_out, c_parked, c_live, c_drop,
                    th2, om2, tau_c)
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


#: The tunables beyond the species table, the 3-D law's (``fountain3d.PARAMS``) with the 2-D units.  name -> (label, lo, hi, step,
#: default, restart, group).  ``restart`` ones change what an id IS at placement (the size distribution, the emission rate): a new run;
#: the others are RUNTIME INPUTS of the law, applied at a step boundary and logged with their step.  ``fl_rho``/``fl_mu`` are the
#: fluid (``FLUIDS``).  A disk's mass is density x pi r^2 (~2.5e-4), 80x the 3-D sphere's, hence the hotter ``temp`` range.
PARAMS = {
    "omega": ("stiffness (compressibility)", 40.0, OMEGA_MAX, 5.0, OMEGA, False, "Physics"),
    "hertz": ("Hertz blend", 0.0, 1.0, 0.05, 0.0, False, "Physics"),
    "coh": ("cohesion", 0.0, 3.0, 0.05, 0.0, False, "Physics"),
    "spin_drag": ("spin drag", 0.0, 20.0, 0.5, SPIN_DRAG, False, "Physics"),
    "temp": ("temperature kT", 0.0, TEMP_MAX, 1.0e-8, 0.0, False, "Physics"),
    "k_run": ("neighbour cap", 4.0, float(K_RUN_MAX), 1.0, float(K_RUN), False, "Physics"),
    "nozzle_speed": ("nozzle speed", 0.3, 2.0, 0.05, 1.0, False, "Physics"),
    "rate": ("nozzle rate", 0.25, 4.0, 0.05, 1.0, True, "Physics"),
    "fl_rho": ("fluid density", 0.0, 1.2, 0.01, FLUIDS[DEFAULT_FLUID][0], False, "Fluid"),
    "fl_mu": ("fluid viscosity", 0.0, 5.0e-4, 1.0e-5, FLUIDS[DEFAULT_FLUID][1], False, "Fluid"),
    "lognormal": ("size distribution (0 uniform, 1 lognormal)", 0.0, 1.0, 1.0, 0.0, True, "Grains"),
    "bi_ratio": ("bidisperse size ratio (large / small)", 1.0, 3.0, 0.05, 1.0, True, "Grains"),
    "bi_frac": ("bidisperse number fraction large", 0.0, 1.0, 0.05, 0.5, True, "Grains"),
    # the hinged wheel (shape "waterwheel" only): paddles and radius are what the body IS (a new run); inertia, damping and the motor are runtime inputs
    "wheel_n": ("wheel paddles", 2.0, 16.0, 1.0, 8.0, True, "Wheel"),
    "wheel_r": ("wheel radius (x)", 0.5, 2.0, 0.05, 1.0, True, "Wheel"),
    "wheel_i": ("wheel inertia (x)", 0.1, 10.0, 0.1, 1.0, False, "Wheel"),
    "wheel_gam": ("hinge damping rate (1/s)", 0.0, 20.0, 0.1, 1.0, False, "Wheel"),
    "wheel_alpha": ("motor (rad/s^2, 0 = free)", -50.0, 50.0, 0.5, 0.0, False, "Wheel"),
}
WHEEL_RADIUS = 0.30                    # the wheel's outer reach at the reference basin (x the basin scale s)
WHEEL_HUB = 0.2                        # hub radius / reach
WHEEL_PADDLE_HALF = 0.012              # paddle half thickness (about one grain radius: it does not scale with the basin)
WHEEL_AREAL_DENSITY = 3.0              # wheel material, x the grain's density (mass per area): sets the reference inertia
WHEEL_NOZZLE_BAR = 0.04                # the inlet's half width at the reference basin (x s): a stream, not a curtain
ROLL_RANGE = (0.0, 0.3)                # rolling resistance per species (roughness)
IRR_RANGE = (0.0, 0.3)                 # size spread per species (irregularity): a uniform width, or the lognormal's sigma


def species_params(params: dict | None = None) -> dict[str, float]:
    """The page's tunables in their own units: the species table (``r{s}`` radius, ``mu{s}`` friction (0-2), ``rho{s}`` density,
    ``e{s}`` restitution, ``swell{s}`` rate and ``swmax{s}`` ratio, ``roll{s}`` roughness, ``irr{s}`` irregularity) and ``PARAMS``
    (stiffness, Hertz, cohesion, spin drag, temperature, neighbour cap, nozzle speed and rate, the fluid, the size distributions) --
    defaults, then the overrides clamped to their ranges."""
    p: dict[str, float] = {k: float(v[4]) for k, v in PARAMS.items()}
    for s, (_name, r, rho, mu, _drag) in enumerate(SPECIES):
        p.update({f"r{s}": r, f"rho{s}": rho, f"mu{s}": mu, f"e{s}": restitution_of(ZETA), f"swell{s}": 0.0, f"swmax{s}": 1.0,
                  f"roll{s}": 0.0, f"irr{s}": 0.0})
    ranges = {"r": (RSCALE_RANGE[0], RSCALE_RANGE[1], True), "rho": DENSITY_RANGE + (False,), "mu": MU_RANGE + (False,),
              "e": RESTITUTION_RANGE + (False,), "swell": (0.0, SWELL_RATE_MAX, False), "swmax": (1.0, SWELL_MAX_RATIO, False),
              "roll": ROLL_RANGE + (False,), "irr": IRR_RANGE + (False,)}
    for k, v in (params or {}).items():
        if k in PARAMS:
            p[k] = min(max(float(v), PARAMS[k][1]), PARAMS[k][2])
            continue
        key = k.rstrip("0123456789")
        if key in ranges and k[len(key):].isdigit() and int(k[len(key):]) < len(SPECIES):
            lo, hi, relative = ranges[key]
            base = SPECIES[int(k[len(key):])][1] if relative else 1.0
            p[k] = min(max(float(v), lo * base), hi * base)
    return p


def restart_param(name: str) -> bool:
    """Does changing tunable ``name`` begin a new run?  The radius (it sets the grid), the size distributions and the emission rate."""
    return (PARAMS[name][5] if name in PARAMS else name.rstrip("0123456789") in ("r", "irr"))


def size_factor_max(p: dict[str, float], s: int) -> float:
    """The largest multiple of species ``s``'s radius any of its grains can have: its uniform width, or the lognormal's tail (the law
    clamps the normal at 3 sigma), times the bidisperse mode's large-size factor (the 3-D law's)."""
    spread = p[f"irr{s}"]
    wide = math.exp(3.0 * spread) if p["lognormal"] > 0.5 else 1.0 + spread
    return wide * math.sqrt(max(p["bi_ratio"], 1.0))


def stiffness_of(p: dict[str, float]) -> float:
    """The contact stiffness the law is given: the page's ``omega`` clamped to the CFL bound (Hertz stiffens the spring up to
    1 + 2 hertz times its linear tangent, so the bound drops with it)."""
    return min(p["omega"], OMEGA_MAX / math.sqrt(1.0 + 2.0 * p["hertz"]))


def measured_temperature(diag) -> float:
    """kT measured from the law's diagnostics: the kinetic + rotational energy over the live grains' THREE degrees of freedom (x, y,
    spin; each holds kT / 2), so ``ke = 1.5 n_live kT`` (the 3-D law has six: ``ke = 3 n_live kT``)."""
    return float(diag["ke"]) / (DOF / 2.0 * max(float(diag["n_live"]), 1.0))


def cell_needed(p: dict[str, float]) -> float:
    """The grid cell the species table needs: twice the largest radius any grain can reach (its size spread and bidisperse mode, and a
    swelling species grows to ``swmax`` x its radius)."""
    return 2.0 * max(p[f"r{s}"] * size_factor_max(p, s) * (p[f"swmax{s}"] if p[f"swell{s}"] > 0.0 else 1.0) for s in range(len(SPECIES)))


def _slot(kind: str, role: str, cx: float, cy: float, rot: float, *p: float) -> dict[str, float]:
    """One SDF primitive slot as uniforms ``{field: value}`` (``p`` pads with zeros to p0..p5)."""
    ps = list(p) + [0.0] * (6 - len(p))
    return {"kind": float(SDF_KINDS[kind]), "role": 0.0 if role == "fluid" else 1.0, "cx": cx, "cy": cy, "rot": rot,
            **{f"p{i}": float(ps[i]) for i in range(6)}}


def _capsule_ramp(high: tuple[float, float], low: tuple[float, float], rad: float) -> dict[str, float]:
    """A capsule ramp from its ``high`` end to its ``low`` end, thickened by ``rad`` (centre, angle and half length)."""
    (hx, hy), (lx, ly) = high, low
    return _slot("capsule", "solid", 0.5 * (hx + lx), 0.5 * (hy + ly), math.atan2(ly - hy, lx - hx), 0.5 * math.hypot(hx - lx, hy - ly), rad)


def sdf_preset(shape: str, geo: dict[str, float]) -> dict[str, Any]:
    """An SDF chamber as slots (pure data) at its DESIGNED proportions: the width is the basin's (``|x| <= w``, the window's half width
    less the 3-cell margin), the HEIGHT is whatever the design needs (``top``), built from ``w`` (which goes as sqrt(ids)) and the grain
    size (``cell``): a circle's diameter, a funnel's slope, the toy's peg rows, throat and ramps.  ``chamber_basin`` makes the view
    follow it.  Also where its inlet goes (``nozzle_y``) -- the inlet is checked to be inside the fluid by ``uniforms``."""
    c = geo.get("cell", CELL)
    w = 0.5 * (geo["view"][1] - geo["view"][0]) - 3.0 * c
    slots: list[dict[str, float]] = []
    nozzle_y = None
    if shape == "sdf_circle":
        top = 2.0 * w
        slots.append(_slot("circle", "fluid", 0.0, 0.5 * top, 0.0, w))
        nozzle_y = 0.5 * top
    elif shape == "sdf_square":
        top = 2.0 * w
        slots.append(_slot("box", "fluid", 0.0, w, 0.0, w, w, 2.0 * c))
        nozzle_y = w
    elif shape == "sdf_triangle":
        ang = [-0.5 * math.pi - math.pi / 3 + TAU * i / 3 for i in range(3)]
        ys = [math.sin(t) for t in ang]
        r = 2.0 * w / (max(math.cos(t) for t in ang) - min(math.cos(t) for t in ang))
        cy = -r * min(ys)
        top = r * (max(ys) - min(ys))
        slots.append(_slot("ngon", "fluid", 0.0, cy, -0.5 * math.pi, r, 3.0))
        nozzle_y = cy
    elif shape == "sdf_annulus":
        big = w
        top = 2.0 * big
        band = min(max(0.25 * big, 4.0 * c), 0.4 * big)                          # the ring's half thickness (a tiny window keeps a ring wide enough to pour into)
        slots.append(_slot("annulus", "fluid", 0.0, big, 0.0, big - band, band))
        nozzle_y = big + big - band
    elif shape == "sdf_hourglass":
        top = 2.7 * w
        slots.append(_slot("hourglass", "fluid", 0.0, 0.5 * top, 0.0, w, max(0.1 * w, 3.5 * c), 0.5 * top, 0.0))
        nozzle_y = 0.75 * top
    elif shape == "sdf_toy":
        # ONE hourglass-profile vessel (a vertical wall to the funnel, a throat, a funnel back out to a vertical wall, the floor y = 0 under
        # it); three capsule ramps (left-down, right-down, left-down) in the upper section; a staggered peg field under the throat.  Built
        # bottom up from what each part needs, so the height is the sum, not a fraction of a window: six peg rows (pitch 4.6 grain
        # diameters) and a 3-cell clearance under the funnel; the funnel (slope 1.7); the ramps (a drop of tan 0.42 across the vessel,
        # each stepped 0.25 w below the last, 6 cells of clearance above the funnel); the inlet 0.12 above the highest ramp.
        rr = 0.9 * c
        throat = max(0.09 * w, 3.5 * c)
        slope, tan, step = 1.7, 0.42, 0.25 * w
        delta = (w - throat) / slope
        px = py = 4.6 * c
        y_lo, peg_rows = 4.0 * c, 6
        pegs_top = y_lo + (peg_rows - 1) * py + 0.9 * c
        yc = pegs_top + 3.0 * c + delta
        edge = w + 1.5 * rr
        y0 = yc + delta + 6.0 * c + tan * (edge + 0.1 * w) + 2.0 * step
        nozzle_y = y0 + 0.12
        top = nozzle_y + 0.12
        slots.append(_slot("hourglass", "fluid", 0.0, yc, 0.0, w, throat, top - yc, slope))
        for i, low_x in enumerate((-0.10 * w, 0.10 * w, -0.10 * w)):
            sign = 1.0 if i % 2 == 0 else -1.0                                           # +: high end at the right wall (the ramp runs left-down)
            high = (sign * edge, y0 - i * step)
            slots.append(_capsule_ramp(high, (low_x, high[1] - tan * abs(high[0] - low_x)), rr))
        cols = 2 * int(w // px) + 1
        slots.append(_slot("pegs", "solid", 0.0, y_lo + 0.5 * (peg_rows - 1) * py, 0.0, px, py, 0.9 * c, cols, peg_rows, 1.0))
    else:
        raise ValueError(f"unknown SDF chamber {shape!r}")
    return {"slots": slots, "nozzle_y": nozzle_y, "top": top, "half": w, "bar": (0.25 if shape == "sdf_toy" else 0.4) * w}


def chamber_basin(shape: str, n: int, aspect: float | None = None, cell: float = CELL) -> dict[str, float]:
    """``basin`` for a chamber shape: the polygon presets and the box are the basin as it always was; an SDF preset keeps its designed
    proportions, so the VIEW (its top: the chamber's top and the 3-cell margin), the grid's rows and the inlet follow the chamber."""
    geo = basin(n, aspect, cell)
    if shape not in SDF_SHAPES:
        return geo
    spec = sdf_preset(shape, geo)
    view = (geo["view"][0], geo["view"][1], geo["view"][2], spec["top"] + 3.0 * cell)
    return {**geo, "view": view, "nozzle_y": spec["nozzle_y"], "rows": max(geo["rows"], math.ceil(spec["top"] / cell) + 1), "top": spec["top"]}


def slot_uniforms(slots: list[dict[str, float]]) -> dict[str, float]:
    """Each slot dict goes to the next free slot of its kind (``SDF_LAYOUT``); the rest are unused (all zero)."""
    names = {v: k for k, v in SDF_KINDS.items()}
    placed: dict[int, dict[str, float]] = {}
    for slot in slots:
        kind = names[int(slot["kind"])]
        free = [k for k in range(SDF_SLOTS) if SDF_LAYOUT[k] == kind and k not in placed]
        if not free:
            raise ValueError(f"a chamber has at most {SDF_LAYOUT.count(kind)} SDF slots of kind {kind}")
        placed[free[0]] = slot
    out = {"sdf_on": 1.0 if slots else 0.0}
    for k in range(SDF_SLOTS):
        out.update({f"q{k}{f}": float(placed.get(k, {}).get(f, 0.0)) for f in SDF_FIELDS})
    return out


def chamber_distance(u: dict[str, float], xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The law's own ``sdf_field`` (the same code, on the NumPy backend) at points: ``(d, nx, ny)``.  Host side: outlines, the inlet check, tests."""
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    q = lambda k, f: AbstractTensor.get_tensor(np.asarray(float(_f32(np.float64(u[f"q{k}{f}"])))))
    d, nx, ny = sdf_field(AbstractTensor.get_tensor(xs), AbstractTensor.get_tensor(ys), q)
    return tuple(np.asarray(v.numpy(), dtype=np.float64) for v in (d, nx, ny))


def fit_inlet(u: dict[str, float], cell: float) -> float:
    """The widest inlet bar (half width at most ``u['bar_half']``) whose whole rectangle (the grains leave within +-BAR_HEIGHT/2 of
    ``nozzle_y``) is deeper than one grain diameter (``cell``) inside the SDF fluid; the narrowest (0.01) if none is."""
    bar = float(u["bar_half"])
    while bar >= 0.01:
        xs = np.linspace(-bar, bar, 41)
        pts = [(xs, np.full_like(xs, u["nozzle_y"] + dy)) for dy in (-0.5 * BAR_HEIGHT, 0.0, 0.5 * BAR_HEIGHT)]
        if min(float(chamber_distance(u, px, py)[0].min()) for px, py in pts) > cell:
            return bar
        bar *= 0.9
    return 0.01                                   # no bar is clear of the walls (a tiny window): the narrowest, the walls push any grain that starts in them out


def zero_contours(field: np.ndarray, x0: float, y0: float, h: float) -> list[list[tuple[float, float]]]:
    """The zero contour of ``field[j, i]`` (the value at ``(x0 + i h, y0 + j h)``) by marching squares with linear interpolation:
    deterministic (cells in raster order, the saddle decided by the cell's mean, polylines started from the smallest edge key).
    Returns polylines of ``(x, y)``; a closed one repeats its first point."""
    pos = field > 0.0
    corners = pos[:-1, :-1].astype(np.int8) + 2 * pos[:-1, 1:] + 4 * pos[1:, 1:] + 8 * pos[1:, :-1]
    # edge keys: ("h", i, j) joins (i, j)-(i + 1, j); ("v", i, j) joins (i, j)-(i, j + 1).  A cell (i, j) has bottom h(i, j), right v(i + 1, j),
    # top h(i, j + 1), left v(i, j).  Corner bits: 1 bl, 2 br, 4 tr, 8 tl.
    table = {1: [("L", "B")], 2: [("B", "R")], 3: [("L", "R")], 4: [("R", "T")], 6: [("B", "T")], 7: [("L", "T")],
             8: [("T", "L")], 9: [("T", "B")], 11: [("T", "R")], 12: [("R", "L")], 13: [("B", "R")], 14: [("B", "L")]}
    adjacency: dict[tuple, list[tuple]] = {}
    for j, i in zip(*np.nonzero((corners != 0) & (corners != 15))):
        i, j, case = int(i), int(j), int(corners[j, i])
        edge = {"B": ("h", i, j), "R": ("v", i + 1, j), "T": ("h", i, j + 1), "L": ("v", i, j)}
        if case in (5, 10):
            centre = 0.25 * (field[j, i] + field[j, i + 1] + field[j + 1, i + 1] + field[j + 1, i])
            # case 5: bl and tr positive; case 10: br and tl.  The mean says whether the positive diagonal is joined through the cell
            cut_negatives = [("B", "R"), ("L", "T")] if case == 5 else [("L", "B"), ("T", "R")]
            cut_positives = [("L", "B"), ("T", "R")] if case == 5 else [("B", "R"), ("L", "T")]
            pairs = cut_negatives if centre > 0.0 else cut_positives
        else:
            pairs = table[case]
        for a, b in pairs:
            adjacency.setdefault(edge[a], []).append(edge[b])
            adjacency.setdefault(edge[b], []).append(edge[a])

    def point(key):
        kind, i, j = key
        i2, j2 = (i + 1, j) if kind == "h" else (i, j + 1)
        va, vb = field[j, i], field[j2, i2]
        t = va / (va - vb) if va != vb else 0.5
        return (x0 + h * (i + t * (i2 - i)), y0 + h * (j + t * (j2 - j)))
    seen: set[tuple] = set()
    lines = []
    # open chains first (from an end), then closed loops (from the smallest key), each walked once
    for start in sorted(adjacency, key=lambda k: (len(adjacency[k]) != 1, k)):
        if start in seen:
            continue
        chain, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [k for k in adjacency[cur] if k != prev and k not in seen]
            if not nxt:
                if len(chain) > 2 and start in adjacency[cur]:
                    chain.append(start)                                  # a closed loop: the walk met its start again
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            chain.append(cur)
        lines.append([point(k) for k in chain])
    return lines


def simplify(line: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Douglas-Peucker, iterative (a closed line is cut at its two farthest-apart points first)."""
    if len(line) < 3:
        return line
    pts = np.asarray(line)
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    if np.allclose(pts[0], pts[-1]):
        far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
        keep[far] = True
        stack = [(0, far), (far, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        a, b = pts[lo], pts[hi]
        seg = b - a
        norm = float(np.hypot(*seg)) or 1.0
        dist = np.abs(seg[0] * (pts[lo + 1:hi, 1] - a[1]) - seg[1] * (pts[lo + 1:hi, 0] - a[0])) / norm
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[lo + 1 + k] = True
            stack += [(lo, lo + 1 + k), (lo + 1 + k, hi)]
    return [tuple(p) for p in pts[keep]]


_OUTLINES: dict[tuple, list[list[list[float]]]] = {}


def chamber_outline(shape: str, n: int, aspect: float | None = None, params: dict | None = None) -> list[list[list[float]]]:
    """The chamber's walls as a list of polylines ``[[x, y], ...]`` for the page to draw: the polyline chambers' own vertices; the
    SDF chambers' zero contour of the composed SDF (marching squares, ``OUTLINE_STEP`` cells fine over the window, simplified to
    ``OUTLINE_TOL``).  Deterministic; ``box`` has none."""
    p = species_params(params)
    cell = cell_needed(p)
    key = (shape, int(n), aspect, round(cell, 9))
    if key in _OUTLINES:
        return _OUTLINES[key]
    geo = chamber_basin(shape, n, aspect, cell)
    if shape in SDF_SHAPES:
        u = uniforms(n=n, aspect=aspect, params=params, shape=shape)
        x0, x1, y0, y1 = geo["view"]
        h = cell * OUTLINE_STEP
        xs = x0 + h * np.arange(int(math.ceil((x1 - x0) / h)) + 1)
        ys = y0 + h * np.arange(int(math.ceil((y1 - y0) / h)) + 1)
        gx, gy = np.meshgrid(xs, ys)
        d = chamber_distance(u, gx.ravel(), gy.ravel())[0].reshape(gx.shape)
        d = np.clip(d, -1.0, 1.0)
        lines = [[[float(round(px, 5)), float(round(py, 5))] for px, py in simplify(line, OUTLINE_TOL)]
                 for line in zero_contours(d, float(xs[0]), float(ys[0]), h)]
    elif shape == "box" or shape in BODY_SHAPES:
        lines = []
    else:
        lines = [[[float(vx), float(vy)] for vx, vy in shape_vertices(shape, geo)]]
    _OUTLINES[key] = lines
    return lines


OUTLINE_STEP = 1.0 / 3.0                 # the outline's sampling step, in grid cells
OUTLINE_TOL = 5.0e-4                     # the outline's simplification tolerance (scene units; a pixel is ~2e-3)


def wheel_geometry(geo: dict[str, float], p: dict[str, float]) -> dict[str, float]:
    """The wheel's uniforms for a basin: the hinge left of the stream (so the stream, falling at x ~ 0, strikes the right half and turns
    the wheel clockwise), reach, hub, paddles, and the reference inertia of its material (paddles as rods about the hinge, a hub disc)."""
    s = geo["scale"]
    reach = WHEEL_RADIUS * s * p["wheel_r"]
    hub = WHEEL_HUB * reach
    count = round(p["wheel_n"])
    ht = WHEEL_PADDLE_HALF
    m_paddle = WHEEL_AREAL_DENSITY * 2.0 * ht * reach
    i_ref = count * m_paddle * reach * reach / 3.0 + 0.5 * math.pi * WHEEL_AREAL_DENSITY * hub ** 4
    return {"body_x": -0.5 * reach, "body_y": 0.45 * s + 0.2 * reach, "body_rh": hub, "body_pl": reach, "body_ht": ht, "body_np": float(count),
            "body_i": i_ref * p["wheel_i"], "body_gam": p["wheel_gam"], "body_alpha": p["wheel_alpha"], "body_tau": 0.0}


def uniforms(*, n: int, mode: int = 0, use_image: float = 0.0, mix: float = 0.5, aspect: float | None = None,
             gravity: tuple[float, float] = (0.0, -G), lid: float = LID_OFF, strata: float = STRATA,
             params: dict | None = None, shape: str = "box") -> dict[str, np.ndarray]:
    """The law's scalar columns (shape ()) as float32-representable values: the run (``mode``, ``use_image``, ``mix``),
    its basin for ``n`` active ids at the screen ``aspect`` (``None``: the reference basin) and the RUNTIME INPUTS
    (``gravity`` vector, ``lid`` height), whose defaults are the plain downward pull of ``G`` and no ceiling."""
    p = species_params(params)
    cell = cell_needed(p)
    geo = chamber_basin(shape, n, aspect, cell)
    out = {"mode": mode, "use_image": use_image, "mix": mix}
    out.update({k: geo[k] for k in ("half", "nozzle_y", "bar_half", "cols", "rows")})
    wheel = shape in BODY_SHAPES
    if wheel:
        out["bar_half"] = WHEEL_NOZZLE_BAR * geo["scale"]
    out.update({"grav_x": gravity[0], "grav_y": gravity[1], "lid": lid, "strata": strata})
    out.update({"fl_rho": p["fl_rho"], "fl_mu": p["fl_mu"], "omega": stiffness_of(p), "hertz": p["hertz"], "coh": p["coh"],
                "spin_drag": p["spin_drag"], "temp": p["temp"], "k_run": min(max(round(p["k_run"]), 1), K_RUN_MAX),
                "nozzle_speed": p["nozzle_speed"], "emit_scale": 1.0 / p["rate"], "lognormal": 1.0 if p["lognormal"] > 0.5 else 0.0,
                "bi_ratio": p["bi_ratio"], "bi_frac": p["bi_frac"]})
    out.update({"cell_scale": cell / CELL, "case_omega": 0.0, "case_domega": 0.0,
                "pivot_x": 0.5 * (geo["view"][0] + geo["view"][1]), "pivot_y": 0.5 * (geo["view"][2] + geo["view"][3])})
    slots: list[dict[str, float]] = []
    spec = None
    if shape in SDF_SHAPES:
        verts = [(0.0, 0.0)] * NV                                                # no polyline: the SDF walls
        spec = sdf_preset(shape, geo)
        slots = spec["slots"]
        if spec["nozzle_y"] is not None:
            out["nozzle_y"] = spec["nozzle_y"]
        out["rows"] = max(out["rows"], math.ceil(spec["top"] / cell) + 1)         # the grid covers the whole chamber
    elif shape == "box" or shape in BODY_SHAPES:
        verts = [(0.0, 0.0)] * NV                                                # no polygon: zero-length segments, exact zeros
    else:
        verts = shape_vertices(shape, geo)
        if len(verts) > NV:
            raise ValueError(f"a chamber outline has at most {NV} vertices")
        # the inlet bar INSIDE the shape: its half-width at most 80% of the chord at the nozzle's height
        if shape in NGON_SHAPES:
            out["nozzle_y"] = 0.5 * (min(v[1] for v in verts) + max(v[1] for v in verts))    # the inlet at the middle of the polygon
        xl, xr = chord(verts, out["nozzle_y"])
        out["bar_half"] = min(geo["bar_half"], 0.8 * 0.5 * (xr - xl))
        if shape in NGON_SHAPES:
            out["rows"] = max(out["rows"], math.ceil(max(v[1] for v in verts) / cell) + 1)    # the grid covers the whole chamber
        verts = verts + [verts[-1]] * (NV - len(verts))
    out.update({f"px{k}": verts[k][0] for k in range(NV)})
    out.update({f"py{k}": verts[k][1] for k in range(NV)})
    out.update(slot_uniforms(slots))
    if wheel:
        out.update(wheel_geometry(geo, p))
    if slots:
        out["bar_half"] = min(geo["bar_half"], spec["bar"])
        out["bar_half"] = fit_inlet(out, cell)                                   # the inlet is inside the fluid
    for s, (_name, r, rho, mu, _drag) in enumerate(SPECIES):                 # multiples of the defaults: exactly 1.0 when untouched
        out.update({f"rscale{s}": p[f"r{s}"] / r, f"fscale{s}": p[f"mu{s}"] / mu if mu else 1.0, f"dscale{s}": p[f"rho{s}"] / rho,
                    f"zscale{s}": zeta_of(p[f"e{s}"]) / ZETA, f"swell{s}": p[f"swell{s}"], f"swmax{s}": p[f"swmax{s}"],
                    f"roll{s}": p[f"roll{s}"], f"irr{s}": p[f"irr{s}"]})
    return {k: _f32(np.float64(v)) for k, v in out.items()}


def initial_columns(n: int, *, mix: float = 0.5, mode: int = 0, use_image: float = 0.0, active: int | None = None,
                    aspect: float | None = None, strata: float = STRATA, params: dict | None = None,
                    shape: str = "box") -> dict[str, np.ndarray]:
    """Round zero for ``n`` ids: the state is zero, ``pid`` is the iota and the image colours are zero.
    Nothing else is per-id.  The basin is that of ``active`` ids (default ``n``) at ``aspect``."""
    zeros = np.zeros(n)
    columns = {name: zeros.copy() for name in STATE + (BODY_STATE if shape in BODY_SHAPES else ()) + ("ir", "ig", "ib")}
    columns["pid"] = np.arange(n, dtype=np.float64)
    columns.update(uniforms(n=int(active or n), mode=mode, use_image=use_image, mix=mix, aspect=aspect, strata=strata, params=params, shape=shape))
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


def make_piece(n: int, active: int | None = None, body: bool = False) -> FountainPiece:
    """A ``FountainPiece`` whose law is ``make_law(n, active)`` (``body``: the hinged-body variant, with its own argument list)."""
    class Sized(FountainPiece):
        law = staticmethod(make_law(n, active, body=body))
        argument_names = BODY_ARGUMENTS if body else ARGUMENTS
        output_names = BODY_OUTPUTS if body else OUTPUTS
    piece = Sized(n, active)
    if body:
        piece.entry += "_body"
    return piece


def run_frames(piece: LawPiece, columns: dict[str, np.ndarray], steps: int) -> dict[str, np.ndarray]:
    """Advance ``steps`` fixed steps through ``piece`` (its compiled product
    when compiled, otherwise the AbstractTensor law); returns the final state,
    the colour columns and ``rad``."""
    names = STATE + (BODY_STATE if "bth" in columns else ())
    state = {name: np.array(columns[name], dtype=np.float64) for name in names}
    extra = {}
    for _ in range(int(steps)):
        outputs = piece(*[state[n] if n in state else columns[n] for n in piece.argument_names])
        produced = dict(zip(piece.output_names, outputs))
        for name in names:
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
        self.shape = "box"                       # the chamber outline (SHAPES); part of the recorded run
        self.fluid = DEFAULT_FLUID               # the fluid's name ("custom" once its density or viscosity is set by hand)
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
                               active=self.n, aspect=self.aspect, strata=self.strata, params=self.params, shape=self.shape)
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
            if name in ("fl_rho", "fl_mu"):
                self.fluid = "custom"
            if restart_param(name):          # the radius sets the grid, the size distributions and the rate what an id is: a new run (the others are runtime inputs)
                if payload.get("aspect"):
                    self.aspect = float(payload["aspect"])
                self.reset()
        elif kind == "fountain_choice":
            name, value = str(payload["name"]), str(payload["value"])
            if name == "shape" and value in dict(CHAMBER_CHOICES):
                self.shape = value
                self.reset()
            elif name == "fluid" and value in FLUIDS:
                self.fluid = value
                self.params["fl_rho"], self.params["fl_mu"] = FLUIDS[value]
            else:
                raise ValueError(f"fountain_choice: unknown choice {name!r} = {value!r}")
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
            self.piece = make_piece(pad, active, body=self.shape in BODY_SHAPES)
            self.piece.compile(columns)
        final = run_frames(self.piece, columns, steps)
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(final[n], dtype=np.float32).tobytes()
                                         for n in STATE + ("cr", "cg", "cb"))).hexdigest()[:16]
        emitted = int((columns["pid"][:active] * (EMIT_SPAN / active) * float(columns["emit_scale"]) <= steps * DT).sum())
        self.verified = {"steps": steps, "lane": self.piece.lane, "digest": digest,
                         "emitted": emitted, "top": float(final["y"].max())}
        return self.verified

    def _control(self, name: str, kind: str = "range", **extra) -> dict[str, Any]:
        """One tunable of ``PARAMS`` as a page control (``fountain_param``), in its group; a restart one carries ``aspect``."""
        label, lo, hi, step, _default, restart, group = PARAMS[name]
        spec = {"id": name, "kind": kind, "label": label, "min": lo, "max": hi, "step": step, "value": species_params(self.params)[name],
                "action": "fountain_param", "name": name, "field": "value", "group": group}
        if restart:
            spec["aspect"] = True
        spec.update(extra)
        return spec

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
            out.append(ctl("roll", "roughness", ROLL_RANGE[0], ROLL_RANGE[1], 0.01))
            out.append(ctl("irr", "irregularity", IRR_RANGE[0], IRR_RANGE[1], 0.01, {"aspect": True}))     # a restart: it sets the radius an id has
        out.append({"id": "lognormal", "kind": "select", "label": "size distribution", "value": int(p["lognormal"] > 0.5),
                    "action": "fountain_param", "name": "lognormal", "field": "value", "group": "Grains", "aspect": True,
                    "options": [[0, "uniform width"], [1, "lognormal sigma"]]})
        for name in ("bi_ratio", "bi_frac"):                       # Grains: the optional two-size (bidisperse) mode
            out.append(self._control(name))
        for name in ("omega", "hertz", "coh", "spin_drag", "temp", "nozzle_speed", "rate", "k_run"):       # Physics
            # the temperature is a logarithmic slider (0 = off, then kT from TEMP_MIN_LOG to TEMP_MAX)
            out.append(self._control(name, scale="log", log_min=TEMP_MIN_LOG) if name == "temp" else self._control(name))
        out.append({"id": "k_run_drop", "kind": "readout", "label": "dropped contacts", "group": "Physics", "diag": "n_drop"})
        out.append({"id": "fluid", "kind": "select", "label": "fluid", "value": self.fluid, "action": "fountain_choice",
                    "name": "fluid", "field": "value", "group": "Fluid",
                    "options": [[k, k] for k in FLUIDS] + ([["custom", "custom"]] if self.fluid == "custom" else [])})
        out.extend([self._control("fl_rho"), self._control("fl_mu")])
        out.append({"id": "shape", "kind": "select", "label": "chamber shape", "value": self.shape, "action": "fountain_choice",
                    "name": "shape", "field": "value", "group": "Chamber", "options": [[k, label] for k, label in CHAMBER_CHOICES]})
        if self.shape in BODY_SHAPES:                            # the hinged wheel: paddles / radius (a new run), inertia, damping, the motor
            out.extend(self._control(name) for name in ("wheel_n", "wheel_r", "wheel_i", "wheel_gam", "wheel_alpha"))
            out.append({"id": "wheel_om", "kind": "readout", "label": "wheel speed (rad/s)", "group": "Wheel", "diag": "body_om"})
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
        return chamber_basin(self.shape, self.n, self.aspect, cell_needed(species_params(self.params)))

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
                "manifest": "fountain_wheel.manifest.json" if self.shape in BODY_SHAPES else "fountain.manifest.json", "size": self.size, "epoch": self.epoch,
                "uniforms": {k: float(v) for k, v in uniforms(n=self.n, mode=self.mode, mix=self.mix, aspect=self.aspect, strata=self.strata,
                                                              use_image=1.0 if self.phase == "replay" else 0.0, params=self.params, shape=self.shape).items()},
                "view": list(geo["view"]), "aspect": self.aspect, "chamber_outline": chamber_outline(self.shape, self.n, self.aspect, self.params),
                "body": self.body_spec(), "speed": self.speed, "named": ["ir", "ig", "ib"], "run": self.run,
                "tilt": tilt_inputs(self.n, self.aspect, cell_needed(species_params(self.params)), self.shape),
            },
        })
        return base

    def body_spec(self) -> dict[str, Any] | None:
        """The hinged body for the page to draw (its SDF outline at the angle the law reports in ``diag.body_th``) and to couple
        (``body_tau`` is the runtime input a machine-sim coupling writes; ``body_om`` is read back from the diagnostics)."""
        if self.shape not in BODY_SHAPES:
            return None
        u = uniforms(n=self.n, aspect=self.aspect, params=self.params, shape=self.shape)
        return {"hinge": [float(u["body_x"]), float(u["body_y"])], "hub": float(u["body_rh"]), "reach": float(u["body_pl"]),
                "half_thickness": float(u["body_ht"]), "paddles": int(round(float(u["body_np"]))), "angle": "body_th",
                "speed": "body_om", "torque_input": "body_tau"}

    def result(self):
        return None
# <purrline>
# ------------------------------------------------------------------------------------------------------------------------
# PURRLINE SAND (WORKLIST.md, "Purrline sand mode", C): a PRODUCT VARIANT of the law, ``make_law_purrline``, with its own argument list,
# manifest (``fountain_purrline.manifest.json``) and shaders.  Everything of it lives between ``# <purrline>`` markers, which
# ``fountain_build`` leaves out of the source digest of the DEFAULT and wheel products: they are the same law, the same shaders and the same
# digests as before this variant existed.  The variant adds two solids to the law, as runtime uniforms:
#   OCCUPANCY GRID (``og_*``): rows x cols (at most 8 x 8) solid square cells of side ``og_cell`` centred on ``(og_cx, og_cy)``, from a per-cell mask
#   packed into the three words ``og_w0..og_w2`` (row r from the TOP is byte r % 3 of word r // 3, column c is bit c of the byte; each word is
#   below 2**24, exact in f32).  Its distance is the nearest occupied cell's (domain repetition + a 3 x 3 check), exact on faces, with the
#   analytic gradient; INSIDE an occupied cell it is the distance to the nearest face that borders a FREE cell, and the gradient points out
#   through it, so a grain caught in a cell that became solid leaves through free space, never into an occupied neighbour.
#   MOVING SOLID BOXES (``mb<k>_*``, six): centre ``cx cy``, half extents ``hw hh``, corner radius ``rc`` and pose velocity ``vx vy``.  A contact of
#   the wall law against a SURFACE THAT MOVES (the waterwheel's contact: relative velocity = the grain's less the box's), infinite mass, position
#   driven: the law holds no state for a box, so nothing a grain does can slow it.
# Both are far/zero when unused (``og_on`` 0, ``mb<k>_on`` 0).  The grains' speed relative to the solid they touch is bounded by the contact's own
# saturation (omega x 6 radii) so a squeeze cannot fire them out of the chamber.
MB_COUNT = 6
OG_MAX = 8
PURRLINE_ROWS, PURRLINE_COLS = 8, 8
PURRLINE_SUMP, PURRLINE_HEAD, PURRLINE_INLET = 2.0, 3.0, 1.8       # the chamber in board cells: the sump under the board, the headroom over it, the inlet over the board
PURRLINE_BOX_ROUND = 0.12                                          # a falling item's corner radius, in cells
PURRLINE_JET = 1.6                                                 # the clear height over the inlet (the jet's apex is ~1.34 above it; absolute, in fountain units)
PURRLINE_POUR = {"nozzle_speed": 0.3, "rate": 0.5}                      # the inlet POURS sand (the slowest nozzle, half the rate), it is not the fountain's jet: the 64k jet's blast is what fired grains through the board
PURRLINE_PUSH = 1.0                                                # a moving box's overlap saturates at this many radii (the wall law's own saturation is 6)
MB_FIELDS = ("on", "cx", "cy", "hw", "hh", "rc", "vx", "vy")
OG_FIELDS = ("on", "cx", "cy", "cell", "cols", "rows", "w0", "w1", "w2")
PURRLINE_UNIFORMS = tuple(f"og_{f}" for f in OG_FIELDS) + tuple(f"mb{k}_{f}" for k in range(MB_COUNT) for f in MB_FIELDS)
PURRLINE_ARGUMENTS = ARGUMENTS + PURRLINE_UNIFORMS
PURRLINE_ALL_UNIFORMS = UNIFORMS + PURRLINE_UNIFORMS
PURRLINE_RUNTIME_INPUTS = RUNTIME_INPUTS + PURRLINE_UNIFORMS
SDF_SHAPES = SDF_SHAPES + ("sdf_purrline",)
_PURRLINE = {"on": False}


def _cell_box(ex, ey, half):
    """A square cell of half side ``half`` at offset ``(ex, ey)``: the signed distance and its analytic gradient.  EXACT on the faces (no epsilon in
    the length; the gradient's guard is on the divisor only)."""
    qx, qy = ex.abs() - half, ey.abs() - half
    ox, oy = qx.clamp(0.0, 1.0e9), qy.clamp(0.0, 1.0e9)
    out = as_float((ox + oy) > 0.0)
    ol = (ox * ox + oy * oy).sqrt()
    inv = 1.0 / (ol + (1.0 - out))                    # out = 0: the divisor is 1 (and the numerators are 0)
    xdom = as_float(qx > qy)
    f = ol + _min(_max(qx, qy), 0.0)
    return f, (out * (ox * inv) + (1.0 - out) * xdom) * _sgn(ex), (out * (oy * inv) + (1.0 - out) * (1.0 - xdom)) * _sgn(ey)


def _bit_scale(index, count):
    """``2 ** -index`` for a whole-number tensor ``index`` in ``[0, count)``, exactly, as a one-hot sum of constants (no pow / exp2)."""
    out = index * 0.0
    for k in range(count):
        out = out + as_float((index - float(k)).abs() < 0.5) * (2.0 ** -k)
    return out


def _sdf_occgrid(lx, ly, cell, cols, rows, w0, w1, w2):
    """The OCCUPANCY GRID at the point ``(lx, ly)`` (relative to the grid's centre): ``(f, gx, gy)``, the signed distance to the NEAREST OCCUPIED CELL
    (negative inside) and its analytic gradient.  The grain's own cell and the 3 x 3 around it are checked (every cell nearer than one cell is among
    them), each by a bit read out of the packed words with exact power-of-two scalings and one-hot picks; a cell off the grid or empty is a SELECTED
    far distance (never a blend).  Inside an occupied cell: the distance to the nearest face that borders a free cell (or the grid's edge) and the
    gradient out through it; with no such face (sealed) the distance is a cell deeper and the gradient points up."""
    c = cell.clamp(1.0e-9, 1.0e9)
    inv = 1.0 / c
    i0 = ((lx + 0.5 * cols * c) * inv).floor()
    j0 = ((0.5 * rows * c - ly) * inv).floor()
    best, bgx, bgy = None, None, None
    occs = []
    for dj in (-1.0, 0.0, 1.0):
        j = j0 + dj
        row_ok = as_float(j > -0.5) * as_float(j < rows - 0.5)
        jc = j.clamp(0.0, float(OG_MAX - 1))
        g1, g2 = as_float(jc > 2.5), as_float(jc > 5.5)
        word = w0 * (1.0 - g1) + w1 * (g1 * (1.0 - g2)) + w2 * g2           # exact: 0/1 products of whole numbers below 2**24
        r3 = jc - 3.0 * (g1 + g2)
        shifted = (word * (as_float(r3 < 0.5) + as_float((r3 - 1.0).abs() < 0.5) * (1.0 / 256.0) + as_float(r3 > 1.5) * (1.0 / 65536.0))).floor()
        rowbits = shifted - 256.0 * (shifted * (1.0 / 256.0)).floor()        # this row's 8 bits (a whole number below 256)
        cy_l = 0.5 * rows * c - (j + 0.5) * c
        for di in (-1.0, 0.0, 1.0):
            i = i0 + di
            ok = row_ok * as_float(i > -0.5) * as_float(i < cols - 0.5)
            ic = i.clamp(0.0, float(OG_MAX - 1))
            t = (rowbits * _bit_scale(ic, OG_MAX)).floor()
            occ = ok * (t - 2.0 * (t * 0.5).floor())
            occs.append(occ)
            cx_l = (i + 0.5) * c - 0.5 * cols * c
            f, gx, gy = _cell_box(lx - cx_l, ly - cy_l, 0.5 * c)
            f = occ * f + (1.0 - occ) * SDF_FAR
            if best is None:
                best, bgx, bgy = f, gx, gy
            else:
                take = f < best
                best = AbstractTensor.where(take, f, best)
                bgx = AbstractTensor.where(take, gx, bgx)
                bgy = AbstractTensor.where(take, gy, bgy)
    own = occs[4]
    u = lx - ((i0 + 0.5) * c - 0.5 * cols * c)
    v = ly - (0.5 * rows * c - (j0 + 0.5) * c)
    h = 0.5 * c
    open_l, open_r, open_t, open_b = 1.0 - occs[3], 1.0 - occs[5], 1.0 - occs[1], 1.0 - occs[7]
    far = 1.0e8
    dl, dr, dt, db = h + u + (1.0 - open_l) * far, h - u + (1.0 - open_r) * far, h - v + (1.0 - open_t) * far, h + v + (1.0 - open_b) * far
    m = _min(_min(dl, dr), _min(dt, db))
    sealed = as_float(m > far * 0.5)
    ix = as_float(dr <= dl) * 2.0 - 1.0
    xnear = as_float(_min(dl, dr) <= _min(dt, db))
    iy = as_float(dt <= db) * 2.0 - 1.0                    # +1: out the top
    in_d = -m * (1.0 - sealed) + sealed * (best - c)
    in_gx = (1.0 - sealed) * xnear * ix
    in_gy = (1.0 - sealed) * (1.0 - xnear) * iy + sealed
    pick = as_float(own > 0.5) > 0.5
    return (AbstractTensor.where(pick, in_d, best), AbstractTensor.where(pick, in_gx, bgx), AbstractTensor.where(pick, in_gy, bgy))


def moving_field(x, y, q, zero):
    """The MOVING SOLID BOXES at ``(x, y)`` (``q(k, field)`` gives box ``k``'s ``MB_FIELDS``): the signed distance to the nearest used box (``SDF_FAR``
    with none), the analytic gradient there (unit, out of the box) and that box's surface velocity.  Returns ``(d, nx, ny, svx, svy)``."""
    d, nx, ny, svx, svy = zero + SDF_FAR, zero, zero, zero, zero
    for k in range(MB_COUNT):
        on, cx, cy, hw, hh, rc, bvx, bvy = (q(k, f) for f in MB_FIELDS)
        f, gx, gy = _sdf_box(x - cx, y - cy, hw, hh, rc)
        used = as_float(on > 0.5)
        cs = used * f + (1.0 - used) * SDF_FAR
        take = cs < d
        d = AbstractTensor.where(take, cs, d)
        nx = AbstractTensor.where(take, gx, nx)
        ny = AbstractTensor.where(take, gy, ny)
        svx = AbstractTensor.where(take, bvx + zero, svx)
        svy = AbstractTensor.where(take, bvy + zero, svy)
    return d, nx, ny, svx, svy


def grid_distance(u: dict, xs, ys):
    """The law's own occupancy-grid distance (the same code on the NumPy backend) at points: ``(d, gx, gy)``.  Host side: tests."""
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    q = lambda name: AbstractTensor.get_tensor(np.asarray(float(_f32(np.float64(u[name])))))
    out = _sdf_occgrid(AbstractTensor.get_tensor(xs) - q("og_cx"), AbstractTensor.get_tensor(ys) - q("og_cy"), q("og_cell"), q("og_cols"), q("og_rows"),
                       q("og_w0"), q("og_w1"), q("og_w2"))
    return tuple(np.asarray(v.numpy(), dtype=np.float64) for v in out)


_sdf_preset_base = sdf_preset
_chamber_outline_base = chamber_outline


def sdf_preset(shape: str, geo: dict[str, float]) -> dict[str, Any]:
    """``sdf_preset`` plus the PURRLINE SAND chamber (WORKLIST.md C): a box chamber exactly as wide as the board, so sand can only pass THROUGH it; bottom
    up: the SUMP (``PURRLINE_SUMP`` cells), the board (8 x 8 cells; the occupancy grid, not a slot) and the headroom with the inlet ``PURRLINE_INLET``
    cells above the board.  One board cell is ``2 w / 8`` wide.  ``map``: the board's centre in fountain units and one cell's width."""
    if shape != "sdf_purrline":
        return _sdf_preset_base(shape, geo)
    c = geo.get("cell", CELL)
    w = min(0.5 * (geo["view"][1] - geo["view"][0]) - 3.0 * c, geo["half"])
    cell_f = 2.0 * w / PURRLINE_COLS
    nozzle_y = (PURRLINE_SUMP + PURRLINE_ROWS + PURRLINE_INLET) * cell_f
    top = max((PURRLINE_SUMP + PURRLINE_ROWS + PURRLINE_HEAD) * cell_f, nozzle_y + PURRLINE_JET)      # the ceiling clears the jet's apex (an ABSOLUTE height: it does not scale with the board)
    cy = (PURRLINE_SUMP + 0.5 * PURRLINE_ROWS) * cell_f
    return {"slots": [_slot("box", "fluid", 0.0, 0.5 * top, 0.0, w, 0.5 * top, 2.0 * c)], "nozzle_y": nozzle_y,
            "top": top, "half": w, "bar": 0.7 * w,
            "map": {"cx": 0.0, "cy": cy, "cell": cell_f, "rows": PURRLINE_ROWS, "cols": PURRLINE_COLS, "sump": PURRLINE_SUMP * cell_f}}


def chamber_outline(shape: str, n: int, aspect: float | None = None, params: dict | None = None) -> list[list[list[float]]]:
    """``chamber_outline``; the purrline chamber's is its box (the board's cells are the page's own board)."""
    if shape != "sdf_purrline":
        return _chamber_outline_base(shape, n, aspect, params)
    spec = sdf_preset(shape, chamber_basin(shape, n, aspect, cell_needed(species_params(params))))
    wd, tp = spec["half"], spec["top"]
    return [[[-wd, 0.0], [wd, 0.0], [wd, tp], [-wd, tp], [-wd, 0.0]]]


def pack_mask(mask) -> tuple[float, float, float]:
    """The 8 x 8 mask (64 values, row-major, row 0 at the TOP) as the three words of the occupancy grid: row ``r`` is the byte ``r mod 3`` of word
    ``r // 3`` (rows 0-2 word 0, 3-5 word 1, 6-7 word 2; each word below 2**24)."""
    bits = list(mask)
    if len(bits) != PURRLINE_ROWS * PURRLINE_COLS:
        raise ValueError("pack_mask: give the 8 x 8 mask (64 values)")
    words = [0, 0, 0]
    for r in range(PURRLINE_ROWS):
        for c in range(PURRLINE_COLS):
            if bits[r * PURRLINE_COLS + c]:
                words[r // 3] |= 1 << (8 * (r % 3) + c)
    return float(words[0]), float(words[1]), float(words[2])


def purrline_inputs(game: dict, mapping: dict[str, float]) -> dict[str, float]:
    """THE PAGE'S ONE-WAY WRITE: the purrline game's state (``state['purrline']``: ``occupancy``, ``rows``, ``cols``, ``cell``, ``origin``, ``falling``)
    as the variant's runtime inputs.  ``mapping`` is the chamber's ``sdf_preset(..)['map']`` (the board's centre ``cx, cy`` in fountain units and one
    board cell's width ``cell``).  Board units (x right, y DOWN, a cell ``cell`` wide, the top-left corner ``origin``) map to fountain units (y UP) by
    a scale ``k = mapping.cell / game.cell`` about the board's centre, y flipped.  The occupied cells fill the grid's mask; the ``MB_COUNT`` lowest
    falling items are the moving boxes, each the size of the cell it will fill, centred on the item, carrying its velocity (``vx, -vy`` x k).  Unused
    boxes are written off.  ``web/compute.js`` ``purrlineInputs`` is this function."""
    cb = float(game.get("cell", 1.0))
    ox, oy = (float(v) for v in game.get("origin", (0.0, 0.0)))
    rows, cols = int(game["rows"]), int(game["cols"])
    k = float(mapping["cell"]) / cb
    bcx, bcy = ox + 0.5 * cols * cb, oy + 0.5 * rows * cb
    mask = [0] * (PURRLINE_ROWS * PURRLINE_COLS)
    for r in range(rows):
        for c in range(cols):
            mask[PURRLINE_COLS * r + c] = 1 if game["occupancy"][r * cols + c] else 0
    w0, w1, w2 = pack_mask(mask)
    out = {"og_on": 1.0, "og_cx": float(mapping["cx"]), "og_cy": float(mapping["cy"]), "og_cell": float(mapping["cell"]), "og_cols": float(cols),
           "og_rows": float(rows), "og_w0": w0, "og_w1": w1, "og_w2": w2}
    items = sorted(game.get("falling", ()), key=lambda it: -float(it["y"]))[:MB_COUNT]
    half = 0.5 * float(mapping["cell"])
    for slot in range(MB_COUNT):
        if slot < len(items):
            it = items[slot]
            vals = (1.0, float(mapping["cx"]) + (float(it["x"]) - bcx) * k, float(mapping["cy"]) - (float(it["y"]) - bcy) * k, half, half,
                    PURRLINE_BOX_ROUND * float(mapping["cell"]), float(it.get("vx", 0.0)) * k, -float(it.get("vy", 0.0)) * k)
        else:
            vals = (0.0,) * len(MB_FIELDS)
        out.update({f"mb{slot}_{n}": v for n, v in zip(MB_FIELDS, vals)})
    return {name: float(_f32(np.float64(v))) for name, v in out.items()}


def purrline_defaults(mapping: dict[str, float]) -> dict[str, float]:
    """The variant's uniforms for a FULL board and no falling item."""
    return purrline_inputs({"rows": PURRLINE_ROWS, "cols": PURRLINE_COLS, "cell": 1.0, "origin": [0.0, 0.0], "occupancy": [1] * 64, "falling": []}, mapping)


def purrline_columns(n: int, *, active: int | None = None, aspect: float | None = None, params: dict | None = None, **kw) -> dict[str, np.ndarray]:
    """``initial_columns`` for the purrline chamber plus the variant's uniforms (a full board)."""
    cols = initial_columns(n, active=active, aspect=aspect, params=params, shape="sdf_purrline", **kw)
    mapping = sdf_preset("sdf_purrline", chamber_basin("sdf_purrline", int(n if active is None else active), aspect, cell_needed(species_params(params))))["map"]
    cols.update({k: np.float64(v) for k, v in purrline_defaults(mapping).items()})
    return cols


def make_law_purrline(n: int, active: int | None = None):
    """The fountain law with the occupancy grid and the moving boxes (arguments ``PURRLINE_ARGUMENTS``; outputs ``OUTPUTS``).  The same law as
    ``make_law``: the hooks between the ``purrline`` markers in ``fountain_step`` run only while this wrapper's call is traced or evaluated."""
    law = make_law(n, active)

    def purrline_step(*args):
        _PURRLINE["on"] = True
        try:
            return law(*args)
        finally:
            _PURRLINE["on"] = False
    return purrline_step


def purrline_spec(size: str = "4k", aspect: float | None = None, params: dict | None = None) -> dict[str, Any]:
    """What a page needs to run the purrline sand chamber at ``size``: the uniforms (the variant's, a full board, no moving boxes), the view, the
    chamber outline, the tilt inputs and the board map."""
    active = SIZES[size][0]
    params = {**PURRLINE_POUR, **(params or {})}
    p = species_params(params)
    geo = chamber_basin("sdf_purrline", active, aspect, cell_needed(p))
    mapping = sdf_preset("sdf_purrline", geo)["map"]
    u = uniforms(n=active, aspect=aspect, params=params, shape="sdf_purrline")
    return {"shape": "sdf_purrline", "size": size, "uniforms": {**{k: float(v) for k, v in u.items()}, **purrline_defaults(mapping)}, "view": list(geo["view"]),
            "chamber_outline": chamber_outline("sdf_purrline", active, aspect, params),
            "tilt": tilt_inputs(active, aspect, cell_needed(p), "sdf_purrline"), "map": mapping}
# </purrline>
