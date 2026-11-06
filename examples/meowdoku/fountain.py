"""Fountain: a soft-sphere (DEM) granular fountain that fills a basin, takes an image, fills it again.

ONE batched per-step law (``fountain_step``), authored once over spans of
particle ids in the repository's AbstractTensor style, compiled by the
repository to native, wasm and (generated) WebGPU compute.

PHYSICS
-------
* ``N`` disks (ids ``0..N-1``) of four SPECIES that differ in radius, density
  (so mass), friction and drag.  Contacts are SOFT SPHERES: a normal
  spring-dashpot (stiffness proportional to the pair's reduced mass, damping
  from a damping ratio) plus tangential Coulomb friction (regularised with a
  tanh of the contact's slip speed) acting on the slip of the contact point,
  spin included, so disks roll.  The basin is a floor and two side walls that
  are contacts of the same law (a wall is a disk of infinite mass).  Gravity
  and linear air drag act on every particle.
* Contacts WRITE FORCES (and torques); the integrator (semi-implicit Euler,
  fixed ``DT``) is the only thing that moves anything.  Each particle sums
  the force its neighbours put on IT; both sides of a pair evaluate the same
  symmetric expressions, so they are equal and opposite without any
  per-pair bookkeeping, impulse or position correction.
* The FOUNTAIN is an emitter: every id has a deterministic emit time, nozzle
  offset and launch velocity (``layout``).  Until its age reaches its emit time
  a particle is parked, takes no part in contacts and is not drawn; at that
  step it receives its launch velocity.  There are no targets and no slots:
  the pile forms and settles by contact forces, so regular and irregular
  packing emerge from the species mix.

NEIGHBOUR SEARCH (in the law, from tensor ops; deterministic)
-------------------------------------------------------------
A uniform grid of cells of the largest diameter.  Each step: cell id per
particle (``cell * N + id`` as ONE key; parked particles get the cell past the
grid), a stable BITONIC SORT of the keys as a gather + compare-exchange
network (``log2(N)(log2(N)+1)/2`` stages; keys are unique, so the result is
unique and no run-to-run ordering can differ), then for each of the three
rows of the 3x3 neighbourhood the contiguous run of sorted keys covering its
three cells is found by two binary searches (gathers).  The contact loop walks
the first ``K`` entries of each run.  No atomics and no scatter anywhere, so
the floating-point result does not depend on thread scheduling.

COLOUR
------
Phase 1: hue from the ANGLE of the id's launch vector, as a trig ramp
(``mode 0``: ``atan2``) or as a rotation of the chroma vector about the grey
axis with cos/sin read off the vector (``mode 1``, no ``atan2``).  The image
step samples the picked image at every id's ACTUAL rest position (read from
the settled pile) into per-id colours ``ir ig ib``; phase 2 replays the
identical simulation (same initial columns, same fixed ``DT``) with
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
N_FULL = 4096                          # a power of two: the sort is a bitonic network
N_SMALL = 256
WIDTH_FULL = 2.0                       # basin width for N_FULL; others scale with sqrt(N)
FLOOR_Y = 0.0
NOZZLE = (0.0, 1.25)
VIEW = (-1.05, 1.05, -0.05, 2.6)       # x0 x1 y0 y1 of the page's window
EMIT_SPAN = 6.0                        # seconds over which ids are emitted
SETTLE_SECONDS = 14.0                  # sim time by which the pile has come to rest
DT = 1.0 / 360.0
G = 1.0
OMEGA = 150.0                          # contact stiffness: kn = m * OMEGA^2 (OMEGA * DT stays well under 1: a grain feels several contacts)
ZETA = 0.7                             # normal damping ratio
VREF = 0.02                            # tangential slip speed over which friction saturates
SPIN_DRAG = 3.0                        # 1/s
GRID_ROWS = 56
K_RUN = 10                             # candidates walked per row run

#: species: name, radius, density, friction, linear drag (1/s)
SPECIES = (
    ("grain", 0.0090, 1.0, 0.35, 0.10),
    ("bead",  0.0105, 1.6, 0.20, 0.06),
    ("dust",  0.0120, 0.7, 0.55, 0.25),
    ("pearl", 0.0140, 2.2, 0.10, 0.04),
)
SPECIES_WEIGHT = (0.35, 0.30, 0.20, 0.15)
CELL = 2.0 * max(s[1] for s in SPECIES)

STATE = ("x", "y", "vx", "vy", "w", "age")
CONSTANTS = ("pid", "t_emit", "vx0", "vy0", "radius", "mass", "mu", "drag", "ir", "ig", "ib")
UNIFORMS = ("mode", "use_image")
OUTPUTS = ("x_next", "y_next", "vx_next", "vy_next", "w_next", "age_next", "cr", "cg", "cb")
ARGUMENTS = STATE + CONSTANTS + UNIFORMS
#: The columns the layout endpoint sends the page (float32, in this order).
LAYOUT_COLUMNS = ("x", "y", "vx", "vy", "t_emit", "vx0", "vy0", "pid", "radius", "mass", "mu", "drag", "species")

TAU = 2.0 * math.pi


def basin(n: int) -> dict[str, float]:
    """Geometry for ``n`` particles: basin width and the grid over it."""
    width = WIDTH_FULL * math.sqrt(n / N_FULL)
    cols = int(math.ceil(width / CELL))
    return {"width": width, "half": width / 2.0, "cols": cols, "rows": GRID_ROWS,
            "cells": cols * GRID_ROWS}


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
    (fx, fy, torque_arm_force) where the third value is the tangential force
    (torque = -radius * it)."""
    vn = vrx * nx + vry * ny
    cn = 2.0 * ZETA * (kn * m_eff).sqrt()
    fn = (kn * delta - cn * vn).clamp(0.0, 1.0e9)
    tx, ty = -ny, nx
    vt = vrx * tx + vry * ty - spin_slip
    ft = -(mu * fn) * (vt * (1.0 / VREF)).tanh()
    return fn * nx + ft * tx, fn * ny + ft * ty, ft


def make_law(n: int):
    """The fountain law for ``n`` particles (a power of two)."""
    geo = basin(n)
    half, cols, rows, cells = geo["half"], geo["cols"], geo["rows"], geo["cells"]
    stages = []
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            stages.append((k, j))
            j //= 2
        k *= 2
    search_steps = int(math.ceil(math.log2(n + 1))) + 1

    def fountain_step(x, y, vx, vy, w, age, pid, t_emit, vx0, vy0, radius, mass, mu, drag, ir, ig, ib,
                      mode, use_image):
        """One fixed step of every particle (see the module docstring)."""
        active = as_float(age >= t_emit)
        just = active * as_float((age - DT) < t_emit)
        # ---- grid cell, one sortable key per particle
        cx = ((x + half) * (1.0 / CELL)).floor().clamp(0.0, cols - 1.0)
        cy = ((y - FLOOR_Y) * (1.0 / CELL)).floor().clamp(0.0, rows - 1.0)
        # only particles inside the grid take part: parked ones, and the dilute spray above the
        # grid's top, get the cell past the end (sorted last, in nobody's neighbourhood)
        gridded = active * as_float(y < rows * CELL + FLOOR_Y)
        cell = AbstractTensor.where(gridded > 0.5, cy * float(cols) + cx, cy * 0.0 + float(cells))
        key = cell * float(n) + pid
        # ---- stable bitonic sort of the keys: gathers and compare-exchanges
        for (kk, jj) in stages:
            bit = (pid * (1.0 / jj)).floor() - 2.0 * (pid * (1.0 / (2 * jj))).floor()
            up = 1.0 - ((pid * (1.0 / kk)).floor() - 2.0 * (pid * (1.0 / (2 * kk))).floor())
            partner = pid + jj * (1.0 - 2.0 * bit)
            other = key.index_select(0, partner)
            keep_min = AbstractTensor.where(bit < 0.5, up, 1.0 - up)
            lo_key = key.minimum(other)
            hi_key = key.maximum(other)
            key = AbstractTensor.where(keep_min > 0.5, lo_key, hi_key)
        # ---- the run of sorted keys covering each of the three rows of the 3x3 block
        fx = drag * 0.0
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
            for target in (c_lo * float(n), (c_hi + 1.0) * float(n)):
                lo = cx * 0.0
                hi = cx * 0.0 + float(n)
                for _ in range(search_steps):
                    mid = ((lo + hi) * 0.5).floor()
                    probe = key.index_select(0, mid.clamp(0.0, top))
                    live = as_float(lo < hi)
                    less = as_float(probe < target)
                    lo = AbstractTensor.where(live * less > 0.5, mid + 1.0, lo)
                    hi = AbstractTensor.where(live * (1.0 - less) > 0.5, mid, hi)
                ends.append(lo)
            runs.append((ends[0], ends[1], ok))
        # ---- particle-particle contacts: forces from the neighbours' state
        for lo, hi, ok in runs:
            for slot in range(K_RUN):
                idx = lo + float(slot)
                valid = as_float(idx < hi) * ok * gridded
                kj = key.index_select(0, idx.clamp(0.0, top))
                j = kj - (kj * (1.0 / n)).floor() * float(n)
                valid = valid * as_float((j - pid).abs() > 0.5)
                xj = x.index_select(0, j)
                yj = y.index_select(0, j)
                vxj = vx.index_select(0, j)
                vyj = vy.index_select(0, j)
                wj = w.index_select(0, j)
                rj = radius.index_select(0, j)
                mj = mass.index_select(0, j)
                muj = mu.index_select(0, j)
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
        floor_delta = (radius - (y - FLOOR_Y)).clamp(0.0, 1.0e9) * active
        gx, gy, gt = _contact(floor_delta, y * 0.0, y * 0.0 + 1.0, vx, vy, radius * w, kw, mass, mu)
        touch = as_float(floor_delta > 0.0)
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        left_delta = (radius - (x + half)).clamp(0.0, 1.0e9) * active
        gx, gy, gt = _contact(left_delta, y * 0.0 + 1.0, y * 0.0, vx, vy, radius * w, kw, mass, mu)
        touch = as_float(left_delta > 0.0)
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        right_delta = (radius - (half - x)).clamp(0.0, 1.0e9) * active
        gx, gy, gt = _contact(right_delta, y * 0.0 - 1.0, y * 0.0, vx, vy, radius * w, kw, mass, mu)
        touch = as_float(right_delta > 0.0)
        fx = fx + gx * touch
        fy = fy + gy * touch
        tq = tq - radius * gt * touch
        # ---- the integrator: gravity, drag, the forces above; emission gives the launch velocity
        inertia = 0.5 * mass * radius * radius
        vxn = AbstractTensor.where(just > 0.5, vx0, vx + active * (fx / mass - drag * vx) * DT)
        vyn = AbstractTensor.where(just > 0.5, vy0, vy + active * (fy / mass - G - drag * vy) * DT)
        wn = (w + active * (tq / inertia - SPIN_DRAG * w) * DT) * (1.0 - just)
        xn = x + active * vxn * DT
        yn = y + active * vyn * DT
        age_next = age + DT
        # ---- colour from the angle of the launch vector
        speed = (vx0 * vx0 + vy0 * vy0 + 1.0e-12).sqrt()
        # theta: the launch angle from the vertical (+-0.7 rad).  The hue turns four times as
        # fast as theta so the fan spans most of the wheel: trig = atan2, rotation = the
        # double-angle formulas applied twice to cos/sin read off the vector (no atan2).
        cth = vy0 / speed
        sth = vx0 / speed
        c2 = cth * cth - sth * sth
        s2 = 2.0 * cth * sth
        c = c2 * c2 - s2 * s2
        s = 2.0 * c2 * s2
        hue = vx0.atan2(vy0) * (4.0 / TAU) + 0.5
        root3 = 0.5773502691896258
        rot_r = (c * (2.0 / 3.0)) * 0.85 + 0.5
        rot_g = (c * (-1.0 / 3.0) + s * root3) * 0.85 + 0.5
        rot_b = (c * (-1.0 / 3.0) - s * root3) * 0.85 + 0.5
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
        return xn, yn, vxn, vyn, wn, age_next, cr, cg, cb

    return fountain_step


# ------------------------------------------------------------------- layout
def _f32(a) -> np.ndarray:
    """Rounded through float32: the page's buffers and the reference see one value."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


MONO_WEIGHT = (0.6, 0.4, 0.0, 0.0)       # mix 0: near-monodisperse grains and beads (a crystal wants to form)
POLY_WEIGHT = (0.15, 0.15, 0.35, 0.35)   # mix 1: all four species, widely polydisperse


def layout(n: int = N_FULL, *, seed: int = 7, mix: float = 0.5) -> dict[str, np.ndarray]:
    """The round-zero columns for ``n`` particles, all values exactly float32-
    representable.  Everything is bulk array work over ids.  ``mix`` blends the
    species weights from near-monodisperse to polydisperse: it sets only WHO is in
    the pile; where each one ends up is the physics' doing."""
    rng = np.random.default_rng(seed)
    weight = (1.0 - mix) * np.asarray(MONO_WEIGHT) + mix * np.asarray(POLY_WEIGHT)
    species = np.searchsorted(np.cumsum(weight) / weight.sum(), rng.random(n), side="right").clip(0, 3)
    radius = np.asarray([s[1] for s in SPECIES])[species]
    density = np.asarray([s[2] for s in SPECIES])[species]
    mass = density * math.pi * radius ** 2
    mu = np.asarray([s[3] for s in SPECIES])[species]
    drag = np.asarray([s[4] for s in SPECIES])[species]
    # The emitter: ids leave in a seeded order, evenly spaced in time, and CONSECUTIVE ids start
    # at positions along the nozzle bar a golden ratio apart (a low-discrepancy sequence), so the
    # stream is dilute: neighbours in time are far apart in space.
    order = rng.permutation(n)
    rank = np.empty(n)
    rank[order] = np.arange(n)
    t_emit = (rank + rng.random(n)) / n * EMIT_SPAN
    bar_x = (rank * 0.6180339887498949) % 1.0
    bar_y = (rank * 0.7548776662466927) % 1.0
    theta = (rng.random(n) * 2.0 - 1.0) * 0.7
    speed = 0.9 + rng.random(n) * 0.7
    columns = {
        "x": NOZZLE[0] + (bar_x - 0.5) * 1.6, "y": NOZZLE[1] + (bar_y - 0.5) * 0.12,
        "vx": np.zeros(n), "vy": np.zeros(n), "w": np.zeros(n), "age": np.zeros(n),
        "pid": np.arange(n, dtype=np.float64), "t_emit": t_emit,
        "vx0": speed * np.sin(theta), "vy0": speed * np.cos(theta),
        "radius": radius, "mass": mass, "mu": mu, "drag": drag,
        "ir": np.zeros(n), "ig": np.zeros(n), "ib": np.zeros(n),
    }
    out = {name: _f32(value) for name, value in columns.items()}
    out["species"] = species.astype(np.float64)
    return out


def uniforms(*, mode: int = 0, use_image: float = 0.0) -> dict[str, np.ndarray]:
    """The law's scalar columns (shape ()) as float32-representable values."""
    return {"mode": _f32(np.float64(mode)), "use_image": _f32(np.float64(use_image))}


def initial_columns(n: int = N_FULL, *, seed: int = 7, mix: float = 0.5, mode: int = 0,
                    use_image: float = 0.0) -> dict[str, np.ndarray]:
    base = layout(n, seed=seed, mix=mix)
    base.update(uniforms(mode=mode, use_image=use_image))
    return base


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
    rounded).  The page's runtime is the same formula."""
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

    def __init__(self, batch: int):
        super().__init__(batch)
        # ``LawPiece`` files its native product under entry + source digest +
        # argument list; the law and its shapes depend on the particle count.
        self.entry = f"fountain_step_{int(batch)}"


def make_piece(n: int) -> FountainPiece:
    """A ``FountainPiece`` whose law is ``make_law(n)``."""
    class Sized(FountainPiece):
        law = staticmethod(make_law(n))
    return Sized(n)


def run_frames(piece: LawPiece, columns: dict[str, np.ndarray], steps: int) -> dict[str, np.ndarray]:
    """Advance ``steps`` fixed steps through ``piece`` (its compiled product
    when compiled, otherwise the AbstractTensor law); returns the final state
    and colour columns."""
    state = {name: np.array(columns[name], dtype=np.float64) for name in STATE}
    colours = {}
    for _ in range(int(steps)):
        outputs = piece(*[state[n] if n in state else columns[n] for n in ARGUMENTS])
        produced = dict(zip(OUTPUTS, outputs))
        for name in STATE:
            state[name] = produced[name + "_next"]
        colours = {name: produced[name] for name in ("cr", "cg", "cb")}
    return {**state, **colours}


# -------------------------------------------------------------------- game
#: The free board the game occupies in the main scene (board units): the world
#: window ``VIEW`` mapped onto it.
BOARD = (10.5, 10.5 * (VIEW[3] - VIEW[2]) / (VIEW[1] - VIEW[0]))
#: The two particle counts the compiler builds products for (``fountain_build.SIZES``).
SIZES = {"full": N_FULL, "small": N_SMALL}


class FountainGame(Game):
    """Server side of the fountain: phase, seed, mix, size, the image hand-off
    and verification by the law's native product.  It does not stream
    particles: the main page's generic compute runtime runs the compiled law
    (WebGPU compute or the wasm product, as the compiler's verdict and the
    browser allow) and draws the particles in the shared scene.

    The state declares a ``compute`` product (manifest, size, layout URL,
    epoch, uniforms) and ``controls_ui`` (buttons, file pickers, ranges and
    selects for the shared controls bar).

    Actions (``/api/action``): ``fountain_start`` (phase 1; optional ``seed``,
    ``mix``, ``mode``), ``fountain_mode``, ``fountain_speed``,
    ``fountain_size``, ``fountain_image`` (per-id colours as base64 bytes
    ``rgb``, three per id), ``fountain_replay`` (phase 2) and
    ``fountain_verify`` (run the law ``steps`` steps from round zero,
    natively when the native lane compiles, and report a digest)."""

    game = "fountain"
    input = "compute"
    title = "Fountain"

    def __init__(self, board_id: str = "", *, n: int = N_FULL):
        super().__init__(board_id)
        self.n = n
        self.seed, self.mix, self.mode, self.speed = 7, 0.5, 0, 1
        self.phase = "fill"
        self.epoch = 1                       # bumped whenever the page must restart the fountain
        self.image: np.ndarray | None = None
        self.piece: FountainPiece | None = None
        self.verified: dict[str, Any] | None = None
        self.status = "fountain"

    @property
    def size_tag(self) -> str:
        return next((tag for tag, count in SIZES.items() if count == self.n), "full")

    def columns(self, *, image: bool = False) -> dict[str, np.ndarray]:
        cols = initial_columns(self.n, seed=self.seed, mix=self.mix, mode=self.mode,
                               use_image=1.0 if image else 0.0)
        if image and self.image is not None:
            cols["ir"], cols["ig"], cols["ib"] = (_f32(self.image[:, c] / 255.0) for c in range(3))
        return cols

    def reset(self) -> None:
        self.phase, self.verified = "fill", None
        self.epoch += 1

    def do(self, payload: dict) -> None:
        kind = str(payload.get("type"))
        if kind == "fountain_start":
            self.seed = int(payload.get("seed", self.seed))
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
            self.n = SIZES[tag]
            self.image, self.piece = None, None
            self.reset()
        elif kind == "fountain_image":
            import base64
            raw = np.frombuffer(base64.b64decode(str(payload["rgb"])), dtype=np.uint8)
            if raw.size != self.n * 3:
                raise ValueError(f"fountain_image: expected {self.n * 3} bytes, got {raw.size}")
            self.image = raw.reshape(self.n, 3).copy()
        elif kind == "fountain_replay":
            if self.image is None:
                raise ValueError("fountain_replay: supply an image first")
            self.phase, self.verified = "replay", None
            self.epoch += 1
        elif kind == "fountain_verify":
            self.verify(int(payload.get("steps", 600)), image=bool(payload.get("image", False)))
        else:
            raise ValueError(f"unknown fountain action {kind!r}")

    def verify(self, steps: int, *, image: bool = False) -> dict[str, Any]:
        columns = self.columns(image=image)
        if self.piece is None:
            self.piece = make_piece(self.n)
            self.piece.compile(columns)
        final = run_frames(self.piece, columns, steps)
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(final[n], dtype=np.float32).tobytes()
                                         for n in STATE + ("cr", "cg", "cb"))).hexdigest()[:16]
        live = columns["t_emit"] <= steps * DT
        self.verified = {"steps": steps, "lane": self.piece.lane, "digest": digest,
                         "emitted": int(live.sum()), "top": float(final["y"][live].max()) if live.any() else 0.0}
        return self.verified

    def controls_ui(self) -> list[dict[str, Any]]:
        sample = {"fit": "extent", "read": ["x", "y"], "into": ["ir", "ig", "ib"], "raster_scale": 1,
                  "raster": [96, 72], "post": {"action": "fountain_image", "field": "rgb"}}
        return [
            {"id": "start", "kind": "button", "label": "Start", "action": "fountain_start"},
            {"id": "image", "kind": "file", "label": "Image", "sample": sample},
            {"id": "camera", "kind": "file", "label": "Camera", "capture": "environment", "sample": sample},
            {"id": "replay", "kind": "button", "label": "Replay", "action": "fountain_replay",
             "enabled": self.image is not None},
            {"id": "mix", "kind": "range", "label": "poly mix", "min": 0, "max": 1, "step": 0.05,
             "value": self.mix, "action": "fountain_start", "field": "mix"},
            {"id": "mode", "kind": "select", "label": "colour", "value": self.mode, "action": "fountain_mode",
             "field": "mode", "options": [[0, "trig (atan2)"], [1, "rotation"]]},
            {"id": "size", "kind": "select", "label": "ids", "value": self.size_tag, "action": "fountain_size",
             "field": "size", "options": [["full", str(N_FULL)], ["small", str(N_SMALL)]]},
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
            "status": f"{text} · {self.n} ids" + (f" · server {verified['lane']}: {verified['emitted']} emitted, "
                                                  f"top {verified['top']:.3f}, digest {verified['digest']}" if verified else ""),
            "n": self.n, "phase": self.phase, "seed": self.seed, "mix": self.mix, "mode": self.mode,
            "has_image": self.image is not None, "verified": verified,
            "controls_ui": self.controls_ui(),
            "compute": {
                "manifest": "fountain.manifest.json", "size": self.size_tag, "epoch": self.epoch,
                "layout": f"/api/fountain/layout?n={self.n}&seed={self.seed}&mix={self.mix}",
                "uniforms": {"mode": float(self.mode), "use_image": 1.0 if self.phase == "replay" else 0.0},
                "speed": self.speed, "named": ["ir", "ig", "ib"],
            },
        })
        return base

    def result(self):
        return None
