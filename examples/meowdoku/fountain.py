"""Fountain: a multi-species particle fountain that fills a basin, takes an image, fills it again.

A game built to make the deployment layer's compute decision visible.  The
whole game is ONE batched per-step particle law (``fountain_step``), authored
once over spans of particle ids in the repository's AbstractTensor style.
There is no neighbour search: every particle reads only its own columns and a
few scalars, so one compute-shader invocation per particle is exactly the
batched program's deployment.

DESIGN
------
* ``N`` particles with ids ``0..N-1`` and four SPECIES that differ in radius,
  mass and drag.  Species 0 and 1 sit on a REGULAR hexagonal lattice; species
  2 and 3 sit on the same lattice JITTERED by up to ``JITTER`` of a cell
  (the irregular lattice).  The ``mix`` knob is the probability that a slot
  belongs to an irregular species, so ``mix=0`` is a clean hex crystal and
  ``mix=1`` is all jittered.
* SLOT ASSIGNMENT (batched, deterministic): the lattice points are visited in
  the order of a seeded hash of the point (``argsort`` of one uniform draw per
  point), and particle ``id`` takes the ``id``-th point of that order.  So an
  id's slot is fixed by (seed, mix) alone and the basin fills in a scattered,
  reproducible order.
* THE FOUNTAIN: each id has an emit time ``t_emit`` (a seeded draw over
  ``EMIT_SPAN`` seconds) and a flight time ``T``.  Its launch velocity is the
  closed-form solution of ballistic flight WITH linear drag from the nozzle to
  its slot in time ``T`` -- computed once at layout, in bulk -- so it arrives
  at its slot; the law then integrates it (gravity and drag), detects
  landing (descending through the slot's height), bleeds momentum by
  ``rest`` and lets a spring-damper (stiffness ``k``, damping ratio
  ``zeta``, species mass) settle it into the slot.  That is phase 1 and
  "fills once with ids in locations".
* COLOUR in phase 1 comes from an ANGLE -- the slot's direction from the
  basin centre -- in one of two ways chosen by the uniform ``mode``:
  ``mode 0``  trig:      hue = atan2(dy, dx), then the hue ramp;
  ``mode 1``  rotation:  cos and sin are dx/r and dy/r (no atan2), and the
              chroma vector (2/3, -1/3, -1/3) is rotated about the grey axis
              by that angle with the closed-form rotation matrix.
* IMAGE STEP: the page samples the user's image at every id's recorded slot
  (``slot_u``, ``slot_v``: the slot in basin fractions) into a per-id colour
  buffer (``ir ig ib``).  PHASE 2 resets the state, sets the uniform
  ``use_image`` to 1 and runs the fountain again with the same ids and
  emitter; the law now emits the mapped colour, so the final settle shows
  the image.  The colour mix is ``hue * (1 - use) + image * use``, which for
  ``use = 1`` is exactly the image sample.

The server (``FountainGame``) owns phase, seed, mix, the image hand-off and
the law's native product for verification; it does NOT stream particles.
The page runs the identical law as a compiler-generated WebGPU compute
shader (or its WebAssembly product when the compiler's verdict, or the
browser, says no GPU): see ``web/fountain.js`` and ``fountain_build.py``.
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
#: Basin (settling lattice): ``COLS`` x ``ROWS`` hexagonal cells, ``COLS`` wide.
COLS, ROWS = 176, 136
N_FULL = COLS * ROWS                  # 23936 particles
WIDTH = 1.92                          # basin width, world units (x in [-WIDTH/2, WIDTH/2])
BASIN_Y0 = 0.08                       # lowest slot row
CELL = WIDTH / COLS                   # lattice spacing a
ROW_PITCH = CELL * math.sqrt(3.0) / 2.0
HEIGHT = (ROWS - 1) * ROW_PITCH       # slot rows span [BASIN_Y0, BASIN_Y0 + HEIGHT]
NOZZLE = (0.0, 1.5)
VIEW = (-1.05, 1.05, -0.05, 2.0)      # x0 x1 y0 y1 of the page's window
JITTER = 0.30                         # irregular species: +- JITTER * CELL
EMIT_SPAN = 8.0                       # seconds over which ids are emitted
FLIGHT = (1.0, 1.6)                   # flight time range T
DT = 1.0 / 120.0
G = 3.0
SPRING_K, ZETA, REST = 900.0, 0.4, 0.12
CENTER = (0.0, BASIN_Y0 + HEIGHT / 2.0)

#: species: name, radius (in cells), mass, linear drag, irregular?
SPECIES = (
    ("grain",  0.55, 1.0, 0.05, False),
    ("bead",   0.70, 2.0, 0.03, False),
    ("dust",   0.50, 0.6, 0.15, True),
    ("pearl",  0.80, 3.0, 0.02, True),
)
#: the share of each species within its lattice class
SPECIES_WEIGHT = (0.7, 0.3, 0.7, 0.3)

STATE = ("x", "y", "vx", "vy", "age", "landed")
CONSTANTS = ("t_emit", "sx", "sy", "mass", "drag", "ir", "ig", "ib")
UNIFORMS = ("dt", "g", "k", "zeta", "rest", "ccx", "ccy", "mode", "use_image")
OUTPUTS = ("x_next", "y_next", "vx_next", "vy_next", "age_next", "landed_next", "cr", "cg", "cb")
ARGUMENTS = STATE + CONSTANTS + UNIFORMS
#: The columns the layout endpoint sends the page (float32, in this order).
LAYOUT_COLUMNS = ("t_emit", "sx", "sy", "mass", "drag", "vx", "vy", "species", "radius", "slot_u", "slot_v")

TAU = 2.0 * math.pi


# ------------------------------------------------------------------ the law
def _ramp(h, offset):
    """One channel of the HSV ramp at hue ``h`` in [0, 1] (full saturation, value)."""
    f = h + offset
    f = f - f.floor()
    return ((f * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0)


def fountain_step(x, y, vx, vy, age, landed, t_emit, sx, sy, mass, drag, ir, ig, ib,
                  dt, g, k, zeta, rest, ccx, ccy, mode, use_image):
    """One fixed step of every particle.

    Flying: gravity ``g`` (y up) and linear drag.  Landed: spring-damper to
    the slot.  A particle is idle until its age reaches ``t_emit``.  Returns
    the next state and the colour (see the module docstring for the two
    colour modes and the image mix)."""
    live = as_float(age >= t_emit)
    spring = landed > 0.5
    damp = 2.0 * zeta * (k * mass).sqrt()
    ax = AbstractTensor.where(spring, (-(k * (x - sx)) - damp * vx) / mass, -(drag * vx))
    ay = AbstractTensor.where(spring, (-(k * (y - sy)) - damp * vy) / mass, -(g + drag * vy))
    vxn = vx + live * ax * dt
    vyn = vy + live * ay * dt
    xn = x + live * vxn * dt
    yn = y + live * vyn * dt
    # landing: a flying live particle descends through its slot's height
    touch = live * as_float(landed < 0.5) * as_float(vyn < 0.0) * as_float(yn <= sy)
    bleed = 1.0 + touch * (rest - 1.0)
    vxn = vxn * bleed
    vyn = vyn * bleed
    landed_next = landed + touch * (1.0 - landed)
    age_next = age + dt
    # colour from the slot's angle about the basin centre
    dx = sx - ccx
    dy = sy - ccy
    hue = dy.atan2(dx) * (1.0 / TAU) + 0.5
    r = (dx * dx + dy * dy).sqrt() + 1.0e-9
    c = dx / r
    s = dy / r
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
    return xn, yn, vxn, vyn, age_next, landed_next, cr, cg, cb


# ------------------------------------------------------------------- layout
def _f32(a) -> np.ndarray:
    """Rounded through float32: the page's buffers and the reference see one value."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def lattice(cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Hex lattice points (x, y), row-major, ``cols`` * ``rows`` of them."""
    cell = WIDTH / cols
    pitch = cell * math.sqrt(3.0) / 2.0
    jj, ii = np.divmod(np.arange(cols * rows), cols)
    x = (ii + 0.5 * (jj % 2) + 0.25 - cols / 2.0 + 0.5) * cell
    y = BASIN_Y0 + jj * pitch
    return x.astype(np.float64), y.astype(np.float64)


def layout(cols: int = COLS, rows: int = ROWS, *, seed: int = 7, mix: float = 0.5) -> dict[str, np.ndarray]:
    """The round-zero columns for ``cols * rows`` particles, all float64 values
    that are exactly float32-representable.  Everything is a bulk array
    computation over ids; nothing is looped over particles."""
    n = cols * rows
    cell = WIDTH / cols
    rng = np.random.default_rng(seed)
    px, py = lattice(cols, rows)
    order = np.argsort(rng.random(n), kind="stable")        # id -> lattice point
    slot_x, slot_y = px[order], py[order]
    # species: the lattice class first (mix), then the share within the class
    irregular = rng.random(n) < float(mix)
    pick = rng.random(n)
    weight = np.asarray(SPECIES_WEIGHT)
    within = np.where(pick < weight[np.where(irregular, 2, 0)], 0, 1)
    species = np.where(irregular, 2, 0) + within
    jitter = (rng.random((2, n)) * 2.0 - 1.0) * JITTER * cell * irregular
    slot_x = slot_x + jitter[0]
    slot_y = slot_y + jitter[1]
    mass = np.asarray([s[2] for s in SPECIES])[species]
    drag = np.asarray([s[3] for s in SPECIES])[species]
    radius = np.asarray([s[1] for s in SPECIES])[species] * cell
    # the emitter: a time and a flight time per id, then the launch velocity
    t_emit = rng.random(n) * EMIT_SPAN
    flight = FLIGHT[0] + rng.random(n) * (FLIGHT[1] - FLIGHT[0])
    decay = 1.0 - np.exp(-drag * flight)
    vx0 = (slot_x - NOZZLE[0]) * drag / decay
    vy0 = (slot_y - NOZZLE[1] + G * flight / drag) * drag / decay - G / drag
    zeros = np.zeros(n)
    columns = {
        "x": np.full(n, NOZZLE[0]), "y": np.full(n, NOZZLE[1]), "vx": vx0, "vy": vy0,
        "age": zeros.copy(), "landed": zeros.copy(),
        "t_emit": t_emit, "sx": slot_x, "sy": slot_y, "mass": mass, "drag": drag,
        "ir": zeros.copy(), "ig": zeros.copy(), "ib": zeros.copy(),
    }
    out = {name: _f32(value) for name, value in columns.items()}
    out["species"] = species.astype(np.int32)
    out["radius"] = _f32(radius)
    out["slot_u"] = _f32((slot_x + WIDTH / 2.0) / WIDTH)
    out["slot_v"] = _f32(1.0 - (slot_y - BASIN_Y0) / ((rows - 1) * pitch_of(cols)))
    return out


def pitch_of(cols: int) -> float:
    return WIDTH / cols * math.sqrt(3.0) / 2.0


def uniforms(*, mode: int = 0, use_image: float = 0.0, cols: int = COLS, rows: int = ROWS) -> dict[str, np.ndarray]:
    """The law's scalar columns (shape ()) as float32-representable values."""
    centre_y = BASIN_Y0 + (rows - 1) * pitch_of(cols) / 2.0
    values = {"dt": DT, "g": G, "k": SPRING_K, "zeta": ZETA, "rest": REST, "ccx": CENTER[0],
              "ccy": centre_y, "mode": float(mode), "use_image": float(use_image)}
    return {name: _f32(np.float64(value)) for name, value in values.items()}


def sample_raster(raster: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-pixel sample of an ``(h, w, 3)`` uint8 raster at basin fractions
    (``u`` across, ``v`` down), as float64 in [0, 1] (``byte / 255``, float32
    rounded).  The page's ``sampleRaster`` is the same formula."""
    h, w = raster.shape[:2]
    px = np.clip((np.asarray(u) * w).astype(np.int64), 0, w - 1)
    py = np.clip((np.asarray(v) * h).astype(np.int64), 0, h - 1)
    return _f32(raster[py, px, :3] / 255.0)


def initial_columns(cols: int = COLS, rows: int = ROWS, *, seed: int = 7, mix: float = 0.5,
                    mode: int = 0, use_image: float = 0.0) -> dict[str, np.ndarray]:
    base = layout(cols, rows, seed=seed, mix=mix)
    base.update(uniforms(mode=mode, use_image=use_image, cols=cols, rows=rows))
    return base


# ------------------------------------------------------------------- piece
class FountainPiece(LawPiece):
    """The fountain law as a ``LawPiece`` (the repository's law-product machinery)."""

    entry = "fountain_step"
    law = staticmethod(fountain_step)
    argument_names = ARGUMENTS
    output_names = OUTPUTS

    def __init__(self, batch: int):
        super().__init__(batch)
        # ``LawPiece`` files its native product under entry + source digest +
        # argument list, and a game's shapes are fixed; the fountain runs at
        # several particle counts, so the count joins the entry.
        self.entry = f"fountain_step_{int(batch)}"


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
class FountainGame(Game):
    """Server side of the fountain: phase, seed, mix, the image hand-off and
    verification by the law's native product.  It does not stream particles:
    the page runs the law itself, so a tick costs nothing.

    Actions (``/api/action``): ``fountain_start`` (phase 1, optional ``seed``,
    ``mix``, ``mode``), ``fountain_image`` (per-id colours as base64 bytes
    ``rgb``, three per id), ``fountain_replay`` (phase 2), ``fountain_verify``
    (run the law ``steps`` steps from round zero, natively when the native
    lane compiles, and report a digest of the state and of the colours)."""

    game = "fountain"
    input = "fountain"
    title = "Fountain"

    def __init__(self, board_id: str = "", *, cols: int = COLS, rows: int = ROWS):
        super().__init__(board_id)
        self.cols, self.rows = cols, rows
        self.seed, self.mix, self.mode = 7, 0.5, 0
        self.phase = "idle"
        self.image: np.ndarray | None = None
        self.piece: FountainPiece | None = None
        self.verified: dict[str, Any] | None = None
        self.status = "fountain: the page runs the law; start a fill"

    @property
    def n(self) -> int:
        return self.cols * self.rows

    def columns(self, *, image: bool = False) -> dict[str, np.ndarray]:
        cols = initial_columns(self.cols, self.rows, seed=self.seed, mix=self.mix, mode=self.mode,
                               use_image=1.0 if image else 0.0)
        if image and self.image is not None:
            cols["ir"], cols["ig"], cols["ib"] = (_f32(self.image[:, c] / 255.0) for c in range(3))
        return cols

    def reset(self) -> None:
        self.phase = "idle"
        self.image = None
        self.verified = None

    def do(self, payload: dict) -> None:
        kind = str(payload.get("type"))
        if kind == "fountain_start":
            self.seed = int(payload.get("seed", self.seed))
            self.mix = float(payload.get("mix", self.mix))
            self.mode = int(payload.get("mode", self.mode))
            self.phase, self.verified = "fill", None
        elif kind == "fountain_image":
            import base64
            raw = np.frombuffer(base64.b64decode(str(payload["rgb"])), dtype=np.uint8)
            if raw.size != self.n * 3:
                raise ValueError(f"fountain_image: expected {self.n * 3} bytes, got {raw.size}")
            self.image = raw.reshape(self.n, 3).copy()
            self.phase = "mapped"
        elif kind == "fountain_replay":
            if self.image is None:
                raise ValueError("fountain_replay: supply an image first")
            self.phase = "replay"
        elif kind == "fountain_verify":
            self.verify(int(payload.get("steps", 600)), image=bool(payload.get("image", False)))
        else:
            raise ValueError(f"unknown fountain action {kind!r}")

    def verify(self, steps: int, *, image: bool = False) -> dict[str, Any]:
        columns = self.columns(image=image)
        if self.piece is None:
            self.piece = FountainPiece(self.n)
            self.piece.compile(columns)
        final = run_frames(self.piece, columns, steps)
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(final[n], dtype=np.float32).tobytes()
                                         for n in STATE + ("cr", "cg", "cb"))).hexdigest()[:16]
        self.verified = {"steps": steps, "lane": self.piece.lane, "digest": digest,
                         "landed": int(np.sum(final["landed"] > 0.5))}
        return self.verified

    def state(self) -> dict[str, Any]:
        base = self.base_state(1, 1)
        base.update({
            "input": self.input, "page": "fountain.html", "n": self.n, "cols": self.cols, "rows": self.rows,
            "phase": self.phase, "seed": self.seed, "mix": self.mix, "mode": self.mode,
            "has_image": self.image is not None, "verified": self.verified,
        })
        return base

    def result(self):
        return None
