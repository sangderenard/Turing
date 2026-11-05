"""A procedural vertical platformer ("climber") as one dt-system entity law.

The same engine as the scroller and Space Invaders (``entities.py``): entity
lanes with a fixed slot each, a branch-free step law compiled to a NumPy
product, the dt controller subdividing each tick, and the player moved
sideways by ``entity_motion.damped_axis`` (the law the page predicts with).
Gravity pulls toward +y, the bottom of the screen.

Lanes (contiguous blocks):
    0              the player (a frog); feet at ``y + PLAYER_RY``
    1..24          brick platforms; a brick is square, so a platform's width
                   ``rx`` is its drawn half size and its top is ``y - ry``
    25..28         star pickups (alive only while one is on the way down)

The player is integrated, never placed: semi-implicit gravity each step, and
when the player is falling and its feet cross a platform's top during the step
over that platform's width, the velocity becomes ``-JUMP`` (the bounce).  The
only input is sideways (``move_axis`` / a held pointer's ``steer_x``).

The world scrolls: when the player rises above ``SCROLL_Y`` every lane's y
increases by the overshoot (``climbed`` and the score grow by it).  A platform
that falls off the bottom is recycled above the highest remaining platform at
a gap drawn from the law's ``noise`` (rank-cumulated, so several recycled in
one step stack); the gap range grows with the climb but never reaches the jump
height (``JUMP**2 / (2 GRAVITY)``).  Falling below the bottom ends the run.

Columns: lane, slot, rx, ry (static); alive, x, y, vx, vy, born, score, lives,
climbed, clock (state; ``vy`` of a platform is the scroll velocity of the
step); field_w, seed (settings); move_axis, steer_x (controls).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from meowdoku.entities import EntityGame, LawPiece, as_float, block_hits, first_of, noise
from meowdoku.entity_motion import damped_axis
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

HEIGHT = 16.0
WIDTH = 9.0
#: The field is portrait (phone-shaped) whatever the screen (see scroller.py).
PORTRAIT = 0.62
PLAYER, PLATFORM, PICKUP = 0.0, 1.0, 2.0
PLATFORMS = (1, 25)
PICKUPS = (25, 29)
LANES = PICKUPS[1]

GRAVITY = 24.0
JUMP = 12.0                                  # bounce velocity: apex JUMP**2 / (2 GRAVITY) = 3.0 above the feet
FALL_MAX = 20.0
SPEED = 9.0
TAU = 0.06
REACH = 0.2
MARGIN = 0.4
PLAYER_RX, PLAYER_RY = 0.3, 0.35
FOOT = 0.2                                   # half width of the feet that land
SCROLL_Y = 0.4 * HEIGHT                      # the world scrolls to keep the player above this line
GAP_MIN = 1.1
GAP_MAX0, GAP_MAX_GROWTH, GAP_MAX_LIMIT = 1.5, 0.01, 2.5      # centre-to-centre; below the 3.0 jump
PICKUP_RX = 0.25
PICKUP_CHANCE = 0.4
PICKUP_LIFT = 0.65
POINTS_PER_UNIT = 10.0
POINTS_PER_PICKUP = 100.0

CONTROLS = {"move_axis": "move-axis-x", "steer_x": "pointer-board-x"}


def platformer_step(lane, slot, rx, ry,
                    alive, x, y, vx, vy, born, score, lives, climbed, clock,
                    field_w, seed,
                    move_axis, steer_x, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    f = as_float
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_plat, is_pick = slot == PLAYER, slot == PLATFORM, slot == PICKUP
    zero = x * 0.0
    clock_next = clock + dt

    # Sideways: the shared damped motion law.
    ship_x, ship_vx = damped_axis(x, vx, move_axis, steer_x, zero + MARGIN, field_w - MARGIN,
                                  SPEED, TAU, REACH, dt)
    me = is_player & live
    py, pvy = (f(me) * y).sum(), (f(me) * vy).sum()
    pxn = (f(me) * ship_x).sum()

    # Vertical: gravity, then the bounce when the feet cross a platform's top.
    fall = (pvy + GRAVITY * dt).clip(-FALL_MAX, FALL_MAX)
    py_new = py + fall * dt
    top = y - ry
    crossed = (is_plat & live & playing & (fall > 0.0)
               & (py + PLAYER_RY <= top) & (py_new + PLAYER_RY >= top)
               & ((x - pxn).abs() < rx + FOOT))
    landed = f(crossed).sum() > 0.0
    pvy_next = where(landed, -JUMP, fall)

    # The world scrolls down by however far the player rose above the line.
    overshoot = (SCROLL_Y - py_new).clip(0.0, 100.0) * f(playing)
    climbed_next = climbed + overshoot
    x1 = where(is_player & playing, ship_x, x)
    y1 = where(is_player, where(playing, py_new, y), y) + overshoot * f(live)

    # Pickups: touched by the player, or fallen off the bottom.
    touched = is_pick & live & playing & (block_hits(x1, y1, rx, ry, live, None, PICKUPS, (0, 1)) > 0.0)
    pick_out = is_pick & (y1 > HEIGHT + 0.5)

    # Platforms off the bottom are recycled above the highest remaining one.
    gone = is_plat & live & playing & (y1 > HEIGHT + 0.6)
    anchor = where(is_plat & live & ~gone, y1, 1.0e9).min()
    gap_max = (GAP_MAX0 + GAP_MAX_GROWTH * climbed_next).clip(GAP_MAX0, GAP_MAX_LIMIT)
    gap = GAP_MIN + (gap_max - GAP_MIN) * noise(lane * 3.1 + climbed_next * 7.3 + seed + 11.0)
    new_y = anchor - where(gone, gap, 0.0).cumsum(dim=0)
    new_x = rx + (field_w - 2.0 * rx) * noise(lane * 1.7 + climbed_next * 5.9 + seed + 2.0)

    # A pickup may ride the first platform recycled this step.
    first_gone = first_of(gone, lane)
    any_gone = f(first_gone).sum() > 0.0
    gx, gy = (f(first_gone) * new_x).sum(), (f(first_gone) * new_y).sum()
    spawn_pick = (first_of(is_pick & ~live, lane) & any_gone
                  & (noise(climbed_next * 3.3 + seed + 9.0) < PICKUP_CHANCE))

    x2 = where(gone, new_x, where(spawn_pick, gx, x1))
    y2 = where(gone, new_y, where(spawn_pick, gy - PICKUP_LIFT, y1))
    born2 = where(gone | spawn_pick, clock, born)

    # The run ends when the player falls below the bottom.
    dead = is_player & live & (y2 > HEIGHT + PLAYER_RY)
    live2 = (live & ~touched & ~pick_out & ~dead) | spawn_pick
    lives_next = lives - f(dead).sum()
    score_next = score + POINTS_PER_UNIT * overshoot + POINTS_PER_PICKUP * f(touched).sum()

    vx2 = where(is_player, ship_vx, 0.0)
    vy2 = where(is_player, pvy_next, overshoot / dt)

    # The longest step that cannot carry the feet through a platform's
    # thickness, and no more than half the damping time of the sideways law.
    window = 2.0 * where(is_plat, ry, 1.0e9).min()
    dt_limit = (0.9 * window / (pvy_next.abs() + 1e-6)).clip(0.0, 0.5 * TAU) + zero
    residual = f(live2 & is_plat).sum() + zero
    return (f(live2), x2, y2, vx2, vy2, born2, score_next, lives_next, climbed_next, clock_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "vx", "vy", "born", "score", "lives", "climbed", "clock")
SETTINGS = ("field_w", "seed")


class PlatformerPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "platformer_step"
    law = staticmethod(platformer_step)
    argument_names = ("lane", "slot", "rx", "ry", *STATE, *SETTINGS, *CONTROLS, "dt")
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


def field_width(aspect: float | None) -> float:
    if not aspect or aspect <= 0.0:
        return WIDTH
    return float(np.clip(round(HEIGHT * min(aspect, PORTRAIT) * 2.0) / 2.0, 9.0, HEIGHT * PORTRAIT + 0.5))


def initial_columns(width: float = WIDTH, seed: int = 0) -> dict[str, np.ndarray]:
    lane = np.arange(LANES, dtype=np.float64)
    slot = np.zeros(LANES)
    rx, ry = np.full(LANES, PLAYER_RX), np.full(LANES, PLAYER_RY)
    count = PLATFORMS[1] - PLATFORMS[0]
    slot[PLATFORMS[0]:PLATFORMS[1]] = PLATFORM
    rx[PLATFORMS[0]:PLATFORMS[1]] = 0.28 + 0.04 * (np.arange(count) % 4)      # widths 0.28 .. 0.40
    ry[PLATFORMS[0]:PLATFORMS[1]] = rx[PLATFORMS[0]:PLATFORMS[1]]             # a brick is square
    slot[PICKUPS[0]:PICKUPS[1]], rx[PICKUPS[0]:PICKUPS[1]], ry[PICKUPS[0]:PICKUPS[1]] = PICKUP, PICKUP_RX, PICKUP_RX
    rng = np.random.default_rng(int(seed))
    x, y = np.zeros(LANES), np.zeros(LANES)
    below = HEIGHT - 0.8
    for k in range(*PLATFORMS):
        x[k] = width / 2.0 if k == PLATFORMS[0] else rng.uniform(rx[k], width - rx[k])
        y[k] = below
        below -= rng.uniform(GAP_MIN, GAP_MAX0)
    x[0] = width / 2.0
    y[0] = y[PLATFORMS[0]] - ry[PLATFORMS[0]] - PLAYER_RY - 0.02          # standing on the first brick
    alive = (slot != PICKUP).astype(np.float64)
    return {
        "lane": lane, "slot": slot, "rx": rx, "ry": ry,
        "alive": alive, "x": x, "y": y, "vx": np.zeros(LANES), "vy": np.zeros(LANES),
        "born": np.zeros(LANES), "score": np.zeros(LANES), "lives": np.ones(LANES),
        "climbed": np.zeros(LANES), "clock": np.zeros(LANES),
        "field_w": np.full(LANES, float(width)), "seed": np.full(LANES, float(int(seed) % 1000)),
        "move_axis": np.zeros(LANES), "steer_x": np.full(LANES, -1.0),
    }


class Platformer(EntityGame):
    game = "platformer"
    title = "Climber (free board)"
    height = HEIGHT
    CONTROLS = tuple(CONTROLS)
    PLAYER_LANE = 0

    def __init__(self, board_id: str = "", aspect: float | None = None, seed: int | None = None):
        self.width = field_width(aspect)
        self.seed = int(time.time_ns() % 1000) if seed is None else int(seed)
        super().__init__(board_id)

    def make_piece(self):
        return PlatformerPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.width, self.seed)

    def motion(self) -> dict[str, Any]:
        return {"speed": SPEED, "tau": TAU, "reach": REACH, "x": [MARGIN, self.width - MARGIN]}

    def reset(self) -> None:
        self.seed = (self.seed + 1) % 1000          # a new climb each run
        super().reset()

    def layout(self, aspect: float) -> None:
        width = field_width(aspect)
        if width == self.width:
            return
        s = self.session
        s.column("x")[...] += (width - self.width) / 2.0
        s.column("field_w")[...] = width
        self.epoch += 1
        self.width = width

    def palette(self) -> dict[str, list[float]]:
        return {"floor": [0.55, 0.78, 0.95], "frog": [0.45, 0.9, 0.35], "brick": [0.75, 0.42, 0.25],
                "star": [1.0, 0.85, 0.2]}

    def entities(self) -> list[dict[str, Any]]:
        names = ("slot", "alive", "x", "y", "vx", "vy", "born", "rx")
        col = {name: self.session.column(name) for name in names}
        clock = float(self.session.column("clock")[0])
        out = []
        for k in np.flatnonzero(col["alive"] > 0.0):
            slot = col["slot"][k]
            if slot == PLAYER:
                mesh, size = "frog", PLAYER_RY
            elif slot == PLATFORM:
                mesh, size = "brick", float(col["rx"][k])
            else:
                mesh, size = "star", PICKUP_RX
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": float(col["vx"][k]),
                        "vy": float(col["vy"][k]) if slot != PLAYER else 0.0,
                        "mesh": mesh, "color": mesh, "size": size, "roll": 0.0,
                        "lean": 0.25 if slot == PLAYER else 0.55, "owner": -1,
                        "age": clock - float(col["born"][k]),
                        "spin": 1.5 if slot == PICKUP else 0.0, "bob": 0.0})
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]), "detail": f"climbed {s.column('climbed')[0]:.0f} m"}

    def describe(self) -> str:
        s = self.session
        seconds, calls = s.last_round
        head = "GAME OVER · R to restart" if s.column("lives")[0] <= 0 else "climb!"
        return (f"{head} · height {s.column('climbed')[0]:.0f} · score {int(s.column('score')[0])} · "
                f"tick {seconds * 1000:.0f} ms = {calls} law steps")
