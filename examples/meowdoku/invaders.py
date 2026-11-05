"""Space Invaders as one dt-system entity law on a free board.

The state is a span of entity LANES.  Every lane has a fixed SLOT (what may
live there) and static geometry; whether something lives there, and where,
is state.  The step law is masked span arithmetic with no branch, the same
form as ``meowdoku.program.meowdoku_step``: spawning is "the first free lane
of the right slot", collision is a pairwise overlap matrix over lanes masked
by which slots may touch, and dying moves a lane from ``alive`` to ``boom``.

Lanes:
    0            the player (the rocket)
    1            the player's shot (one in flight, as in the arcade)
    2..5         invader bombs
    6            the saucer
    7..61        the fleet: 5 rows x 11 columns
    62..85       four shields of 3 x 2 blocks, 3 hp each

Columns (one value per lane; game-wide values are broadcast to every lane):
    lane, slot, home_x, home_y, points, rx, ry, home_hp      static
    alive, x, y, vx, vy, hp, boom, born                       entity state
    score, lives, wave, fleet_dir, clock, ufo_clock           game state
    move_axis, fire                                           the CONTROLS

Positions are board units, ``x`` across and ``y`` down the board; there are
no cells.  ``dt_limit`` is the time the fastest live entity takes to cross
the thinnest overlap window, so the dt controller (at cfl 0.5) never steps a
shot over an invader: tileless collision is the dt system's subdivision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meowdoku.entities import EntityGame
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

WIDTH, HEIGHT = 15.0, 13.0
PLAYER, SHOT, BOMB, UFO, INVADER, SHIELD = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
BOMB_LANES = range(2, 6)
UFO_LANE = 6
FLEET_ROWS, FLEET_COLS = 5, 11
FLEET0 = 7
SHIELD0 = FLEET0 + FLEET_ROWS * FLEET_COLS
SHIELDS, SHIELD_COLS, SHIELD_ROWS = 4, 3, 2
LANES = SHIELD0 + SHIELDS * SHIELD_COLS * SHIELD_ROWS
INVADERS = float(FLEET_ROWS * FLEET_COLS)

LIVES = 3.0
PLAYER_Y = HEIGHT - 0.9
PLAYER_SPEED = 7.0
SHOT_SPEED = 15.0
BOMB_SPEED = 5.5
UFO_SPEED = 3.2
UFO_Y = 0.75
FLEET_MARGIN = 0.55
FLEET_DROP = 0.42
INVASION_Y = HEIGHT - 1.9
PLAYER_BOOM = 1.6
BOOM = 0.35

#: The control record: state column -> the semantic action a shell fills it
#: from.  ``move_axis`` is -1..1 (keys, or the pointer steering the rocket);
#: ``fire`` is held, and a shot leaves whenever the shot lane is free.
CONTROLS = {"move_axis": "move-axis-x", "fire": "primary-action"}

#: Mesh vocabulary: name -> emoji, extruded by ``web_export`` through the
#: same glyph builder as every other piece (Segoe UI Emoji outlines).
EMOJI = {
    "rocket": "\U0001F680", "alien": "\U0001F47D", "invader": "\U0001F47E", "crab": "\U0001F980",
    "ufo": "\U0001F6F8", "bolt": "⚡", "bomb": "\U0001F4A3", "brick": "\U0001F9F1", "boom": "\U0001F4A5",
}
#: Outline subdivisions per emoji (the rocket's tail collapses at 1).
EMOJI_SUBDIVISIONS = {"rocket": 2}


def _frac(value):
    return value - value.floor()


def _hash(value):
    """A deterministic per-lane noise in [0, 1): the law's dice."""
    return _frac((value * 12.9898 + 78.233).sin() * 43758.5453)


def invaders_step(lane, slot, home_x, home_y, points, rx, ry, home_hp,
                  alive, x, y, vx, vy, hp, boom, born,
                  score, lives, wave, fleet_dir, clock, ufo_clock,
                  move_axis, fire, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    n = int(lane.shape[0])
    f = lambda mask: mask.to_dtype("float")  # noqa: E731
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_shot, is_bomb = slot == PLAYER, slot == SHOT, slot == BOMB
    is_ufo, is_inv, is_shield = slot == UFO, slot == INVADER, slot == SHIELD
    clock_next = clock + dt

    # The fleet: one heading for every invader; fewer invaders, faster fleet.
    fleet = is_inv & live
    count = f(fleet).sum()
    speed = (0.45 + 0.12 * wave) * (1.0 + 4.0 * (1.0 - count / INVADERS))
    step_x = fleet_dir * speed * dt
    beyond = fleet & ((((x + step_x) > (WIDTH - FLEET_MARGIN)) & (fleet_dir > 0.0))
                      | (((x + step_x) < FLEET_MARGIN) & (fleet_dir < 0.0)))
    edge = (f(beyond).sum() > 0.0) & playing
    fleet_dir_next = where(edge, -fleet_dir, fleet_dir)
    fleet_dx = where(edge, 0.0, step_x)
    fleet_dy = where(edge, FLEET_DROP, 0.0)

    # The player's lane: steer from the control, read where it is.
    player_live = is_player & live
    player_x = (f(player_live) * x).sum()
    player_on = f(player_live).sum() > 0.0

    # Spawns.  Shot: whenever fire is held and the one shot lane is free.
    shot_spawn = is_shot & ~live & (fire > 0.0) & player_on & playing
    # Bombs: the dice pick a front invader (nothing alive below it in its
    # column) and the first free bomb lane carries its bomb.
    hx1, hx2 = home_x.reshape((n, 1)), home_x.reshape((1, n))
    hy1, hy2 = home_y.reshape((n, 1)), home_y.reshape((1, n))
    below = (hx1 == hx2) & (hy2 > hy1) & fleet.reshape((1, n))
    front = fleet & ~(f(below).sum(dim=1) > 0.0)
    dice = _hash(lane + clock * 7.31)
    ranked = where(front, dice, -1.0)
    shooter = front & (ranked == ranked.max())
    rate = 0.7 + 0.25 * wave
    trigger = (_hash(clock * 3.17 + 0.5) < rate * dt) & (f(front).sum() > 0.0) & playing
    free_bomb = is_bomb & ~live
    first_bomb = free_bomb & (lane == where(free_bomb, lane, 1.0e9).min())
    bomb_spawn = first_bomb & trigger
    shooter_x = (f(shooter) * x).sum()
    shooter_y = (f(shooter) * y).sum()
    # The saucer: when its clock runs out, from a side the dice pick.
    ufo_spawn = is_ufo & ~live & (ufo_clock <= 0.0) & playing
    from_left = _hash(clock * 1.91 + 2.0) > 0.5
    ufo_x0 = where(from_left, -0.9, WIDTH + 0.9)
    ufo_vx0 = where(from_left, UFO_SPEED, -UFO_SPEED)
    ufo_launched = f(ufo_spawn).sum() > 0.0
    ufo_clock_next = where(ufo_launched, 14.0 + 10.0 * _hash(clock * 5.3), ufo_clock - dt)

    spawn = shot_spawn | bomb_spawn | ufo_spawn
    x1 = where(shot_spawn, player_x, where(bomb_spawn, shooter_x, where(ufo_spawn, ufo_x0, x)))
    y1 = where(shot_spawn, PLAYER_Y - 0.65, where(bomb_spawn, shooter_y + 0.5, where(ufo_spawn, UFO_Y, y)))
    vx1 = where(is_player, move_axis * PLAYER_SPEED,
                where(is_inv, fleet_dir_next * speed, where(ufo_spawn, ufo_vx0, where(spawn, 0.0, vx))))
    vy1 = where(shot_spawn, -SHOT_SPEED, where(bomb_spawn, BOMB_SPEED, where(spawn, 0.0, vy)))
    live1 = live | spawn
    born1 = where(spawn, clock_next, born)

    # Motion: the fleet steps as one; everything else flies its velocity.
    moving = live1 & playing
    x2 = where(moving & is_inv, x1 + fleet_dx, where(moving, x1 + vx1 * dt, x1))
    y2 = where(moving & is_inv, y1 + fleet_dy, where(moving, y1 + vy1 * dt, y1))
    x2 = where(is_player, x2.clip(0.7, WIDTH - 0.7), x2)
    # The rocket reports the velocity it actually moved with: zero when a
    # wall stops it, so a viewer carrying it forward never overshoots.
    vx_out = where(is_player & moving, (x2 - x1) / dt, vx1)
    gone = (((is_shot | is_bomb) & ((y2 < -0.5) | (y2 > HEIGHT + 0.5)))
            | (is_ufo & ((x2 < -1.5) | (x2 > WIDTH + 1.5))))
    live2 = live1 & ~gone

    # Collisions: overlapping boxes between slots that may touch.  Invaders
    # erase shields and are not hurt by them; every other touch harms both.
    X1, X2 = x2.reshape((n, 1)), x2.reshape((1, n))
    Y1, Y2 = y2.reshape((n, 1)), y2.reshape((1, n))
    S1, S2 = slot.reshape((n, 1)), slot.reshape((1, n))
    touch = (((X1 - X2).abs() < (rx.reshape((n, 1)) + rx.reshape((1, n))))
             & ((Y1 - Y2).abs() < (ry.reshape((n, 1)) + ry.reshape((1, n))))
             & live2.reshape((n, 1)) & live2.reshape((1, n)))

    def pair(p, q):
        return ((S1 == p) & (S2 == q)) | ((S1 == q) & (S2 == p))

    may_touch = (pair(SHOT, INVADER) | pair(SHOT, UFO) | pair(SHOT, SHIELD) | pair(SHOT, BOMB)
                 | pair(BOMB, PLAYER) | pair(BOMB, SHIELD) | pair(INVADER, SHIELD) | pair(INVADER, PLAYER))
    harmless = (S1 == INVADER) & ((S2 == SHIELD) | (S2 == PLAYER))
    hits = f(touch & may_touch & ~harmless).sum(dim=1)
    hp2 = where(is_shield, hp - hits, hp)
    dies = live2 & (hits > 0.0) & (~is_shield | (hp2 <= 0.0))
    live3 = live2 & ~dies
    boom3 = where(dies & (is_inv | is_ufo), BOOM,
                  where(dies & is_player, PLAYER_BOOM, (boom - dt).clip(0.0, PLAYER_BOOM)))
    score_next = score + (f(dies & (is_inv | is_ufo)) * points).sum()
    invaded = f(fleet & live3 & (y2 > INVASION_Y)).sum() > 0.0
    lives_after = lives - f(dies & is_player).sum()
    lives_next = where(invaded, 0.0, lives_after).clip(0.0, LIVES)

    # The rocket comes back when its explosion has burned out.
    respawn = is_player & ~live3 & (boom > 0.0) & (boom3 <= 0.0) & (lives_next > 0.0)
    # A cleared wave: the fleet re-forms (a row lower each wave, up to four),
    # bombs in flight vanish, and the fleet is faster and shoots more.
    cleared = (f(is_inv & live3).sum() == 0.0) & playing
    reform = is_inv & cleared
    live4 = (live3 | respawn | reform) & ~(is_bomb & cleared)
    x4 = where(respawn, WIDTH / 2.0, where(reform, home_x, x2))
    y4 = where(reform, home_y + FLEET_DROP * (wave + 1.0).clip(0.0, 4.0), y2)
    born4 = where(respawn | reform, clock_next, born1)
    wave_next = where(cleared, wave + 1.0, wave)
    fleet_dir_out = where(cleared, 1.0, fleet_dir_next)

    # The fastest live mover against the thinnest overlap window (a shot's
    # length plus a brick's): the most a step may move anything.
    pace = where(live4, vx1.abs() + vy1.abs(), 0.0) + where(is_inv & live4, speed, 0.0)
    dt_limit = 0.5 / (pace.max() + 1e-6) + x * 0.0
    residual = f(is_inv & live4).sum() + x * 0.0
    return (f(live4), x4, y4, vx_out, vy1, hp2, boom3, born4,
            score_next, lives_next, wave_next, fleet_dir_out, clock_next, ufo_clock_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "vx", "vy", "hp", "boom", "born",
         "score", "lives", "wave", "fleet_dir", "clock", "ufo_clock")


class InvadersPiece:
    """The law as a dt-system piece (``llvm_dt_system.PIECE_API``)."""

    entry = "invaders_step"
    argument_names = ("lane", "slot", "home_x", "home_y", "points", "rx", "ry", "home_hp",
                      *STATE, *CONTROLS, "dt")
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        self.batch = int(batch)
        self._bound = None

    def instantiate(self, columns, outputs=None):
        self._bound = tuple(columns[name] for name in self.argument_names)
        return ()

    def __call__(self, *columns):
        if len(columns) != len(self.argument_names):
            raise TypeError(f"{self.entry}: takes {len(self.argument_names)} columns, got {len(columns)}")
        tensors = [AbstractTensor.get_tensor(np.asarray(column, dtype=np.float64)) for column in columns]
        produced = invaders_step(*tensors)
        return tuple(np.broadcast_to(np.asarray(value.numpy(), dtype=np.float64), (self.batch,)).copy()
                     for value in produced)


def layout() -> dict[str, np.ndarray]:
    """The static columns: each lane's slot, home and size."""
    slot = np.zeros(LANES)
    home_x = np.full(LANES, WIDTH / 2.0)
    home_y = np.full(LANES, PLAYER_Y)
    points = np.zeros(LANES)
    rx = np.full(LANES, 0.5)
    ry = np.full(LANES, 0.38)
    home_hp = np.ones(LANES)
    slot[1], rx[1], ry[1] = SHOT, 0.08, 0.28
    for k in BOMB_LANES:
        slot[k], rx[k], ry[k] = BOMB, 0.16, 0.24
    slot[UFO_LANE], rx[UFO_LANE], ry[UFO_LANE], points[UFO_LANE], home_y[UFO_LANE] = UFO, 0.6, 0.35, 100.0, UFO_Y
    spacing_x, spacing_y = 1.05, 0.92
    left = (WIDTH - spacing_x * (FLEET_COLS - 1)) / 2.0
    for r in range(FLEET_ROWS):
        for c in range(FLEET_COLS):
            k = FLEET0 + r * FLEET_COLS + c
            slot[k], rx[k], ry[k] = INVADER, 0.4, 0.34
            home_x[k], home_y[k] = left + c * spacing_x, 1.9 + r * spacing_y
            points[k] = 30.0 if r == 0 else 20.0 if r < 3 else 10.0
    k = SHIELD0
    for s in range(SHIELDS):
        centre = WIDTH * (s + 0.5) / SHIELDS
        for r in range(SHIELD_ROWS):
            for c in range(SHIELD_COLS):
                slot[k], rx[k], ry[k], home_hp[k] = SHIELD, 0.27, 0.22, 3.0
                home_x[k], home_y[k] = centre + (c - 1) * 0.55, PLAYER_Y - 1.75 + r * 0.45
                k += 1
    return {"lane": np.arange(LANES, dtype=np.float64), "slot": slot, "home_x": home_x, "home_y": home_y,
            "points": points, "rx": rx, "ry": ry, "home_hp": home_hp}


def initial_columns() -> dict[str, np.ndarray]:
    columns = layout()
    slot = columns["slot"]
    starts = (slot == PLAYER) | (slot == INVADER) | (slot == SHIELD)
    columns.update({
        "alive": starts.astype(np.float64), "x": columns["home_x"].copy(), "y": columns["home_y"].copy(),
        "vx": np.zeros(LANES), "vy": np.zeros(LANES), "hp": columns["home_hp"].copy(),
        "boom": np.zeros(LANES), "born": np.zeros(LANES),
        "score": np.zeros(LANES), "lives": np.full(LANES, LIVES), "wave": np.zeros(LANES),
        "fleet_dir": np.ones(LANES), "clock": np.zeros(LANES), "ufo_clock": np.full(LANES, 12.0),
        "move_axis": np.zeros(LANES), "fire": np.zeros(LANES),
    })
    return columns


class SpaceInvaders(EntityGame):
    game = "space_invaders"
    title = "Space Invaders (free board)"
    width, height = WIDTH, HEIGHT
    CONTROLS = tuple(CONTROLS)

    def make_piece(self):
        return InvadersPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns()

    def palette(self) -> dict[str, list[float]]:
        return {"rocket": [0.85, 0.88, 0.95], "alien": [0.35, 0.95, 0.35], "invader": [0.75, 0.35, 0.95],
                "crab": [0.95, 0.35, 0.25], "ufo": [0.95, 0.15, 0.55], "bolt": [1.0, 0.92, 0.25],
                "bomb": [0.2, 0.2, 0.24], "brick": [0.75, 0.42, 0.25], "brick_hurt": [0.45, 0.25, 0.15],
                "boom": [1.0, 0.55, 0.1]}

    def entities(self) -> list[dict[str, Any]]:
        col = {name: self.session.column(name) for name in ("slot", "alive", "x", "y", "vx", "vy", "hp", "boom", "born", "home_y", "rx")}
        clock = float(self.session.column("clock")[0])
        out = []
        for k in range(LANES):
            slot, alive, boom = col["slot"][k], col["alive"][k] > 0.0, col["boom"][k]
            exploding = not alive and boom > 0.0 and (slot != PLAYER or boom > PLAYER_BOOM - 0.5)
            if not (alive or exploding):
                continue
            if exploding:
                mesh, color, size = "boom", "boom", 0.42
            elif slot == PLAYER:
                mesh, color, size = "rocket", "rocket", 0.46
            elif slot == SHOT:
                mesh, color, size = "bolt", "bolt", 0.26
            elif slot == BOMB:
                mesh, color, size = "bomb", "bomb", 0.22
            elif slot == UFO:
                mesh, color, size = "ufo", "ufo", 0.55
            elif slot == INVADER:
                row = (k - FLEET0) // FLEET_COLS
                mesh = "alien" if row == 0 else "invader" if row < 3 else "crab"
                color, size = mesh, 0.38
            else:
                mesh, size = "brick", 0.26
                color = "brick" if col["hp"][k] >= 3.0 else "brick_hurt"
            out.append({
                "id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                "vx": 0.0 if exploding else float(col["vx"][k]), "vy": 0.0 if exploding else float(col["vy"][k]),
                "mesh": mesh, "color": color, "size": size,
                "roll": 0.785 if mesh == "rocket" else 0.0,
                "spin": 2.5 if mesh == "ufo" else 0.0, "bob": 1.0 if slot == INVADER and not exploding else 0.0,
            })
        del clock
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]), "detail": f"reached wave {int(s.column('wave')[0]) + 1}"}

    def describe(self) -> str:
        s = self.session
        lives, score, wave = int(s.column("lives")[0]), int(s.column("score")[0]), int(s.column("wave")[0])
        fleet = int(s.column("alive")[FLEET0:SHIELD0].sum())
        seconds, calls = s.last_round
        head = "GAME OVER · R to restart" if lives <= 0 else f"lives {lives}"
        return (f"{head} · score {score} · wave {wave + 1} · {fleet} invaders · "
                f"tick {seconds * 1000:.0f} ms = {calls} law steps")
