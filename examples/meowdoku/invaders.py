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
    alive, x, y, vx, vy, hp, boom, born, owner                entity state (owner:
                                                              the lane that fired it)
    score, lives, wave, fleet_dir, clock, ufo_clock           game state
    field_w                                                   the field's width
    move_axis, steer_x, fire                                  the CONTROLS

Positions are board units, ``x`` across and ``y`` down the board; there are
no cells.  ``dt_limit`` is the time the fastest live entity takes to cross
the thinnest overlap window, so the dt controller (at cfl 0.5) never steps a
shot over an invader: tileless collision is the dt system's subdivision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meowdoku.entities import EntityGame, LawPiece
from meowdoku.entity_motion import damped_axis
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
PLAYER_SPEED = 9.0
#: The rocket as drawn: its half size, its body box (the hitbox) and the
#: muzzle (where a shot leaves, the nose) -- one geometry for the eye and the
#: law.
SHIP_SIZE = 0.6            # body only (exhaust puff dropped at export): ~1.6 units long
SHIP_RX, SHIP_RY = 0.3, 0.7
MUZZLE = 0.76             # the nose: 1.31 drawn half-sizes ahead, foreshortened by the 0.25 lean
#: The rocket is integrated, not placed: its velocity relaxes toward the
#: commanded velocity with time constant SHIP_TAU (the damping), and a held
#: pointer commands the velocity that would close the gap in SHIP_REACH.
#: Together a damped second-order approach (damping ratio ~0.95): no
#: overshoot to speak of, and no network delay inside the loop, since the
#: pointer's position -- not a speed derived from a stale ship -- is sent.
SHIP_TAU = 0.05
SHIP_REACH = 0.18
SHOT_SPEED = 18.0
BOMB_SPEED = 7.0
UFO_SPEED = 4.0
FLEET_PACE = 0.4
UFO_Y = 0.75
FLEET_MARGIN = 0.55
FLEET_DROP = 0.42
INVASION_Y = HEIGHT - 1.9
PLAYER_BOOM = 1.6
BOOM = 0.35

#: The control record: state column -> the semantic action a shell fills it
#: from.  ``move_axis`` is -1..1 (keys); ``steer_x`` is the board x a held
#: pointer asks the rocket to go to (-1: no pointer, the keys rule);
#: ``fire`` is held, and a shot leaves whenever the shot lane is free.
CONTROLS = {"move_axis": "move-axis-x", "steer_x": "pointer-board-x", "fire": "primary-action"}

#: Mesh vocabulary: name -> emoji, extruded by ``web_export`` through the
#: same glyph builder as every other piece (Segoe UI Emoji outlines).
EMOJI = {
    "rocket": "\U0001F680", "alien": "\U0001F47D", "invader": "\U0001F47E", "crab": "\U0001F980",
    "ufo": "\U0001F6F8", "bolt": "⚡", "bomb": "\U0001F4A3", "brick": "\U0001F9F1", "boom": "\U0001F4A5",
}
#: Outline subdivisions per emoji (the rocket's tail collapses at 1).
EMOJI_SUBDIVISIONS = {"rocket": 2}
#: Glyphs whose outline carries detached decoration (the rocket's exhaust puff
#: floats beside its body): only the largest connected part is the piece.
EMOJI_BODY_ONLY = {"rocket"}


def _frac(value):
    return value - value.floor()


def _hash(value):
    """A deterministic per-lane noise in [0, 1): the law's dice."""
    return _frac((value * 12.9898 + 78.233).sin() * 43758.5453)


def invaders_step(lane, slot, home_x, home_y, points, rx, ry, home_hp,
                  alive, x, y, vx, vy, hp, boom, born, owner,
                  score, lives, wave, fleet_dir, clock, ufo_clock, field_w,
                  move_axis, steer_x, fire, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    n = int(lane.shape[0])
    f = lambda mask: mask.to_dtype("float64")  # noqa: E731
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_shot, is_bomb = slot == PLAYER, slot == SHOT, slot == BOMB
    is_ufo, is_inv, is_shield = slot == UFO, slot == INVADER, slot == SHIELD
    clock_next = clock + dt

    # The fleet: one heading for every invader; fewer invaders, faster fleet.
    fleet = is_inv & live
    count = f(fleet).sum()
    speed = (FLEET_PACE + 0.08 * wave) * (1.0 + 4.0 * (1.0 - count / INVADERS))
    step_x = fleet_dir * speed * dt
    beyond = fleet & ((((x + step_x) > (field_w - FLEET_MARGIN)) & (fleet_dir > 0.0))
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
    rate = 0.45 + 0.15 * wave
    trigger = (_hash(clock * 3.17 + 0.5) < rate * dt) & (f(front).sum() > 0.0) & playing
    free_bomb = is_bomb & ~live
    first_bomb = free_bomb & (lane == where(free_bomb, lane, 1.0e9).min())
    bomb_spawn = first_bomb & trigger
    shooter_x = (f(shooter) * x).sum()
    shooter_lane = (f(shooter) * lane).sum()
    shooter_y = (f(shooter) * y).sum()
    # The saucer: when its clock runs out, from a side the dice pick.
    ufo_spawn = is_ufo & ~live & (ufo_clock <= 0.0) & playing
    from_left = _hash(clock * 1.91 + 2.0) > 0.5
    ufo_x0 = where(from_left, -0.9, field_w + 0.9)
    ufo_vx0 = where(from_left, UFO_SPEED, -UFO_SPEED)
    ufo_launched = f(ufo_spawn).sum() > 0.0
    ufo_clock_next = where(ufo_launched, 14.0 + 10.0 * _hash(clock * 5.3), ufo_clock - dt)

    spawn = shot_spawn | bomb_spawn | ufo_spawn
    x1 = where(shot_spawn, player_x, where(bomb_spawn, shooter_x, where(ufo_spawn, ufo_x0, x)))
    y1 = where(shot_spawn, PLAYER_Y - MUZZLE, where(bomb_spawn, shooter_y + 0.5, where(ufo_spawn, UFO_Y, y)))
    # The rocket: the shared damped motion law (entity_motion.damped_axis),
    # the same text the page runs as WebAssembly to predict its own ship.
    ship_x, ship_v = damped_axis(x, vx, move_axis, steer_x, x * 0.0 + 0.7, field_w - 0.7,
                                 PLAYER_SPEED, SHIP_TAU, SHIP_REACH, dt)
    vx1 = where(is_player, ship_v,
                where(is_inv, fleet_dir_next * speed, where(ufo_spawn, ufo_vx0, where(spawn, 0.0, vx))))
    vy1 = where(shot_spawn, -SHOT_SPEED, where(bomb_spawn, BOMB_SPEED, where(spawn, 0.0, vy)))
    live1 = live | spawn
    # Born at the start of the step: a spawn leaves the muzzle at x1/y1 and flies
    # the whole step, so position = birth point + velocity x (clock - born).
    born1 = where(spawn, clock, born)
    owner1 = where(shot_spawn, 0.0, where(bomb_spawn, shooter_lane, where(ufo_spawn, -1.0, owner)))

    # Motion: the fleet steps as one; everything else flies its velocity.
    moving = live1 & playing
    x2 = where(moving & is_inv, x1 + fleet_dx, where(moving, x1 + vx1 * dt, x1))
    y2 = where(moving & is_inv, y1 + fleet_dy, where(moving, y1 + vy1 * dt, y1))
    x2 = where(is_player & moving, ship_x, x2)
    vx_out = where(is_player & ~moving, 0.0, vx1)
    gone = (((is_shot | is_bomb) & ((y2 < -0.5) | (y2 > HEIGHT + 0.5)))
            | (is_ufo & ((x2 < -1.5) | (x2 > field_w + 1.5))))
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
    x4 = where(respawn, field_w / 2.0, where(reform, home_x, x2))
    vx4 = where(respawn, 0.0, vx_out)
    y4 = where(reform, home_y + FLEET_DROP * (wave + 1.0).clip(0.0, 4.0), y2)
    born4 = where(respawn | reform, clock_next, born1)
    wave_next = where(cleared, wave + 1.0, wave)
    fleet_dir_out = where(cleared, 1.0, fleet_dir_next)

    # The fastest live mover against the thinnest overlap window (a shot's
    # length plus a brick's): the most a step may move anything.
    pace = where(live4, vx1.abs() + vy1.abs(), 0.0) + where(is_inv & live4, speed, 0.0)
    # ... and no step longer than half the rocket's damping time.
    dt_limit = (0.5 / (pace.max() + 1e-6)).clip(0.0, 0.5 * SHIP_TAU) + x * 0.0
    residual = f(is_inv & live4).sum() + x * 0.0
    return (f(live4), x4, y4, vx4, vy1, hp2, boom3, born4, owner1,
            score_next, lives_next, wave_next, fleet_dir_out, clock_next, ufo_clock_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "vx", "vy", "hp", "boom", "born", "owner",
         "score", "lives", "wave", "fleet_dir", "clock", "ufo_clock")
#: Read by the law, written by the host only (a layout change), never stepped.
SETTINGS = ("field_w",)


class InvadersPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "invaders_step"
    law = staticmethod(invaders_step)
    argument_names = ("lane", "slot", "home_x", "home_y", "points", "rx", "ry", "home_hp",
                      *STATE, *SETTINGS, *CONTROLS, "dt")
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


def field_width(aspect: float | None) -> float:
    """The field's width for a screen of ``aspect`` (width / height): the
    height is fixed, the width follows the screen within arcade limits."""
    if not aspect or aspect <= 0.0:
        return WIDTH
    return float(np.clip(round(HEIGHT * aspect * 0.92 * 2.0) / 2.0, 12.5, 26.0))


def layout(width: float = WIDTH) -> dict[str, np.ndarray]:
    """The static columns: each lane's slot, home and size, for a field ``width`` wide."""
    WIDTH = width  # noqa: N806  (the layout below is written against the field width)
    slot = np.zeros(LANES)
    home_x = np.full(LANES, WIDTH / 2.0)
    home_y = np.full(LANES, PLAYER_Y)
    points = np.zeros(LANES)
    rx = np.full(LANES, 0.5)
    ry = np.full(LANES, 0.38)
    home_hp = np.ones(LANES)
    rx[0], ry[0] = SHIP_RX, SHIP_RY                 # the rocket's drawn body
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


def initial_columns(width: float = WIDTH) -> dict[str, np.ndarray]:
    columns = layout(width)
    slot = columns["slot"]
    starts = (slot == PLAYER) | (slot == INVADER) | (slot == SHIELD)
    columns.update({
        "alive": starts.astype(np.float64), "x": columns["home_x"].copy(), "y": columns["home_y"].copy(),
        "vx": np.zeros(LANES), "vy": np.zeros(LANES), "hp": columns["home_hp"].copy(),
        "boom": np.zeros(LANES), "born": np.zeros(LANES), "owner": np.full(LANES, -1.0),
        "score": np.zeros(LANES), "lives": np.full(LANES, LIVES), "wave": np.zeros(LANES),
        "fleet_dir": np.ones(LANES), "clock": np.zeros(LANES), "ufo_clock": np.full(LANES, 12.0),
        "field_w": np.full(LANES, float(width)),
        "move_axis": np.zeros(LANES), "steer_x": np.full(LANES, -1.0), "fire": np.zeros(LANES),
    })
    return columns


class SpaceInvaders(EntityGame):
    game = "space_invaders"
    title = "Space Invaders (free board)"
    width, height = WIDTH, HEIGHT
    CONTROLS = tuple(CONTROLS)

    def __init__(self, board_id: str = "", aspect: float | None = None):
        self.width = field_width(aspect)
        super().__init__(board_id)

    def make_piece(self):
        return InvadersPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.width)

    PLAYER_LANE = 0

    def motion(self) -> dict[str, Any]:
        return {"speed": PLAYER_SPEED, "tau": SHIP_TAU, "reach": SHIP_REACH,
                "x": [0.7, self.width - 0.7]}

    def layout(self, aspect: float) -> None:
        """Fit the field to a new screen shape mid-game: the walls move and
        everything on the field shifts by half the change, so the play stays
        centred.  A host write between rounds, like the controls."""
        width = field_width(aspect)
        if width == self.width:
            return
        shift = (width - self.width) / 2.0
        s = self.session
        for name in ("x", "home_x"):
            s.column(name)[...] += shift
        s.column("field_w")[...] = width
        self.epoch += 1
        self.width = width

    def palette(self) -> dict[str, list[float]]:
        return {"rocket": [0.85, 0.88, 0.95], "alien": [0.35, 0.95, 0.35], "invader": [0.75, 0.35, 0.95],
                "crab": [0.95, 0.35, 0.25], "ufo": [0.95, 0.15, 0.55], "bolt": [1.0, 0.92, 0.25],
                "bomb": [0.2, 0.2, 0.24], "brick": [0.75, 0.42, 0.25], "brick_hurt": [0.45, 0.25, 0.15],
                "boom": [1.0, 0.55, 0.1]}

    def entities(self) -> list[dict[str, Any]]:
        col = {name: self.session.column(name) for name in ("slot", "alive", "x", "y", "vx", "vy", "hp", "boom", "born", "owner", "home_y", "rx")}
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
                mesh, color, size = "rocket", "rocket", SHIP_SIZE
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
                "roll": 0.785 if mesh == "rocket" else 0.0, "lean": 0.25 if slot == PLAYER else 0.55,
                "owner": int(col["owner"][k]) if slot in (SHOT, BOMB) and not exploding else -1,
                "age": clock - float(col["born"][k]),
                "spin": 2.5 if mesh == "ufo" else 0.0, "bob": 1.0 if slot == INVADER and not exploding else 0.0,
            })
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
