"""A top-down bullet-hell scroller as one dt-system entity law.

The same engine as Space Invaders (``entities.py``): entity lanes with a
fixed slot each, a branch-free step law, the dt controller subdividing each
tick so the fastest bullet never steps over the ship's hitbox, and the
player's ship moved by ``entity_motion.damped_axis`` on both axes -- the law
the page predicts with, compiled to WebAssembly.

Lanes (contiguous blocks, so collisions are block against block):
    0              the player ship; a small hitbox under a larger body
    1..24          the player's shots (twin shots on a short reload)
    25..48         enemies: drifters (sway, aimed three-way spreads) and
                   spinners (descend, hold, fire rotating rings, leave)
    49..368        enemy bullets
    369..398       stars: scenery scrolling past, never collide

Columns: lane, slot (static); alive, x, y, vx, vy, hp, boom, born, owner, kind,
age, timer (entity state: ``timer`` is the player's reload and an enemy's
time to its next volley); score, lives, invuln, clock, spawn_clock, field_w
(game-wide); move_axis, move_axis_y, steer_x, steer_y, fire (controls).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meowdoku.entities import EntityGame, LawPiece, as_float, block_hits, first_of, noise
from meowdoku.entity_motion import damped_axis
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

HEIGHT = 16.0
WIDTH = 13.0
PLAYER, SHOT, ENEMY, BULLET, STAR = 0.0, 1.0, 2.0, 3.0, 4.0
SHOTS = (1, 25)
ENEMIES = (25, 49)
BULLETS = (49, 369)
STARS = (369, 399)
LANES = STARS[1]
DRIFTER, SPINNER = 1.0, 2.0

LIVES = 3.0
SPEED = 10.0
TAU = 0.05
REACH = 0.18
MARGIN = 0.5
SHOT_SPEED = 22.0
RELOAD = 0.085
INVULN = 2.0
BOOM = 0.35
BULLET_SPEED = 5.0
SPREAD = 3.0
RING = 12.0
HITBOX = 0.11
#: The ship as drawn: half size, body box (hitbox), muzzle at the nose.
SHIP_SIZE = 0.5            # body only (exhaust puff dropped at export): ~1.3 units long
SHIP_RX, SHIP_RY = 0.22, 0.4
MUZZLE = 0.63             # the nose: 1.31 drawn half-sizes ahead, foreshortened by the 0.25 lean

CONTROLS = {"move_axis": "move-axis-x", "move_axis_y": "move-axis-y", "steer_x": "pointer-board-x",
            "steer_y": "pointer-board-y", "fire": "primary-action"}


def scroller_step(lane, slot, rx, ry,
                  alive, x, y, vx, vy, hp, boom, born, owner, kind, age, timer,
                  score, lives, invuln, clock, spawn_clock, field_w,
                  move_axis, move_axis_y, steer_x, steer_y, fire, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    f = as_float
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_shot, is_enemy = slot == PLAYER, slot == SHOT, slot == ENEMY
    is_bullet, is_star = slot == BULLET, slot == STAR
    clock_next = clock + dt
    zero = x * 0.0

    # The ship: the shared damped motion law on both axes.
    ship_x, ship_vx = damped_axis(x, vx, move_axis, steer_x, zero + MARGIN, field_w - MARGIN,
                                  SPEED, TAU, REACH, dt)
    ship_y, ship_vy = damped_axis(y, vy, move_axis_y, steer_y, zero + MARGIN, zero + HEIGHT - MARGIN,
                                  SPEED, TAU, REACH, dt)
    me = is_player & live
    px, py = (f(me) * x).sum(), (f(me) * y).sum()
    ship_on = f(me).sum() > 0.0

    # Twin shots on a reload: the first two free shot lanes.
    reload = (f(is_player) * timer).sum()
    volley = (fire > 0.0) & ship_on & playing & (reload <= 0.0)
    free_shot = is_shot & ~live
    left = first_of(free_shot, lane)
    right = first_of(free_shot & ~left, lane)
    shot_spawn = (left | right) & volley
    shot_x = px + where(left, -0.2, 0.2)

    # Enemies: one arrives whenever the spawn clock runs out, faster as the
    # run goes on; drifters three times as often as spinners.
    arrive = (spawn_clock <= 0.0) & playing
    enemy_spawn = first_of(is_enemy & ~live, lane) & arrive
    roll = noise(clock * 3.7 + 1.0)
    new_kind = where(roll < 0.75, DRIFTER, SPINNER)
    spawn_clock_next = where(arrive, (1.1 - clock / 200.0).clip(0.35, 1.1) * (0.6 + 0.8 * noise(clock * 1.3)),
                             spawn_clock - dt)
    enemy_x0 = 1.0 + (field_w - 2.0) * noise(clock * 5.1 + 2.0)

    # One volley per step: the ready enemy in the lowest lane fires, into the
    # first free bullet lanes (drifters an aimed spread, spinners a ring).
    ready = is_enemy & live & (timer <= 0.0) & (y > 0.3) & playing
    shooter = first_of(ready, lane)
    fired = f(shooter).sum() > 0.0
    sx, sy = (f(shooter) * x).sum(), (f(shooter) * y).sum()
    skind = (f(shooter) * kind).sum()
    shooter_lane = (f(shooter) * lane).sum()
    count = where(skind == SPINNER, RING, SPREAD)
    free_bullet = is_bullet & ~live
    rank = f(free_bullet).cumsum(dim=0) - 1.0
    bullet_spawn = free_bullet & (rank < count) & fired
    aim = (py - sy).atan2(px - sx)
    angle = where(skind == SPINNER, rank * (6.283185307179586 / RING) + clock * 1.7,
                  aim + (rank - 1.0) * 0.22)
    speed = where(skind == SPINNER, BULLET_SPEED * 0.8, BULLET_SPEED)

    spawn = shot_spawn | enemy_spawn | bullet_spawn
    x1 = where(shot_spawn, shot_x, where(enemy_spawn, enemy_x0, where(bullet_spawn, sx, x)))
    y1 = where(shot_spawn, py - MUZZLE, where(enemy_spawn, -0.8, where(bullet_spawn, sy + 0.2, y)))
    kind1 = where(enemy_spawn, new_kind, kind)
    age1 = where(spawn, 0.0, age + dt)
    hp1 = where(enemy_spawn, where(new_kind == SPINNER, 9.0, 3.0), hp)
    timer1 = where(is_player, where(volley, RELOAD, (timer - dt).clip(0.0, 10.0)),
                   where(enemy_spawn, 0.8, where(shooter, where(skind == SPINNER, 0.55, 1.5), timer - dt)))
    # Enemy paths: drifters fall and sway; spinners descend, hold, then leave.
    drifter = is_enemy & (kind1 == DRIFTER)
    spinner = is_enemy & (kind1 == SPINNER)
    sway = 1.8 * (clock_next * 1.9 + lane * 0.7).sin()
    vx1 = where(is_player, ship_vx, where(drifter, sway, where(bullet_spawn, speed * angle.cos(),
                                     where(spawn | spinner, 0.0, vx))))
    vy1 = where(is_player, ship_vy,
                where(shot_spawn, -SHOT_SPEED,
                      where(drifter, 1.7, where(spinner, where(age1 < 3.0, 1.0, where(age1 < 10.0, 0.0, 1.6)),
                                                where(bullet_spawn, speed * angle.sin(), vy)))))
    live1 = live | spawn
    # Born at the start of the step: a spawn leaves the muzzle at x1/y1 and flies
    # the whole step, so position = birth point + velocity x (clock - born).
    born1 = where(spawn, clock, born)
    owner1 = where(shot_spawn, 0.0, where(bullet_spawn, shooter_lane, where(enemy_spawn, -1.0, owner)))

    moving = (live1 & playing) | is_star
    x2 = where(is_player & moving, ship_x, where(moving, x1 + vx1 * dt, x1))
    y2 = where(is_player & moving, ship_y, where(moving, y1 + vy1 * dt, y1))
    # Stars wrap from the bottom back to the top: the scroll.
    y2 = where(is_star & (y2 > HEIGHT + 0.3), y2 - (HEIGHT + 0.6), y2)
    out = (y2 < -1.5) | (y2 > HEIGHT + 1.5) | (x2 < -1.5) | (x2 > field_w + 1.5)
    live2 = live1 & ~((is_shot | is_enemy | is_bullet) & out)

    # Collisions, block against block.
    enemy_hits = block_hits(x2, y2, rx, ry, live2, None, ENEMIES, SHOTS)
    shot_hits = block_hits(x2, y2, rx, ry, live2, None, SHOTS, ENEMIES)
    exposed = live2 & (invuln <= 0.0)
    player_hits = (block_hits(x2, y2, rx, ry, exposed, None, (0, 1), BULLETS)
                   + block_hits(x2, y2, rx, ry, exposed, None, (0, 1), ENEMIES))
    bullet_hits = block_hits(x2, y2, rx, ry, live2 & ((invuln <= 0.0) | ~is_player), None, BULLETS, (0, 1))
    hp2 = hp1 - enemy_hits
    killed = is_enemy & live2 & (hp2 <= 0.0)
    hit = is_player & live2 & (player_hits > 0.0)
    was_hit = f(hit).sum() > 0.0
    lives_next = lives - f(hit).sum()
    over = lives_next <= 0.0
    # A hit clears the screen of bullets and grants a moment of grace.
    gone = killed | (is_shot & (shot_hits > 0.0)) | (is_bullet & ((bullet_hits > 0.0) | was_hit)) | (hit & over)
    live3 = live2 & ~gone
    boom3 = where(killed | (hit & over), BOOM, (boom - dt).clip(0.0, BOOM))
    invuln_next = where(was_hit, INVULN, (invuln - dt).clip(0.0, INVULN))
    score_next = score + (f(killed) * where(kind1 == SPINNER, 300.0, 100.0)).sum()

    # The longest step that steps nothing over anything it can hit, pair by
    # pair: a shot must not cross an enemy's height in one step, a bullet
    # must not cross the ship's hitbox at their closing speed; and the ship
    # moves at most half its damping time per step.
    pace = vx1.abs() + vy1.abs()
    shot_pace = where(is_shot & live3, pace, 0.0).max()
    bullet_pace = where(is_bullet & live3, pace, 0.0).max() + (f(is_player) * pace).sum()
    shot_window = 2.0 * (0.3 + 0.36)
    bullet_window = 2.0 * (SHIP_RX + 0.12)
    shot_limit = 0.9 * shot_window / (shot_pace + 1e-6)
    bullet_limit = 0.9 * bullet_window / (bullet_pace + 1e-6)
    dt_limit = where(shot_limit < bullet_limit, shot_limit, bullet_limit).clip(0.0, 0.5 * TAU) + zero
    residual = f(is_enemy & live3).sum() + zero
    return (f(live3), x2, y2, vx1, vy1, hp2, boom3, born1, owner1, kind1, age1, timer1,
            score_next, lives_next, invuln_next, clock_next, spawn_clock_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "vx", "vy", "hp", "boom", "born", "owner", "kind", "age", "timer",
         "score", "lives", "invuln", "clock", "spawn_clock")
SETTINGS = ("field_w",)


class ScrollerPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "scroller_step"
    law = staticmethod(scroller_step)
    argument_names = ("lane", "slot", "rx", "ry", *STATE, *SETTINGS, *CONTROLS, "dt")
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


def field_width(aspect: float | None) -> float:
    if not aspect or aspect <= 0.0:
        return WIDTH
    return float(np.clip(round(HEIGHT * aspect * 0.92 * 2.0) / 2.0, 10.0, 26.0))


def initial_columns(width: float = WIDTH) -> dict[str, np.ndarray]:
    lane = np.arange(LANES, dtype=np.float64)
    slot = np.zeros(LANES)
    rx, ry = np.full(LANES, HITBOX), np.full(LANES, HITBOX)
    rx[0], ry[0] = SHIP_RX, SHIP_RY
    slot[SHOTS[0]:SHOTS[1]], rx[SHOTS[0]:SHOTS[1]], ry[SHOTS[0]:SHOTS[1]] = SHOT, 0.1, 0.3
    slot[ENEMIES[0]:ENEMIES[1]], rx[ENEMIES[0]:ENEMIES[1]], ry[ENEMIES[0]:ENEMIES[1]] = ENEMY, 0.42, 0.36
    slot[BULLETS[0]:BULLETS[1]], rx[BULLETS[0]:BULLETS[1]], ry[BULLETS[0]:BULLETS[1]] = BULLET, 0.12, 0.12
    slot[STARS[0]:STARS[1]] = STAR
    rng = np.random.default_rng(7)
    x, y = np.zeros(LANES), np.zeros(LANES)
    vy = np.zeros(LANES)
    x[0], y[0] = width / 2.0, HEIGHT - 2.0
    stars = slice(*STARS)
    x[stars] = rng.uniform(0.2, width - 0.2, STARS[1] - STARS[0])
    y[stars] = rng.uniform(-0.3, HEIGHT + 0.3, STARS[1] - STARS[0])
    vy[stars] = rng.uniform(1.5, 5.0, STARS[1] - STARS[0])
    alive = ((slot == PLAYER) | (slot == STAR)).astype(np.float64)
    return {
        "lane": lane, "slot": slot, "rx": rx, "ry": ry,
        "alive": alive, "x": x, "y": y, "vx": np.zeros(LANES), "vy": vy, "hp": np.zeros(LANES),
        "boom": np.zeros(LANES), "born": np.zeros(LANES), "owner": np.full(LANES, -1.0), "kind": np.zeros(LANES), "age": np.zeros(LANES),
        "timer": np.zeros(LANES), "score": np.zeros(LANES), "lives": np.full(LANES, LIVES),
        "invuln": np.full(LANES, 1.0), "clock": np.zeros(LANES), "spawn_clock": np.full(LANES, 1.5),
        "field_w": np.full(LANES, float(width)),
        **{name: np.zeros(LANES) for name in CONTROLS}, "steer_x": np.full(LANES, -1.0),
        "steer_y": np.full(LANES, -1.0),
    }


class BulletScroller(EntityGame):
    game = "bullet_hell"
    title = "Bullet hell (free board)"
    height = HEIGHT
    CONTROLS = tuple(CONTROLS)
    PLAYER_LANE = 0

    def __init__(self, board_id: str = "", aspect: float | None = None):
        self.width = field_width(aspect)
        super().__init__(board_id)

    def make_piece(self):
        return ScrollerPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.width)

    def motion(self) -> dict[str, Any]:
        return {"speed": SPEED, "tau": TAU, "reach": REACH,
                "x": [MARGIN, self.width - MARGIN], "y": [MARGIN, HEIGHT - MARGIN]}

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
        return {"rocket": [0.85, 0.88, 0.95], "rocket_blink": [0.35, 0.37, 0.45], "bolt": [1.0, 0.92, 0.25],
                "invader": [0.75, 0.35, 0.95], "ufo": [0.95, 0.15, 0.55], "bullet": [1.0, 0.25, 0.35],
                "bullet_ring": [0.35, 0.8, 1.0], "star": [0.55, 0.6, 0.8], "boom": [1.0, 0.55, 0.1]}

    def entities(self) -> list[dict[str, Any]]:
        names = ("slot", "alive", "x", "y", "vx", "vy", "boom", "born", "owner", "kind", "invuln")
        col = {name: self.session.column(name) for name in names}
        clock = float(self.session.column("clock")[0])
        show = (col["alive"] > 0.0) | (col["boom"] > 0.0)
        out = []
        for k in np.flatnonzero(show):
            slot, exploding = col["slot"][k], col["alive"][k] <= 0.0
            if exploding:
                mesh, color, size = "boom", "boom", 0.42
            elif slot == PLAYER:
                blink = col["invuln"][k] > 0.0 and int(clock * 12) % 2 == 0
                mesh, color, size = "rocket", "rocket_blink" if blink else "rocket", SHIP_SIZE
            elif slot == SHOT:
                mesh, color, size = "bolt", "bolt", 0.2
            elif slot == ENEMY:
                spinner = col["kind"][k] == SPINNER
                mesh, color, size = ("ufo", "ufo", 0.55) if spinner else ("invader", "invader", 0.4)
            elif slot == BULLET:
                mesh, color, size = "disc", "bullet", 0.13
            else:
                mesh, color, size = "disc", "star", 0.05
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": 0.0 if exploding else float(col["vx"][k]),
                        "vy": 0.0 if exploding else float(col["vy"][k]),
                        "mesh": mesh, "color": color, "size": size,
                        "roll": 0.785 if mesh == "rocket" else 0.0, "lean": 0.25 if slot == PLAYER else 0.55,
                        "owner": int(col["owner"][k]) if slot in (SHOT, BULLET) and not exploding else -1,
                        "age": clock - float(col["born"][k]),
                        "spin": 2.5 if mesh == "ufo" else 0.0, "bob": 0.0})
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]), "detail": f"survived {s.column('clock')[0]:.0f} s"}

    def describe(self) -> str:
        s = self.session
        lives, score = int(s.column("lives")[0]), int(s.column("score")[0])
        enemies = int(s.column("alive")[ENEMIES[0]:ENEMIES[1]].sum())
        bullets = int(s.column("alive")[BULLETS[0]:BULLETS[1]].sum())
        seconds, calls = s.last_round
        head = "GAME OVER · R to restart" if lives <= 0 else f"lives {lives}"
        return (f"{head} · score {score} · {enemies} enemies · {bullets} bullets · "
                f"tick {seconds * 1000:.0f} ms = {calls} law steps")
