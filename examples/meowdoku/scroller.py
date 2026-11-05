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
    399..406       missiles: ballistic rounds on engine_toy's trajectory law
    407..412       pickups: weapon power-ups dropped by destroyed enemies

WEAPONS: the ship carries ``weapon`` (basic / laser / missile), ``mult``
(1..4: 1 + mult shots a volley) and ``fan`` (0..3: the volley's spread).
Destroyed enemies sometimes drop a pickup -- SPREAD (fan + 1), MULT
(mult + 1), MISSILE, LASER -- that drifts down and is pulled hard toward the
ship inside MAGNET_R (harder the closer), collected on contact.  Missiles are
point masses on ``symbolic_parts.trajectory_batch_step`` (drag on
air-relative velocity, gravity along height), launched with the firing
solution that lands them on the nearest enemy; they hit for MISSILE_DAMAGE.

DEPTH: every lane also has a height ``z`` above the board (0..DEPTH) with
velocity ``vz`` and a depth half-extent ``rz``: the ship climbs and dives on
the same damped motion law, shots leave at its height, aimed spreads aim in
three dimensions, spinner rings stay level, and enemies run a clipped-sine
height routine that holds them at the floor or the ceiling most of the time
(crossing between), so they are reachable.  Hits need overlap in depth too.
The layer is physically compressed (DEPTH units); the page exaggerates the
drawn scale by height.

Columns: lane, slot, rz (static); alive, x, y, z, vx, vy, vz, hp, boom, born, owner, energy, kind,
age, timer (entity state: ``timer`` is the player's reload and an enemy's
time to its next volley); score, lives, invuln, clock, spawn_clock, field_w
(game-wide); move_axis, move_axis_y, move_axis_z, steer_x, steer_y, fire (controls).
"""

from __future__ import annotations

from typing import Any

import numpy as np

import sys
from pathlib import Path

from meowdoku.entities import EntityGame, LawPiece, as_float, block_hits, first_of, noise
from meowdoku.entity_motion import damped_axis
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

# engine_toy's ballistics: the SymPy trajectory law printed as AbstractTensor
# code over a batch (symbolic_parts.trajectory_batch_step), used as is.
_ENGINE_TOY = Path(__file__).resolve().parents[3] / "engine_toy"
if str(_ENGINE_TOY) not in sys.path:
    sys.path.append(str(_ENGINE_TOY))
_TRAJECTORY = None


def trajectory_step(*args):
    global _TRAJECTORY
    if _TRAJECTORY is None:
        from symbolic_parts import trajectory_batch_step
        _TRAJECTORY = trajectory_batch_step()
    return _TRAJECTORY(*args)

HEIGHT = 16.0
WIDTH = 9.0
#: The field is portrait (phone-shaped) whatever the screen: its width follows
#: the screen's shape only up to this width-to-height ratio.
PORTRAIT = 0.62
PLAYER, SHOT, ENEMY, BULLET, STAR, MISSILE_SLOT, PICKUP = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
SHOTS = (1, 25)
ENEMIES = (25, 49)
BULLETS = (49, 369)
STARS = (369, 399)
MISSILES = (399, 407)
PICKUPS = (407, 413)
LANES = PICKUPS[1]
#: Weapons (the ship's ``weapon`` column) and pickup kinds (a pickup's ``kind``).
BASIC, LASER, MISSILE = 0.0, 1.0, 2.0
SPREAD_UP, MULT_UP, MISSILE_UP, LASER_UP = 1.0, 2.0, 3.0, 4.0
FAN_STEP = 0.13            # radians between neighbouring shots per fan level
LASER_SPEED = 2.0          # laser shots fly this many times faster
LASER_RELOAD = 0.05
MISSILE_RELOAD = 0.5
MISSILE_DAMAGE = 3.0
DROP_CHANCE = 0.35
MAGNET_R = 2.2
MAGNET_SPEED = 9.0
#: The missiles' trajectory parameters, in game units (the law is unit-free).
MISSILE_MASS, MISSILE_DIAMETER, MISSILE_DRAG, AIR_DENSITY, GRAVITY = 1.0, 0.3, 0.3, 1.2, 6.0
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
#: The craft's regenerative energy shield (the ``energy`` column, player lane):
#: a hit that finds at least SHIELD_COST of charge is absorbed -- the
#: projectile still dies, the ship keeps its life -- and drains that much; the
#: charge regenerates continuously at SHIELD_REGEN per second up to 1.
SHIELD_COST = 0.5
SHIELD_REGEN = 0.25
#: The ship as drawn: half size, body box (hitbox), muzzle at the nose.
#: Height above the board: the field's third axis (physically compressed).
DEPTH = 4.0
Z_MARGIN = 0.3
#: The enemies' height routine: a sine clipped DWELL times over, so it sits at
#: the floor or the ceiling and crosses quickly between them.
DWELL = 1.8
ROUTINE_RATE = 0.9
SHIP_SIZE = 0.5            # body only (exhaust puff dropped at export): ~1.3 units long
SHIP_RX, SHIP_RY = 0.22, 0.4
MUZZLE = 0.63             # the nose: 1.31 drawn half-sizes ahead, foreshortened by the 0.25 lean

CONTROLS = {"move_axis": "move-axis-x", "move_axis_y": "move-axis-y", "move_axis_z": "move-axis-z",
            "steer_x": "pointer-board-x", "steer_y": "pointer-board-y", "fire": "primary-action"}


def scroller_step(lane, slot, rx, ry, rz,
                  alive, x, y, z, vx, vy, vz, hp, boom, born, owner, energy, kind, age, timer,
                  weapon, mult, fan,
                  score, lives, invuln, clock, spawn_clock, field_w,
                  move_axis, move_axis_y, move_axis_z, steer_x, steer_y, fire, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    f = as_float
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_shot, is_enemy = slot == PLAYER, slot == SHOT, slot == ENEMY
    is_bullet, is_star = slot == BULLET, slot == STAR
    is_missile, is_pickup = slot == MISSILE_SLOT, slot == PICKUP
    clock_next = clock + dt
    zero = x * 0.0

    # The ship: the shared damped motion law on both axes.
    ship_x, ship_vx = damped_axis(x, vx, move_axis, steer_x, zero + MARGIN, field_w - MARGIN,
                                  SPEED, TAU, REACH, dt)
    ship_y, ship_vy = damped_axis(y, vy, move_axis_y, steer_y, zero + MARGIN, zero + HEIGHT - MARGIN,
                                  SPEED, TAU, REACH, dt)
    ship_z, ship_vz = damped_axis(z, vz, move_axis_z, zero - 1.0, zero + Z_MARGIN, zero + DEPTH - Z_MARGIN,
                                  SPEED, TAU, REACH, dt)
    me = is_player & live
    px, py, pz = (f(me) * x).sum(), (f(me) * y).sum(), (f(me) * z).sum()
    ship_on = f(me).sum() > 0.0

    # The ship's weapon: 1 + mult shots a volley, fanned by fan; a laser
    # flies faster on a quicker reload; a missile volley launches mult rounds.
    pw, pm, pf = (f(is_player) * weapon).sum(), (f(is_player) * mult).sum(), (f(is_player) * fan).sum()
    reload = (f(is_player) * timer).sum()
    volley = (fire > 0.0) & ship_on & playing & (reload <= 0.0)
    free_shot = is_shot & ~live
    shots_n = 1.0 + pm
    shot_rank = f(free_shot).cumsum(dim=0) - 1.0
    shot_spawn = free_shot & (shot_rank < shots_n) & volley & (pw != MISSILE)
    shot_c = shot_rank - 0.5 * (shots_n - 1.0)
    shot_x = px + shot_c * 0.18
    shot_angle = shot_c * FAN_STEP * pf
    shot_speed = where(pw == LASER, SHOT_SPEED * LASER_SPEED, SHOT_SPEED)
    # Missiles: the firing solution that lands each on the nearest enemy
    # (time of flight from the horizontal distance, vertical speed to rise
    # the height difference and fall back over it), then the trajectory law.
    enemies_live = is_enemy & live
    gap = ((x - px) * (x - px) + (y - py) * (y - py)).sqrt()
    nearest = first_of(enemies_live & (where(enemies_live, gap, 1.0e9) == where(enemies_live, gap, 1.0e9).min()), lane)
    has_target = f(nearest).sum() > 0.0
    tx = where(has_target, (f(nearest) * x).sum(), px)
    ty = where(has_target, (f(nearest) * y).sum(), py - 8.0)
    tz = where(has_target, (f(nearest) * z).sum(), pz)
    flight = (((tx - px) * (tx - px) + (ty - py) * (ty - py)).sqrt() / 9.0)
    flight = where(flight < 0.45, 0.45, where(flight > 1.6, 1.6, flight))
    free_missile = is_missile & ~live
    missile_rank = f(free_missile).cumsum(dim=0) - 1.0
    missile_spawn = free_missile & (missile_rank < pm) & volley & (pw == MISSILE)
    missile_c = missile_rank - 0.5 * (pm - 1.0)

    # Enemies: one arrives whenever the spawn clock runs out, faster as the
    # run goes on; drifters three times as often as spinners.
    arrive = (spawn_clock <= 0.0) & playing
    enemy_spawn = first_of(is_enemy & ~live, lane) & arrive
    roll = noise(clock * 3.7 + 1.0)
    new_kind = where(roll < 0.75, DRIFTER, SPINNER)
    spawn_clock_next = where(arrive, (1.1 - clock / 200.0).clip(0.35, 1.1) * (0.6 + 0.8 * noise(clock * 1.3)),
                             spawn_clock - dt)
    enemy_x0 = 1.0 + (field_w - 2.0) * noise(clock * 5.1 + 2.0)
    # The enemies' height routine (see DWELL): its value and its slope.
    phase = clock_next * ROUTINE_RATE + lane * 1.3
    wave = DWELL * phase.sin()
    held_wave = where(wave > 1.0, 1.0, where(wave < -1.0, -1.0, wave))
    span_z = 0.5 * (DEPTH - 2.0 * Z_MARGIN)
    routine_z = Z_MARGIN + span_z * (1.0 + held_wave)
    routine_vz = where(wave.abs() < 1.0, span_z * DWELL * ROUTINE_RATE * phase.cos(), 0.0)

    # One volley per step: the ready enemy in the lowest lane fires, into the
    # first free bullet lanes (drifters an aimed spread, spinners a ring).
    ready = is_enemy & live & (timer <= 0.0) & (y > 0.3) & playing
    shooter = first_of(ready, lane)
    fired = f(shooter).sum() > 0.0
    sx, sy, sz = (f(shooter) * x).sum(), (f(shooter) * y).sum(), (f(shooter) * z).sum()
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
    # Aimed spreads climb or dive toward the ship height over the flight
    # their horizontal distance takes; rings stay level.
    reach_xy = ((px - sx) * (px - sx) + (py - sy) * (py - sy)).sqrt() + 0.5
    climb = where(skind == SPINNER, 0.0, speed * (pz - sz) / reach_xy)

    spawn = shot_spawn | enemy_spawn | bullet_spawn | missile_spawn
    x1 = where(shot_spawn, shot_x, where(enemy_spawn, enemy_x0, where(bullet_spawn, sx,
               where(missile_spawn, px + missile_c * 0.3, x))))
    y1 = where(shot_spawn | missile_spawn, py - MUZZLE, where(enemy_spawn, -0.8, where(bullet_spawn, sy + 0.2, y)))
    z1 = where(shot_spawn | missile_spawn, pz, where(enemy_spawn, routine_z, where(bullet_spawn, sz, z)))
    kind1 = where(enemy_spawn, new_kind, kind)
    age1 = where(spawn, 0.0, age + dt)
    hp1 = where(enemy_spawn, where(new_kind == SPINNER, 9.0, 3.0), hp)
    reload_of = where(pw == LASER, LASER_RELOAD, where(pw == MISSILE, MISSILE_RELOAD, RELOAD))
    timer1 = where(is_player, where(volley, reload_of, (timer - dt).clip(0.0, 10.0)),
                   where(enemy_spawn, 0.8, where(shooter, where(skind == SPINNER, 0.55, 1.5), timer - dt)))
    # Enemy paths: drifters fall and sway; spinners descend, hold, then leave.
    drifter = is_enemy & (kind1 == DRIFTER)
    spinner = is_enemy & (kind1 == SPINNER)
    sway = 1.8 * (clock_next * 1.9 + lane * 0.7).sin()
    vx1 = where(is_player, ship_vx, where(drifter, sway, where(bullet_spawn, speed * angle.cos(),
                where(shot_spawn, shot_speed * shot_angle.sin(),
                      where(missile_spawn, (tx - x1) / flight, where(spawn | spinner, 0.0, vx))))))
    vy1 = where(is_player, ship_vy,
                where(shot_spawn, -shot_speed * shot_angle.cos(), where(missile_spawn, (ty - y1) / flight,
                      where(drifter, 1.7, where(spinner, where(age1 < 3.0, 1.0, where(age1 < 10.0, 0.0, 1.6)),
                                                where(bullet_spawn, speed * angle.sin(), vy))))))
    vz1 = where(is_player, ship_vz, where(is_enemy, routine_vz, where(bullet_spawn, climb,
                where(missile_spawn, (tz - z1) / flight + 0.5 * GRAVITY * flight, where(spawn, 0.0, vz)))))
    # Pickups drift down; inside MAGNET_R they are pulled hard toward the
    # ship, harder the closer (velocity set toward it).
    to_x, to_y, to_z = px - x, py - y, pz - z
    reach_p = (to_x * to_x + to_y * to_y + to_z * to_z).sqrt() + 1e-6
    pull = is_pickup & live & ship_on & (reach_p < MAGNET_R)
    pull_speed = MAGNET_SPEED * (1.0 + 2.0 * (1.0 - reach_p / MAGNET_R))
    vx1 = where(is_pickup, where(pull, pull_speed * to_x / reach_p, 0.0), vx1)
    vy1 = where(is_pickup, where(pull, pull_speed * to_y / reach_p, 1.2), vy1)
    vz1 = where(is_pickup, where(pull, pull_speed * to_z / reach_p, 0.0), vz1)
    live1 = live | spawn
    # Born at the start of the step: a spawn leaves the muzzle at x1/y1 and flies
    # the whole step, so position = birth point + velocity x (clock - born).
    born1 = where(spawn, clock, born)
    owner1 = where(shot_spawn | missile_spawn, 0.0, where(bullet_spawn, shooter_lane, where(enemy_spawn, -1.0, owner)))

    moving = (live1 & playing) | is_star
    x2 = where(is_player & moving, ship_x, where(moving, x1 + vx1 * dt, x1))
    y2 = where(is_player & moving, ship_y, where(moving, y1 + vy1 * dt, y1))
    z2 = where(is_player & moving, ship_z, where(is_enemy & moving, routine_z, where(moving, z1 + vz1 * dt, z1)))
    # Missiles fly engine_toy's trajectory law; height is the law's vertical
    # (y) axis, the field's y its z.
    mx, mz_, my, mvx, mvz_, mvy = trajectory_step(x1, z1, y1, vx1, vz1, vy1, dt, MISSILE_MASS, MISSILE_DIAMETER,
                                                  MISSILE_DRAG, AIR_DENSITY, GRAVITY, 0.0, 0.0, 0.0)[:6]
    flying = is_missile & moving
    x2, y2, z2 = where(flying, mx, x2), where(flying, my, y2), where(flying, mz_, z2)
    vx1, vy1, vz1 = where(flying, mvx, vx1), where(flying, mvy, vy1), where(flying, mvz_, vz1)
    # Stars wrap from the bottom back to the top: the scroll.
    y2 = where(is_star & (y2 > HEIGHT + 0.3), y2 - (HEIGHT + 0.6), y2)
    out = (y2 < -1.5) | (y2 > HEIGHT + 1.5) | (x2 < -1.5) | (x2 > field_w + 1.5) | (z2 < -0.6) | (z2 > DEPTH + 0.6)
    grounded = is_missile & (z2 < 0.0)                  # a missile that reaches the board is spent
    live2 = live1 & ~((is_shot | is_enemy | is_bullet | is_missile | is_pickup) & out) & ~grounded

    # Collisions, block against block.
    enemy_hits = (block_hits(x2, y2, rx, ry, live2, None, ENEMIES, SHOTS, z2, rz)
                  + MISSILE_DAMAGE * block_hits(x2, y2, rx, ry, live2, None, ENEMIES, MISSILES, z2, rz))
    missile_hits = block_hits(x2, y2, rx, ry, live2, None, MISSILES, ENEMIES, z2, rz)
    collected = block_hits(x2, y2, rx, ry, live2, None, PICKUPS, (0, 1), z2, rz)
    got = is_pickup & live2 & (collected > 0.0)
    got_kind = lambda k: f(got & (kind == k)).sum() > 0.0  # noqa: E731
    weapon_next = where(is_player, where(got_kind(MISSILE_UP), MISSILE, where(got_kind(LASER_UP), LASER, weapon)), weapon)
    mult_next = where(is_player & got_kind(MULT_UP), (mult + 1.0).clip(1.0, 4.0), mult)
    fan_next = where(is_player & got_kind(SPREAD_UP), (fan + 1.0).clip(0.0, 3.0), fan)
    shot_hits = block_hits(x2, y2, rx, ry, live2, None, SHOTS, ENEMIES, z2, rz)
    exposed = live2 & (invuln <= 0.0)
    player_hits = (block_hits(x2, y2, rx, ry, exposed, None, (0, 1), BULLETS, z2, rz)
                   + block_hits(x2, y2, rx, ry, exposed, None, (0, 1), ENEMIES, z2, rz))
    bullet_hits = block_hits(x2, y2, rx, ry, live2 & ((invuln <= 0.0) | ~is_player), None, BULLETS, (0, 1), z2, rz)
    hp2 = hp1 - enemy_hits
    killed = is_enemy & live2 & (hp2 <= 0.0)
    touched = is_player & live2 & (player_hits > 0.0)
    # The craft's shield takes a hit while it holds enough charge (the bullet
    # still dies); only an unshielded hit costs a life and clears the screen.
    absorbed = touched & (energy >= SHIELD_COST)
    hit = touched & ~absorbed
    shielded = f(absorbed).sum() > 0.0
    energy_next = where(is_player, where(absorbed, energy - SHIELD_COST,
                                         (energy + SHIELD_REGEN * dt).clip(0.0, 1.0)), energy)
    was_hit = f(hit).sum() > 0.0
    lives_next = lives - f(hit).sum()
    over = lives_next <= 0.0
    # A hit clears the screen of bullets and grants a moment of grace.
    gone = (killed | (is_shot & (shot_hits > 0.0)) | (is_missile & (missile_hits > 0.0)) | got
            | (is_bullet & ((bullet_hits > 0.0) | was_hit)) | (hit & over))
    live3 = live2 & ~gone
    # A destroyed enemy sometimes drops a pickup (into the first free pickup
    # lane, where the first of the step's kills died).
    killer = first_of(killed, lane)
    dice = noise(clock * 7.3 + score * 0.01 + 3.0)
    dropping = (f(killer).sum() > 0.0) & (dice < DROP_CHANCE) & playing
    drop = first_of(is_pickup & ~live3, lane) & dropping
    drop_kind = (1.0 + (noise(clock * 2.9 + 11.0) * 4.0).floor()).clip(1.0, 4.0)
    kx, ky, kz = (f(killer) * x2).sum(), (f(killer) * y2).sum(), (f(killer) * z2).sum()
    x2, y2, z2 = where(drop, kx, x2), where(drop, ky, y2), where(drop, kz, z2)
    vx1, vy1, vz1 = where(drop, 0.0, vx1), where(drop, 1.2, vy1), where(drop, 0.0, vz1)
    kind1 = where(drop, drop_kind, kind1)
    born1 = where(drop, clock, born1)
    age1 = where(drop, 0.0, age1)
    live3 = live3 | drop
    boom3 = where(killed | (hit & over), BOOM, (boom - dt).clip(0.0, BOOM))
    invuln_next = where(was_hit, INVULN, where(shielded, 0.4, (invuln - dt).clip(0.0, INVULN)))
    score_next = score + (f(killed) * where(kind1 == SPINNER, 300.0, 100.0)).sum()

    # The longest step that steps nothing over anything it can hit, pair by
    # pair: a shot must not cross an enemy's height in one step, a bullet
    # must not cross the ship's hitbox at their closing speed; and the ship
    # moves at most half its damping time per step.
    pace = vx1.abs() + vy1.abs() + vz1.abs()
    shot_pace = where(is_shot & live3, pace, 0.0).max()
    bullet_pace = where(is_bullet & live3, pace, 0.0).max() + (f(is_player) * pace).sum()
    shot_window = 2.0 * (0.3 + 0.36)
    bullet_window = 2.0 * (SHIP_RX + 0.12)
    shot_limit = 0.9 * shot_window / (shot_pace + 1e-6)
    bullet_limit = 0.9 * bullet_window / (bullet_pace + 1e-6)
    dt_limit = where(shot_limit < bullet_limit, shot_limit, bullet_limit).clip(0.0, 0.5 * TAU) + zero
    residual = f(is_enemy & live3).sum() + zero
    return (f(live3), x2, y2, z2, vx1, vy1, vz1, hp2, boom3, born1, owner1, energy_next, kind1, age1, timer1,
            weapon_next, mult_next, fan_next,
            score_next, lives_next, invuln_next, clock_next, spawn_clock_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "z", "vx", "vy", "vz", "hp", "boom", "born", "owner", "energy", "kind", "age", "timer",
         "weapon", "mult", "fan",
         "score", "lives", "invuln", "clock", "spawn_clock")
SETTINGS = ("field_w",)


class ScrollerPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "scroller_step"
    law = staticmethod(scroller_step)
    argument_names = ("lane", "slot", "rx", "ry", "rz", *STATE, *SETTINGS, *CONTROLS, "dt")
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


def field_width(aspect: float | None) -> float:
    if not aspect or aspect <= 0.0:
        return WIDTH
    return float(np.clip(round(HEIGHT * min(aspect, PORTRAIT) * 2.0) / 2.0, 9.0, HEIGHT * PORTRAIT + 0.5))


def initial_columns(width: float = WIDTH) -> dict[str, np.ndarray]:
    lane = np.arange(LANES, dtype=np.float64)
    slot = np.zeros(LANES)
    rx, ry, rz = np.full(LANES, HITBOX), np.full(LANES, HITBOX), np.full(LANES, 0.18)
    rx[0], ry[0], rz[0] = SHIP_RX, SHIP_RY, 0.35
    rz[SHOTS[0]:SHOTS[1]], rz[ENEMIES[0]:ENEMIES[1]] = 0.25, 0.45
    missiles, pickups = slice(*MISSILES), slice(*PICKUPS)
    slot[missiles], rx[missiles], ry[missiles], rz[missiles] = MISSILE_SLOT, 0.18, 0.3, 0.25
    slot[pickups], rx[pickups], ry[pickups], rz[pickups] = PICKUP, 0.35, 0.35, 0.4
    slot[SHOTS[0]:SHOTS[1]], rx[SHOTS[0]:SHOTS[1]], ry[SHOTS[0]:SHOTS[1]] = SHOT, 0.1, 0.3
    slot[ENEMIES[0]:ENEMIES[1]], rx[ENEMIES[0]:ENEMIES[1]], ry[ENEMIES[0]:ENEMIES[1]] = ENEMY, 0.42, 0.36
    slot[BULLETS[0]:BULLETS[1]], rx[BULLETS[0]:BULLETS[1]], ry[BULLETS[0]:BULLETS[1]] = BULLET, 0.12, 0.12
    slot[STARS[0]:STARS[1]] = STAR
    rng = np.random.default_rng(7)
    x, y = np.zeros(LANES), np.zeros(LANES)
    vy = np.zeros(LANES)
    x[0], y[0] = width / 2.0, HEIGHT - 2.0
    z = np.zeros(LANES)
    z[0] = DEPTH / 2.0
    stars = slice(*STARS)
    x[stars] = rng.uniform(0.2, width - 0.2, STARS[1] - STARS[0])
    y[stars] = rng.uniform(-0.3, HEIGHT + 0.3, STARS[1] - STARS[0])
    vy[stars] = rng.uniform(1.5, 5.0, STARS[1] - STARS[0])
    z[stars] = rng.uniform(0.0, DEPTH, STARS[1] - STARS[0])
    alive = ((slot == PLAYER) | (slot == STAR)).astype(np.float64)
    return {
        "lane": lane, "slot": slot, "rx": rx, "ry": ry, "rz": rz,
        "alive": alive, "x": x, "y": y, "z": z, "vx": np.zeros(LANES), "vy": vy, "vz": np.zeros(LANES),
        "hp": np.zeros(LANES),
        "boom": np.zeros(LANES), "born": np.zeros(LANES), "owner": np.full(LANES, -1.0),
        "energy": (slot == PLAYER).astype(np.float64), "kind": np.zeros(LANES), "age": np.zeros(LANES),
        "timer": np.zeros(LANES), "weapon": np.full(LANES, BASIC), "mult": np.ones(LANES), "fan": np.zeros(LANES),
        "score": np.zeros(LANES), "lives": np.full(LANES, LIVES),
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
                "x": [MARGIN, self.width - MARGIN], "y": [MARGIN, HEIGHT - MARGIN],
                "z": [Z_MARGIN, DEPTH - Z_MARGIN], "depth": DEPTH,
                # Twin shots from the nose on a reload, as the law fires them.
                "weapon": {"muzzles": [[-0.2, -MUZZLE], [0.2, -MUZZLE]], "velocity": [0.0, -SHOT_SPEED],
                           "reload": RELOAD, "max_in_flight": SHOTS[1] - SHOTS[0], "mesh": "bolt", "color": "bolt",
                           "size": 0.2}}

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
                "bullet_ring": [0.35, 0.8, 1.0], "star": [0.55, 0.6, 0.8], "boom": [1.0, 0.55, 0.1],
                "laser": [0.3, 1.0, 0.9], "missile": [1.0, 0.75, 0.3],
                "pickup_S": [0.35, 0.95, 0.45], "pickup_X": [1.0, 0.85, 0.2],
                "pickup_M": [1.0, 0.5, 0.2], "pickup_L": [0.3, 0.95, 1.0]}

    def entities(self) -> list[dict[str, Any]]:
        names = ("slot", "alive", "x", "y", "z", "vx", "vy", "vz", "boom", "born", "owner", "kind", "invuln", "weapon")
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
                laser = float(col["weapon"][0]) == LASER
                mesh, color, size = "bolt", "laser" if laser else "bolt", 0.24 if laser else 0.2
            elif slot == ENEMY:
                spinner = col["kind"][k] == SPINNER
                mesh, color, size = ("ufo", "ufo", 0.55) if spinner else ("invader", "invader", 0.4)
            elif slot == BULLET:
                mesh, color, size = "disc", "bullet", 0.13
            elif slot == MISSILE_SLOT:
                mesh, color, size = "rocket", "missile", 0.24
            elif slot == PICKUP:
                letter = {SPREAD_UP: "S", MULT_UP: "X", MISSILE_UP: "M", LASER_UP: "L"}.get(float(col["kind"][k]), "S")
                mesh, color, size = letter, f"pickup_{letter}", 0.34
            else:
                mesh, color, size = "disc", "star", 0.05
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": 0.0 if exploding else float(col["vx"][k]),
                        "vy": 0.0 if exploding else float(col["vy"][k]),
                        "z": float(col["z"][k]), "vz": 0.0 if exploding else float(col["vz"][k]),
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

    HOSTILE_LANES = (ENEMIES, BULLETS)

    def describe(self) -> str:
        s = self.session
        lives, score = int(s.column("lives")[0]), int(s.column("score")[0])
        enemies = int(s.column("alive")[ENEMIES[0]:ENEMIES[1]].sum())
        bullets = int(s.column("alive")[BULLETS[0]:BULLETS[1]].sum())
        seconds, calls = s.last_round
        head = "GAME OVER · R to restart" if lives <= 0 else f"lives {lives}"
        weapon = {BASIC: "blaster", LASER: "laser", MISSILE: "missiles"}.get(float(s.column("weapon")[0]), "blaster")
        head += f" · {weapon} x{1 + int(s.column('mult')[0])} fan {int(s.column('fan')[0])}"
        return (f"{head} · score {score} · {enemies} enemies · {bullets} bullets · "
                f"tick {seconds * 1000:.0f} ms = {calls} law steps")
