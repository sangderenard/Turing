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
    369..398       clouds: frosted-glass scenery at their own heights, drifting
                   faster the higher they float (parallax); never collide
    399..406       missiles: ballistic rounds on engine_toy's trajectory law
    407..412       pickups: weapon power-ups dropped by destroyed enemies
    413..424       smoke: black clouds left where a craft was destroyed; they
                   drift back with the scroll, sink toward the land and fade
    425..428       flashes: lightning, at the striking cloud and the craft struck

STORMS: smoke drifting into the same column as a cloud (overlap on the
board, any height) turns it into a RAIN cloud (``kind`` RAIN) until it
scrolls away.  A rain cloud strikes at random (STRIKE_RATE a second); the
strike chains to every craft within CHAIN_R of the cloud: enemies take
LIGHTNING_DAMAGE, the player's shield gains LIGHTNING_CHARGE points.

BOSS: lane 48 (the last enemy lane, reserved) is the level's boss, flown by
the trained policy on a mirrored view of the board (the server writes its
``pax``/``pay``/``paz``/``pfire`` controls before each landing).  It arrives
after BOSS_TIME seconds, or sooner once the player has cruised CALM_TIME
seconds with the shield above CALM_SHIELD (``calm``).  While it lives and the
player stays calm, network-flown ACES join every ACE_EVERY seconds.
Destroying it clears the level (``boss_phase`` 2: a win).

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
SMOKES = (413, 425)
FLASHES = (425, 429)
LANES = FLASHES[1]
FLASH = 8.0
RAIN = 1.0
STRIKE_RATE = 0.35
CHAIN_R = 2.6
LIGHTNING_DAMAGE = 4.0
LIGHTNING_CHARGE = 40.0
FLASH_LIFE = 0.22
BOSS_LANE = ENEMIES[1] - 1
SMOKE = 7.0
ACE, BOSS_KIND = 3.0, 4.0
BOSS_TIME = 75.0
CALM_TIME = 20.0
CALM_SHIELD = 0.85
ACE_EVERY = 7.0
BOSS_HP = 120.0
ACE_HP = 6.0
PILOT_SPEED, PILOT_TAU = 6.5, 0.12
SMOKE_LIFE = 4.0
#: Weapons (the ship's ``weapon`` column) and pickup kinds (a pickup's ``kind``).
BASIC, LASER, MISSILE = 0.0, 1.0, 2.0
SPREAD_UP, MULT_UP, MISSILE_UP, LASER_UP, SHIELD_UP, HEALTH_UP = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
HEALTH_PICKUP = 40.0
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

#: VITALS.  ``lives`` holds health points (0..HEALTH_MAX, slow regeneration)
#: and ``energy`` the shield charge as a fraction of SHIELD_POINTS (fast
#: regeneration).  Damage drains the shield first; what it cannot hold comes
#: off health; the run ends at zero health.
HEALTH_MAX = 100.0
HEALTH_REGEN = 0.5            # health points a second
SHIELD_POINTS = 200.0
SHIELD_RATE = 5.0             # shield points a second (a full refill takes 40 s)
SHIELD_PAUSE = 1.5            # no regeneration for this long after any hit (the player's ``age`` counts it)
BULLET_DAMAGE = 10.0
RAM_DAMAGE = 25.0
LIVES = HEALTH_MAX            # the run starts at full health
#: PACE scales how often enemies arrive and fire: the user plays at USER_PACE,
#: training and the demo at 1.  LEVEL steps difficulty up by LEVEL_STEP each.
USER_PACE = 0.35
LEVEL_STEP = 0.15
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
SHIELD_COST = BULLET_DAMAGE / SHIELD_POINTS      # (historical name: one bullet, as charge)
SHIELD_REGEN = SHIELD_RATE / SHIELD_POINTS
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

#: Glyphs this game adds to the exported meshes (web_export.py): the cloud.
EMOJI = {"cloud": "\u2601"}

CONTROLS = {"move_axis": "move-axis-x", "move_axis_y": "move-axis-y", "move_axis_z": "move-axis-z",
            "steer_x": "pointer-board-x", "steer_y": "pointer-board-y", "fire": "primary-action"}


def scroller_step(lane, slot, rx, ry, rz,
                  alive, x, y, z, vx, vy, vz, hp, boom, born, owner, energy, kind, age, timer,
                  weapon, mult, fan, pax, pay, paz, pfire, calm, boss_phase, ace_clock,
                  score, lives, invuln, clock, spawn_clock, field_w, pace, level,
                  move_axis, move_axis_y, move_axis_z, steer_x, steer_y, fire, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    f = as_float
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_shot, is_enemy = slot == PLAYER, slot == SHOT, slot == ENEMY
    is_bullet, is_star = slot == BULLET, slot == STAR
    is_missile, is_pickup = slot == MISSILE_SLOT, slot == PICKUP
    is_smoke = slot == SMOKE
    is_flash = slot == FLASH
    is_boss_lane = lane == BOSS_LANE
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
    enemy_spawn = first_of(is_enemy & ~live & ~is_boss_lane, lane) & arrive
    roll = noise(clock * 3.7 + 1.0)
    new_kind = where(roll < 0.75, DRIFTER, SPINNER)
    tempo = pace * (1.0 + LEVEL_STEP * (level - 1.0))       # arrivals and volleys scale with it
    spawn_clock_next = where(arrive, (1.1 - clock / 200.0).clip(0.35, 1.1) * (0.6 + 0.8 * noise(clock * 1.3)) / tempo,
                             spawn_clock - dt)
    enemy_x0 = 1.0 + (field_w - 2.0) * noise(clock * 5.1 + 2.0)
    # The enemies' height routine (see DWELL): its value and its slope.
    phase = clock_next * ROUTINE_RATE + lane * 1.3
    wave = DWELL * phase.sin()
    held_wave = where(wave > 1.0, 1.0, where(wave < -1.0, -1.0, wave))
    span_z = 0.5 * (DEPTH - 2.0 * Z_MARGIN)
    routine_z = Z_MARGIN + span_z * (1.0 + held_wave)
    routine_vz = where(wave.abs() < 1.0, span_z * DWELL * ROUTINE_RATE * phase.cos(), 0.0)

    # The boss arrives on time, or early once the player has cruised calm;
    # while it lives and the player stays calm, aces join.
    shield_now = (f(is_player) * energy).sum()
    calm_next = where(playing & (shield_now >= CALM_SHIELD), calm + dt, 0.0 * calm)
    phase0 = (f(is_player) * boss_phase).sum()
    boss_due = playing & (phase0 == 0.0) & ((clock >= BOSS_TIME) | ((f(is_player) * calm).sum() >= CALM_TIME))
    boss_spawn = is_boss_lane & ~live & boss_due
    ace_ready = playing & (phase0 == 1.0) & ((f(is_player) * ace_clock).sum() <= 0.0) & (shield_now >= CALM_SHIELD)
    ace_spawn = first_of(is_enemy & ~live & ~is_boss_lane & ~enemy_spawn, lane) & ace_ready
    ace_clock_next = where(ace_ready, ACE_EVERY / tempo, (ace_clock - dt).clip(0.0, 100.0))
    piloted = (is_boss_lane | (is_enemy & (kind == ACE))) & live
    # One volley per step: the ready enemy in the lowest lane fires, into the
    # first free bullet lanes (drifters an aimed spread, spinners a ring).
    ready = is_enemy & live & (timer <= 0.0) & (y > 0.3) & playing & (~piloted | (pfire > 0.0))
    shooter = first_of(ready, lane)
    fired = f(shooter).sum() > 0.0
    sx, sy, sz = (f(shooter) * x).sum(), (f(shooter) * y).sum(), (f(shooter) * z).sum()
    skind = (f(shooter) * kind).sum()
    shooter_lane = (f(shooter) * lane).sum()
    count = where(skind == SPINNER, RING, where(skind == BOSS_KIND, 5.0, SPREAD))
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

    arrival = enemy_spawn | boss_spawn | ace_spawn
    spawn = shot_spawn | arrival | bullet_spawn | missile_spawn
    x1 = where(shot_spawn, shot_x, where(enemy_spawn | ace_spawn, enemy_x0, where(boss_spawn, 0.5 * field_w,
               where(bullet_spawn, sx, where(missile_spawn, px + missile_c * 0.3, x)))))
    y1 = where(shot_spawn | missile_spawn, py - MUZZLE, where(arrival, -0.8, where(bullet_spawn, sy + 0.2, y)))
    z1 = where(shot_spawn | missile_spawn, pz, where(arrival, routine_z, where(bullet_spawn, sz, z)))
    kind1 = where(enemy_spawn, new_kind, where(boss_spawn, BOSS_KIND, where(ace_spawn, ACE, kind)))
    age1 = where(spawn, 0.0, age + dt)
    level_hp = 1.0 + 0.5 * (level - 1.0)
    hp1 = where(enemy_spawn, where(new_kind == SPINNER, 9.0, 3.0),
                where(boss_spawn, BOSS_HP * level_hp, where(ace_spawn, ACE_HP * level_hp, hp)))
    reload_of = where(pw == LASER, LASER_RELOAD, where(pw == MISSILE, MISSILE_RELOAD, RELOAD))
    timer1 = where(is_player, where(volley, reload_of, (timer - dt).clip(0.0, 10.0)),
                   where(arrival, 0.8 / tempo,
                         where(shooter, where(skind == SPINNER, 0.55, where(skind == BOSS_KIND, 0.45,
                                                                            where(skind == ACE, 1.1, 1.5))) / tempo,
                               timer - dt)))
    # Enemy paths: drifters fall and sway; spinners descend, hold, then leave.
    drifter = is_enemy & (kind1 == DRIFTER) & ~is_boss_lane
    spinner = is_enemy & (kind1 == SPINNER) & ~is_boss_lane
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
    # Piloted craft (the boss, aces): the shared damped motion law under the
    # pilot's controls, kept to the upper field.
    pilot_x, pilot_vx = damped_axis(x1, vx, pax, zero - 1.0, zero + MARGIN, field_w - MARGIN, PILOT_SPEED, PILOT_TAU, REACH, dt)
    pilot_y, pilot_vy = damped_axis(y1, vy, pay, zero - 1.0, zero + 0.6, zero + 0.55 * HEIGHT, PILOT_SPEED, PILOT_TAU, REACH, dt)
    pilot_z, pilot_vz = damped_axis(z1, vz, paz, zero - 1.0, zero + Z_MARGIN, zero + DEPTH - Z_MARGIN, PILOT_SPEED, PILOT_TAU, REACH, dt)
    flown = piloted & ~arrival
    vx1, vy1, vz1 = where(flown, pilot_vx, vx1), where(flown, pilot_vy, vy1), where(flown, pilot_vz, vz1)
    live1 = live | spawn
    # Born at the start of the step: a spawn leaves the muzzle at x1/y1 and flies
    # the whole step, so position = birth point + velocity x (clock - born).
    born1 = where(spawn, clock, born)
    owner1 = where(shot_spawn | missile_spawn, 0.0, where(bullet_spawn, shooter_lane, where(enemy_spawn, -1.0, owner)))

    moving = (live1 & playing) | is_star
    x2 = where(is_player & moving, ship_x, where(moving, x1 + vx1 * dt, x1))
    y2 = where(is_player & moving, ship_y, where(moving, y1 + vy1 * dt, y1))
    z2 = where(is_player & moving, ship_z, where(is_enemy & moving, routine_z, where(moving, z1 + vz1 * dt, z1)))
    x2 = where(flown & moving, pilot_x, x2)
    y2 = where(flown & moving, pilot_y, y2)
    z2 = where(flown & moving, pilot_z, z2)
    # Missiles fly engine_toy's trajectory law; height is the law's vertical
    # (y) axis, the field's y its z.
    mx, mz_, my, mvx, mvz_, mvy = trajectory_step(x1, z1, y1, vx1, vz1, vy1, dt, MISSILE_MASS, MISSILE_DIAMETER,
                                                  MISSILE_DRAG, AIR_DENSITY, GRAVITY, 0.0, 0.0, 0.0)[:6]
    flying = is_missile & moving
    x2, y2, z2 = where(flying, mx, x2), where(flying, my, y2), where(flying, mz_, z2)
    vx1, vy1, vz1 = where(flying, mvx, vx1), where(flying, mvy, vy1), where(flying, mvz_, vz1)
    # Stars wrap from the bottom back to the top: the scroll.
    wrapped = is_star & (y2 > HEIGHT + 0.3)
    y2 = where(wrapped, y2 - (HEIGHT + 0.6), y2)
    soaked_cloud = block_hits(x2, y2, rx, ry, live1 | is_star, None, STARS, SMOKES) > 0.0      # same column, any height
    kind1 = where(is_star, where(wrapped, 0.0, where(soaked_cloud, RAIN, kind1)), kind1)
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
    bullet_strikes = block_hits(x2, y2, rx, ry, exposed, None, (0, 1), BULLETS, z2, rz)
    rams = block_hits(x2, y2, rx, ry, exposed, None, (0, 1), ENEMIES, z2, rz)        # the boss lane is an enemy lane
    player_hits = bullet_strikes + rams
    bullet_hits = block_hits(x2, y2, rx, ry, live2 & ((invuln <= 0.0) | ~is_player), None, BULLETS, (0, 1), z2, rz)
    # Lightning: one rain cloud may strike this step; it chains to every
    # craft within CHAIN_R of the cloud on the board.
    raining = is_star & (kind1 == RAIN)
    strike_dice = noise(clock * 5.7 + lane * 1.31 + 9.0)
    striker = first_of(raining & (strike_dice < STRIKE_RATE * dt) & playing, lane)
    striking = f(striker).sum() > 0.0
    lx, ly, lz = (f(striker) * x2).sum(), (f(striker) * y2).sum(), (f(striker) * z2).sum()
    reach_l = ((x2 - lx) * (x2 - lx) + (y2 - ly) * (y2 - ly)).sqrt()
    struck = striking & live2 & (is_enemy | is_player) & (reach_l < CHAIN_R)
    zapped = f(struck & is_player).sum() > 0.0
    hp2 = hp1 - enemy_hits - LIGHTNING_DAMAGE * f(struck & is_enemy)
    killed = is_enemy & live2 & (hp2 <= 0.0)
    touched = is_player & live2 & (player_hits > 0.0)
    # Damage drains the shield's points first, the rest comes off health;
    # shield and health pickups refill them; both regenerate (health slowly).
    damage = (f(is_player & live2) * (BULLET_DAMAGE * bullet_strikes + RAM_DAMAGE * rams)).sum()
    shield_pts = (f(is_player) * energy).sum() * SHIELD_POINTS
    soaked = where(damage < shield_pts, damage, shield_pts)
    wound = damage - soaked
    refill = got_kind(SHIELD_UP)
    since_hit = (f(is_player) * age).sum()
    recharging = playing & (damage <= 0.0) & (since_hit >= SHIELD_PAUSE)
    shield_after = where(refill, SHIELD_POINTS, shield_pts - soaked + SHIELD_RATE * dt * f(recharging)
                         + LIGHTNING_CHARGE * f(zapped))          # lightning charges the shield
    energy_next = where(is_player, (shield_after / SHIELD_POINTS).clip(0.0, 1.0), energy)
    healed = where(got_kind(HEALTH_UP), HEALTH_PICKUP, 0.0)
    health = (lives - wound + healed + HEALTH_REGEN * dt * f(lives > 0.0)).clip(0.0, HEALTH_MAX)
    lives_next = where(playing, health, lives)
    over = lives_next <= 0.0
    hit = touched & (wound > 0.0)
    shielded = (damage > 0.0) & (wound <= 0.0)
    was_hit = wound > 0.0
    # A hit clears the screen of bullets and grants a moment of grace.
    gone = (killed | (is_shot & (shot_hits > 0.0)) | (is_missile & (missile_hits > 0.0)) | got
            | (is_bullet & (bullet_hits > 0.0)) | (hit & over))
    live3 = live2 & ~gone
    # A destroyed enemy sometimes drops a pickup (into the first free pickup
    # lane, where the first of the step's kills died).
    killer = first_of(killed, lane)
    dice = noise(clock * 7.3 + score * 0.01 + 3.0)
    dropping = (f(killer).sum() > 0.0) & (dice < DROP_CHANCE) & playing
    drop = first_of(is_pickup & ~live3, lane) & dropping
    drop_kind = (1.0 + (noise(clock * 2.9 + 11.0) * 6.0).floor()).clip(1.0, 6.0)
    kx, ky, kz = (f(killer) * x2).sum(), (f(killer) * y2).sum(), (f(killer) * z2).sum()
    x2, y2, z2 = where(drop, kx, x2), where(drop, ky, y2), where(drop, kz, z2)
    vx1, vy1, vz1 = where(drop, 0.0, vx1), where(drop, 1.2, vy1), where(drop, 0.0, vz1)
    kind1 = where(drop, drop_kind, kind1)
    born1 = where(drop, clock, born1)
    age1 = where(drop, 0.0, age1)
    age1 = where(is_player & (damage > 0.0), 0.0, age1)      # the player's age: time since its last hit
    live3 = live3 | drop
    # Smoke where a craft was destroyed: the first free smoke lane, at the
    # step's first kill; it drifts back with the scroll, sinks and fades.
    puff = first_of(is_smoke & ~live3, lane) & (f(killer).sum() > 0.0)
    x2, y2, z2 = where(puff, kx, x2), where(puff, ky, y2), where(puff, (kz * 0.5).clip(0.15, 1.2), z2)
    vx1 = where(puff, 0.3 * (noise(clock * 4.1 + 5.0) - 0.5), vx1)
    vy1 = where(puff, 2.4, vy1)
    vz1 = where(puff, -0.08, vz1)
    born1 = where(puff, clock, born1)
    age1 = where(puff, 0.0, age1)
    live3 = (live3 | puff) & ~(is_smoke & (age1 > SMOKE_LIFE))
    nearest_struck = first_of(struck & (where(struck, reach_l, 1.0e9) == where(struck, reach_l, 1.0e9).min()), lane)
    tx_l, ty_l, tz_l = (f(nearest_struck) * x2).sum(), (f(nearest_struck) * y2).sum(), (f(nearest_struck) * z2).sum()
    hit_any = f(nearest_struck).sum() > 0.0
    flash_rank = f(is_flash & ~live3).cumsum(dim=0) - 1.0
    flash_at_cloud = is_flash & ~live3 & (flash_rank == 0.0) & striking
    flash_at_craft = is_flash & ~live3 & (flash_rank == 1.0) & striking & hit_any
    flashing = flash_at_cloud | flash_at_craft
    x2 = where(flash_at_cloud, lx, where(flash_at_craft, tx_l, x2))
    y2 = where(flash_at_cloud, ly, where(flash_at_craft, ty_l, y2))
    z2 = where(flash_at_cloud, lz, where(flash_at_craft, tz_l, z2))
    vx1, vy1, vz1 = where(flashing, 0.0, vx1), where(flashing, 0.0, vy1), where(flashing, 0.0, vz1)
    born1 = where(flashing, clock, born1)
    age1 = where(flashing, 0.0, age1)
    live3 = (live3 | flashing) & ~(is_flash & (age1 > FLASH_LIFE))
    boss_killed = (f(killed & is_boss_lane).sum() > 0.0)
    boss_phase_next = where(boss_killed, 2.0, where(boss_due, 1.0, boss_phase))
    boom3 = where(killed | (hit & over), BOOM, (boom - dt).clip(0.0, BOOM))
    invuln_next = where(was_hit, 0.6, where(shielded, 0.25, (invuln - dt).clip(0.0, INVULN)))
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
            weapon_next, mult_next, fan_next, pax, pay, paz, pfire, calm_next, boss_phase_next, ace_clock_next,
            score_next, lives_next, invuln_next, clock_next, spawn_clock_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "z", "vx", "vy", "vz", "hp", "boom", "born", "owner", "energy", "kind", "age", "timer",
         "weapon", "mult", "fan", "pax", "pay", "paz", "pfire", "calm", "boss_phase", "ace_clock",
         "score", "lives", "invuln", "clock", "spawn_clock")
SETTINGS = ("field_w", "pace", "level")


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


def initial_columns(width: float = WIDTH, pace: float = 1.0, level: int = 1) -> dict[str, np.ndarray]:
    lane = np.arange(LANES, dtype=np.float64)
    slot = np.zeros(LANES)
    rx, ry, rz = np.full(LANES, HITBOX), np.full(LANES, HITBOX), np.full(LANES, 0.18)
    rx[0], ry[0], rz[0] = SHIP_RX, SHIP_RY, 0.35
    rz[SHOTS[0]:SHOTS[1]], rz[ENEMIES[0]:ENEMIES[1]] = 0.25, 0.45
    missiles, pickups = slice(*MISSILES), slice(*PICKUPS)
    slot[missiles], rx[missiles], ry[missiles], rz[missiles] = MISSILE_SLOT, 0.18, 0.3, 0.25
    slot[pickups], rx[pickups], ry[pickups], rz[pickups] = PICKUP, 0.35, 0.35, 0.4
    slot[slice(*SMOKES)] = SMOKE
    rx[slice(*SMOKES)], ry[slice(*SMOKES)] = 0.5, 0.5
    rx[slice(*STARS)], ry[slice(*STARS)] = 0.55, 0.45                  # a cloud's footprint on the board
    slot[slice(*FLASHES)] = FLASH
    rx[BOSS_LANE], ry[BOSS_LANE], rz[BOSS_LANE] = 1.1, 0.75, 0.75      # the boss: a big hull
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
    z[stars] = rng.uniform(0.05, 1.2, STARS[1] - STARS[0])      # low clouds, close over the land
    vy[stars] = 0.8 + 0.7 * z[stars]          # clouds: higher drifts faster (parallax)
    alive = ((slot == PLAYER) | (slot == STAR)).astype(np.float64)
    return {
        "lane": lane, "slot": slot, "rx": rx, "ry": ry, "rz": rz,
        "alive": alive, "x": x, "y": y, "z": z, "vx": np.zeros(LANES), "vy": vy, "vz": np.zeros(LANES),
        "hp": np.zeros(LANES),
        "boom": np.zeros(LANES), "born": np.zeros(LANES), "owner": np.full(LANES, -1.0),
        "energy": (slot == PLAYER).astype(np.float64), "kind": np.zeros(LANES), "age": np.zeros(LANES),
        "timer": np.zeros(LANES), "weapon": np.full(LANES, BASIC), "mult": np.ones(LANES), "fan": np.zeros(LANES),
        **{name: np.zeros(LANES) for name in ("pax", "pay", "paz", "pfire", "calm", "boss_phase", "ace_clock")},
        "score": np.zeros(LANES), "lives": np.full(LANES, LIVES),
        "invuln": np.full(LANES, 1.0), "clock": np.zeros(LANES), "spawn_clock": np.full(LANES, 1.5),
        "field_w": np.full(LANES, float(width)), "pace": np.full(LANES, float(pace)), "level": np.full(LANES, float(level)),
        **{name: np.zeros(LANES) for name in CONTROLS}, "steer_x": np.full(LANES, -1.0),
        "steer_y": np.full(LANES, -1.0),
    }


class BulletScroller(EntityGame):
    game = "bullet_hell"
    title = "Bullet hell (free board)"
    height = HEIGHT
    CONTROLS = tuple(CONTROLS)
    PLAYER_LANE = 0

    def __init__(self, board_id: str = "", aspect: float | None = None, level: int = 1):
        self.width = field_width(aspect)
        self.level = max(1, int(level))
        super().__init__(board_id)

    def make_piece(self):
        return ScrollerPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.width, self.pace(), self.level)

    def pace(self) -> float:
        """Full speed for training episodes and the demo; USER_PACE for a person."""
        return 1.0 if (getattr(self, "autopilot", False) or not getattr(self, "learns", True)) else USER_PACE

    def drive_settings(self) -> None:
        """Before each landing: the pace for whoever drives now, and the pilots."""
        column = self.session.column("pace")
        if column[0] != self.pace():
            column[...] = self.pace()
        self.fly_pilots()

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

    @property
    def emissive(self) -> dict[str, list[float]]:
        return {"lightning": [2.6, 2.8, 3.4]}

    def palette(self) -> dict[str, list[float]]:
        return {"rocket": [0.85, 0.88, 0.95], "rocket_blink": [0.35, 0.37, 0.45], "bolt": [1.0, 0.92, 0.25],
                "invader": [0.75, 0.35, 0.95], "ufo": [0.95, 0.15, 0.55], "bullet": [1.0, 0.25, 0.35],
                "bullet_ring": [0.35, 0.8, 1.0], "star": [0.55, 0.6, 0.8], "boom": [1.0, 0.55, 0.1],
                "cloud": [0.96, 0.98, 1.0], "boss": [0.95, 0.2, 0.25], "ace": [0.3, 1.0, 0.55],
                "rain": [0.5, 0.52, 0.56], "lightning": [0.85, 0.9, 1.0],
                "smoke_0": [0.04, 0.04, 0.05], "smoke_1": [0.07, 0.07, 0.08], "smoke_2": [0.1, 0.1, 0.11],
                "smoke_3": [0.13, 0.13, 0.14],
                "laser": [0.3, 1.0, 0.9], "missile": [1.0, 0.75, 0.3],
                "pickup_S": [0.35, 0.95, 0.45], "pickup_X": [1.0, 0.85, 0.2],
                "pickup_M": [1.0, 0.5, 0.2], "pickup_L": [0.3, 0.95, 1.0],
                "pickup_O": [0.35, 0.6, 1.0], "pickup_H": [1.0, 0.3, 0.35]}

    def entities(self) -> list[dict[str, Any]]:
        names = ("slot", "alive", "x", "y", "z", "vx", "vy", "vz", "boom", "born", "owner", "kind", "invuln", "weapon")
        # (clouds and flashes are always shown; flashes glow -- see emissive)
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
                kind_k = float(col["kind"][k])
                if kind_k == BOSS_KIND:
                    mesh, color, size = "ufo", "boss", 1.35
                elif kind_k == ACE:
                    mesh, color, size = "alien", "ace", 0.5
                elif kind_k == SPINNER:
                    mesh, color, size = "ufo", "ufo", 0.55
                else:
                    mesh, color, size = "invader", "invader", 0.4
            elif slot == SMOKE:
                # black smoke, thinning as it ages
                step = min(3, int(4.0 * float(clock - col["born"][k]) / SMOKE_LIFE))
                mesh, color, size = "cloud", f"smoke_{step}", 0.42 + 0.1 * step
            elif slot == BULLET:
                mesh, color, size = "disc", "bullet", 0.13
            elif slot == MISSILE_SLOT:
                mesh, color, size = "rocket", "missile", 0.24
            elif slot == PICKUP:
                letter = {SPREAD_UP: "S", MULT_UP: "X", MISSILE_UP: "M", LASER_UP: "L",
                          SHIELD_UP: "O", HEALTH_UP: "H"}.get(float(col["kind"][k]), "S")
                mesh, color, size = letter, f"pickup_{letter}", 0.34
            elif slot == FLASH:
                mesh, color, size = "bolt", "lightning", 0.55
            else:
                # A cloud: frosted glass, larger the higher it floats; grey
                # when smoke has turned it to rain.
                rain = float(col["kind"][k]) == RAIN
                mesh, color, size = "cloud", "rain" if rain else "cloud", (0.34 if rain else 0.26) + 0.08 * float(col["z"][k])
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": 0.0 if exploding else float(col["vx"][k]),
                        "vy": 0.0 if exploding else float(col["vy"][k]),
                        "z": float(col["z"][k]), "vz": 0.0 if exploding else float(col["vz"][k]),
                        "mesh": mesh, "color": color, "size": size,
                        # Top-down: the ship lies flat, nose up the screen (the
                        # direction of illusory travel); a missile faces its
                        # flight (the page aims it along its velocity each frame).
                        "roll": 0.785 if mesh == "rocket" else 0.0,
                        "lean": 0.0 if slot in (PLAYER, MISSILE_SLOT) else 0.55,
                        "face": bool(slot == MISSILE_SLOT and not exploding),
                        # drawn with the glass (blended), not in the opaque pass
                        "translucent": bool(slot in (STAR, SMOKE)),
                        "spin": 9.0 if slot == FLASH else (2.5 if mesh == "ufo" else 0.0),
                        "owner": int(col["owner"][k]) if slot in (SHOT, BULLET) and not exploding else -1,
                        "age": clock - float(col["born"][k]),
                        "bob": 0.0})
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None:
            return None
        if float(s.column("boss_phase")[0]) == 2.0:
            return {"score": float(s.column("score")[0]), "won": True, "level": self.level,
                    "detail": f"level {self.level} cleared in {s.column('clock')[0]:.0f} s"}
        if s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]), "detail": f"survived {s.column('clock')[0]:.0f} s"}

    def reset(self) -> None:
        """A cleared level's reset starts the next level."""
        if self.session.state is not None and float(self.session.column("boss_phase")[0]) == 2.0:
            self.level += 1
        super().reset()

    def fly_pilots(self) -> None:
        """Before each landing: the trained policy flies the boss and the aces
        on a mirrored view of the board (autopilot.pilot_controls)."""
        s = self.session
        if s.state is None:
            return
        alive, kind = s.column("alive"), s.column("kind")
        lanes = [BOSS_LANE] if alive[BOSS_LANE] > 0.0 else []
        lanes += [int(k) for k in range(ENEMIES[0], BOSS_LANE) if alive[k] > 0.0 and kind[k] == ACE]
        if not lanes:
            return
        from meowdoku.autopilot import pilot_controls
        for k in lanes:
            ax, ay, az, fire = pilot_controls(self, k)
            for name, value in (("pax", ax), ("pay", ay), ("paz", az), ("pfire", fire)):
                s.column(name)[k] = value

    HOSTILE_LANES = (ENEMIES, BULLETS)
    # Lo-fi arcade: the backdrop shows through (no floor slab); the glass
    # plate and the arcade ring are on by default so the lights glow into it.
    SCENE = {"floor": False, "glass": True, "arcade": True, "terrain": True}
    SCENE_PREFER = {"glass": True, "arcade": True}
    HEALTH_MAX = HEALTH_MAX

    def describe(self) -> str:
        s = self.session
        lives, score = int(s.column("lives")[0]), int(s.column("score")[0])
        enemies = int(s.column("alive")[ENEMIES[0]:ENEMIES[1]].sum())
        bullets = int(s.column("alive")[BULLETS[0]:BULLETS[1]].sum())
        seconds, calls = s.last_round
        head = ("GAME OVER · R to restart" if lives <= 0
                else f"level {self.level} · health {lives} · shield {s.column('energy')[0] * SHIELD_POINTS:.0f}")
        weapon = {BASIC: "blaster", LASER: "laser", MISSILE: "missiles"}.get(float(s.column("weapon")[0]), "blaster")
        head += f" · {weapon} x{1 + int(s.column('mult')[0])} fan {int(s.column('fan')[0])}"
        phase = float(s.column("boss_phase")[0])
        if phase == 1.0:
            boss_hp = float(s.column("hp")[BOSS_LANE])
            head += f" · BOSS {boss_hp:.0f} hp"
        elif phase == 2.0:
            head = f"LEVEL {self.level} CLEARED · " + head
        return (f"{head} · score {score} · {enemies} enemies · {bullets} bullets · "
                f"tick {seconds * 1000:.0f} ms = {calls} law steps")
