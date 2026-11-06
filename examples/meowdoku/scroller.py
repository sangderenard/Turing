"""A dogfight on a sphere, drawn as a top-down bullet hell, as one dt-system entity law.

THE SPHERE: every body flies great circles on a sphere (sphere_motion): a
unit position, a unit heading and a ground speed per lane.  The player turns
(left/right: yaw on the shared damped motion law), throttles (up/down) and
climbs; the screen is the PATCH about the player, its heading up the screen
-- the ship never moves on screen, the world turns and flows beneath it (the
page re-reads the land's relief at the turned, travelling coordinate).  A
FIXED POPULATION of enemies lives on the whole sphere and hunts the player
HEAD ON (racing wide to get in front, then turning in face to face); one
destroyed returns later, far away.  Collisions and drawing use the patch
coordinates ``x, y`` (derived every step); scenery left far behind is
rebuilt ahead.  (Lane notes below that speak of scrolling or of the top of
the screen describe the old flat field.)

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
    429..432       anti-aircraft stations: yellow ground emplacements riding the
                   terrain's scroll; each fires a yellow shell straight at where
                   the player IS (no lead); shots, missiles and rams destroy them;
                   one scrolled past returns at the top, rebuilt elsewhere

STORMS: smoke soaking into a cloud -- overlapping it in three dimensions,
at its own height, for SOAK_TIME (the cloud's ``timer`` counts it) -- turns
it into a RAIN cloud (``kind`` RAIN) until it scrolls away; rare, because
smoke must really meet a cloud.  A rain cloud strikes at random
(STRIKE_RATE a second); the strike chains to every craft within CHAIN_R of
the cloud: enemies take LIGHTNING_DAMAGE, the player LIGHTNING_HIT (shield
first, then health).

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
time to its next volley); ux, uy, uz, hx, hy, hz, spd (the sphere), yaw,
vyaw, vspd (the player's turn and throttle); score, lives, invuln, clock, field_w
(game-wide); move_axis, move_axis_y, move_axis_z, steer_x, steer_y, fire (controls).
"""

from __future__ import annotations

from typing import Any

import numpy as np

import sys
from pathlib import Path

from meowdoku.entities import EntityGame, LawPiece, as_float, block_hits, first_of, noise
from meowdoku.entity_motion import damped_axis, ground_travel
from meowdoku import sphere_motion as sm
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
STATIONS = (429, 433)
BOMBS = (433, 439)
BLASTS = (439, 443)
LANES = BLASTS[1]
STATION, STATION_KIND = 9.0, 5.0
#: ALTERNATE FIRE: bombs (``kind`` is the style) and their blasts.  A DUMB
#: bomb falls on the trajectory law; a PARACHUTE bomb opens its canopy after
#: CHUTE_DEPLOY and hangs, drifting slowly -- a second press air-bursts it; a
#: SEEKER flies powered at the nearest ground emplacement.  Each blast hurts
#: what is inside its own radius and leaves a glowing flash in its colour.
BOMB_SLOT, BLAST_SLOT = 10.0, 11.0
DUMB, CHUTE, SEEKER = 0.0, 1.0, 2.0
BOMB_DIAMETER, BOMB_DRAG = 0.3, 0.3
CHUTE_DEPLOY, CHUTE_DIAMETER, CHUTE_DRAG = 0.35, 3.2, 1.4
SEEK_SPD, SEEK_TURN = 6.0, 3.0
BLAST_R = {DUMB: 1.8, CHUTE: 2.6, SEEKER: 1.3}
BLAST_DAMAGE = {DUMB: 6.0, CHUTE: 4.0, SEEKER: 10.0}
BLAST_LIFE = 0.5
BOMB_LIFE = 14.0
FLAK_SPEED = 6.0
FLAK_RELOAD = 2.4
STATION_HP = 4.0
AA_QUIET = 6.0               # stations hold fire until enemies have not hurt the player this long
FLASH = 8.0
RAIN, THUNDER = 1.0, 2.0
STRIKE_RATE = 0.35           # a rain cloud's strikes a second
CHAIN_R = 2.6
#: THUNDERSTORMS form naturally while enemies have not damaged the player for
#: STORM_QUIET seconds (the player's ``age`` is the time since its last hit):
#: STORM_RATE a second, at most STORM_MAX at once -- larger and higher clouds
#: that strike more often and chain farther.
STORM_QUIET = 10.0
STORM_RATE = 0.06
STORM_MAX = 2
THUNDER_RATE = 2.4          # storms strike often: the nearest craft in reach, else the ground (harmless)
THUNDER_CHAIN_R = 3.5
LIGHTNING_DAMAGE = 4.0
LIGHTNING_HIT = 25.0
SOAK_TIME = 1.5          # smoke must linger in a cloud this long to make rain
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
DEMO_EASY_S, DEMO_CHAOS_S = 60.0, 180.0
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
#: THE SPHERE.  Every body flies great circles on a sphere of RADIUS (its
#: unit position ``ux, uy, uz``, unit heading ``hx, hy, hz``, ground speed
#: ``spd``); the screen is the patch about the player (sphere_motion.project),
#: its heading up the screen, the player at (field_w / 2, PLAYER_Y).
RADIUS = 30.0
PLAYER_Y = 11.0
PATCH_R = 15.0               # bodies this far from the player are drawn
#: The player's flight: left/right TURN (yaw, the damped law: TURN_RATE rad/s
#: at most), up/down THROTTLE (speed SPD_MIN..SPD_MAX, held when released).
#: Yaw is unwrapped from YAW0 so a held finger's target is never negative.
YAW0, YAW_LO, YAW_HI = 1000.0, 0.0, 2000.0
TURN_RATE, TURN_TAU, TURN_REACH = 2.0, 0.12, 0.35
SPD_MIN, SPD_MAX, SPD0 = 2.5, 7.0, 4.0
THROTTLE, THROTTLE_TAU = 4.0, 0.2
#: FUEL (the player's ``fuel``, 0..1): burnt in proportion to the throttle
#: above CRUISE (a full burn empties the tank in 1 / FUEL_BURN seconds),
#: refilled only at or below CRUISE; with the tank dry the throttle tops out
#: at CRUISE.
CRUISE = 3.2
FUEL_BURN, FUEL_REFILL = 1.0 / 12.0, 1.0 / 10.0
#: The guns' climb per unit of horizontal flight at full pitch (the page
#: draws height at half scale: this matches the cockpit's sight line).
AIM_SLOPE = 1.2
#: The fixed population: POP_BASE + POP_STEP * level enemies (at most
#: POP_MAX); one destroyed returns RESPAWN / tempo seconds later, FAR_MIN..
#: FAR_MAX radians of arc away.  They hunt head on: steering for a point LEAD
#: ahead on the player's path, holding course for a pass inside PASS_R.
POP_BASE, POP_STEP, POP_MAX = 5.0, 3.0, 18.0
RESPAWN = 6.0
FAR_MIN, FAR_MAX = 0.7, 1.6
ARRIVE = 14.0                # the boss and aces arrive this far ahead
LEAD, PASS_R, HUNT_GAIN = 18.0, 3.5, 1.5
TURN_IN = 15.0
SWING, BEHIND_BOOST = 0.9, 1.8          # behind: race wide (this azimuth off the nose) to get in front
DRIFTER_TURN, SPINNER_TURN = 1.5, 0.9
DRIFTER_SPD, SPINNER_SPD = 3.6, 2.4
PILOT_SPD, PILOT_THROTTLE, PILOT_TURN = 4.5, 1.5, 1.8
SIGHT, FIRE_RANGE, RING_RANGE, FLAK_RANGE = 0.45, 10.0, 6.0, 11.0
SHOT_LIFE, BULLET_LIFE, PICKUP_LIFE = 0.6, 3.5, 14.0
#: Scenery (clouds, emplacements) left farther than SCENERY_FAR behind is
#: rebuilt AHEAD in front of the player.
AHEAD, SCENERY_FAR = 13.0, 16.0
STATION_Z = 0.0             # emplacements sit on the land (the page sets them on its relief)
CLOUD_DRIFT, STORM_DRIFT, SMOKE_DRIFT = 0.25, 0.5, 0.25
SHIP_SIZE = 0.5            # body only (exhaust puff dropped at export): ~1.3 units long
SHIP_RX, SHIP_RY = 0.22, 0.4
MUZZLE = 0.63             # the nose: 1.31 drawn half-sizes ahead, foreshortened by the 0.25 lean

#: Glyphs this game adds to the exported meshes (web_export.py): the cloud.
EMOJI = {"cloud": "\u2601", "parachute": "\U0001FA82"}

CONTROLS = {"move_axis": "move-axis-x", "move_axis_y": "move-axis-y", "move_axis_z": "move-axis-z",
            "steer_x": "pointer-board-x", "steer_y": "pointer-board-y", "fire": "primary-action",
            "alt_fire": "secondary-action", "bomb_style": "bomb-style"}


def nearest_of_block(u, z, live, rows, cols, radius):
    """For each lane in ``rows``, the position and height of the nearest live
    lane in ``cols`` (great-circle distance), as full-length spans (zero
    outside ``rows``), and whether it has one.  The rows-by-cols block only,
    as ``block_hits`` computes its pairs."""
    r0, r1 = rows
    c0, c1 = cols
    n = int(u[0].shape[0])
    a, b = r1 - r0, c1 - c0
    ru = [u[i][r0:r1].reshape((a, 1)) for i in range(3)]
    cu = [u[i][c0:c1].reshape((1, b)) for i in range(3)]
    cross = (ru[1] * cu[2] - ru[2] * cu[1], ru[2] * cu[0] - ru[0] * cu[2], ru[0] * cu[1] - ru[1] * cu[0])
    gap = radius * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt().atan2(
        ru[0] * cu[0] + ru[1] * cu[1] + ru[2] * cu[2])
    gap = where(live[c0:c1].reshape((1, b)) + (gap * 0.0 > 1.0), gap, 1.0e9)
    best = gap.min(dim=1).reshape((a, 1))
    pick = as_float(gap == best) * as_float(best < 1.0e8)
    pick = pick / (pick.sum(dim=1).reshape((a, 1)) + 1e-12)

    def spread(values):
        parts = [u[0][:r0] * 0.0] if r0 > 0 else []
        parts.append(values)
        if r1 < n:
            parts.append(u[0][r1:] * 0.0)
        return AbstractTensor.cat(parts, dim=0) if len(parts) > 1 else values

    target = tuple(spread((pick * cu[i]).sum(dim=1)) for i in range(3))
    height = spread((pick * z[c0:c1].reshape((1, b))).sum(dim=1))
    has = spread(as_float(best.reshape((a,)) < 1.0e8)) > 0.0
    return target, height, has


def scroller_step(lane, slot, rx, ry, rz,
                  alive, x, y, z, vx, vy, vz, ux, uy, uz, hx, hy, hz, spd, yaw, vyaw, vspd,
                  hp, boom, born, owner, energy, kind, age, timer,
                  weapon, mult, fan, pax, pay, paz, pfire, calm, boss_phase, ace_clock,
                  score, lives, invuln, clock, alt_prev, fuel, map_u, map_v, map_gx, map_gy, field_w, pace, level,
                  move_axis, move_axis_y, move_axis_z, steer_x, steer_y, fire, alt_fire, bomb_style, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    f = as_float
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_shot, is_enemy = slot == PLAYER, slot == SHOT, slot == ENEMY
    is_bullet, is_star = slot == BULLET, slot == STAR
    is_missile, is_pickup = slot == MISSILE_SLOT, slot == PICKUP
    is_smoke = slot == SMOKE
    is_flash = slot == FLASH
    is_station = slot == STATION
    is_bomb, is_blast = slot == BOMB_SLOT, slot == BLAST_SLOT
    is_boss_lane = lane == BOSS_LANE
    clock_next = clock + dt
    zero = x * 0.0
    P = f(is_player)
    cx, cy = field_w * 0.5, zero + PLAYER_Y
    u = (ux, uy, uz)
    h = (hx, hy, hz)

    # ------------------------------------------------ the player: turn, throttle, height
    # The shared damped motion law on three axes: YAW (left/right turn the
    # ship; a held finger steers toward its bearing from the ship), SPEED
    # (up/down throttle; released, the speed holds) and HEIGHT.
    finger = (steer_x - cx).atan2(cy - steer_y)
    yaw_target = where(steer_x >= 0.0, yaw + finger, zero - 1.0)
    ship_yaw, ship_vyaw = damped_axis(yaw, vyaw, move_axis, yaw_target, zero + YAW_LO, zero + YAW_HI,
                                      TURN_RATE, TURN_TAU, TURN_REACH, dt)
    tank = (P * fuel).sum()
    top_speed = where(tank > 1e-3, zero + SPD_MAX, zero + CRUISE)
    ship_spd, ship_vspd = damped_axis(spd, vspd, zero - move_axis_y, zero - 1.0, zero + SPD_MIN, top_speed,
                                      THROTTLE, THROTTLE_TAU, REACH, dt)
    ship_z, ship_vz = damped_axis(z, vz, move_axis_z, zero - 1.0, zero + Z_MARGIN, zero + DEPTH - Z_MARGIN,
                                  SPEED, TAU, REACH, dt)
    me = is_player & live
    ship_on = f(me).sum() > 0.0
    # The player's frame at the start of the step, turned by this step's yaw.
    u0 = ((P * ux).sum(), (P * uy).sum(), (P * uz).sum())
    h0 = ((P * hx).sum(), (P * hy).sum(), (P * hz).sum())
    dyaw = (P * (ship_yaw - yaw)).sum()
    h0 = sm.turn(u0, h0, dyaw)
    r0 = sm.right_of(u0, h0)
    pz = (P * z).sum()
    pspd = (P * ship_spd).sum()
    # Where every body is relative to the player.
    dist = sm.arc(u, u0, RADIUS)
    to_player = sm.toward(u, u0)
    bearing_p = sm.bearing(u, h, to_player)

    # ------------------------------------------------ the player's guns
    pw, pm, pf = (P * weapon).sum(), (P * mult).sum(), (P * fan).sum()
    reload = (P * timer).sum()
    volley = (fire > 0.0) & ship_on & playing & (reload <= 0.0)
    free_shot = is_shot & ~live
    shots_n = 1.0 + pm
    shot_rank = f(free_shot).cumsum(dim=0) - 1.0
    shot_spawn = free_shot & (shot_rank < shots_n) & volley & (pw != MISSILE)
    shot_c = shot_rank - 0.5 * (shots_n - 1.0)
    shot_u, shot_h = sm.offset(u0, h0, zero + MUZZLE, shot_c * 0.18, RADIUS)
    shot_h = sm.turn(shot_u, shot_h, shot_c * FAN_STEP * pf)
    shot_speed = where(pw == LASER, SHOT_SPEED * LASER_SPEED, zero + SHOT_SPEED) + pspd
    # Missiles: the firing solution that lands each on the nearest enemy
    # (time of flight from the arc to it, vertical speed to rise the height
    # difference and fall back over it), then engine_toy's trajectory law.
    enemies_live = is_enemy & live
    gap = where(enemies_live, dist, 1.0e9)
    nearest = first_of(enemies_live & (gap == gap.min()), lane)
    has_target = f(nearest).sum() > 0.0
    tu = ((f(nearest) * ux).sum(), (f(nearest) * uy).sum(), (f(nearest) * uz).sum())
    t_dist = where(has_target, (f(nearest) * dist).sum(), zero + 8.0)
    tz = where(has_target, (f(nearest) * z).sum(), pz)
    flight = (t_dist / 9.0).clip(0.45, 1.6)
    free_missile = is_missile & ~live
    missile_rank = f(free_missile).cumsum(dim=0) - 1.0
    missile_spawn = free_missile & (missile_rank < pm) & volley & (pw == MISSILE)
    missile_c = missile_rank - 0.5 * (pm - 1.0)
    missile_u, missile_h0 = sm.offset(u0, h0, zero + MUZZLE, missile_c * 0.3, RADIUS)
    missile_h = sm.vwhere(has_target, sm.toward(missile_u, tu), missile_h0)
    # Alternate fire, on the press (not the hold): with a parachute bomb of
    # the player's hanging, the press bursts it; else a bomb of the chosen
    # style leaves the belly, carrying the ship's speed.  A seeker's target is
    # the nearest live ground emplacement (else the nearest enemy).
    alt_now = (P * alt_fire).sum() > 0.0
    press = alt_now & ((P * alt_prev).sum() <= 0.0) & ship_on & playing
    hanging = is_bomb & live & (kind == CHUTE) & (age >= CHUTE_DEPLOY)
    burst_chutes = hanging & press
    drop_bomb = press & ~(f(hanging).sum() > 0.0)
    bomb_spawn = first_of(is_bomb & ~live, lane) & drop_bomb
    style = ((P * bomb_style).sum().clip(0.0, 2.0) + 0.5).floor()
    ground_live = is_station & live
    ggap = where(ground_live, dist, 1.0e9)
    ground_t = first_of(ground_live & (ggap == ggap.min()), lane)
    has_ground = f(ground_t).sum() > 0.0
    G_ = where(has_ground, f(ground_t), f(nearest))
    gu = ((G_ * ux).sum(), (G_ * uy).sum(), (G_ * uz).sum())
    gz = (G_ * z).sum()
    bomb_u, bomb_h = sm.offset(u0, h0, zero - 0.2, zero, RADIUS)
    bomb_h = sm.vwhere((style == SEEKER) & (G_.sum() > 0.0), sm.toward(bomb_u, gu), bomb_h)

    # ------------------------------------------------ the population
    # A FIXED population lives on the sphere: POP enemy lanes (more each
    # level); one destroyed returns RESPAWN seconds later, far away, heading
    # for the player.  The boss lane and the lanes past POP are for the boss
    # and the aces.
    tempo = pace * (1.0 + LEVEL_STEP * (level - 1.0))       # returns and volleys scale with it
    pop = (POP_BASE + POP_STEP * level).clip(POP_BASE, POP_MAX)
    in_pop = is_enemy & ~is_boss_lane & ((lane - ENEMIES[0]) < pop)
    respawn = in_pop & ~live & (age >= RESPAWN / tempo) & playing
    roll = noise(lane * 3.7 + clock * 1.3 + 1.0)
    new_kind = where(roll < 0.75, DRIFTER, SPINNER)
    far_arc = FAR_MIN + (FAR_MAX - FAR_MIN) * noise(lane * 1.9 + clock * 0.7)
    far_az = 6.283185307179586 * noise(lane * 2.3 + clock * 1.1 + 4.0)
    far_u = sm.place(u0, h0, far_arc, far_az)
    # The enemies' height routine (see DWELL): its value and its slope.
    phase = clock_next * ROUTINE_RATE + lane * 1.3
    wave = DWELL * phase.sin()
    held_wave = where(wave > 1.0, 1.0, where(wave < -1.0, -1.0, wave))
    span_z = 0.5 * (DEPTH - 2.0 * Z_MARGIN)
    routine_z = Z_MARGIN + span_z * (1.0 + held_wave)
    routine_vz = where(wave.abs() < 1.0, span_z * DWELL * ROUTINE_RATE * phase.cos(), 0.0)

    # The boss arrives on time, or early once the player has cruised calm --
    # dead ahead, coming head on; while it lives and the player stays calm,
    # aces join, ahead and to the sides.
    shield_now = (P * energy).sum()
    calm_next = where(playing & (shield_now >= CALM_SHIELD), calm + dt, 0.0 * calm)
    phase0 = (P * boss_phase).sum()
    boss_due = playing & (phase0 == 0.0) & ((clock >= BOSS_TIME) | ((P * calm).sum() >= CALM_TIME))
    boss_spawn = is_boss_lane & ~live & boss_due
    ace_ready = playing & (phase0 == 1.0) & ((P * ace_clock).sum() <= 0.0) & (shield_now >= CALM_SHIELD)
    ace_spawn = first_of(is_enemy & ~live & ~is_boss_lane & ~in_pop, lane) & ace_ready
    ace_clock_next = where(ace_ready, ACE_EVERY / tempo, (ace_clock - dt).clip(0.0, 100.0))
    arrive_az = where(boss_spawn, zero, 1.6 * (noise(lane * 4.1 + clock * 2.7) - 0.5))
    arrive_u = sm.place(u0, h0, zero + ARRIVE / RADIUS, arrive_az)
    arrival = respawn | boss_spawn | ace_spawn
    arrival_u = sm.vwhere(respawn, far_u, arrive_u)
    piloted = (is_boss_lane | (is_enemy & (kind == ACE))) & live

    # ------------------------------------------------ enemy fire
    # One volley per step: the ready shooter in the lowest lane.  Craft fire
    # FORWARD along their heading (a fan on the nose) when the player is in
    # their sights; spinners ring the sky when close; anti-aircraft fires one
    # shell straight at where the player IS (no lead), only while the
    # enemies have not been hurting the player (AA_QUIET).
    aa_awake = (P * age).sum() >= AA_QUIET
    in_sights = (bearing_p.abs() < SIGHT) & (dist < FIRE_RANGE)
    craft_ready = is_enemy & where(kind == SPINNER, dist < RING_RANGE, in_sights) & (~piloted | (pfire > 0.0))
    piloted_ready = piloted & (pfire > 0.0) & (dist < FIRE_RANGE)
    ready = ((craft_ready & ~piloted) | piloted_ready | (is_station & aa_awake & (dist < FLAK_RANGE))) & live & (timer <= 0.0) & playing
    shooter = first_of(ready, lane)
    fired = f(shooter).sum() > 0.0
    S = f(shooter)
    su = ((S * ux).sum(), (S * uy).sum(), (S * uz).sum())
    sh = ((S * hx).sum(), (S * hy).sum(), (S * hz).sum())
    sz = (S * z).sum()
    s_dist = (S * dist).sum()
    s_spd = (S * spd).sum()
    skind = (S * kind).sum()
    shooter_lane = (S * lane).sum()
    count = where(skind == SPINNER, RING, where(skind == BOSS_KIND, 5.0, where(skind == STATION_KIND, 1.0, SPREAD)))
    free_bullet = is_bullet & ~live
    rank = f(free_bullet).cumsum(dim=0) - 1.0
    bullet_spawn = free_bullet & (rank < count) & fired
    ring = skind == SPINNER
    flak = skind == STATION_KIND
    spread_angle = where(ring, rank * (6.283185307179586 / RING) + clock * 1.7, (rank - 0.5 * (count - 1.0)) * 0.22)
    bullet_h = sm.vwhere(flak, sm.toward(su, u0), sm.turn(su, sh, spread_angle + zero))
    bullet_speed = where(ring, zero + BULLET_SPEED * 0.8, where(flak, zero + FLAK_SPEED, BULLET_SPEED + s_spd))
    # Forward and flak fire climb or dive toward the ship's height over the
    # flight the arc takes; rings stay level.
    climb = where(ring, 0.0, bullet_speed * (pz - sz) / (s_dist + 0.5))

    # ------------------------------------------------ headings and speeds
    # Enemies hunt HEAD ON: each steers for a point on the player's path a
    # lead ahead (its own lead), meeting the player face to face; once it has
    # passed close by (the player behind it) it holds its course, then turns
    # back for another pass.
    # In front of the player (within 90 degrees of its nose) an enemy flies
    # straight at it, face to face; behind it, it first swings wide for a
    # point well ahead on the player's path (its own lead), to come back
    # around and meet it head on.
    seen_at = sm.bearing((u0[0] + zero, u0[1] + zero, u0[2] + zero), (h0[0] + zero, h0[1] + zero, h0[2] + zero),
                         sm.toward((u0[0] + zero, u0[1] + zero, u0[2] + zero), u))
    in_front = seen_at.abs() < 1.5707963267948966
    lead = LEAD * (1.0 + noise(lane * 3.1 + 7.0))
    side = where(lane - 2.0 * (lane * 0.5).floor() > 0.5, zero + SWING, zero - SWING)
    meet = sm.place((u0[0] + zero, u0[1] + zero, u0[2] + zero), (h0[0] + zero, h0[1] + zero, h0[2] + zero),
                    (lead + dist * 0.5) / RADIUS, side)
    # Head on only from far enough out to come round (TURN_IN) in a cone off
    # the player's nose, or once already facing it; else keep racing for
    # the meeting point.
    facing = sm.dot(h, (h0[0] + zero, h0[1] + zero, h0[2] + zero)) < -0.3
    head_on = (seen_at.abs() < 0.9) & ((dist > TURN_IN) | facing)
    want = where(head_on, bearing_p, sm.bearing(u, h, sm.toward(u, meet)))
    passed = (dist < PASS_R) & (bearing_p.abs() > 1.6)
    hunt = is_enemy & live & ~piloted
    agility = where(kind == SPINNER, SPINNER_TURN, DRIFTER_TURN)
    hunt_turn = where(passed, zero, (HUNT_GAIN * want).clip(-1.0, 1.0) * agility)
    pilot_turn = pax.clip(-1.0, 1.0) * PILOT_TURN
    seeking = is_bomb & live & (kind == SEEKER)
    own_u, own_z, own_has = nearest_of_block(u, z, ground_live, BOMBS, STATIONS, RADIUS)
    seek_u = sm.vwhere(own_has, own_u, (gu[0] + zero, gu[1] + zero, gu[2] + zero))
    seek_z = where(own_has, own_z, gz + zero)
    seek_turn = (2.0 * sm.bearing(u, h, sm.toward(u, seek_u))).clip(-1.0, 1.0) * SEEK_TURN
    turn_angle = where(is_player, zero + dyaw, where(hunt, hunt_turn * dt, where(piloted, pilot_turn * dt,
                       where(seeking, seek_turn * dt, zero))))
    h_turned = sm.turn(u, h, turn_angle)
    # Pickups wait where they fell; inside MAGNET_R they fly hard at the ship.
    pull = is_pickup & live & ship_on & (dist < MAGNET_R)
    pull_speed = MAGNET_SPEED * (1.0 + 2.0 * (1.0 - dist / MAGNET_R))
    h_turned = sm.vwhere(pull, to_player, h_turned)
    hunt_spd = (where(kind == SPINNER, SPINNER_SPD, DRIFTER_SPD) * (0.8 + 0.4 * tempo.clip(0.0, 1.0))
                * where(facing, 1.0, BEHIND_BOOST))              # racing round until it faces the player
    pilot_spd = (PILOT_SPD - PILOT_THROTTLE * pay.clip(-1.0, 1.0))
    spd1 = where(is_player, ship_spd, where(hunt, hunt_spd, where(piloted, pilot_spd,
                 where(is_pickup, where(pull, pull_speed, 0.0), spd))))

    # ------------------------------------------------ spawns
    spawn = shot_spawn | arrival | bullet_spawn | missile_spawn | bomb_spawn
    u1 = sm.vwhere(bomb_spawn, bomb_u, u)
    u1 = sm.vwhere(shot_spawn, shot_u, sm.vwhere(missile_spawn, missile_u, sm.vwhere(bullet_spawn, su,
                   sm.vwhere(arrival, arrival_u, u1))))
    h_turned = sm.vwhere(bomb_spawn, bomb_h, h_turned)
    h1 = sm.vwhere(shot_spawn, shot_h, sm.vwhere(missile_spawn, missile_h, sm.vwhere(bullet_spawn, bullet_h,
                   sm.vwhere(arrival, sm.toward(arrival_u, u0), h_turned))))
    spd1 = where(bomb_spawn, where(style == SEEKER, zero + SEEK_SPD, pspd), spd1)
    spd1 = where(shot_spawn, shot_speed, where(missile_spawn, t_dist / flight, where(bullet_spawn, bullet_speed,
                 where(arrival, where(is_boss_lane | ace_spawn, zero + PILOT_SPD,
                       where(new_kind == SPINNER, SPINNER_SPD, DRIFTER_SPD) * (0.8 + 0.4 * tempo.clip(0.0, 1.0))), spd1))))
    z1 = where(shot_spawn | missile_spawn, pz, where(arrival, routine_z, where(bullet_spawn, sz, z)))
    z1 = where(bomb_spawn, (pz - 0.15).clip(0.05, DEPTH), z1)
    kind1 = where(respawn, new_kind, where(boss_spawn, BOSS_KIND, where(ace_spawn, ACE, kind)))
    kind1 = where(bomb_spawn, style, kind1)
    age1 = where(spawn, 0.0, age + dt)
    level_hp = 1.0 + 0.5 * (level - 1.0)
    hp1 = where(respawn, where(new_kind == SPINNER, 9.0, 3.0),
                where(boss_spawn, BOSS_HP * level_hp, where(ace_spawn, ACE_HP * level_hp, hp)))
    reload_of = where(pw == LASER, LASER_RELOAD, where(pw == MISSILE, MISSILE_RELOAD, RELOAD))
    timer1 = where(is_player, where(volley, reload_of, (timer - dt).clip(0.0, 10.0)),
                   where(arrival, 0.8 / tempo,
                         where(shooter, where(skind == SPINNER, 0.9, where(skind == BOSS_KIND, 0.45,
                                                                           where(skind == ACE, 1.1,
                                                                                 where(skind == STATION_KIND, FLAK_RELOAD, 1.5)))) / tempo,
                               timer - dt)))
    vz1 = where(is_player, ship_vz, where(is_enemy, routine_vz, where(bullet_spawn, climb,
                where(missile_spawn, (tz - z1) / flight + 0.5 * GRAVITY * flight, where(spawn, 0.0, vz)))))
    vz1 = where(is_pickup, 0.0, vz1)
    # The guns follow the nose's pitch: a shot climbs or dives at the slope
    # the pitch command (the height axis) asks for -- point and shoot.
    pitch = (P * move_axis_z).sum().clip(-1.0, 1.0)
    vz1 = where(shot_spawn, shot_speed * AIM_SLOPE * pitch, vz1)
    live1 = live | spawn
    born1 = where(spawn, clock, born)
    owner1 = where(shot_spawn | missile_spawn | bomb_spawn, 0.0, where(bullet_spawn, shooter_lane, where(respawn, -1.0, owner)))

    # ------------------------------------------------ flight
    # Everything flies its great circle at its speed; the player's height and
    # the enemies' routine are set, the rest climb at their own rates.
    moving = (live1 & playing) | is_star | is_station
    step = where(moving, spd1 * dt, zero)
    # Missiles fly engine_toy's trajectory law in their own frame (forward,
    # height, side): the law's horizontal advance is flown along the heading.
    mx, mz_, my, mvx, mvz_, mvy = trajectory_step(zero, z1, zero, spd1, vz1, zero, dt, MISSILE_MASS, MISSILE_DIAMETER,
                                                  MISSILE_DRAG, AIR_DENSITY, GRAVITY, 0.0, 0.0, 0.0)[:6]
    flying = is_missile & moving
    step = where(flying, mx, step)
    # Bombs: the same trajectory law, a canopy's drag once a parachute opens
    # (it slows, hangs and drifts down); a seeker flies powered, diving at
    # its target over the time the arc takes.
    canopy = (kind1 == CHUTE) & (age1 >= CHUTE_DEPLOY)
    diameter = where(canopy, zero + CHUTE_DIAMETER, zero + BOMB_DIAMETER)
    drag_k = where(canopy, zero + CHUTE_DRAG, zero + BOMB_DRAG)
    bx_, bz_, by_, bvx_, bvz_, bvy_ = trajectory_step(zero, z1, zero, spd1, vz1, zero, dt, MISSILE_MASS, diameter,
                                                      drag_k, AIR_DENSITY, GRAVITY, 0.0, 0.0, 0.0)[:6]
    falling = is_bomb & moving & (kind1 != SEEKER)
    seeker_live = is_bomb & moving & (kind1 == SEEKER)
    seek_time = (sm.arc(u1, seek_u, RADIUS) / SEEK_SPD).clip(0.15, 10.0)
    seek_vz = (seek_z - z1) / seek_time
    step = where(falling, bx_, step)
    u2, h2 = sm.advance(u1, h1, step, RADIUS)
    spd2 = where(flying, mvx, where(falling, bvx_, spd1))
    z2 = where(is_player & moving, ship_z, where(is_enemy & moving, routine_z,
               where(flying, mz_, where(falling, bz_, where(seeker_live, z1 + seek_vz * dt,
                                                            where(moving, z1 + vz1 * dt, z1))))))
    vz1 = where(flying, mvz_, where(falling, bvz_, where(seeker_live, seek_vz, vz1)))
    paz_z, paz_vz = damped_axis(z1, vz, paz, zero - 1.0, zero + Z_MARGIN, zero + DEPTH - Z_MARGIN,
                                SPEED, TAU, REACH, dt)
    flown = piloted & ~arrival
    z2, vz1 = where(flown & moving, paz_z, z2), where(flown, paz_vz, vz1)
    # The player's frame at the end of the step: the patch everyone is seen in.
    u0n = ((P * u2[0]).sum(), (P * u2[1]).sum(), (P * u2[2]).sum())
    h0n = ((P * h2[0]).sum(), (P * h2[1]).sum(), (P * h2[2]).sum())
    dist2 = sm.arc(u2, u0n, RADIUS)
    # Scenery keeps the sky and the land around the player populated: a cloud
    # or an emplacement left far behind is rebuilt ahead (the same lanes, a
    # fixed population of scenery).
    rebuild_az = 2.0 * (noise(lane * 0.77 + clock * 3.3) - 0.5)
    ahead_u = sm.place(u0n, h0n, zero + AHEAD / RADIUS, rebuild_az)
    rebuilt = is_station & (dist2 > SCENERY_FAR) & playing
    wrapped = is_star & (dist2 > SCENERY_FAR)
    renewed = rebuilt | wrapped
    u2 = sm.vwhere(renewed, ahead_u, u2)
    h2 = sm.vwhere(renewed, sm.turn(ahead_u, sm.toward(ahead_u, u0n), zero + 6.283185307179586 * noise(lane * 1.3 + clock)), h2)
    X, Y = sm.project(u2, u0n, h0n, RADIUS)
    x2, y2 = cx + X, cy - Y

    # A cloud soaks while smoke overlaps it in 3-D (its own height too); held
    # for SOAK_TIME it turns to rain.  The cloud's timer counts the soaking.
    in_cloud = block_hits(x2, y2, rx, ry, live1 | is_star, None, STARS, SMOKES, z2, rz) > 0.0
    soak = where(is_star, where(wrapped, 0.0, where(in_cloud, timer + dt, (timer - 0.2 * dt).clip(0.0, 10.0))), timer1)
    timer1 = where(is_star, soak, timer1)
    kind1 = where(is_star, where(wrapped, 0.0, where((soak >= SOAK_TIME) & (kind1 == 0.0), RAIN, kind1)), kind1)
    # A quiet sky (no damage for STORM_QUIET) may grow a thunderstorm: a
    # plain cloud ahead, chosen by the law's dice, rises, swells and darkens.
    quiet = (P * age).sum() >= STORM_QUIET
    storms = f(is_star & (kind1 == THUNDER)).sum()
    storm_dice = noise(clock * 2.3 + 17.0)
    grows = playing & quiet & (storms < STORM_MAX) & (storm_dice < STORM_RATE * dt)
    candidate = is_star & (kind1 == 0.0) & (noise(lane * 1.7 + clock * 0.9) > 0.5) & (Y > 2.0)
    storm_born = first_of(candidate, lane) & grows
    kind1 = where(storm_born, THUNDER, kind1)
    z2 = where(storm_born, 2.6 + 0.8 * noise(clock * 4.7 + 3.0), z2)
    spd2 = where(storm_born, zero + STORM_DRIFT, spd2)
    z2 = where(rebuilt, zero + STATION_Z, z2)
    out = (z2 < -0.6) | (z2 > DEPTH + 0.6)
    expired = ((is_shot & (age1 > where(pw == LASER, SHOT_LIFE / LASER_SPEED, SHOT_LIFE)))
               | (is_bullet & (age1 > BULLET_LIFE)) | (is_pickup & (age1 > PICKUP_LIFE)))
    grounded = is_missile & (z2 < 0.0)                  # a missile that reaches the land is spent
    expired = expired | (is_bomb & (age1 > BOMB_LIFE)) | (is_blast & (age1 > BLAST_LIFE))
    live2 = (live1 & ~((is_shot | is_enemy | is_bullet | is_missile | is_pickup) & out) & ~grounded & ~expired) | rebuilt

    # ------------------------------------------------ collisions, block against block (in the patch)
    enemy_hits = (block_hits(x2, y2, rx, ry, live2, None, ENEMIES, SHOTS, z2, rz)
                  + MISSILE_DAMAGE * block_hits(x2, y2, rx, ry, live2, None, ENEMIES, MISSILES, z2, rz))
    missile_hits = (block_hits(x2, y2, rx, ry, live2, None, MISSILES, ENEMIES, z2, rz)
                    + block_hits(x2, y2, rx, ry, live2, None, MISSILES, STATIONS, z2, rz))
    station_hits = (block_hits(x2, y2, rx, ry, live2, None, STATIONS, SHOTS, z2, rz)
                    + MISSILE_DAMAGE * block_hits(x2, y2, rx, ry, live2, None, STATIONS, MISSILES, z2, rz))
    collected = block_hits(x2, y2, rx, ry, live2, None, PICKUPS, (0, 1), z2, rz)
    got = is_pickup & live2 & (collected > 0.0)
    got_kind = lambda k: f(got & (kind == k)).sum() > 0.0  # noqa: E731
    weapon_next = where(is_player, where(got_kind(MISSILE_UP), MISSILE, where(got_kind(LASER_UP), LASER, weapon)), weapon)
    mult_next = where(is_player & got_kind(MULT_UP), (mult + 1.0).clip(1.0, 4.0), mult)
    fan_next = where(is_player & got_kind(SPREAD_UP), (fan + 1.0).clip(0.0, 3.0), fan)
    shot_hits = (block_hits(x2, y2, rx, ry, live2, None, SHOTS, ENEMIES, z2, rz)
                 + block_hits(x2, y2, rx, ry, live2, None, SHOTS, STATIONS, z2, rz))
    exposed = live2 & (invuln <= 0.0)
    bullet_strikes = block_hits(x2, y2, rx, ry, exposed, None, (0, 1), BULLETS, z2, rz)
    rams = (block_hits(x2, y2, rx, ry, exposed, None, (0, 1), ENEMIES, z2, rz)       # the boss lane is an enemy lane
            + block_hits(x2, y2, rx, ry, exposed, None, (0, 1), STATIONS, z2, rz))
    player_hits = bullet_strikes + rams
    bullet_hits = block_hits(x2, y2, rx, ry, live2 & ((invuln <= 0.0) | ~is_player), None, BULLETS, (0, 1), z2, rz)
    # Lightning: one rain cloud may strike this step; it chains to every
    # craft within CHAIN_R of the cloud on the board.
    raining = is_star & ((kind1 == RAIN) | (kind1 == THUNDER))
    strike_dice = noise(clock * 5.7 + lane * 1.31 + 9.0)
    rate = where(kind1 == THUNDER, THUNDER_RATE, STRIKE_RATE)
    striker = first_of(raining & (strike_dice < rate * dt) & playing & (dist2 < SCENERY_FAR), lane)
    striking = f(striker).sum() > 0.0
    L = f(striker)
    lx, ly = (L * x2).sum(), (L * y2).sum()
    lu = ((L * u2[0]).sum(), (L * u2[1]).sum(), (L * u2[2]).sum())
    lz = (L * z2).sum()
    reach_l = ((x2 - lx) * (x2 - lx) + (y2 - ly) * (y2 - ly)).sqrt()
    storm_strike = (L * kind1).sum() == THUNDER
    chain = where(storm_strike, THUNDER_CHAIN_R, CHAIN_R)
    in_reach = striking & live2 & (is_enemy | is_player) & (reach_l < chain)
    # A rain cloud's strike chains to everything in reach; a thunderstorm's
    # takes the closest craft only (nothing in reach: a harmless ground strike).
    closest = first_of(in_reach & (where(in_reach, reach_l, 1.0e9) == where(in_reach, reach_l, 1.0e9).min()), lane)
    struck = where(storm_strike, closest, in_reach)
    zapped = f(struck & is_player).sum() > 0.0
    # Bombs go off on the land, on a seeker's target, or (a parachute) at the
    # second press; each blast hurts every craft and emplacement inside its
    # style's radius (in 3-D: the patch and height).
    seek_arrived = is_bomb & live2 & (kind1 == SEEKER) & (sm.arc(u2, seek_u, RADIUS) < 0.6) & (z2 < seek_z + 0.6)
    blows = is_bomb & live2 & ((z2 <= 0.0) | seek_arrived | burst_chutes)
    blast_hurt = zero
    for style_k in (DUMB, CHUTE, SEEKER):
        radius_k = BLAST_R[style_k]
        mine = blows & (kind1 == style_k)
        reach_x = where(is_bomb, zero + radius_k, rx)
        reach_z = where(is_bomb, zero + 0.8 * radius_k, rz)
        live_k = (mine | ((is_enemy | is_station) & live2))
        blast_hurt = blast_hurt + BLAST_DAMAGE[style_k] * (
            block_hits(x2, y2, reach_x, reach_x, live_k, None, ENEMIES, BOMBS, z2.clip(0.0, DEPTH), reach_z)
            + block_hits(x2, y2, reach_x, reach_x, live_k, None, STATIONS, BOMBS, z2.clip(0.0, DEPTH), reach_z))
    hp2 = where(rebuilt, STATION_HP, hp1 - enemy_hits - station_hits - LIGHTNING_DAMAGE * f(struck & is_enemy)
                - blast_hurt)
    killed = (is_enemy | is_station) & live2 & (hp2 <= 0.0)
    touched = is_player & live2 & (player_hits > 0.0)
    # Damage drains the shield's points first, the rest comes off health;
    # shield and health pickups refill them; both regenerate (health slowly).
    damage = ((f(is_player & live2) * (BULLET_DAMAGE * bullet_strikes + RAM_DAMAGE * rams)).sum()
              + LIGHTNING_HIT * f(zapped))                       # lightning is a hazard
    shield_pts = (P * energy).sum() * SHIELD_POINTS
    soaked = where(damage < shield_pts, damage, shield_pts)
    wound = damage - soaked
    refill = got_kind(SHIELD_UP)
    since_hit = (P * age).sum()
    recharging = playing & (damage <= 0.0) & (since_hit >= SHIELD_PAUSE)
    shield_after = where(refill, SHIELD_POINTS, shield_pts - soaked + SHIELD_RATE * dt * f(recharging))
    energy_next = where(is_player, (shield_after / SHIELD_POINTS).clip(0.0, 1.0), energy)
    healed = where(got_kind(HEALTH_UP), HEALTH_PICKUP, 0.0)
    health = (lives - wound + healed + HEALTH_REGEN * dt * f(lives > 0.0)).clip(0.0, HEALTH_MAX)
    lives_next = where(playing, health, lives)
    over = lives_next <= 0.0
    hit = touched & (wound > 0.0)
    shielded = (damage > 0.0) & (wound <= 0.0)
    was_hit = wound > 0.0
    gone = (killed | blows | (is_shot & (shot_hits > 0.0)) | (is_missile & (missile_hits > 0.0)) | got
            | (is_bullet & (bullet_hits > 0.0)) | (hit & over))
    live3 = live2 & ~gone
    # A destroyed craft sometimes drops a pickup (into the first free pickup
    # lane, where the first of the step's kills died) and always leaves smoke.
    killer = first_of(killed, lane)
    K = f(killer)
    ku = ((K * u2[0]).sum(), (K * u2[1]).sum(), (K * u2[2]).sum())
    kh = ((K * h2[0]).sum(), (K * h2[1]).sum(), (K * h2[2]).sum())
    kx, ky, kz = (K * x2).sum(), (K * y2).sum(), (K * z2).sum()
    dice = noise(clock * 7.3 + score * 0.01 + 3.0)
    dropping = (K.sum() > 0.0) & (dice < DROP_CHANCE) & playing
    drop = first_of(is_pickup & ~live3, lane) & dropping
    drop_kind = (1.0 + (noise(clock * 2.9 + 11.0) * 6.0).floor()).clip(1.0, 6.0)
    puff = first_of(is_smoke & ~live3, lane) & (K.sum() > 0.0)
    left_at = drop | puff
    u2 = sm.vwhere(left_at, ku, u2)
    h2 = sm.vwhere(left_at, kh, h2)
    x2, y2 = where(left_at, kx, x2), where(left_at, ky, y2)
    z2 = where(drop, kz, where(puff, (kz * 0.5).clip(0.15, 1.2), z2))
    spd2 = where(drop, 0.0, where(puff, zero + SMOKE_DRIFT, spd2))
    vz1 = where(drop, 0.0, where(puff, 0.12, vz1))          # smoke rises toward the cloud layer
    kind1 = where(drop, drop_kind, kind1)
    born1 = where(left_at, clock, born1)
    age1 = where(left_at, 0.0, age1)
    age1 = where(is_player & (damage > 0.0), 0.0, age1)      # the player's age: time since its last hit
    age1 = where(killed & is_enemy, 0.0, age1)                # an enemy's age, dead: time to its return
    live3 = (live3 | drop | puff) & ~(is_smoke & (age1 > SMOKE_LIFE))
    # Lightning ends: at the striking cloud and the nearest craft struck.
    nearest_struck = first_of(struck & (where(struck, reach_l, 1.0e9) == where(struck, reach_l, 1.0e9).min()), lane)
    T = f(nearest_struck)
    tu_l = ((T * u2[0]).sum(), (T * u2[1]).sum(), (T * u2[2]).sum())
    tx_l, ty_l, tz_l = (T * x2).sum(), (T * y2).sum(), (T * z2).sum()
    hit_any = T.sum() > 0.0
    flash_rank = f(is_flash & ~live3).cumsum(dim=0) - 1.0
    flash_at_cloud = is_flash & ~live3 & (flash_rank == 0.0) & striking
    flash_at_craft = is_flash & ~live3 & (flash_rank == 1.0) & striking & hit_any
    flashing = flash_at_cloud | flash_at_craft
    u2 = sm.vwhere(flash_at_cloud, lu, sm.vwhere(flash_at_craft, tu_l, u2))
    x2 = where(flash_at_cloud, lx, where(flash_at_craft, tx_l, x2))
    y2 = where(flash_at_cloud, ly, where(flash_at_craft, ty_l, y2))
    z2 = where(flash_at_cloud, lz, where(flash_at_craft, tz_l, z2))
    spd2, vz1 = where(flashing, 0.0, spd2), where(flashing, 0.0, vz1)
    born1 = where(flashing, clock, born1)
    age1 = where(flashing, 0.0, age1)
    live3 = (live3 | flashing) & ~(is_flash & (age1 > FLASH_LIFE))
    # A blast's flash: the first two free blast lanes take the step's first
    # two bursts, glowing in their style's colour (kind) for BLAST_LIFE.
    first_blow = first_of(blows, lane)
    second_blow = first_of(blows & ~first_blow, lane)
    blast_rank = f(is_blast & ~live3).cumsum(dim=0) - 1.0
    for rank_k, which in ((0.0, first_blow), (1.0, second_blow)):
        W = f(which)
        takes = is_blast & ~live3 & (blast_rank == rank_k) & (W.sum() > 0.0)
        wu = ((W * u2[0]).sum(), (W * u2[1]).sum(), (W * u2[2]).sum())
        u2 = sm.vwhere(takes, wu, u2)
        x2, y2 = where(takes, (W * x2).sum(), x2), where(takes, (W * y2).sum(), y2)
        z2 = where(takes, (W * z2).sum().clip(0.0, DEPTH), z2)
        kind1 = where(takes, (W * kind1).sum(), kind1)
        spd2, vz1 = where(takes, 0.0, spd2), where(takes, 0.0, vz1)
        born1, age1 = where(takes, clock, born1), where(takes, 0.0, age1)
        live3 = live3 | takes
    boss_killed = (f(killed & is_boss_lane).sum() > 0.0)
    boss_phase_next = where(boss_killed, 2.0, where(boss_due, 1.0, boss_phase))
    boom3 = where(killed | (hit & over), BOOM, (boom - dt).clip(0.0, BOOM))
    invuln_next = where(was_hit, 0.6, where(shielded, 0.25, (invuln - dt).clip(0.0, INVULN)))
    score_next = score + (f(killed) * where(kind1 == SPINNER, 300.0, 100.0)).sum()

    # ------------------------------------------------ what the page sees
    # Apparent velocity in the patch (relative flight plus the frame's turn);
    # the player sits still at the patch's centre.
    X2, Y2 = x2 - cx, cy - y2
    pvx, pvy = sm.patch_velocity(h2, spd2, X2, Y2, h0n, pspd, u0n, (P * ship_vyaw).sum())
    vx2 = where(is_player, zero, pvx)
    vy2 = where(is_player, zero, -pvy)
    x2 = where(is_player, cx, x2)
    y2 = where(is_player, cy, y2)

    # The longest step that steps nothing over anything it can hit, pair by
    # pair: a shot must not cross an enemy's body in one step, a bullet must
    # not cross the ship's hitbox at their closing speed; the ship moves at
    # most half its damping time per step.
    pace_k = spd2.abs() + vz1.abs()
    shot_pace = where(is_shot & live3, pace_k, 0.0).max() + pspd
    bullet_pace = where(is_bullet & live3, pace_k, 0.0).max() + pspd + (P * vz1.abs()).sum()
    shot_window = 2.0 * (0.3 + 0.36)
    bullet_window = 2.0 * (SHIP_RX + 0.12)
    shot_limit = 0.9 * shot_window / (shot_pace + 1e-6)
    bullet_limit = 0.9 * bullet_window / (bullet_pace + 1e-6)
    dt_limit = where(shot_limit < bullet_limit, shot_limit, bullet_limit).clip(0.0, 0.5 * TAU) + zero
    residual = f(is_enemy & live3).sum() + zero
    yaw_next = where(is_player, ship_yaw, yaw)
    alt_prev_next = where(is_player, f(alt_now) + zero, alt_prev)
    burn = ((pspd - CRUISE) / (SPD_MAX - CRUISE)).clip(0.0, 1.0) * FUEL_BURN
    refill = where(pspd <= CRUISE + 1e-6, zero + FUEL_REFILL, zero)
    fuel_next = where(is_player & playing, (fuel - burn * dt + refill * dt).clip(0.0, 1.0), fuel)
    # The land under the ship: one law (entity_motion.ground_travel) the page
    # runs too -- the ground frame turns with the ship's yaw step and the map
    # point travels at its speed.
    gu, gv, ggx, ggy = ground_travel(map_u, map_v, map_gx, map_gy, zero + dyaw, zero + pspd, dt)
    flying_on = is_player & playing
    map_u_next, map_v_next = where(flying_on, gu, map_u), where(flying_on, gv, map_v)
    map_gx_next, map_gy_next = where(flying_on, ggx, map_gx), where(flying_on, ggy, map_gy)
    vyaw_next = where(is_player, ship_vyaw, vyaw)
    vspd_next = where(is_player, ship_vspd, vspd)
    return (f(live3), x2, y2, z2, vx2, vy2, vz1, u2[0], u2[1], u2[2], h2[0], h2[1], h2[2], spd2,
            yaw_next, vyaw_next, vspd_next,
            hp2, boom3, born1, owner1, energy_next, kind1, age1, timer1,
            weapon_next, mult_next, fan_next, pax, pay, paz, pfire, calm_next, boss_phase_next, ace_clock_next,
            score_next, lives_next, invuln_next, clock_next, alt_prev_next, fuel_next,
            map_u_next, map_v_next, map_gx_next, map_gy_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "z", "vx", "vy", "vz", "ux", "uy", "uz", "hx", "hy", "hz", "spd", "yaw", "vyaw", "vspd",
         "hp", "boom", "born", "owner", "energy", "kind", "age", "timer",
         "weapon", "mult", "fan", "pax", "pay", "paz", "pfire", "calm", "boss_phase", "ace_clock",
         "score", "lives", "invuln", "clock", "alt_prev", "fuel", "map_u", "map_v", "map_gx", "map_gy")
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
    rx[slice(*STARS)], ry[slice(*STARS)], rz[slice(*STARS)] = 0.55, 0.45, 0.3   # a cloud's extent
    rz[slice(*SMOKES)] = 0.35
    slot[slice(*FLASHES)] = FLASH
    stations = slice(*STATIONS)
    slot[stations], rx[stations], ry[stations], rz[stations] = STATION, 0.35, 0.35, 0.35
    slot[slice(*BOMBS)], rx[slice(*BOMBS)], ry[slice(*BOMBS)], rz[slice(*BOMBS)] = BOMB_SLOT, 0.2, 0.2, 0.2
    slot[slice(*BLASTS)] = BLAST_SLOT
    rx[BOSS_LANE], ry[BOSS_LANE], rz[BOSS_LANE] = 1.1, 0.75, 0.75      # the boss: a big hull
    slot[SHOTS[0]:SHOTS[1]], rx[SHOTS[0]:SHOTS[1]], ry[SHOTS[0]:SHOTS[1]] = SHOT, 0.1, 0.3
    slot[ENEMIES[0]:ENEMIES[1]], rx[ENEMIES[0]:ENEMIES[1]], ry[ENEMIES[0]:ENEMIES[1]] = ENEMY, 0.42, 0.36
    slot[BULLETS[0]:BULLETS[1]], rx[BULLETS[0]:BULLETS[1]], ry[BULLETS[0]:BULLETS[1]] = BULLET, 0.12, 0.12
    slot[STARS[0]:STARS[1]] = STAR
    rng = np.random.default_rng(7)
    # On the sphere: the player at the pole heading "north"; the patch's
    # right is +x, ahead +y (sphere_motion.right_of).  Scenery starts spread
    # over the patch; the enemies start away, returning one after another.
    ux, uy, uz = np.zeros(LANES), np.zeros(LANES), np.ones(LANES)
    hx, hy, hz = np.zeros(LANES), np.ones(LANES), np.zeros(LANES)
    spd = np.zeros(LANES)
    z = np.zeros(LANES)

    def put(lanes, X, Y, heading):
        d = np.hypot(X, Y) + 1e-12
        a = d / RADIUS
        ux[lanes], uy[lanes], uz[lanes] = np.sin(a) * X / d, np.sin(a) * Y / d, np.cos(a)
        u = np.stack([ux[lanes], uy[lanes], uz[lanes]], axis=1)
        g = np.stack([np.sin(heading), np.cos(heading), np.zeros_like(heading)], axis=1)
        t = g - (g * u).sum(axis=1, keepdims=True) * u
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        hx[lanes], hy[lanes], hz[lanes] = t[:, 0], t[:, 1], t[:, 2]

    stars = slice(*STARS)
    n_star = STARS[1] - STARS[0]
    z[stars] = rng.uniform(0.05, 1.2, n_star)      # low clouds, close over the land
    r, a = SCENERY_FAR * np.sqrt(rng.uniform(0.0, 0.8, n_star)), rng.uniform(0.0, 2 * np.pi, n_star)
    put(stars, r * np.sin(a), r * np.cos(a), rng.uniform(0.0, 2 * np.pi, n_star))
    spd[stars] = CLOUD_DRIFT * (1.0 + z[stars])
    stations = slice(*STATIONS)
    n_st = STATIONS[1] - STATIONS[0]
    put(stations, rng.uniform(-3.5, 3.5, n_st), np.linspace(3.0, AHEAD, n_st), np.zeros(n_st))
    z[stations] = STATION_Z
    z[0] = DEPTH / 2.0
    spd[0] = SPD0
    alive = ((slot == PLAYER) | (slot == STAR) | (slot == STATION)).astype(np.float64)
    # The population returns one after another: each enemy lane starts dead
    # with its return clock nearly run (half a second apart).
    tempo = pace * (1.0 + LEVEL_STEP * (level - 1.0))
    age = np.zeros(LANES)
    enemies = np.arange(ENEMIES[0], ENEMIES[1])
    age[enemies] = RESPAWN / tempo - 0.5 * (enemies - ENEMIES[0])
    yaw = np.zeros(LANES)
    yaw[0] = YAW0
    return {
        "lane": lane, "slot": slot, "rx": rx, "ry": ry, "rz": rz,
        "alive": alive, "x": np.full(LANES, width / 2.0), "y": np.full(LANES, PLAYER_Y), "z": z,
        "vx": np.zeros(LANES), "vy": np.zeros(LANES), "vz": np.zeros(LANES),
        "ux": ux, "uy": uy, "uz": uz, "hx": hx, "hy": hy, "hz": hz, "spd": spd,
        "yaw": yaw, "vyaw": np.zeros(LANES), "vspd": np.zeros(LANES),
        "hp": np.where(slot == STATION, STATION_HP, 0.0),
        "boom": np.zeros(LANES), "born": np.zeros(LANES), "owner": np.full(LANES, -1.0),
        "energy": (slot == PLAYER).astype(np.float64), "kind": np.where(slot == STATION, STATION_KIND, 0.0),
        "age": age,
        "timer": np.where(slot == STATION, rng.uniform(0.5, FLAK_RELOAD, LANES), 0.0), "weapon": np.full(LANES, BASIC), "mult": np.ones(LANES), "fan": np.zeros(LANES),
        **{name: np.zeros(LANES) for name in ("pax", "pay", "paz", "pfire", "calm", "boss_phase", "ace_clock")},
        "score": np.zeros(LANES), "lives": np.full(LANES, LIVES),
        "invuln": np.full(LANES, 1.0), "clock": np.zeros(LANES), "alt_prev": np.zeros(LANES), "fuel": np.ones(LANES),
        # the land's map point and the ground frame's heading (unit; "up" the map)
        "map_u": np.zeros(LANES), "map_v": np.zeros(LANES), "map_gx": np.zeros(LANES), "map_gy": np.full(LANES, -1.0),
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
        if getattr(self, "autopilot", False) and self.session.state is not None:
            # The demo plays the player's early game first (USER_PACE for
            # DEMO_EASY_S), then ramps to full chaos by DEMO_CHAOS_S -- so what
            # it learns from its runs starts where the player starts.
            t = float(self.session.column("clock")[0])
            ramp = min(1.0, max(0.0, (t - DEMO_EASY_S) / (DEMO_CHAOS_S - DEMO_EASY_S)))
            return USER_PACE + (1.0 - USER_PACE) * ramp
        return 1.0 if not getattr(self, "learns", True) else USER_PACE

    def drive_settings(self) -> None:
        """Before each landing: the pace for whoever drives now, and the pilots."""
        column = self.session.column("pace")
        if column[0] != self.pace():
            column[...] = self.pace()
        self.fly_pilots()

    def motion(self) -> dict[str, Any]:
        """The player's three damped axes as the law runs them: YAW (turn;
        a held finger steers for its bearing from the ship), SPD (throttle on
        the up/down axis, inverted) and Z (height); and the patch the page
        draws in (the player fixed at its centre)."""
        return {"speed": SPEED, "tau": TAU, "reach": REACH,
                "z": [Z_MARGIN, DEPTH - Z_MARGIN], "depth": DEPTH,
                "yaw": [YAW_LO, YAW_HI], "spd": [SPD_MIN, self._top_speed()],
                "bind": {"yaw": {"move": "move_axis", "finger": True, "speed": TURN_RATE, "tau": TURN_TAU,
                                 "reach": TURN_REACH},
                         "spd": {"move": "move_axis_y", "sign": -1.0, "speed": THROTTLE, "tau": THROTTLE_TAU}},
                "patch": {"cx": self.width / 2.0, "cy": PLAYER_Y, "radius": RADIUS},
                # the land's travel law (entity_motion.ground_travel): stepped
                # here, predicted by the page, adopted with the ship
                "travel": ["map_u", "map_v", "map_gx", "map_gy"],
                # Twin shots from the nose on a reload, as the law fires them.
                "weapon": {"muzzles": [[-0.2, -MUZZLE], [0.2, -MUZZLE]], "velocity": [0.0, -SHOT_SPEED],
                           "reload": RELOAD, "max_in_flight": SHOTS[1] - SHOTS[0], "mesh": "bolt", "color": "bolt",
                           "size": 0.2}}

    def layout(self, aspect: float) -> None:
        width = field_width(aspect)
        if width == self.width:
            return
        s = self.session
        s.column("field_w")[...] = width             # the patch recentres on the next step
        self.epoch += 1
        self.width = width

    @property
    def emissive(self) -> dict[str, list[float]]:
        # Lightning and the three bomb styles' blasts glow (emissive materials;
        # the page also lights the scene from each).
        return {"lightning": [2.6, 2.8, 3.4], "blast_dumb": [3.2, 1.7, 0.5], "blast_chute": [2.4, 2.6, 3.2],
                "blast_seeker": [3.0, 0.8, 2.6]}

    def palette(self) -> dict[str, list[float]]:
        return {"rocket": [0.85, 0.88, 0.95], "rocket_blink": [0.35, 0.37, 0.45], "bolt": [1.0, 0.92, 0.25],
                "invader": [0.75, 0.35, 0.95], "ufo": [0.95, 0.15, 0.55], "bullet": [1.0, 0.25, 0.35],
                "bullet_ring": [0.35, 0.8, 1.0], "star": [0.55, 0.6, 0.8], "boom": [1.0, 0.55, 0.1],
                "cloud": [0.96, 0.98, 1.0], "boss": [0.95, 0.2, 0.25], "ace": [0.3, 1.0, 0.55],
                "rain": [0.5, 0.52, 0.56], "lightning": [0.85, 0.9, 1.0], "storm": [0.28, 0.29, 0.33],
                "station": [1.0, 0.82, 0.1], "flak": [1.0, 0.9, 0.2],
                "bomb_dumb": [0.3, 0.32, 0.36], "bomb_chute": [0.85, 0.85, 0.8], "chute": [0.95, 0.95, 0.9],
                "seeker": [1.0, 0.45, 0.2],
                "blast_dumb": [1.0, 0.6, 0.2], "blast_chute": [0.9, 0.95, 1.0], "blast_seeker": [1.0, 0.3, 0.9],
                "smoke_0": [0.04, 0.04, 0.05], "smoke_1": [0.07, 0.07, 0.08], "smoke_2": [0.1, 0.1, 0.11],
                "smoke_3": [0.13, 0.13, 0.14],
                "laser": [0.3, 1.0, 0.9], "missile": [1.0, 0.75, 0.3],
                "pickup_S": [0.35, 0.95, 0.45], "pickup_X": [1.0, 0.85, 0.2],
                "pickup_M": [1.0, 0.5, 0.2], "pickup_L": [0.3, 0.95, 1.0],
                "pickup_O": [0.35, 0.6, 1.0], "pickup_H": [1.0, 0.3, 0.35]}

    def entities(self) -> list[dict[str, Any]]:
        names = ("slot", "alive", "x", "y", "z", "vx", "vy", "vz", "boom", "born", "owner", "kind", "invuln", "weapon",
                 "ux", "uy", "uz", "hx", "hy", "hz")
        # (clouds and flashes are always shown; flashes glow -- see emissive)
        col = {name: self.session.column(name) for name in names}
        clock = float(self.session.column("clock")[0])
        # Only the patch: bodies within PATCH_R of the player (the rest of the
        # sphere's population is state, not drawn).  Each body's heading in the
        # patch (right positive from up the screen) orients it.
        u = np.stack([col["ux"], col["uy"], col["uz"]], axis=1)
        hd = np.stack([col["hx"], col["hy"], col["hz"]], axis=1)
        u0, h0 = u[0], hd[0]
        r0 = np.cross(h0, u0)
        reach = RADIUS * np.arctan2(np.linalg.norm(np.cross(u, u0), axis=1), u @ u0)
        heading = np.arctan2(hd @ r0, hd @ h0)
        # ... and within the land's width either side of the ship (no body
        # floats off the edge of the drawn board).
        show = (((col["alive"] > 0.0) | (col["boom"] > 0.0)) & (reach < PATCH_R)
                & (np.abs(col["x"] - self.width / 2.0) < self.width / 2.0 + 0.6))
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
                mesh, color, size = "cloud", f"smoke_{step}", 0.62 + 0.16 * step
            elif slot == BULLET:
                flak = STATIONS[0] <= int(col["owner"][k]) < STATIONS[1]
                mesh, color, size = ("disc", "flak", 0.15) if flak else ("disc", "bullet", 0.13)
            elif slot == MISSILE_SLOT:
                mesh, color, size = "rocket", "missile", 0.24
            elif slot == PICKUP:
                letter = {SPREAD_UP: "S", MULT_UP: "X", MISSILE_UP: "M", LASER_UP: "L",
                          SHIELD_UP: "O", HEALTH_UP: "H"}.get(float(col["kind"][k]), "S")
                mesh, color, size = letter, f"pickup_{letter}", 0.34
            elif slot == STATION:
                mesh, color, size = "bomb", "station", 0.42
            elif slot == FLASH:
                mesh, color, size = "bolt", "lightning", 0.55
            elif slot == BOMB_SLOT:
                style_k = float(col["kind"][k])
                if style_k == SEEKER:
                    mesh, color, size = "rocket", "seeker", 0.26
                elif style_k == CHUTE and clock - float(col["born"][k]) >= CHUTE_DEPLOY:
                    mesh, color, size = "parachute", "chute", 0.42
                else:
                    mesh, color, size = "bomb", "bomb_" + ("chute" if style_k == CHUTE else "dumb"), 0.24
            elif slot == BLAST_SLOT:
                style_k = float(col["kind"][k])
                grow = 0.6 + 1.6 * min(1.0, float(clock - col["born"][k]) / BLAST_LIFE)
                mesh, color = "boom", {DUMB: "blast_dumb", CHUTE: "blast_chute", SEEKER: "blast_seeker"}.get(style_k, "blast_dumb")
                size = BLAST_R.get(style_k, 1.5) * 0.5 * grow
            else:
                # A cloud: frosted glass, larger the higher it floats; grey
                # when smoke has turned it to rain.
                kind_k = float(col["kind"][k])
                if kind_k == THUNDER:
                    mesh, color, size = "cloud", "storm", 0.9 + 0.12 * float(col["z"][k])
                elif kind_k == RAIN:
                    mesh, color, size = "cloud", "rain", 0.34 + 0.08 * float(col["z"][k])
                else:
                    mesh, color, size = "cloud", "cloud", 0.26 + 0.08 * float(col["z"][k])
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": 0.0 if exploding else float(col["vx"][k]),
                        "vy": 0.0 if exploding else float(col["vy"][k]),
                        "z": float(col["z"][k]), "vz": 0.0 if exploding else float(col["vz"][k]),
                        "mesh": mesh, "color": color, "size": size,
                        # Top-down: the ship lies flat, nose up the screen (the
                        # direction of illusory travel); a missile faces its
                        # flight (the page aims it along its velocity each frame).
                        # Nothing leans toward the camera: every body lies in
                        # its own flight plane, as it would really fly.
                        # Craft point along their own heading on the sphere
                        # (the rocket glyph's nose sits at 45 degrees).
                        "roll": (0.785 if slot == PLAYER else
                                 0.785 - float(heading[k]) if mesh == "rocket" else
                                 -float(heading[k]) if slot == ENEMY and not exploding else 0.0),
                        "lean": 0.0,
                        "face": False,
                        # drawn with the glass (blended), not in the opaque pass
                        "translucent": bool(slot in (STAR, SMOKE)),
                        "ribbon": bool(slot == FLASH),       # a lightning end: the page draws the bolt
                        "ground": bool(slot == STATION),     # emplacements stand on the land's relief
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

    HOSTILE_LANES = (ENEMIES, BULLETS, STATIONS)
    # Lo-fi arcade: the backdrop shows through (no floor slab); the glass
    # plate and the arcade ring are on by default so the lights glow into it.
    SCENE = {"floor": False, "glass": True, "arcade": True, "terrain": True}
    SCENE_PREFER = {"glass": True, "arcade": True}           # projection glass: off unless the viewer turns it on
    HEALTH_MAX = HEALTH_MAX

    #: The sun's direction in the planet's frame: fixed, so the local sky is
    #: where the player is -- fly round the globe through dawn, day, dusk
    #: and night.
    SUN = (0.82, 0.18, 0.55)

    def _top_speed(self) -> float:
        """The throttle's top as the law applies it: SPD_MAX, CRUISE with the tank dry."""
        if self.session.state is None:
            return SPD_MAX
        return SPD_MAX if float(self.session.column("fuel")[0]) > 1e-3 else CRUISE

    def state(self) -> dict[str, Any]:
        st = super().state()
        if self.session.state is not None:
            c = self.session.column
            # The flight gauges: throttle (0 at SPD_MIN .. 1 at SPD_MAX), the
            # fuel tank, cruise as a throttle fraction; and the boss's health.
            span = SPD_MAX - SPD_MIN
            st["flight"] = {"throttle": (float(c("spd")[0]) - SPD_MIN) / span, "fuel": float(c("fuel")[0]),
                            "cruise": (CRUISE - SPD_MIN) / span, "min": SPD_MIN, "max": SPD_MAX}
            boss_live = float(c("alive")[BOSS_LANE]) > 0.0 and float(c("boss_phase")[0]) == 1.0
            st["boss"] = ({"hp": max(0.0, float(c("hp")[BOSS_LANE])),
                           "max": BOSS_HP * (1.0 + 0.5 * (self.level - 1.0))} if boss_live else None)
            # The player's place and heading on the globe (and the yaw they
            # were read at), for the page's sun and sky.
            st["globe"] = {"u": [float(c("ux")[0]), float(c("uy")[0]), float(c("uz")[0])],
                           "h": [float(c("hx")[0]), float(c("hy")[0]), float(c("hz")[0])],
                           "yaw": float(c("yaw")[0]), "sun": list(self.SUN), "radius": RADIUS}
        return st

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
