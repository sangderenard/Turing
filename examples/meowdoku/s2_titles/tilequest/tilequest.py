"""Tile Quest: a top-down tile-map adventure, one dt-system entity law.

An original game of the top-down tile-adventure genre.  A procedurally
generated map (walls, a river with bridges, trees, ponds, a walled village
with gates and rooms, monster lairs) is walked by KINDS of entity that differ
only in their kind row (``walkers.Kind``): the player, villagers who commute
between homes and workplaces, slimes, wolves and goblins that wander, hunt or
flee, and towers that shoot the nearest monster.  RPG/rogue (the player and
the monsters), tower defence (the towers on the bridge approaches) and
city/traffic (the villagers on the road network) are the same law.

The substrate is ``walkers.py``: every entity's choice network runs batched in
the law and picks a motivation (wander, seek, flee, follow, attack, rest);
pathing is by flow fields precomputed here at map build (work, home, town,
lair, avoidance) and stored as dt-system columns on the map's own lanes.

Lanes (contiguous blocks):
    0            the player
    1..12        villagers
    13..18       slimes
    19..22       wolves
    23..26       goblins
    27..31       towers
    32..36       arrows (arrow ``i`` belongs to tower ``i``)
    37..         the map: one lane per cell, row-major (``tile`` and the five
                 flow fields' ``fdx``/``fdy``/``fdd`` columns hold the map)

Columns: the kind row's constants per lane (``team speed sense hpmax dmg rad
reload_t respawn_t needrate goalf0 goalf1 refugef mends``), ``home_x/y``,
the map (``tile``, ``fdx0..4`` ``fdy0..4`` ``fdd0..4``), the walkers' state
(``alive x y vx vy hp hurt cool timer need phase mot think wx wy born``), the
game's (``score kills trips losses clock lives``), the CONTROLS and ``dt``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from meowdoku import walkers
from meowdoku.entities import EntityGame, LawPiece, as_float, noise
from meowdoku.entity_motion import damped_axis
from meowdoku.walkers import ATTACK, FLEE, FOLLOW, REST, SEEK, WANDER, Kind, FlowField
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

# ---------------------------------------------------------------- the world
WIDTH, HEIGHT = 28, 18
GRASS, WALL, WATER, TREE, PATH, HOME, WORK, LAIR, DOOR = range(9)
BLOCKED = (WALL, WATER, TREE)
#: Flow fields, in column order.
F_WORK, F_HOME, F_TOWN, F_LAIR, F_AVOID = range(5)
NF = 5

PLAYER, VILLAGER, SLIME, WOLF, GOBLIN, TOWER, ARROW, MAP = range(8)
N_VILLAGERS, N_SLIMES, N_WOLVES, N_GOBLINS, N_TOWERS = 12, 6, 4, 4, 5
V0 = 1
S0 = V0 + N_VILLAGERS
W0 = S0 + N_SLIMES
G0 = W0 + N_WOLVES
T0 = G0 + N_GOBLINS
A0 = T0 + N_TOWERS
ENT = A0 + N_TOWERS
MAP0 = ENT
LANES = ENT + WIDTH * HEIGHT

PLAYER_SPEED, PLAYER_TAU, PLAYER_REACH = 5.0, 0.06, 0.18
PLAYER_HP = 6.0
WALK_TAU = 0.12
MELEE_REACH, MELEE_DAMAGE = 1.35, 1.5
ARROW_SPEED, ARROW_DAMAGE, ARROW_LIFE = 11.0, 1.0, 1.1
SEPARATION = 0.8
DWELL = 2.0

# ----------------------------------------------------------- the kind rows
# Hidden units are gates (tanh(w.f + b) + 1) / 2 in [0, 1]; logits are b + w.gates.
# Reading a net: "fear" is on when a foe is near and no ally is; a villager
# FOLLOWs its path unless afraid (FLEE), weary or dwelling (REST).
KINDS: dict[int, Kind] = {
    PLAYER: Kind("player", PLAYER, team=0, speed=PLAYER_SPEED, sense=6.0, hp=PLAYER_HP, radius=0.3, reload=0.35),
    VILLAGER: Kind("villager", VILLAGER, team=0, speed=1.8, sense=5.0, hp=3.0, radius=0.28, respawn=15.0,
                   needrate=0.04, goal=(F_WORK, F_HOME), refuge=F_HOME, mends=0.3, net={
        "hidden": {
            "fear": ({"foe_prox": 4.0, "foe_close": 2.0, "ally_prox": -1.0}, -1.2),
            "dwell": ({"busy": 4.0}, -2.0),
            "weary": ({"need": 6.0}, -4.0),
            "wounded": ({"health": -5.0}, 2.0),
        },
        "out": {
            "wander": ({"fear": -2.0, "dwell": -1.0}, 0.5),
            "follow": ({"fear": -3.0, "dwell": -3.0, "weary": -2.5}, 1.4),
            "flee": ({"fear": 4.0}, -1.0),
            "rest": ({"dwell": 3.0, "weary": 2.5, "wounded": 0.8, "fear": -2.0}, 0.2),
        }}),
    SLIME: Kind("slime", SLIME, team=1, speed=1.3, sense=4.0, hp=2.0, damage=0.5, radius=0.3, reload=1.0,
                respawn=6.0, needrate=0.02, refuge=F_LAIR, mends=0.4, net={
        "hidden": {
            "spot": ({"foe_prox": 5.0}, -1.5),
            "near": ({"foe_close": 6.0}, -1.0),
            "hurt": ({"health": -6.0}, 3.0),
            "home": ({"refuge_near": 5.0}, -3.0),
        },
        "out": {
            "wander": ({"spot": -1.5, "hurt": -3.0}, 1.0),
            "seek": ({"spot": 2.5, "hurt": -3.0}, -0.5),
            "attack": ({"near": 4.0, "hurt": -3.0}, -1.5),
            "flee": ({"hurt": 4.0, "home": -2.5}, -2.0),
            "rest": ({"hurt": 2.0, "home": 3.0}, -2.5),
        }}),
    WOLF: Kind("wolf", WOLF, team=1, speed=3.0, sense=6.0, hp=3.0, damage=1.0, radius=0.32, reload=0.9,
               respawn=10.0, needrate=0.02, goal=(F_TOWN, F_TOWN), refuge=F_LAIR, mends=0.5, net={
        "hidden": {
            "spot": ({"foe_prox": 5.0}, -1.0),
            "near": ({"foe_close": 6.0}, -1.0),
            "far": ({"goal_far": 8.0}, -0.8),
            "hurt": ({"health": -6.0}, 2.5),
            "home": ({"refuge_near": 5.0}, -3.0),
        },
        "out": {
            "follow": ({"far": 1.6, "spot": -2.5, "hurt": -3.0}, -0.3),
            "wander": ({"spot": -1.5, "hurt": -3.0}, 0.5),
            "seek": ({"spot": 3.0, "hurt": -3.0}, -1.0),
            "attack": ({"near": 4.5, "spot": 0.5, "hurt": -3.0}, -1.5),
            "flee": ({"hurt": 5.0, "home": -2.5}, -2.0),
            "rest": ({"hurt": 2.0, "home": 3.0}, -2.5),
        }}),
    GOBLIN: Kind("goblin", GOBLIN, team=1, speed=2.1, sense=5.0, hp=3.0, damage=1.0, radius=0.3, reload=1.1,
                 respawn=9.0, needrate=0.02, goal=(F_TOWN, F_TOWN), refuge=F_LAIR, mends=0.4, net={
        "hidden": {
            "spot": ({"foe_prox": 5.0}, -1.0),
            "near": ({"foe_close": 6.0}, -1.0),
            "far": ({"goal_far": 8.0}, -0.8),
            "hurt": ({"health": -6.0}, 2.2),
            "brave": ({"ally_prox": 4.0}, -1.0),
            "home": ({"refuge_near": 5.0}, -3.0),
        },
        "out": {
            "follow": ({"far": 1.6, "spot": -2.5, "hurt": -3.0}, -0.3),
            "wander": ({"spot": -1.5, "hurt": -3.0}, 0.5),
            "seek": ({"spot": 3.0, "hurt": -3.0}, -1.0),
            "attack": ({"near": 4.5, "spot": 0.5, "hurt": -3.0, "brave": 0.5}, -1.5),
            "flee": ({"hurt": 5.0, "brave": -1.5, "home": -2.5}, -2.0),
            "rest": ({"hurt": 2.0, "home": 3.0}, -2.5),
        }}),
    TOWER: Kind("tower", TOWER, team=0, speed=0.0, sense=6.5, hp=12.0, radius=0.4, reload=0.8, respawn=25.0, net={
        "hidden": {"spot": ({"foe_prox": 20.0}, 0.0)},
        "out": {"attack": ({"spot": 3.0}, -1.0), "rest": ({}, 1.0)},
    }),
    ARROW: Kind("arrow", ARROW, team=-1, hp=1.0, radius=0.12),
}

# ------------------------------------------------------------- the columns
CONTROLS = {"move_axis": "move-axis-x", "move_axis_y": "move-axis-y", "steer_x": "pointer-board-x",
            "steer_y": "pointer-board-y", "fire": "primary-action"}
KIND_COLUMNS = ("team", "speed", "sense", "hpmax", "dmg", "rad", "reload_t", "respawn_t", "needrate",
                "goalf0", "goalf1", "refugef", "mends")
FIELD_COLUMNS = tuple(f"{n}{i}" for n in ("fdx", "fdy", "fdd") for i in range(NF))
STATIC = ("lane", "kind", *KIND_COLUMNS, "home_x", "home_y", "tile", *FIELD_COLUMNS)
STATE = ("alive", "x", "y", "vx", "vy", "hp", "hurt", "cool", "timer", "need", "phase", "mot", "think",
         "wx", "wy", "born", "score", "kills", "trips", "losses", "clock", "lives")
ARGUMENTS = (*STATIC, *STATE, *CONTROLS, "dt")


def tilequest_step(*columns):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span (``dt`` a scalar)."""
    c = dict(zip(ARGUMENTS, columns))
    f = as_float
    n, w, h, m0 = ENT, WIDTH, HEIGHT, MAP0
    # dt as an entity-lane span: a scalar natively, a column in the NumPy product.
    dt = (c["x"] * 0.0 + c["dt"])[0:n]

    def e(name):                     # the entity lanes of a column
        return c[name][0:n]

    lane, kind, team, speed, sense = e("lane"), e("kind"), e("team"), e("speed"), e("sense")
    hpmax, dmg, rad, reload_t, respawn_t = e("hpmax"), e("dmg"), e("rad"), e("reload_t"), e("respawn_t")
    needrate, mends = e("needrate"), e("mends")
    home_x, home_y = e("home_x"), e("home_y")
    alive, x, y, vx, vy, hp = e("alive"), e("x"), e("y"), e("vx"), e("vy"), e("hp")
    hurt, cool, timer, need, phase = e("hurt"), e("cool"), e("timer"), e("need"), e("phase")
    mot, think, wx, wy, born = e("mot"), e("think"), e("wx"), e("wy"), e("born")
    clock, lives = e("clock"), e("lives")
    zero = x * 0.0

    is_player, is_villager, is_slime = kind == PLAYER, kind == VILLAGER, kind == SLIME
    is_wolf, is_goblin, is_tower, is_arrow = kind == WOLF, kind == GOBLIN, kind == TOWER, kind == ARROW
    is_monster = is_slime | is_wolf | is_goblin
    is_walker = is_player | is_villager | is_monster
    folk = is_player | is_villager | is_tower
    live = alive > 0.0
    playing = lives > 0.0
    clock_n = clock + dt

    # ------------------------------------------------------------ the map
    lanes_total = int(c["tile"].shape[0])
    tile = c["tile"]
    solid_tile = f((tile == WALL) | (tile == WATER) | (tile == TREE))
    arrow_stop = f((tile == WALL) | (tile == TREE))
    fdx = walkers.stack_columns([c[f"fdx{i}"] for i in range(NF)])
    fdy = walkers.stack_columns([c[f"fdy{i}"] for i in range(NF)])
    fdd = walkers.stack_columns([c[f"fdd{i}"] for i in range(NF)])

    def blocked_at(px, py, table):
        cell = walkers.cell_lane(px, py, w, h, m0)
        outside = (px < 0.25) | (px > w - 0.25) | (py < 0.25) | (py > h - 0.25)
        return (walkers.gather(table, cell) > 0.5) | outside

    cell = walkers.cell_lane(x, y, w, h, m0)
    goal_field = where(phase > 0.5, e("goalf1"), e("goalf0"))
    gdx, gdy = walkers.field_at(fdx, goal_field, cell, lanes_total), walkers.field_at(fdy, goal_field, cell, lanes_total)
    gdist = walkers.field_at(fdd, goal_field, cell, lanes_total)
    refuge = e("refugef")
    rdx, rdy = walkers.field_at(fdx, refuge, cell, lanes_total), walkers.field_at(fdy, refuge, cell, lanes_total)
    rdist = walkers.field_at(fdd, refuge, cell, lanes_total)
    avoid = zero + float(F_AVOID)
    adx, ady = walkers.field_at(fdx, avoid, cell, lanes_total), walkers.field_at(fdy, avoid, cell, lanes_total)
    clearance = walkers.field_at(fdd, avoid, cell, lanes_total)
    push = ((1.5 - clearance) * (1.0 / 1.5)).clip(0.0, 1.0)

    # ------------------------------------------------- lane against lane
    ddx, ddy, dd = walkers.pairwise(x, y)

    def row(v):
        return v.reshape((n, 1))

    def col(v):
        return v.reshape((1, n))

    other = row(lane) != col(lane)
    seeker = live & (is_walker | is_tower)
    target = live & (is_walker | is_tower)
    # A foe is one that can be SEEN: nothing on the line between that stops
    # the seer's way to it.  A walker seeks directly only along a line it can
    # walk (water stops it); a tower shoots along a line an arrow can fly
    # (water does not).
    walk_line = walkers.line_clear(row(x), row(y), ddx, ddy, solid_tile, w, h, m0)
    shot_line = walkers.line_clear(row(x), row(y), ddx, ddy, arrow_stop, w, h, m0)
    sees = where(row(is_tower), shot_line, walk_line)
    foe_valid = row(seeker) & col(target) & (row(team) != col(team)) & sees
    foe_d, fux, fuy = walkers.nearest(ddx, ddy, dd, foe_valid)
    ally_valid = row(seeker) & col(target) & (row(team) == col(team)) & other
    ally_d, _, _ = walkers.nearest(ddx, ddy, dd, ally_valid)
    crowd_valid = row(live) & col(live) & (row(kind) == col(kind)) & other & (dd < 1.2)
    crowd = (f(crowd_valid).sum(dim=1) * (1.0 / 3.0)).clip(0.0, 1.0)
    mover_live = live & is_walker
    sep_x, sep_y = walkers.separation_of(ddx, ddy, dd, row(mover_live) & col(mover_live) & other, SEPARATION)

    # ------------------------------------------------ the choice network
    feats = {
        "bias": zero + 1.0,
        "foe_prox": (1.0 - foe_d * (1.0 / (sense + 1e-6))).clip(0.0, 1.0),
        "foe_close": (1.0 - foe_d * (1.0 / 1.5)).clip(0.0, 1.0),
        "ally_prox": (1.0 - ally_d * (1.0 / 3.0)).clip(0.0, 1.0),
        "goal_far": (gdist * (1.0 / 30.0)).clip(0.0, 1.0),
        "health": (hp * (1.0 / (hpmax + 1e-6))).clip(0.0, 1.0),
        "need": need.clip(0.0, 1.0),
        "crowd": crowd,
        "phase": phase,
        "busy": timer.clip(0.0, 1.0),
        "refuge_near": (1.0 - rdist * (1.0 / 6.0)).clip(0.0, 1.0),
    }
    feats["foe_ahead"] = (wx * fux + wy * fuy) * feats["foe_prox"]
    choice = walkers.choose_motivation(KINDS, kind, feats, zero)
    rethink = (think - dt) <= 0.0
    mot_n = where(rethink, choice, mot)
    think_n = where(rethink, 0.25 + 0.15 * noise(lane * 3.7 + clock_n * 9.1), think - dt)

    def is_m(k):
        return mot_n == float(k)

    # ------------------------------------------------- motivation -> heading
    wander_on = is_m(WANDER)
    turn = where(wander_on, (noise(lane * 2.3 + clock_n * 5.1) - 0.5) * 30.0 * dt, zero)
    rwx, rwy = walkers.rotate(wx, wy, turn)
    wander_x, wander_y = walkers.unit(rwx + adx * (1.5 * push), rwy + ady * (1.5 * push))
    seek_x, seek_y = walkers.unit(fux + adx * (1.5 * push), fuy + ady * (1.5 * push))
    flee_x, flee_y = walkers.unit(rdx * 0.8 - fux + adx * (1.5 * push), rdy * 0.8 - fuy + ady * (1.5 * push))
    follow_x, follow_y = walkers.unit(gdx + adx * (0.4 * push), gdy + ady * (0.4 * push))
    is_seek = is_m(SEEK) | is_m(ATTACK)
    cmd_x = where(wander_on, wander_x, where(is_seek, seek_x, where(is_m(FLEE), flee_x, where(is_m(FOLLOW), follow_x, zero))))
    cmd_y = where(wander_on, wander_y, where(is_seek, seek_y, where(is_m(FLEE), flee_y, where(is_m(FOLLOW), follow_y, zero))))
    pace = where(wander_on, walkers.PACE[WANDER], where(is_m(SEEK), walkers.PACE[SEEK], where(is_m(FLEE), walkers.PACE[FLEE],
            where(is_m(FOLLOW), walkers.PACE[FOLLOW], where(is_m(ATTACK), walkers.PACE[ATTACK], zero)))))
    moving_cmd = (cmd_x * cmd_x + cmd_y * cmd_y) > 0.25
    k_head = dt / (0.15 + dt)
    hx, hy = walkers.unit(rwx + (cmd_x - rwx) * k_head, rwy + (cmd_y - rwy) * k_head)
    wx_n = where(moving_cmd, hx, rwx)
    wy_n = where(moving_cmd, hy, rwy)

    # ---------------------------------------------------------- movement
    walker = live & is_walker & playing
    npc = walker & ~is_player
    want_x = cmd_x * (speed * pace) + sep_x * 1.2
    want_y = cmd_y * (speed * pace) + sep_y * 1.2
    k_vel = dt / (WALK_TAU + dt)
    vx_w = vx + (want_x - vx) * k_vel
    vy_w = vy + (want_y - vy) * k_vel
    lo, hi_x, hi_y = zero + 0.35, zero + (w - 0.35), zero + (h - 0.35)
    ship_x, ship_vx = damped_axis(x, vx, e("move_axis"), e("steer_x"), lo, hi_x, PLAYER_SPEED, PLAYER_TAU, PLAYER_REACH, dt)
    ship_y, ship_vy = damped_axis(y, vy, e("move_axis_y"), e("steer_y"), lo, hi_y, PLAYER_SPEED, PLAYER_TAU, PLAYER_REACH, dt)
    nx = where(is_player, ship_x, x + vx_w * dt)
    ny = where(is_player, ship_y, y + vy_w * dt)
    vxn = where(is_player, ship_vx, vx_w)
    vyn = where(is_player, ship_vy, vy_w)
    # Walls: each axis alone, the leading edge of the body against the tile.
    stop_x = walker & blocked_at(nx + where(vxn >= 0.0, rad, -rad), y, solid_tile)
    stop_y = walker & blocked_at(x, ny + where(vyn >= 0.0, rad, -rad), solid_tile)
    x1 = where(walker, where(stop_x, x, nx), x)
    y1 = where(walker, where(stop_y, y, ny), y)
    vx1 = where(walker, where(stop_x, 0.0, vxn), zero)
    vy1 = where(walker, where(stop_y, 0.0, vyn), zero)

    # ---------------------------------------- arrows: the towers' shots
    shoots = is_tower & live & playing & is_m(ATTACK) & ((cool - dt) <= 0.0) & (foe_d < sense)
    free_back = walkers.shift_block(f(is_arrow & ~live), A0, N_TOWERS, T0) > 0.5
    fire_t = shoots & free_back
    fire_a = (walkers.shift_block(f(fire_t), T0, N_TOWERS, A0) > 0.5) & is_arrow
    src_x, src_y = walkers.shift_block(x, T0, N_TOWERS, A0), walkers.shift_block(y, T0, N_TOWERS, A0)
    aim_x, aim_y = walkers.shift_block(fux, T0, N_TOWERS, A0), walkers.shift_block(fuy, T0, N_TOWERS, A0)
    flying = live & is_arrow & playing
    ax, ay = x + vx * dt, y + vy * dt
    spent_wall = flying & (walkers.gather(arrow_stop, walkers.cell_lane(ax, ay, w, h, m0)) > 0.5)
    lost = flying & ((ax < 0.0) | (ax > w) | (ay < 0.0) | (ay > h) | ((timer - dt) <= 0.0))

    # ----------------------------------------------------------- combat
    # Contact blows: a monster touching a folk lane with its blow ready.
    touch = dd < (row(rad) + col(rad) + 0.1)
    blow = (row(live) & col(live) & row(is_monster) & col(folk) & (row(cool) <= 0.0) & (col(hurt) <= 0.0) & touch)
    struck = f(blow).sum(dim=1) > 0.0
    contact = (f(blow) * row(dmg)).sum(dim=0)
    # The player's sword: a swing hurts every monster within reach.
    pxs = (f(is_player & live) * x).sum()
    pys = (f(is_player & live) * y).sum()
    ground = (x - pxs) * (x - pxs) + (y - pys) * (y - pys)
    swing = is_player & live & playing & (e("fire") > 0.0) & (cool <= 0.0)
    swung = f(swing).sum() > 0.0
    melee = f(swung & live & is_monster & (ground < MELEE_REACH * MELEE_REACH)) * MELEE_DAMAGE
    # Arrows against monsters.
    ahit = f(row(live) & col(live) & row(is_arrow) & col(is_monster) & (dd < (row(rad) + col(rad))))
    arrow_dmg = ahit.sum(dim=0) * ARROW_DAMAGE
    arrow_spent = ahit.sum(dim=1) > 0.0

    resting = live & is_m(REST)
    hp_m = (hp + f(resting) * mends * dt).clip(0.0, 1.0e6)
    hp_m = where(hp_m > hpmax, hpmax, hp_m)
    hp1 = hp_m - contact - melee - arrow_dmg
    dies = live & ~is_arrow & (hp1 <= 0.0)
    hurt_n = where(contact > 0.0, 0.7, (hurt - dt).clip(0.0, 9.0))
    cool_n = where(struck | swing, reload_t, where(fire_t, reload_t, (cool - dt).clip(0.0, 9.0)))
    arrow_dead = (flying & (spent_wall | lost | arrow_spent))

    # ----------------------------------- need, arrival, dwelling, respawn
    need_n = where(live, where(is_m(REST), need - 0.35 * dt, need + needrate * dt).clip(0.0, 1.0), need)
    arrived = is_villager & live & playing & (gdist < 0.5) & (timer <= 0.0)
    phase_n = where(arrived, 1.0 - phase, phase)
    dwell = DWELL + 3.0 * noise(lane * 1.1 + clock_n * 3.3)
    respawnable = (respawn_t > 0.0) & ~live & ~is_player & ~is_arrow & playing & ((timer - dt) <= 0.0)
    timer_n = where(live, (timer - dt).clip(0.0, 1.0e6), timer - dt)
    timer_n = where(arrived, dwell, timer_n)
    timer_n = where(dies, respawn_t * (0.7 + 0.6 * noise(lane * 5.9 + clock_n * 2.3)), timer_n)
    timer_n = where(fire_a, ARROW_LIFE, timer_n)
    spawn = respawnable

    alive_walker = (live & ~dies) | spawn
    alive_arrow = (live & ~arrow_dead) | fire_a
    live_n = where(is_arrow, alive_arrow, alive_walker)

    # ----------------------------------------------- the next state
    ax_n = where(fire_a, src_x + aim_x * 0.4, where(flying, ax, x))
    ay_n = where(fire_a, src_y + aim_y * 0.4, where(flying, ay, y))
    x_n = where(is_arrow, ax_n, where(spawn, home_x, x1))
    y_n = where(is_arrow, ay_n, where(spawn, home_y, y1))
    vx_n = where(is_arrow, where(fire_a, aim_x * ARROW_SPEED, where(flying & ~arrow_dead, vx, zero)), where(spawn, zero, vx1))
    vy_n = where(is_arrow, where(fire_a, aim_y * ARROW_SPEED, where(flying & ~arrow_dead, vy, zero)), where(spawn, zero, vy1))
    hp_n = where(spawn, hpmax, where(is_arrow, hp, hp1))
    hurt_f = where(spawn, 1.0, hurt_n)
    cool_f = where(spawn, 0.5, cool_n)
    need_f = where(spawn, zero, need_n)
    phase_f = where(spawn, zero, phase_n)
    mot_f = where(spawn, zero, mot_n)
    think_f = where(spawn, zero, think_n)
    wx_f, wy_f = where(spawn | is_arrow, wx, wx_n), where(spawn | is_arrow, wy, wy_n)
    born_f = where(spawn | fire_a, clock_n, born)

    kills_d = f(dies & is_monster).sum()
    trips_d = f(arrived & (phase < 0.5)).sum()
    losses_d = f(dies & is_villager).sum()
    hero_hp = (f(is_player) * hp1).sum()

    # The longest step that cannot carry anything through a wall or a body.
    reach = where(live, speed * 1.3, zero) + where(live & is_arrow, ARROW_SPEED, zero)
    dt_limit = (0.3 / (reach.max() + 1e-6)).clip(0.0, 0.5 * PLAYER_TAU)

    def wide(name, head):            # entity lanes' new values, then the map lanes (unchanged)
        return walkers.widen(head, c[name][n:])

    def whole(name, delta):          # a game-wide column: the same value on every lane
        return c[name] + delta

    full_x = c["x"]
    return (
        wide("alive", f(live_n)), wide("x", x_n), wide("y", y_n), wide("vx", vx_n), wide("vy", vy_n),
        wide("hp", hp_n), wide("hurt", hurt_f), wide("cool", cool_f), wide("timer", timer_n),
        wide("need", need_f), wide("phase", phase_f), wide("mot", mot_f), wide("think", think_f),
        wide("wx", wx_f), wide("wy", wy_f), wide("born", born_f),
        whole("score", kills_d * 10.0 + trips_d - losses_d * 5.0), whole("kills", kills_d),
        whole("trips", trips_d), whole("losses", losses_d), c["clock"] + c["dt"],
        (c["lives"] * 0.0 + hero_hp),
        f(live_n).sum() + full_x * 0.0,
        dt_limit + full_x * 0.0,
    )


class TileQuestPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "tilequest_step"
    law = staticmethod(tilequest_step)
    argument_names = ARGUMENTS
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")
    source_modules = ("meowdoku.walkers",)

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


# ------------------------------------------------------------ map building

@dataclass
class TileMap:
    """A generated map and everything precomputed on it."""

    seed: int
    grid: np.ndarray                          # (HEIGHT, WIDTH) tile types
    fields: list[FlowField]
    homes: list[tuple[int, int]]
    works: list[tuple[int, int]]
    lairs: list[tuple[int, int]]
    towers: list[tuple[float, float]]
    start: tuple[float, float]

    def passable(self) -> np.ndarray:
        return ~np.isin(self.grid, BLOCKED)


def _ring(grid, x0, y0, x1, y1, tile=WALL):
    grid[y0, x0:x1 + 1] = tile
    grid[y1, x0:x1 + 1] = tile
    grid[y0:y1 + 1, x0] = tile
    grid[y0:y1 + 1, x1] = tile


def _carve(grid, a, b):
    """An L-shaped road from cell ``a`` to ``b`` (x first): grass, trees and
    water become path (a bridge over water), a wall becomes a door; houses,
    workplaces, lairs and doors stay."""
    (ax, ay), (bx, by) = a, b
    cells = [(x, ay) for x in range(min(ax, bx), max(ax, bx) + 1)] + \
            [(bx, y) for y in range(min(ay, by), max(ay, by) + 1)]
    for x, y in cells:
        if grid[y, x] in (WALL,):
            grid[y, x] = DOOR
        elif grid[y, x] in (GRASS, TREE, WATER):
            grid[y, x] = PATH


def make_map(seed: int) -> TileMap:
    rng = np.random.default_rng(int(seed))
    grid = np.zeros((HEIGHT, WIDTH), dtype=int)
    # The river runs top to bottom between the village and the lairs.
    phase = rng.uniform(0.0, 6.28)
    for y in range(HEIGHT):
        left = 13 + int(round(1.0 * np.sin(y * 0.45 + phase)))
        grid[y, left:left + 2] = WATER
    # The walled village on the west bank; two rooms of lairs on the east.
    _ring(grid, 2, 3, 10, 14)
    _ring(grid, 19, 2, 26, 7)
    _ring(grid, 19, 10, 26, 15)
    homes = [(3, 5), (3, 7), (3, 9), (3, 11), (4, 13)]
    works = [(8, 5), (8, 8), (8, 11), (7, 13)]
    lairs = [(23, 3), (24, 5), (22, 6), (23, 12), (24, 14), (22, 11)]
    for cx, cy in homes:
        grid[cy, cx] = HOME
    for cx, cy in works:
        grid[cy, cx] = WORK
    for cx, cy in lairs:
        grid[cy, cx] = LAIR
    # Roads: the village streets, then three bridges and the rooms' doors.
    for (cx, cy) in homes:
        _carve(grid, (6, cy), (cx, cy))
    for (cx, cy) in works:
        _carve(grid, (6, cy), (cx, cy))
    _carve(grid, (6, 4), (6, 14))
    for y in (5, 9, 12):
        _carve(grid, (6, y), (18, y))
    _carve(grid, (18, 5), (19, 5))
    _carve(grid, (18, 12), (19, 12))
    _carve(grid, (19, 5), (23, 5))
    _carve(grid, (19, 12), (23, 12))
    _carve(grid, (23, 3), (23, 5))
    _carve(grid, (22, 11), (22, 12))
    _carve(grid, (23, 12), (24, 14))
    # Ponds and trees on the open grass outside the walls.
    for _ in range(3):
        px, py = int(rng.integers(16, 25)), int(rng.integers(8, 10))
        if (grid[py:py + 2, px:px + 2] == GRASS).all():
            grid[py:py + 2, px:px + 2] = WATER
    towers = [(11.5, 3.5), (11.5, 6.5), (11.5, 8.5), (11.5, 10.5), (11.5, 13.5)]
    keep_clear = {(int(tx), int(ty)) for tx, ty in towers}
    for _ in range(40):
        tx, ty = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        outside = not (2 <= tx <= 10 and 3 <= ty <= 14) and not (19 <= tx <= 26 and (2 <= ty <= 7 or 10 <= ty <= 15))
        if grid[ty, tx] == GRASS and outside and (tx, ty) not in keep_clear and (tx, ty) != (9, 8):
            grid[ty, tx] = TREE
    for tx, ty in keep_clear:
        if grid[ty, tx] in BLOCKED:
            grid[ty, tx] = GRASS
    passable = ~np.isin(grid, BLOCKED)
    fields = [FlowField.build(passable, [(cx, cy) for cx, cy in works]),
              FlowField.build(passable, [(cx, cy) for cx, cy in homes]),
              FlowField.build(passable, [(cx, cy) for cx, cy in homes + works]),
              FlowField.build(passable, [(cx, cy) for cx, cy in lairs]),
              FlowField.clearance(passable)]
    return TileMap(int(seed), grid, fields, homes, works, lairs, towers, (9.5, 8.5))


def initial_columns(game_map: TileMap) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(game_map.seed + 7919)
    kind = np.full(LANES, float(MAP))
    for lanes, kid in (((0, 1), PLAYER), ((V0, S0), VILLAGER), ((S0, W0), SLIME), ((W0, G0), WOLF),
                       ((G0, T0), GOBLIN), ((T0, A0), TOWER), ((A0, ENT), ARROW)):
        kind[lanes[0]:lanes[1]] = kid
    columns = {"lane": np.arange(LANES, dtype=np.float64), "kind": kind}
    columns.update(walkers.kind_columns(KINDS, kind))
    home_x, home_y = np.zeros(LANES), np.zeros(LANES)
    home_x[0], home_y[0] = game_map.start
    for i in range(N_VILLAGERS):
        cx, cy = game_map.homes[i % len(game_map.homes)]
        home_x[V0 + i], home_y[V0 + i] = cx + 0.5, cy + 0.5
    for lanes in ((S0, W0), (W0, G0), (G0, T0)):
        for i, lane in enumerate(range(*lanes)):
            cx, cy = game_map.lairs[(i + lanes[0]) % len(game_map.lairs)]
            home_x[lane], home_y[lane] = cx + 0.5, cy + 0.5
    for i, (tx, ty) in enumerate(game_map.towers):
        home_x[T0 + i], home_y[T0 + i] = tx, ty
    tile = np.zeros(LANES)
    tile[MAP0:] = game_map.grid.reshape(-1)
    columns.update({"home_x": home_x, "home_y": home_y, "tile": tile})
    columns.update(walkers.field_columns(game_map.fields, MAP0, LANES))
    angle = rng.uniform(0.0, 2.0 * np.pi, LANES)
    alive = np.zeros(LANES)
    alive[0:A0] = 1.0
    timer = np.zeros(LANES)
    timer[V0:S0] = rng.uniform(0.0, 4.0, N_VILLAGERS)
    hp = columns["hpmax"].copy()
    columns.update({
        "alive": alive, "x": home_x.copy(), "y": home_y.copy(), "vx": np.zeros(LANES), "vy": np.zeros(LANES),
        "hp": hp, "hurt": np.zeros(LANES), "cool": np.zeros(LANES), "timer": timer, "need": np.zeros(LANES),
        "phase": np.zeros(LANES), "mot": np.zeros(LANES), "think": rng.uniform(0.0, 0.25, LANES),
        "wx": np.cos(angle), "wy": np.sin(angle), "born": np.zeros(LANES),
        "score": np.zeros(LANES), "kills": np.zeros(LANES), "trips": np.zeros(LANES), "losses": np.zeros(LANES),
        "clock": np.zeros(LANES), "lives": np.full(LANES, PLAYER_HP),
        "move_axis": np.zeros(LANES), "move_axis_y": np.zeros(LANES),
        "steer_x": np.full(LANES, -1.0), "steer_y": np.full(LANES, -1.0), "fire": np.zeros(LANES),
    })
    return columns


# ------------------------------------------------------------------ the game

#: Mesh vocabulary beyond ``invaders.EMOJI`` (brick, bolt) and the side
#: scroller's (ground): name -> emoji, exported by ``web_export``.
EMOJI = {"water": "\U0001F7E6", "tree": "\U0001F332", "home": "\U0001F3E0", "work": "\U0001F3ED",
         "lair": "\U0001F480", "door": "\U0001F6AA", "wizard": "\U0001F9D9", "villager": "\U0001F9D1",
         "slime": "\U0001F9A0", "wolf": "\U0001F43A", "goblin": "\U0001F47A", "tower": "\U0001F5FC"}
EMOJI_SUBDIVISIONS: dict[str, int] = {}

#: tile type -> (mesh, colour); grass is the floor itself.
TILE_LOOK = {WALL: ("brick", "wall"), WATER: ("water", "water"), TREE: ("tree", "tree"), PATH: ("ground", "path"),
             HOME: ("home", "home"), WORK: ("work", "work"), LAIR: ("lair", "lair"), DOOR: ("door", "door")}
#: kind -> (mesh, colour, size)
KIND_LOOK = {PLAYER: ("wizard", "wizard", 0.42), VILLAGER: ("villager", "villager", 0.36),
             SLIME: ("slime", "slime", 0.34), WOLF: ("wolf", "wolf", 0.4), GOBLIN: ("goblin", "goblin", 0.38),
             TOWER: ("tower", "tower", 0.5), ARROW: ("bolt", "bolt", 0.2)}


class TileQuest(EntityGame):
    game = "tilequest"
    title = "Tile Quest (free board)"
    width, height = float(WIDTH), float(HEIGHT)
    CONTROLS = tuple(CONTROLS)
    PLAYER_LANE = 0
    HEALTH_MAX = PLAYER_HP
    #: Walls stop the server's player where the page's wall-free prediction
    #: walks on; a gap beyond this keeps the server's word.
    ADOPT_BOUND = 0.25
    SCENE = {"floor": True, "glass": False, "arcade": False}
    HOSTILE_LANES = ((S0, T0),)

    def __init__(self, board_id: str = "", aspect: float | None = None, seed: int | None = None):
        self.seed = int(time.time_ns() % 1000) if seed is None else int(seed)
        self.map = make_map(self.seed)
        self._tiles = None
        super().__init__(board_id)

    def make_piece(self):
        return TileQuestPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.map)

    def motion(self) -> dict[str, Any]:
        return {"speed": PLAYER_SPEED, "tau": PLAYER_TAU, "reach": PLAYER_REACH,
                "x": [0.35, WIDTH - 0.35], "y": [0.35, HEIGHT - 0.35]}

    def reset(self) -> None:
        self.seed = (self.seed + 1) % 1000          # a new map each run
        self.map = make_map(self.seed)
        self._tiles = None
        super().reset()

    def palette(self) -> dict[str, list[float]]:
        return {"floor": [0.32, 0.56, 0.30], "wall": [0.58, 0.58, 0.64], "water": [0.20, 0.45, 0.85],
                "tree": [0.10, 0.42, 0.16], "path": [0.74, 0.64, 0.44], "home": [0.92, 0.56, 0.36],
                "work": [0.82, 0.76, 0.34], "lair": [0.58, 0.22, 0.44], "door": [0.56, 0.36, 0.20],
                "wizard": [0.52, 0.42, 0.96], "villager": [0.96, 0.82, 0.62], "slime": [0.30, 0.92, 0.42],
                "wolf": [0.66, 0.66, 0.72], "goblin": [0.46, 0.78, 0.26], "tower": [0.88, 0.88, 0.94],
                "bolt": [1.0, 0.92, 0.25]}

    def tiles(self) -> list[dict[str, Any]]:
        """The map's solid and decorated cells as flat entities (ids past the law's lanes)."""
        if self._tiles is None:
            out = []
            for cy in range(HEIGHT):
                for cx in range(WIDTH):
                    look = TILE_LOOK.get(int(self.map.grid[cy, cx]))
                    if look is None:
                        continue
                    out.append({"id": f"{10000 + cy * WIDTH + cx}:0", "x": cx + 0.5, "y": cy + 0.5, "vx": 0.0, "vy": 0.0,
                                "mesh": look[0], "color": look[1], "size": 0.5, "roll": 0.0, "lean": 0.0,
                                "owner": -1, "age": 99.0, "spin": 0.0, "bob": 0.0})
            self._tiles = out
        return self._tiles

    def entities(self) -> list[dict[str, Any]]:
        names = ("kind", "alive", "x", "y", "vx", "vy", "born", "hurt")
        col = {name: self.session.column(name)[:ENT] for name in names}
        clock = float(self.session.column("clock")[0])
        out = list(self.tiles())
        for k in np.flatnonzero(col["alive"] > 0.0):
            mesh, color, size = KIND_LOOK[int(col["kind"][k])]
            moving = abs(col["vx"][k]) + abs(col["vy"][k]) > 0.1
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": float(col["vx"][k]), "vy": float(col["vy"][k]),
                        "mesh": mesh, "color": color, "size": size, "roll": 0.0,
                        "lean": 0.25 if k == 0 else 0.45 if int(col["kind"][k]) != ARROW else 0.0, "owner": -1,
                        "age": clock - float(col["born"][k]), "spin": 0.0,
                        "bob": 1.0 if moving and int(col["kind"][k]) != ARROW else 0.0})
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]),
                "detail": f"{int(s.column('kills')[0])} monsters, {int(s.column('trips')[0])} commutes"}

    def describe(self) -> str:
        s = self.session
        seconds, calls = s.last_round
        alive = s.column("alive")[:ENT]
        head = "GAME OVER · R for a new map" if s.column("lives")[0] <= 0 else f"hp {s.column('lives')[0]:.1f}"
        return (f"{head} · kills {int(s.column('kills')[0])} · commutes {int(s.column('trips')[0])} · "
                f"villagers {int(alive[V0:S0].sum())}/{N_VILLAGERS} · monsters {int(alive[S0:T0].sum())} · "
                f"tick {seconds * 1000:.0f} ms = {calls} law steps")
