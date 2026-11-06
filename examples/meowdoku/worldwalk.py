"""World Walk: a walking treadmill over a small earth, one dt-system entity law.

An original game of entities with motives on a small planet.  The land is the
bullet-hell globe's (``entity_motion.terrain_height``), its towns and roads are
``roads.plan``'s; the screen is the same patch about the player (a treadmill:
the wizard stays at the centre of the screen and the curved earth turns
beneath) and the page draws the land, roads and lights from the same scene
contract.  The new thing is who lives on it: KINDS of walker that differ only
in their kind row (``walkers.Kind``) and its small choice network, all run
batched in the one law --

    settlers   commute between towns along the roads (traffic): they follow
               the cheap road edges, rest, and flee raiders to the nearest town
    builders   walk the same roads and found a house-light at a road point
               every so often, one at a time and not on top of another
               (city growth); raiders burn them down
    raiders    leave their lairs for the nearest town, hunt what they see,
               flee when hurt and mend at the lair (RPG / rogue)
    beasts     wander near their lair, hunt the wizard and the folk they spot
    towers     one at each town (the roads' sites), shoot the nearest raider or
               beast with an arrow of their own (tower defence)
    wizard     the player: turn, throttle, swing at what is close

Cells are the Fibonacci nodes ``roads.py`` builds the roads over.  The flow
fields (``walkers.SphereField``) are Dijkstra on that node graph with the
roads' slope cost, water and cliffs barred, road edges cheap; they live in
dt-system columns, one lane per node (the node's position is that lane's
``ux, uy, uz``), and a walker finds its node by walking the node graph from
the one it stood on (``walkers.climb``) and steers to the field's next node.

Lanes (contiguous blocks):
    0            the wizard
    1..14        settlers            15..18   builders
    19..26       raiders             27..34   beasts
    35..46       towers              47..58   arrows (arrow ``i`` is tower ``i``'s)
    59..106      lights (builder ``b`` owns 12: founded one at a time)
    107..        the map: one lane per node (``tile``, ``comp``, ``nb0..7``, the
                 five fields' ``fnx/fny/fnz/fdd`` columns hold the map)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from meowdoku import roads as road_plan
from meowdoku import scroller as sc
from meowdoku import sphere_motion as sm
from meowdoku import walkers
from meowdoku.entities import EntityGame, LawPiece, as_float, noise
from meowdoku.entity_motion import bank_turn, damped_axis, ground_travel
from meowdoku.walkers import ATTACK, FLEE, FOLLOW, REST, SEEK, WANDER, Kind
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

# ---------------------------------------------------------------- the world
RADIUS, PLAYER_Y, PATCH_R = sc.RADIUS, sc.PLAYER_Y, sc.PATCH_R
NODES, NEIGHBOURS = road_plan.NODES, road_plan.NEIGHBOURS
WORLD_SEED = 7                                  # the bullet-hell globe's roads
LAND, WATER, ROAD = 0.0, 1.0, 2.0               # a node's ``tile``
ROAD_DISCOUNT = 0.3                             # a road edge costs this much of its walk
#: Flow fields, in column order.
F_TOWN_A, F_TOWN_B, F_TOWN, F_LAIR, F_AVOID = range(5)
NF = 5

PLAYER, SETTLER, BUILDER, RAIDER, BEAST, TOWER, ARROW, LIGHT, MAP = range(9)
N_SETTLERS, N_BUILDERS, N_RAIDERS, N_BEASTS, N_TOWERS, LIGHTS_PER = 14, 4, 8, 8, 12, 12
N_LIGHTS = N_BUILDERS * LIGHTS_PER
V0 = 1
B0 = V0 + N_SETTLERS
R0 = B0 + N_BUILDERS
M0 = R0 + N_RAIDERS
T0 = M0 + N_BEASTS
A0 = T0 + N_TOWERS
L0 = A0 + N_TOWERS
ENT = L0 + N_LIGHTS
MAP0 = ENT
LANES = ENT + NODES

SPD_MIN, SPD_MAX, SPD0 = 0.0, 4.0, 1.6          # the wizard's throttle (treadmill speed, units a second)
PLAYER_HP = 6.0
WALK_TAU = 0.12
MELEE_REACH, MELEE_DAMAGE = 1.5, 1.5
ARROW_SPEED, ARROW_DAMAGE, ARROW_LIFE = 9.0, 1.0, 1.2
SEPARATION = 0.8
DWELL = 2.0
BUILD_EVERY, BUILD_GAP = 9.0, 1.1               # a builder founds a light this often, this far from any other
STEP_CAP = sc.STEP_CAP

# ----------------------------------------------------------- the kind rows
# Hidden units are gates (tanh(w.f + b) + 1) / 2 in [0, 1]; logits are b + w.gates.
KINDS: dict[int, Kind] = {
    PLAYER: Kind("wizard", PLAYER, team=0, speed=SPD_MAX, sense=6.0, hp=PLAYER_HP, radius=0.3, reload=0.35),
    SETTLER: Kind("settler", SETTLER, team=0, speed=1.6, sense=4.0, hp=3.0, radius=0.28, respawn=15.0,
                  needrate=0.04, goal=(F_TOWN_B, F_TOWN_A), refuge=F_TOWN, mends=0.3, net={
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
    # A builder is bolder than a settler (it only flees when a foe is close) and never rests
    # for want of breath: it works the roads.
    BUILDER: Kind("builder", BUILDER, team=0, speed=1.3, sense=4.0, hp=4.0, radius=0.3, respawn=20.0,
                  goal=(F_TOWN_B, F_TOWN_A), refuge=F_TOWN, mends=0.3, net={
        "hidden": {
            "fear": ({"foe_prox": 3.0, "foe_close": 4.0, "ally_prox": -1.5}, -2.0),
            "dwell": ({"busy": 4.0}, -2.0),
            "wounded": ({"health": -5.0}, 2.0),
        },
        "out": {
            "wander": ({"fear": -2.0, "dwell": -1.0}, 0.2),
            "follow": ({"fear": -3.0, "dwell": -3.0, "wounded": -1.0}, 1.5),
            "flee": ({"fear": 4.0}, -1.0),
            "rest": ({"dwell": 3.0, "wounded": 1.5, "fear": -2.0}, 0.0),
        }}),
    RAIDER: Kind("raider", RAIDER, team=1, speed=2.0, sense=6.0, hp=3.0, damage=1.0, radius=0.3, reload=1.1,
                 respawn=12.0, needrate=0.02, goal=(F_TOWN, F_TOWN), refuge=F_LAIR, mends=0.4, net={
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
    BEAST: Kind("beast", BEAST, team=1, speed=2.6, sense=5.0, hp=3.0, damage=1.0, radius=0.32, reload=0.9,
                respawn=10.0, needrate=0.02, goal=(F_LAIR, F_LAIR), refuge=F_LAIR, mends=0.5, net={
        "hidden": {
            "spot": ({"foe_prox": 5.0}, -1.0),
            "near": ({"foe_close": 6.0}, -1.0),
            "hurt": ({"health": -6.0}, 2.5),
            "home": ({"refuge_near": 5.0}, -3.0),
        },
        "out": {
            "wander": ({"spot": -1.5, "hurt": -3.0}, 1.0),
            "seek": ({"spot": 3.0, "hurt": -3.0}, -1.0),
            "attack": ({"near": 4.5, "spot": 0.5, "hurt": -3.0}, -1.5),
            "flee": ({"hurt": 5.0, "home": -2.5}, -2.0),
            "rest": ({"hurt": 2.0, "home": 3.0}, -2.5),
        }}),
    TOWER: Kind("tower", TOWER, team=0, speed=0.0, sense=7.0, hp=12.0, radius=0.4, reload=0.8, respawn=25.0, net={
        "hidden": {"spot": ({"foe_prox": 20.0}, 0.0)},
        "out": {"attack": ({"spot": 3.0}, -1.0), "rest": ({}, 1.0)},
    }),
    ARROW: Kind("arrow", ARROW, team=-1, hp=1.0, radius=0.12),
    LIGHT: Kind("light", LIGHT, team=0, hp=2.0, radius=0.35, mends=0.06),
}

# ------------------------------------------------------------- the columns
#: The wizard's controls are the bullet-hell ship's (turn, throttle, bank), so the page flies
#: it with the same bound axes and predicts it on the same compiled laws.
CONTROLS = {name: sc.CONTROLS[name] for name in ("move_axis", "move_axis_y", "steer_x", "steer_y", "fire", "move_axis_r")}
KIND_COLUMNS = ("team", "speed", "sense", "hpmax", "dmg", "rad", "reload_t", "respawn_t", "needrate",
                "goalf0", "goalf1", "refugef", "mends")
NODE_COLUMNS = ("tile", "comp", *(f"nb{k}" for k in range(NEIGHBOURS)))
FIELD_COLUMNS = tuple(f"{n}{i}" for n in ("fnx", "fny", "fnz", "fdd") for i in range(NF))
STATIC = ("lane", "kind", *KIND_COLUMNS, "owner", "home_node", *NODE_COLUMNS, *FIELD_COLUMNS)
STATE = ("alive", "x", "y", "vx", "vy", "ux", "uy", "uz", "hx", "hy", "hz", "spd", "yaw", "vyaw", "vspd",
         "mdx", "mdy", "mdz", "gspd", "bank", "vbank", "hp", "hurt", "cool", "timer", "need", "phase", "mot", "think", "born", "node",
         "map_u", "map_v", "map_gx", "map_gy",
         "score", "kills", "trips", "losses", "built", "lives", "clock")
SETTINGS = ("field_w",)
ARGUMENTS = (*STATIC, *STATE, *SETTINGS, *CONTROLS, "dt")


def worldwalk_step(*columns):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span (``dt`` a scalar)."""
    c = dict(zip(ARGUMENTS, columns))
    f = as_float
    n = ENT
    lanes_total = int(c["tile"].shape[0])
    dt = (c["x"] * 0.0 + c["dt"])[0:n]

    def e(name):                     # the entity lanes of a column
        return c[name][0:n]

    lane, kind, team, speed, sense = e("lane"), e("kind"), e("team"), e("speed"), e("sense")
    hpmax, dmg, rad, reload_t, respawn_t = e("hpmax"), e("dmg"), e("rad"), e("reload_t"), e("respawn_t")
    needrate, mends, owner, home_node = e("needrate"), e("mends"), e("owner"), e("home_node")
    alive, hp, hurt, cool, timer, need, phase = e("alive"), e("hp"), e("hurt"), e("cool"), e("timer"), e("need"), e("phase")
    mot, think, born, node = e("mot"), e("think"), e("born"), e("node")
    spd, yaw, vyaw, vspd, bank, vbank = e("spd"), e("yaw"), e("vyaw"), e("vspd"), e("bank"), e("vbank")
    clock, lives = e("clock"), e("lives")
    u = (e("ux"), e("uy"), e("uz"))
    h = (e("hx"), e("hy"), e("hz"))
    zero = alive * 0.0
    zero3 = (zero, zero, zero)

    is_player, is_settler, is_builder = kind == PLAYER, kind == SETTLER, kind == BUILDER
    is_raider, is_beast, is_tower = kind == RAIDER, kind == BEAST, kind == TOWER
    is_arrow, is_light = kind == ARROW, kind == LIGHT
    is_monster = is_raider | is_beast
    is_walker = is_player | is_settler | is_builder | is_monster
    folk = is_player | is_settler | is_builder | is_tower | is_light
    live = alive > 0.0
    playing = lives > 0.0
    clock_n = clock + dt
    P = f(is_player)

    # --------------------------------------------- the wizard: turn, throttle
    # The bullet-hell ship's own axes on the shared damped law: rudder (a held finger steers
    # for its bearing from the wizard), roll banks the walk into a turn, up/down the throttle.
    cx, cy = e("field_w") * 0.5, zero + PLAYER_Y
    steer_x, steer_y = e("steer_x"), e("steer_y")
    finger = (steer_x - cx).atan2(cy - steer_y)
    yaw_target = where(steer_x >= 0.0, yaw + finger, zero - 1.0)
    rudder_yaw, ship_vyaw = damped_axis(yaw, vyaw, e("move_axis_r"), yaw_target, zero + sc.YAW_LO, zero + sc.YAW_HI,
                                        sc.RUDDER_RATE, sc.TURN_TAU, sc.TURN_REACH, dt)
    level_target = where(e("move_axis").abs() < 0.05, zero + sc.BANK_MAX, zero - 1.0)
    ship_bank, ship_vbank = damped_axis(bank, vbank, e("move_axis"), level_target, zero, zero + 2.0 * sc.BANK_MAX,
                                        sc.ROLL_RATE, sc.ROLL_TAU, sc.ROLL_REACH, dt)
    ship_yaw = bank_turn(rudder_yaw, ship_bank, zero + sc.BANK_MAX, zero + sc.BANK_GAIN, dt)[0]
    ship_spd, ship_vspd = damped_axis(spd, vspd, zero - e("move_axis_y"), zero - 1.0, zero + SPD_MIN, zero + SPD_MAX,
                                      sc.THROTTLE, sc.THROTTLE_TAU, sc.REACH, dt)
    me = is_player & live & playing
    ship_on = f(me).sum() > 0.0
    u0 = ((P * u[0]).sum(), (P * u[1]).sum(), (P * u[2]).sum())
    h0 = ((P * h[0]).sum(), (P * h[1]).sum(), (P * h[2]).sum())
    dyaw = f(ship_on) * (P * (ship_yaw - yaw)).sum()
    pspd = f(ship_on) * (P * ship_spd).sum()
    h0t = sm.turn(u0, h0, dyaw)

    # --------------------------------------------------------- the node map
    nux, nuy, nuz = c["ux"], c["uy"], c["uz"]          # the nodes' positions (on the map lanes)
    tile = c["tile"]
    fnx = walkers.stack_columns([c[f"fnx{i}"] for i in range(NF)])
    fny = walkers.stack_columns([c[f"fny{i}"] for i in range(NF)])
    fnz = walkers.stack_columns([c[f"fnz{i}"] for i in range(NF)])
    fdd = walkers.stack_columns([c[f"fdd{i}"] for i in range(NF)])

    def toward_next(field):
        """Unit tangent from each lane to its node's next node in ``field``, and the field's distance."""
        nxt = (walkers.field_at(fnx, field, node, lanes_total), walkers.field_at(fny, field, node, lanes_total),
               walkers.field_at(fnz, field, node, lanes_total))
        return sm.toward(u, nxt), walkers.field_at(fdd, field, node, lanes_total)

    goal_field = where(phase > 0.5, e("goalf1"), e("goalf0"))
    gdir, gdist = toward_next(goal_field)
    rdir, rdist = toward_next(e("refugef"))
    adir, clearance = toward_next(zero + float(F_AVOID))
    push = ((2.2 - clearance) * (1.0 / 2.2)).clip(0.0, 1.0)
    comp = walkers.gather(c["comp"], node)
    on_road = walkers.gather(tile, node) > 1.5

    # ------------------------------------------------- lane against lane
    ex, ey, ez, dd = walkers.sphere_pairwise(u[0], u[1], u[2], RADIUS)

    def row(v):
        return v.reshape((n, 1))

    def col(v):
        return v.reshape((1, n))

    other = row(lane) != col(lane)
    seeker = live & (is_walker | is_tower)
    target = live & (is_walker | is_tower | is_light)
    # A walker sees only what stands on its own walkable land (water cuts it off); a tower's
    # arrow flies over water.
    same_land = (row(comp) == col(comp)) & (row(comp) >= 0.0)
    sees = row(is_tower) | same_land
    foe_valid = row(seeker) & col(target) & (row(team) != col(team)) & sees
    foe_d, fux, fuy, fuz = walkers.nearest3(ex, ey, ez, dd, foe_valid)
    ally_valid = row(seeker) & col(target) & (row(team) == col(team)) & other
    ally_d, _ax, _ay, _az = walkers.nearest3(ex, ey, ez, dd, ally_valid)
    crowd_valid = row(live) & col(live) & (row(kind) == col(kind)) & other & (dd < 1.2)
    crowd = (f(crowd_valid).sum(dim=1) * (1.0 / 3.0)).clip(0.0, 1.0)
    mover_live = live & is_walker
    sep = walkers.separation3(ex, ey, ez, dd, row(mover_live) & col(mover_live) & other, SEPARATION)

    # ------------------------------------------------ the choice network
    feats = {
        "bias": zero + 1.0,
        "foe_prox": (1.0 - foe_d * (1.0 / (sense + 1e-6))).clip(0.0, 1.0),
        "foe_close": (1.0 - foe_d * (1.0 / 1.5)).clip(0.0, 1.0),
        "ally_prox": (1.0 - ally_d * (1.0 / 3.0)).clip(0.0, 1.0),
        "goal_far": (gdist * (1.0 / 40.0)).clip(0.0, 1.0),
        "health": (hp * (1.0 / (hpmax + 1e-6))).clip(0.0, 1.0),
        "need": need.clip(0.0, 1.0),
        "crowd": crowd,
        "phase": phase,
        "busy": timer.clip(0.0, 1.0),
        "refuge_near": (1.0 - rdist * (1.0 / 6.0)).clip(0.0, 1.0),
    }
    feats["foe_ahead"] = sm.dot(h, (fux, fuy, fuz)) * feats["foe_prox"]
    choice = walkers.choose_motivation(KINDS, kind, feats, zero)
    rethink = (think - dt) <= 0.0
    mot_n = where(rethink, choice, mot)
    think_n = where(rethink, 0.25 + 0.15 * noise(lane * 3.7 + clock_n * 9.1), think - dt)

    def is_m(k):
        return mot_n == float(k)

    # ------------------------------------------------- motivation -> heading
    foe_u = (fux, fuy, fuz)
    wander_on = is_m(WANDER)
    turn = where(wander_on, (noise(lane * 2.3 + clock_n * 5.1) - 0.5) * 30.0 * dt, zero)
    rh = sm.turn(u, h, turn)
    wander_c = sm.tangent(u, sm.add(rh, sm.scale(adir, 1.5 * push)))
    seek_c = sm.tangent(u, sm.add(foe_u, sm.scale(adir, 1.5 * push)))
    flee_c = sm.tangent(u, sm.add(sm.add(sm.scale(rdir, 0.8), sm.scale(foe_u, -1.0)), sm.scale(adir, 1.5 * push)))
    follow_c = sm.tangent(u, sm.add(gdir, sm.scale(adir, 0.4 * push)))
    is_seek = is_m(SEEK) | is_m(ATTACK)
    cmd = sm.vwhere(wander_on, wander_c, sm.vwhere(is_seek, seek_c, sm.vwhere(is_m(FLEE), flee_c,
                    sm.vwhere(is_m(FOLLOW), follow_c, zero3))))
    pace = where(wander_on, walkers.PACE[WANDER], where(is_m(SEEK), walkers.PACE[SEEK], where(is_m(FLEE), walkers.PACE[FLEE],
            where(is_m(FOLLOW), walkers.PACE[FOLLOW], where(is_m(ATTACK), walkers.PACE[ATTACK], zero)))))
    moving_cmd = sm.dot(cmd, cmd) > 0.25
    k_head = dt / (0.15 + dt)
    blended = sm.tangent(u, sm.add(rh, sm.scale(sm.add(cmd, sm.scale(rh, -1.0)), k_head)))
    face = sm.vwhere(moving_cmd, blended, rh)

    # ---------------------------------------------------------- movement
    walker = live & is_walker & playing
    npc = walker & ~is_player
    flying = live & is_arrow & playing
    k_vel = dt / (WALK_TAU + dt)
    spd_w = spd + (speed * pace - spd) * k_vel
    vel = sm.add(sm.scale(face, spd_w), sm.scale(sep, 1.2))
    vlen = sm.dot(vel, vel).sqrt()
    walk_dir = sm.tangent(u, vel)
    move_dir = sm.vwhere(is_player, (h0t[0] + zero, h0t[1] + zero, h0t[2] + zero), sm.vwhere(is_arrow, h, walk_dir))
    move_dist = where(is_player & walker, pspd * dt, where(npc, vlen * dt, where(flying, spd * dt, zero)))
    u1, hm1 = sm.advance(u, move_dir, move_dist, RADIUS)
    node1 = walkers.climb(node, u1[0], u1[1], u1[2], nux, nuy, nuz, [c[f"nb{k}"] for k in range(NEIGHBOURS)])
    # Water bars the walkers (the wizard levitates): they stop at a shore and turn aside.
    stopped = npc & (walkers.gather(tile, node1) > 0.5) & (walkers.gather(tile, node1) < 1.5)
    u2 = sm.vwhere(stopped, u, u1)
    node2 = where(stopped, node, node1)
    aside = (f(noise(lane * 3.1) > 0.5) * 2.0 - 1.0) * 1.1
    face_turned = sm.vwhere(stopped, sm.turn(u, face, aside), face)
    h_next = sm.vwhere(is_player & walker, hm1, sm.vwhere(is_arrow, hm1, sm.tangent(u2, face_turned)))
    moves = npc | (is_player & walker) | flying
    h_next = sm.vwhere(moves, h_next, h)
    ground_speed = where(stopped, zero, move_dist / dt)

    # ---------------------------------------- arrows: the towers' shots
    shoots = is_tower & live & playing & is_m(ATTACK) & ((cool - dt) <= 0.0) & (foe_d < sense)
    free_back = walkers.shift_block(f(is_arrow & ~live), A0, N_TOWERS, T0) > 0.5
    fire_t = shoots & free_back
    fire_a = (walkers.shift_block(f(fire_t), T0, N_TOWERS, A0) > 0.5) & is_arrow
    src = tuple(walkers.shift_block(v, T0, N_TOWERS, A0) for v in u)
    aim = tuple(walkers.shift_block(v, T0, N_TOWERS, A0) for v in foe_u)
    arrow_u, arrow_h = sm.advance(src, aim, zero + 0.4, RADIUS)
    lost = flying & ((timer - dt) <= 0.0)

    # ----------------------------------------------------------- combat
    touch = dd < (row(rad) + col(rad) + 0.1)
    blow = (row(live) & col(live) & row(is_monster) & col(folk) & (row(cool) <= 0.0) & (col(hurt) <= 0.0) & touch
            & row(playing))
    struck = f(blow).sum(dim=1) > 0.0
    contact = (f(blow) * row(dmg)).sum(dim=0)
    to_wizard = sm.arc(u, u0, RADIUS)
    swing = is_player & live & playing & (e("fire") > 0.0) & (cool <= 0.0)
    swung = f(swing).sum() > 0.0
    melee = f(swung & live & is_monster & (to_wizard < MELEE_REACH)) * MELEE_DAMAGE
    ahit = f(row(live) & col(live) & row(is_arrow) & col(is_monster) & (dd < (row(rad) + col(rad))))
    arrow_dmg = ahit.sum(dim=0) * ARROW_DAMAGE
    arrow_spent = ahit.sum(dim=1) > 0.0

    resting = live & is_m(REST)
    grows = live & is_light
    hp_m = (hp + f(resting | grows) * mends * dt).clip(0.0, 1.0e6)
    hp_m = where(hp_m > hpmax, hpmax, hp_m)
    hp1 = hp_m - contact - melee - arrow_dmg
    dies = live & ~is_arrow & (hp1 <= 0.0)
    hurt_n = where(contact > 0.0, 0.7, (hurt - dt).clip(0.0, 9.0))

    # ---------------------------------------- builders found lights
    near_light = f(row(live) & col(live) & col(is_light) & (dd < BUILD_GAP)).sum(dim=1) > 0.0
    wants = is_builder & live & playing & (cool <= 0.0) & on_road & ~near_light & is_m(FOLLOW)
    free_light = is_light & ~live
    claim = row(wants) & col(free_light) & (col(owner) == row(lane))
    first = where(claim, col(lane) + zero.reshape((n, 1)), 1.0e9).min(dim=1)
    pick = f(claim & (col(lane) == first.reshape((n, 1))))
    founded = pick.sum(dim=1) > 0.0                      # per builder
    takes = pick.sum(dim=0) > 0.0                        # per light lane

    def from_builder(values):
        return (pick * row(values)).sum(dim=0)

    light_u = tuple(from_builder(v) for v in u)
    light_h = tuple(from_builder(v) for v in h)
    light_node = from_builder(node)
    cool_n = where(struck | swing | fire_t, reload_t, where(founded, BUILD_EVERY + 4.0 * noise(lane * 1.7 + clock_n), (cool - dt).clip(0.0, 99.0)))

    # ----------------------------------- need, arrival, dwelling, respawn
    need_n = where(live, where(is_m(REST), need - 0.35 * dt, need + needrate * dt).clip(0.0, 1.0), need)
    commuter = is_settler | is_builder
    arrived = commuter & live & playing & (gdist < 0.5) & (timer <= 0.0)
    phase_n = where(arrived, 1.0 - phase, phase)
    dwell = DWELL + 3.0 * noise(lane * 1.1 + clock_n * 3.3)
    spawn = (respawn_t > 0.0) & ~live & ~is_player & ~is_arrow & playing & ((timer - dt) <= 0.0)
    timer_n = where(live, (timer - dt).clip(0.0, 1.0e6), timer - dt)
    timer_n = where(arrived, dwell, timer_n)
    timer_n = where(dies, respawn_t * (0.7 + 0.6 * noise(lane * 5.9 + clock_n * 2.3)), timer_n)
    timer_n = where(fire_a, ARROW_LIFE, timer_n)

    # ----------------------------------------------- the next state
    alive_walker = (live & ~dies) | spawn | takes
    alive_arrow = (live & ~(lost | arrow_spent)) | fire_a
    live_n = where(is_arrow, alive_arrow, alive_walker)
    home_u = (walkers.gather(nux, home_node), walkers.gather(nuy, home_node), walkers.gather(nuz, home_node))
    u_n = sm.vwhere(takes, light_u, sm.vwhere(spawn, home_u, sm.vwhere(fire_a, arrow_u, u2)))
    h_n = sm.vwhere(takes, light_h, sm.vwhere(fire_a, arrow_h, h_next))
    h_n = sm.tangent(u_n, sm.vwhere(spawn, h, h_n))
    node_n = where(takes, light_node, where(spawn, home_node, node2))
    spd_n = where(is_player, pspd + zero, where(npc, spd_w, where(fire_a, ARROW_SPEED + zero, where(spawn | takes, zero, spd))))
    hp_n = where(takes, hpmax * 0.5, where(spawn, hpmax, where(is_arrow, hp, hp1)))
    hurt_f = where(spawn, 1.0, hurt_n)
    cool_f = where(spawn, 0.5, where(takes, 0.0, cool_n))
    need_f = where(spawn, zero, need_n)
    phase_f = where(spawn, zero, phase_n)
    mot_f = where(spawn, zero, mot_n)
    think_f = where(spawn, zero, think_n)
    born_f = where(spawn | fire_a | takes, clock_n, born)

    # --------------------------------------------- what the page sees
    # Every lane in the patch about the wizard's new frame (sphere_motion.project): the
    # wizard sits at its centre; the earth turns beneath.
    u0n = ((P * u_n[0]).sum(), (P * u_n[1]).sum(), (P * u_n[2]).sum())
    h0n = ((P * h_n[0]).sum(), (P * h_n[1]).sum(), (P * h_n[2]).sum())
    Xn, Yn = sm.project(u_n, u0n, h0n, RADIUS)
    x_n = where(is_player, cx, cx + Xn)
    y_n = where(is_player, cy, cy - Yn)
    mdir = sm.tangent(u_n, sm.vwhere(moves, hm1, h_n))
    pvx, pvy = sm.patch_velocity(mdir, ground_speed, x_n - cx, cy - y_n, h0n, pspd, u0n, dyaw / dt)
    vx_n = where(is_player, zero, pvx)
    vy_n = where(is_player, zero, -pvy)

    kills_d = f(dies & is_monster).sum()
    trips_d = f(arrived & (phase < 0.5)).sum()
    losses_d = f(dies & commuter).sum()
    built_d = f(founded).sum()
    hero_hp = (P * hp1).sum()

    # The longest step that cannot carry anything past what it can hit: an arrow crosses 0.3.
    reach = where(live, speed * 1.3, zero) + where(live & is_arrow, ARROW_SPEED + zero, zero) + pspd
    dt_limit = (0.3 / (reach.max() + 1e-6)).clip(0.0, STEP_CAP)

    # The land under the wizard (the page runs the same law).
    gu, gv, ggx, ggy = ground_travel(e("map_u"), e("map_v"), e("map_gx"), e("map_gy"), zero + dyaw, zero + pspd, dt)
    land_on = is_player & playing
    map_u_n, map_v_n = where(land_on, gu, e("map_u")), where(land_on, gv, e("map_v"))
    map_gx_n, map_gy_n = where(land_on, ggx, e("map_gx")), where(land_on, ggy, e("map_gy"))

    def wide(name, head):            # entity lanes' new values, then the map lanes (unchanged)
        return walkers.widen(head, c[name][n:])

    def whole(name, delta):          # a game-wide column: the same value on every lane
        return c[name] + delta

    full_x = c["x"]
    return (
        wide("alive", f(live_n)), wide("x", x_n), wide("y", y_n), wide("vx", vx_n), wide("vy", vy_n),
        wide("ux", u_n[0]), wide("uy", u_n[1]), wide("uz", u_n[2]), wide("hx", h_n[0]), wide("hy", h_n[1]), wide("hz", h_n[2]),
        wide("spd", spd_n), wide("yaw", where(is_player, ship_yaw, yaw)), wide("vyaw", where(is_player, ship_vyaw, vyaw)),
        wide("vspd", where(is_player, ship_vspd, vspd)),
        # the direction each walker is walking and its ground speed: the state the page carries it on from
        wide("mdx", mdir[0]), wide("mdy", mdir[1]), wide("mdz", mdir[2]), wide("gspd", ground_speed),
        wide("bank", where(is_player, ship_bank, bank)),
        wide("vbank", where(is_player, ship_vbank, vbank)),
        wide("hp", hp_n), wide("hurt", hurt_f), wide("cool", cool_f), wide("timer", timer_n),
        wide("need", need_f), wide("phase", phase_f), wide("mot", mot_f), wide("think", think_f),
        wide("born", born_f), wide("node", node_n),
        wide("map_u", map_u_n), wide("map_v", map_v_n), wide("map_gx", map_gx_n), wide("map_gy", map_gy_n),
        whole("score", kills_d * 10.0 + trips_d + built_d * 3.0 - losses_d * 5.0), whole("kills", kills_d),
        whole("trips", trips_d), whole("losses", losses_d), whole("built", built_d),
        (c["lives"] * 0.0 + hero_hp), c["clock"] + c["dt"],
        f(live_n).sum() + full_x * 0.0,
        dt_limit + full_x * 0.0,
    )


class WorldWalkPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "worldwalk_step"
    law = staticmethod(worldwalk_step)
    argument_names = ARGUMENTS
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")
    source_modules = ("meowdoku.walkers", "meowdoku.sphere_motion", "meowdoku.scroller", "meowdoku.roads")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


# ------------------------------------------------------------ world building

class World:
    """The globe's node graph and everything precomputed on it (cached: it is one world)."""

    def __init__(self):
        nodes, height, dry, nearest = road_plan.world()

        def cost(a, b):
            return road_plan.edge_cost(nodes, height, dry, RADIUS, a, b)

        self.graph = walkers.SphereGraph.build(nodes, nearest, dry, cost)
        self.nodes, self.dry = nodes, dry
        sites, lines, _points = road_plan.plan(WORLD_SEED, RADIUS, sites=N_TOWERS, road_points=160)
        self.sites, self.lines = sites, lines
        self.site_nodes = [self.node_of(p) for p in sites]
        # roads: the polylines' vertices are nodes, consecutive ones are edges
        road_nodes, self.road_edges = set(), set()
        for line in lines:
            ids = [self.node_of(p) for p in line]
            road_nodes.update(ids)
            self.road_edges.update(zip(ids[:-1], ids[1:]))
        self.road_nodes = road_nodes
        self.comp = walkers.sphere_components(self.graph)
        labels, counts = np.unique(self.comp[self.comp >= 0], return_counts=True)
        self.main = int(labels[np.argmax(counts)])
        self.towns = [i for i, node in enumerate(self.site_nodes) if self.comp[node] == self.main]
        self.towns_a, self.towns_b = self.towns[0::2], self.towns[1::2]
        self.lairs = self._lairs(4)
        fields = [
            self._field([self.site_nodes[i] for i in self.towns_a]),
            self._field([self.site_nodes[i] for i in self.towns_b]),
            self._field([self.site_nodes[i] for i in self.towns]),
            self._field(self.lairs),
            walkers.SphereField.clearance(self.graph, RADIUS),
        ]
        self.fields = fields
        self.tile = np.where(dry, LAND, WATER)
        self.tile[sorted(road_nodes)] = np.where(dry[sorted(road_nodes)], ROAD, WATER)

    def node_of(self, p) -> int:
        return int(np.argmax(self.nodes @ np.asarray(p)))

    def _field(self, goals):
        return walkers.SphereField.build(self.graph, goals, discount=self.road_edges, factor=ROAD_DISCOUNT)

    def _lairs(self, count: int) -> list[int]:
        """Lairs: dry nodes of the main land farthest from every town (farthest-point sampling)."""
        near = np.full(len(self.nodes), np.inf)
        for node in self.site_nodes:
            near = np.minimum(near, np.arccos(np.clip(self.nodes @ self.nodes[node], -1.0, 1.0)))
        usable = self.comp == self.main
        chosen = []
        for _ in range(count):
            pick = int(np.argmax(np.where(usable, near, -1.0)))
            chosen.append(pick)
            near = np.minimum(near, np.arccos(np.clip(self.nodes @ self.nodes[pick], -1.0, 1.0)))
        return chosen


_WORLD: World | None = None


def world() -> World:
    global _WORLD
    if _WORLD is None:
        _WORLD = World()
    return _WORLD


def initial_columns(width: float = sc.WIDTH, seed: int = 0) -> dict[str, np.ndarray]:
    w = world()
    rng = np.random.default_rng(int(seed) + 7919)
    kind = np.full(LANES, float(MAP))
    for lanes, kid in (((0, 1), PLAYER), ((V0, B0), SETTLER), ((B0, R0), BUILDER), ((R0, M0), RAIDER), ((M0, T0), BEAST),
                       ((T0, A0), TOWER), ((A0, L0), ARROW), ((L0, ENT), LIGHT)):
        kind[lanes[0]:lanes[1]] = kid
    columns = {"lane": np.arange(LANES, dtype=np.float64), "kind": kind}
    columns.update(walkers.kind_columns(KINDS, kind))
    # where each lane lives: commuters at the A towns, raiders and beasts at the lairs, a tower at each site
    home = np.zeros(LANES)
    owner = np.full(LANES, -1.0)
    a_nodes = [w.site_nodes[i] for i in w.towns_a]
    for lane in range(V0, R0):
        home[lane] = MAP0 + a_nodes[(lane - V0) % len(a_nodes)]
    for lane in range(R0, T0):
        home[lane] = MAP0 + w.lairs[(lane - R0) % len(w.lairs)]
    for i in range(N_TOWERS):
        home[T0 + i] = MAP0 + w.site_nodes[i]
        owner[A0 + i] = T0 + i
    for i in range(N_LIGHTS):
        owner[L0 + i] = B0 + i // LIGHTS_PER
    start = a_nodes[0]
    home[0] = MAP0 + start
    other = [w.site_nodes[i] for i in w.towns if w.site_nodes[i] != start]
    toward_town = w.nodes[other[0]] if other else np.array([0.0, 0.0, 1.0])
    ux, uy, uz = np.zeros(LANES), np.zeros(LANES), np.zeros(LANES)
    uz[:ENT] = 1.0                                   # (a body without a place waits at the pole)
    ux[MAP0:], uy[MAP0:], uz[MAP0:] = w.nodes[:, 0], w.nodes[:, 1], w.nodes[:, 2]
    node = np.zeros(LANES)
    for lane in range(ENT):
        if home[lane] > 0.0 or lane == 0:
            idx = int(home[lane]) - MAP0
            ux[lane], uy[lane], uz[lane] = w.nodes[idx]
            node[lane] = home[lane]
    hx, hy, hz = np.zeros(LANES), np.zeros(LANES), np.ones(LANES)
    for lane in range(ENT):
        p = np.array([ux[lane], uy[lane], uz[lane]])
        if lane == 0 or lane >= ENT:
            g = toward_town
        else:
            g = rng.normal(size=3)
        t = g - (g @ p) * p
        t /= np.linalg.norm(t) + 1e-12
        hx[lane], hy[lane], hz[lane] = t
    for lane in range(L0, ENT):                      # lights wait, dark, at their builders' places
        ux[lane], uy[lane], uz[lane] = 0.0, 0.0, 1.0
        hx[lane], hy[lane], hz[lane] = 0.0, 1.0, 0.0
    columns.update({"owner": owner, "home_node": home})
    tile = np.zeros(LANES)
    tile[MAP0:] = w.tile
    columns["tile"] = tile
    columns.update(walkers.sphere_node_columns(w.graph, w.comp, MAP0, LANES))
    columns.update(walkers.sphere_field_columns(w.fields, w.nodes, MAP0, LANES))
    alive = np.zeros(LANES)
    alive[0:A0] = 1.0
    timer = np.zeros(LANES)
    timer[V0:R0] = rng.uniform(0.0, 4.0, R0 - V0)
    cool = np.zeros(LANES)
    cool[B0:R0] = rng.uniform(2.0, BUILD_EVERY, N_BUILDERS)
    yaw = np.zeros(LANES)
    yaw[0] = sc.YAW0
    spd = np.zeros(LANES)
    spd[0] = SPD0
    columns.update({
        "alive": alive, "x": np.full(LANES, width / 2.0), "y": np.full(LANES, PLAYER_Y), "vx": np.zeros(LANES),
        "vy": np.zeros(LANES), "ux": ux, "uy": uy, "uz": uz, "hx": hx, "hy": hy, "hz": hz, "spd": spd,
        "yaw": yaw, "vyaw": np.zeros(LANES), "vspd": np.zeros(LANES), "bank": np.full(LANES, sc.BANK_MAX),
        "mdx": hx.copy(), "mdy": hy.copy(), "mdz": hz.copy(), "gspd": np.zeros(LANES),
        "vbank": np.zeros(LANES), "hp": columns["hpmax"].copy(), "hurt": np.zeros(LANES), "cool": cool,
        "timer": timer, "need": np.zeros(LANES), "phase": np.zeros(LANES), "mot": np.zeros(LANES),
        "think": rng.uniform(0.0, 0.25, LANES), "born": np.zeros(LANES), "node": node,
        "map_u": np.zeros(LANES), "map_v": np.zeros(LANES), "map_gx": np.zeros(LANES), "map_gy": np.full(LANES, -1.0),
        "score": np.zeros(LANES), "kills": np.zeros(LANES), "trips": np.zeros(LANES), "losses": np.zeros(LANES),
        "built": np.zeros(LANES), "lives": np.full(LANES, PLAYER_HP), "clock": np.zeros(LANES),
        "field_w": np.full(LANES, float(width)),
        **{name: np.zeros(LANES) for name in CONTROLS}, "steer_x": np.full(LANES, -1.0), "steer_y": np.full(LANES, -1.0),
    })
    return columns


# ------------------------------------------------------------------ the game

#: Mesh vocabulary this game adds to ``web_export``'s shared tables: name -> emoji.
EMOJI = {"builder": "\U0001F477"}
EMOJI_SUBDIVISIONS: dict[str, int] = {}

#: kind -> (mesh, colour, size)
KIND_LOOK = {PLAYER: ("wizard", "wizard", 0.42), SETTLER: ("villager", "villager", 0.36),
             BUILDER: ("builder", "builder", 0.38), RAIDER: ("goblin", "goblin", 0.4),
             BEAST: ("wolf", "wolf", 0.42), TOWER: ("tower", "tower", 0.5), ARROW: ("bolt", "bolt", 0.2)}


class WorldWalk(EntityGame):
    game = "worldwalk"
    title = "World Walk (small earth)"
    height = sc.HEIGHT
    CONTROLS = tuple(CONTROLS)
    PLAYER_LANE = 0
    HEALTH_MAX = PLAYER_HP
    HOSTILE_LANES = ((R0, T0),)
    SCENE = {"floor": False, "glass": True, "arcade": True, "terrain": True}
    SCENE_PREFER = {"glass": True, "arcade": True}
    SUN = sc.SUN_DIR

    def __init__(self, board_id: str = "", aspect: float | None = None, seed: int = 0):
        self.width = sc.field_width(aspect)
        self.seed = int(seed)
        super().__init__(board_id)

    def make_piece(self):
        return WorldWalkPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.width, self.seed)

    def motion(self) -> dict[str, Any]:
        """The wizard's turn, throttle and bank as the law runs them (the bullet-hell ship's
        bound axes), the patch the page draws in, and the globe's land."""
        return {"speed": sc.SPEED, "tau": sc.TAU, "reach": sc.REACH,
                "yaw": [sc.YAW_LO, sc.YAW_HI], "spd": [SPD_MIN, SPD_MAX], "bank": [0.0, 2.0 * sc.BANK_MAX],
                "bank_turn": {"level": sc.BANK_MAX, "gain": sc.BANK_GAIN},
                "bind": {"yaw": {"move": "move_axis_r", "finger": True, "speed": sc.RUDDER_RATE, "tau": sc.TURN_TAU,
                                 "reach": sc.TURN_REACH},
                         "bank": {"move": "move_axis", "level": sc.BANK_MAX, "speed": sc.ROLL_RATE, "tau": sc.ROLL_TAU,
                                  "reach": sc.ROLL_REACH},
                         "spd": {"move": "move_axis_y", "sign": -1.0, "speed": sc.THROTTLE, "tau": sc.THROTTLE_TAU}},
                "patch": {"cx": self.width / 2.0, "cy": PLAYER_Y, "radius": RADIUS},
                "travel": ["map_u", "map_v", "map_gx", "map_gy", "ux", "uy", "uz", "hx", "hy", "hz"],
                "sphere": {"radius": RADIUS, "air": [0.0, 0.0], "relief": 0.75, "sea": 0.48},
                # a walker, an arrow, a light is carried on between snapshots along its walking direction at its
                # ground speed (entity_motion.sphere_advance), as the step moved it; the page runs the same core
                "bodies": {"step": sc.STEP_CAP, "flight": {"body": sc.FLY_BODY, "craft": sc.FLY_CRAFT, "missile": sc.FLY_MISSILE,
                                                           "bomb": sc.FLY_BOMB, "canopy": sc.FLY_CANOPY},
                           "ballistic": {"mass": sc.MISSILE_MASS, "air_density": sc.AIR_DENSITY, "gravity": sc.GRAVITY,
                                         "missile": [sc.MISSILE_DIAMETER, sc.MISSILE_DRAG],
                                         "bomb": [sc.BOMB_DIAMETER, sc.BOMB_DRAG], "canopy": [sc.CHUTE_DIAMETER, sc.CHUTE_DRAG]}}}

    def layout(self, aspect: float) -> None:
        width = sc.field_width(aspect)
        if width == self.width:
            return
        self.session.column("field_w")[...] = width
        self.epoch += 1
        self.width = width

    @property
    def emissive(self) -> dict[str, list[float]]:
        return {"city": [3.0, 2.4, 1.4]}

    def palette(self) -> dict[str, list[float]]:
        return {"wizard": [0.52, 0.42, 0.96], "villager": [0.96, 0.82, 0.62], "builder": [0.95, 0.62, 0.2],
                "goblin": [0.46, 0.78, 0.26], "wolf": [0.66, 0.66, 0.72], "tower": [0.88, 0.88, 0.94],
                "bolt": [1.0, 0.92, 0.25], "city": [1.0, 0.86, 0.55]}

    def entities(self) -> list[dict[str, Any]]:
        names = ("kind", "alive", "x", "y", "vx", "vy", "born", "hurt", "hp", "hpmax", "ux", "uy", "uz", "hx", "hy", "hz",
                 "mdx", "mdy", "mdz", "gspd")
        col = {name: self.session.column(name)[:ENT] for name in names}
        clock = float(self.session.column("clock")[0])
        u = np.stack([col["ux"], col["uy"], col["uz"]], axis=1)
        u0 = u[0]
        reach = RADIUS * np.arctan2(np.linalg.norm(np.cross(u, u0), axis=1), u @ u0)
        show = ((col["alive"] > 0.0) & (reach < PATCH_R) & (np.abs(col["x"] - self.width / 2.0) < self.width / 2.0 + 0.6))
        out = []
        for k in np.flatnonzero(show):
            kid = int(col["kind"][k])
            if kid == LIGHT:
                glow = float(col["hp"][k] / max(col["hpmax"][k], 1e-9))
                mesh, color, size = "disc", "city", 0.06 + 0.06 * glow
            else:
                mesh, color, size = KIND_LOOK[kid]
            moving = abs(col["vx"][k]) + abs(col["vy"][k]) > 0.1
            body = {"u": [float(col["ux"][k]), float(col["uy"][k]), float(col["uz"][k])],
                    "h": [float(col["mdx"][k]), float(col["mdy"][k]), float(col["mdz"][k])],
                    "spd": float(col["gspd"][k]), "turn": 0.0, "flight": sc.FLY_BODY}
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "body": body, "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": float(col["vx"][k]), "vy": float(col["vy"][k]), "z": 0.0, "vz": 0.0,
                        "mesh": mesh, "color": color, "size": size, "roll": 0.0,
                        "lean": 0.0 if kid in (ARROW, LIGHT) else 0.45, "face": False, "translucent": False,
                        "ribbon": False, "ground": True, "glow": kid == LIGHT, "spin": 0.0, "owner": -1,
                        "age": clock - float(col["born"][k]),
                        "bob": 1.0 if moving and kid not in (ARROW, LIGHT) else 0.0})
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]),
                "detail": f"{int(s.column('kills')[0])} raiders, {int(s.column('trips')[0])} commutes, "
                          f"{int(s.column('built')[0])} lights"}

    def state(self) -> dict[str, Any]:
        st = super().state()
        if self.session.state is not None:
            c = self.session.column
            if not hasattr(self, "_roads_published"):
                _s, lines, _p = road_plan.plan(WORLD_SEED, RADIUS, sites=N_TOWERS, road_points=160)
                self._roads_published = [[[round(float(v), 5) for v in pt] for pt in line] for line in lines]
            st["roads"] = self._roads_published
            st["globe"] = {"u": [float(c("ux")[0]), float(c("uy")[0]), float(c("uz")[0])],
                           "h": [float(c("hx")[0]), float(c("hy")[0]), float(c("hz")[0])],
                           "yaw": float(c("yaw")[0]), "sun": list(self.SUN), "radius": RADIUS}
        return st

    def describe(self) -> str:
        s = self.session
        seconds, calls = s.last_round
        alive = s.column("alive")[:ENT]
        head = "GAME OVER · R to restart" if s.column("lives")[0] <= 0 else f"hp {s.column('lives')[0]:.1f}"
        return (f"{head} · kills {int(s.column('kills')[0])} · commutes {int(s.column('trips')[0])} · "
                f"lights {int(alive[L0:ENT].sum())} · settlers {int(alive[V0:B0].sum())}/{N_SETTLERS} · "
                f"raiders+beasts {int(alive[R0:T0].sum())} · tick {seconds * 1000:.0f} ms = {calls} law steps")
