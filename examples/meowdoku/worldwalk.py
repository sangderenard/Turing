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
    wizard     the player, a WALKER (no ship, no throttle): click and it walks the
               cheapest path there; or WASD/arrows/thumbpads walk it directly and
               it STOPS when released; it swings at what is close
    houses, walls, towers, markets   built by the player on nodes beside a road
               (``Ground.buildable``): towers shoot, houses are settlers' homes and
               shelter the folk about them, markets shelter further, walls bar the node

WALKING IS ONE LAW.  A walker's speed is ``entity_motion.walk_speed``: its base speed times the
node's SURFACE MATERIAL factor over (1 + 6 * slope) -- grass 1, road 1.6, sand 0.8, rock 0.55, water
0.05 (a swim: "all but blocked", never blocked), a wall 0.  The player, every settler, builder, raider and
beast walk at it; the flow fields and the click path cost an edge by the time it takes to cross at those
very speeds (the same function, run on arrays), so paths prefer roads and avoid water unless it is the
only way.  The page predicts the player with the same law (``walk_speed``, ``walk_drive``, ``walk_route``
are ``entity_motion.PREDICTORS`` cores, with ``sphere_turn`` and ``sphere_advance``).

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
    35..52       towers (12 at the sites, 6 to build)   53..70   arrows (arrow ``i`` is tower ``i``'s)
    71..118      lights (builder ``b`` owns 12: founded one at a time)
    119..142     build slots (a house, wall or market takes one; a slot is dead until built)
    143..        the map: one lane per node (``mat`` the surface material, ``slope``, ``comp``,
                 ``nb0..7``, the five fields' ``fnx/fny/fnz/fdd`` columns hold the map; the click path's
                 waypoints are ``rp_x/rp_y/rp_z`` on the first map lanes)
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
from meowdoku.entity_motion import WALK_MATERIALS, ground_travel, walk_drive, walk_route, walk_speed
from meowdoku.walkers import ATTACK, FLEE, FOLLOW, REST, SEEK, WANDER, Kind
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

# ---------------------------------------------------------------- the world
RADIUS, PLAYER_Y, PATCH_R = sc.RADIUS, sc.PLAYER_Y, sc.PATCH_R
NODES, NEIGHBOURS = road_plan.NODES, road_plan.NEIGHBOURS
WORLD_SEED = 7                                  # the bullet-hell globe's roads
#: A node's surface MATERIAL (``entity_motion.WALK_MATERIALS``: the factors on a walker's speed).
GRASS, ROAD, SAND, ROCK, WATER, WALL = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
LAND = GRASS
MATERIAL_NAMES = {GRASS: "grass", ROAD: "road", SAND: "sand", ROCK: "rock", WATER: "water", WALL: "wall"}
SAND_HEIGHT, ROCK_HEIGHT = 0.02, 0.21           # the land's law: below, a shore of sand; above, bare rock (and any steeper than roads climb)
ROAD_DISCOUNT = 0.3                             # (the old road discount; roads are a material now: kept for the field tests)
#: Flow fields, in column order.
F_TOWN_A, F_TOWN_B, F_TOWN, F_LAIR, F_AVOID = range(5)
NF = 5

PLAYER, SETTLER, BUILDER, RAIDER, BEAST, TOWER, ARROW, LIGHT, HOUSE, PALISADE, MARKET, SLOT, MAP = range(13)
N_SETTLERS, N_BUILDERS, N_RAIDERS, N_BEASTS, N_SITES, LIGHTS_PER = 14, 4, 8, 8, 12, 12
N_BUILD_TOWERS, N_STRUCTS = 6, 24               # towers the player can raise; slots for houses, walls and markets
N_TOWERS = N_SITES + N_BUILD_TOWERS
N_LIGHTS = N_BUILDERS * LIGHTS_PER
V0 = 1
B0 = V0 + N_SETTLERS
R0 = B0 + N_BUILDERS
M0 = R0 + N_RAIDERS
T0 = M0 + N_BEASTS
A0 = T0 + N_TOWERS
L0 = A0 + N_TOWERS
S0 = L0 + N_LIGHTS
ENT = S0 + N_STRUCTS
MAP0 = ENT
LANES = ENT + NODES

WALK_BASE = 2.2                                 # the wizard's walk on grass, units a second (road 1.6x: WALK_TOP)
WALK_TOP = WALK_BASE * 1.6
PLAYER_TAU = 0.15                               # the wizard eases to its pace over this long
TURN_RATE, ROUTE_TURN = 2.4, 5.0                # radians a second: free walking (full stick), turning toward a waypoint
ROUTE_REACH = 0.6                               # a waypoint is reached within this arc
INPUT_DEAD = 0.05                               # a stick past this is free walking: it cancels a click path
PATH_MAX = 256                                  # waypoints a click path holds (on the first map lanes)
HOUSE_REACH, MARKET_REACH, SHELTER_MEND = 1.8, 3.2, 0.6   # the folk about a house/market mend (hit points a second)
SNAP_REACH = 2.5                                # a tap places on the nearest buildable node within this arc
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
    PLAYER: Kind("wizard", PLAYER, team=0, speed=WALK_TOP, sense=6.0, hp=PLAYER_HP, radius=0.3, reload=0.35),
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
    # What the player builds (a slot is a dead lane until a tile takes it).  A wall is no one's target: it
    # only bars its node.  A house or a market can be burned like a light.
    HOUSE: Kind("house", HOUSE, team=0, hp=6.0, radius=0.45, mends=0.05),
    PALISADE: Kind("wall", PALISADE, team=0, hp=1.0, radius=0.45),
    MARKET: Kind("market", MARKET, team=0, hp=8.0, radius=0.5, mends=0.05),
    SLOT: Kind("slot", SLOT, team=-1, hp=1.0),
}

# ------------------------------------------------------------- the columns
#: The wizard's controls: ``move_axis`` turns (left/right), ``move_axis_y`` walks (up is forward, down backs
#: up), ``fire`` swings.  ``steer_x``/``steer_y`` stay in the page's control vocabulary but are unused (a tap is
#: the ``worldwalk_goto`` action, not a held finger).  Nothing held: the wizard stands.
CONTROLS = {name: sc.CONTROLS[name] for name in ("move_axis", "move_axis_y", "steer_x", "steer_y", "fire")}
KIND_COLUMNS = ("team", "speed", "sense", "hpmax", "dmg", "rad", "reload_t", "respawn_t", "needrate",
                "goalf0", "goalf1", "refugef", "mends")
NODE_COLUMNS = ("mat", "slope", "comp", *(f"nb{k}" for k in range(NEIGHBOURS)))
#: The click path: waypoint ``k``'s position on map lane ``MAP0 + k``; its length on the wizard's lane.
PATH_COLUMNS = ("rp_x", "rp_y", "rp_z", "rp_len")
FIELD_COLUMNS = tuple(f"{n}{i}" for n in ("fnx", "fny", "fnz", "fdd") for i in range(NF))
STATIC = ("lane", "kind", *KIND_COLUMNS, "owner", "home_node", *NODE_COLUMNS, *PATH_COLUMNS, *FIELD_COLUMNS)
STATE = ("alive", "x", "y", "vx", "vy", "ux", "uy", "uz", "hx", "hy", "hz", "spd", "yaw", "vyaw", "vspd",
         "mdx", "mdy", "mdz", "gspd", "bank", "vbank", "hp", "hurt", "cool", "timer", "need", "phase", "mot", "think", "born", "node",
         "map_u", "map_v", "map_gx", "map_gy",
         "score", "kills", "trips", "losses", "built", "lives", "clock", "route", "wp")
SETTINGS = ("field_w",)
ARGUMENTS = (*STATIC, *STATE, *SETTINGS, *CONTROLS, "dt")


def worldwalk_step(*columns):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span (``dt`` a scalar)."""
    c = dict(zip(ARGUMENTS, columns))
    f = as_float
    n = ENT
    lanes_total = int(c["mat"].shape[0])
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
    is_house, is_market = kind == HOUSE, kind == MARKET
    is_monster = is_raider | is_beast
    is_walker = is_player | is_settler | is_builder | is_monster
    folk = is_player | is_settler | is_builder | is_tower | is_light | is_house | is_market
    live = alive > 0.0
    playing = lives > 0.0
    clock_n = clock + dt
    P = f(is_player)

    # ------------------------------------------------------ the wizard: walk
    # Not a ship: it has no throttle to hold.  Its pace is ``walk_speed`` at the node it stands on (the surface
    # material and slope) times what it is asked for (``walk_drive``): the stick (up/down walks, left/right turns,
    # ``walk_route``), or -- with the stick at rest and a click path laid -- the way along that path (turning toward
    # each waypoint, ``walk_route``).  Nothing asked, it stops.  The page runs these same cores.
    cx, cy = e("field_w") * 0.5, zero + PLAYER_Y
    me = is_player & live & playing
    ship_on = f(me).sum() > 0.0

    def mine(v):                     # the wizard's lane of a span
        return (P * v).sum()

    u0 = (mine(u[0]), mine(u[1]), mine(u[2]))
    h0 = (mine(h[0]), mine(h[1]), mine(h[2]))
    dt0 = mine(dt)
    stick_turn, stick_walk = mine(e("move_axis")), mine(e("move_axis_y"))
    held = f((stick_turn.abs() > INPUT_DEAD) | (stick_walk.abs() > INPUT_DEAD))
    route_on = mine(e("route")) * (1.0 - held)            # free walking cancels a click path
    wp0 = mine(e("wp"))
    way = zero + float(MAP0) + wp0.clip(0.0, PATH_MAX - 1.0)
    waypoint = (mine(walkers.gather(c["rp_x"], way)), mine(walkers.gather(c["rp_y"], way)),
                mine(walkers.gather(c["rp_z"], way)))
    mat_here = walkers.gather(c["mat"], node)               # the surface under each lane, at the node it stood on
    slope_here = walkers.gather(c["slope"], node)
    top = mine(walk_speed(zero + WALK_BASE, mat_here, slope_here)[0])
    turn_angle, align, reached = walk_route(u0[0], u0[1], u0[2], h0[0], h0[1], h0[2], waypoint[0], waypoint[1], waypoint[2],
                                            route_on, stick_turn, TURN_RATE, ROUTE_TURN, ROUTE_REACH, dt0, RADIUS)
    walk_pace = walk_drive(mine(spd), top, 0.0 - stick_walk, route_on, align, dt0, PLAYER_TAU)[0]
    on = f(ship_on)
    dyaw = on * turn_angle
    pspd = on * walk_pace
    h0t = sm.turn(u0, h0, dyaw)
    wp_n = wp0 + on * route_on * reached
    route_n = on * route_on * f(wp_n < mine(e("rp_len")))   # the last waypoint reached: the path is done

    # --------------------------------------------------------- the node map
    nux, nuy, nuz = c["ux"], c["uy"], c["uz"]          # the nodes' positions (on the map lanes)
    mat = c["mat"]
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
    on_road = (mat_here > 0.5) & (mat_here < 1.5)

    # ------------------------------------------------- lane against lane
    ex, ey, ez, dd = walkers.sphere_pairwise(u[0], u[1], u[2], RADIUS)

    def row(v):
        return v.reshape((n, 1))

    def col(v):
        return v.reshape((1, n))

    other = row(lane) != col(lane)
    seeker = live & (is_walker | is_tower)
    target = live & (is_walker | is_tower | is_light | is_house | is_market)
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
    spd_w = spd + (walk_speed(speed * pace, mat_here, slope_here)[0] - spd) * k_vel
    vel = sm.add(sm.scale(face, spd_w), sm.scale(sep, 1.2))
    vlen = sm.dot(vel, vel).sqrt()
    walk_dir = sm.tangent(u, vel)
    move_dir = sm.vwhere(is_player, (h0t[0] + zero, h0t[1] + zero, h0t[2] + zero), sm.vwhere(is_arrow, h, walk_dir))
    move_dist = where(is_player & walker, pspd * dt, where(npc, vlen * dt, where(flying, spd * dt, zero)))
    u1, hm1 = sm.advance(u, move_dir, move_dist, RADIUS)
    node1 = walkers.climb(node, u1[0], u1[1], u1[2], nux, nuy, nuz, [c[f"nb{k}"] for k in range(NEIGHBOURS)])
    # Water does not bar a walker -- it swims, at a twentieth of its pace (walk_speed) -- but a wall does: it stops
    # at the wall's node and turns aside.
    stopped = (npc | (is_player & walker)) & (walkers.gather(mat, node1) > 4.5)
    u2 = sm.vwhere(stopped, u, u1)
    node2 = where(stopped, node, node1)
    aside = (f(noise(lane * 3.1) > 0.5) * 2.0 - 1.0) * 1.1
    face_turned = sm.vwhere(stopped, sm.turn(u, face, aside), face)
    h_next = sm.vwhere(is_player & walker, sm.vwhere(stopped, move_dir, hm1),
                       sm.vwhere(is_arrow, hm1, sm.tangent(u2, face_turned)))
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
    grows = live & (is_light | is_house | is_market)
    # the folk about a house (or, further, a market) mend
    shelter_reach = where(is_market, zero + MARKET_REACH, zero + HOUSE_REACH)
    sheltered = f(row(live & (is_player | is_settler | is_builder)) & col(live & (is_house | is_market))
                  & (dd < col(shelter_reach))).sum(dim=1) > 0.0
    hp_m = (hp + f(resting | grows) * mends * dt + f(sheltered) * SHELTER_MEND * dt).clip(0.0, 1.0e6)
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
    spd_n = where(is_player, where(stopped, zero, pspd + zero), where(npc, spd_w, where(fire_a, ARROW_SPEED + zero, where(spawn | takes, zero, spd))))
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
    reach = where(live, speed * 1.3, zero) + where(live & is_arrow, ARROW_SPEED + zero, zero) + pspd.abs()
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
        wide("spd", spd_n), wide("yaw", where(is_player, yaw + dyaw, yaw)), wide("vyaw", where(is_player, zero + dyaw / dt0, vyaw)),
        wide("vspd", where(is_player, zero, vspd)),
        # the direction each walker is walking and its ground speed: the state the page carries it on from
        wide("mdx", mdir[0]), wide("mdy", mdir[1]), wide("mdz", mdir[2]), wide("gspd", ground_speed),
        wide("bank", bank),
        wide("vbank", vbank),
        wide("hp", hp_n), wide("hurt", hurt_f), wide("cool", cool_f), wide("timer", timer_n),
        wide("need", need_f), wide("phase", phase_f), wide("mot", mot_f), wide("think", think_f),
        wide("born", born_f), wide("node", node_n),
        wide("map_u", map_u_n), wide("map_v", map_v_n), wide("map_gx", map_gx_n), wide("map_gy", map_gy_n),
        whole("score", kills_d * 10.0 + trips_d + built_d * 3.0 - losses_d * 5.0), whole("kills", kills_d),
        whole("trips", trips_d), whole("losses", losses_d), whole("built", built_d),
        (c["lives"] * 0.0 + hero_hp), c["clock"] + c["dt"],
        wide("route", where(is_player, zero + route_n, e("route"))), wide("wp", where(is_player, zero + wp_n, e("wp"))),
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
    """The globe's node graph and everything precomputed on it (cached: it is one world).

    The SURFACE of the world is a per-node material (grass, road, sand, rock, water; a wall once built) and
    slope, from the terrain law (``roads.world``'s heights) and the road plan.  Walking an edge costs the time it
    takes at ``entity_motion.walk_speed``'s speeds, so every flow field and click path is priced by the law the
    walkers walk by.  What a game changes (walls) lives in its own ``Ground`` (``World.ground()``)."""

    def __init__(self):
        nodes, height, dry, nearest = road_plan.world()

        def cost(a, b):
            return road_plan.edge_cost(nodes, height, dry, RADIUS, a, b)

        self.graph = walkers.SphereGraph.build(nodes, nearest, dry, cost)     # land connectivity (sight, clearance)
        self.nodes, self.dry, self.height = nodes, dry, height
        sites, lines, _points = road_plan.plan(WORLD_SEED, RADIUS, sites=N_SITES, road_points=160)
        self.sites, self.lines = sites, lines
        self.site_nodes = [self.node_of(p) for p in sites]
        # roads: the polylines' vertices are nodes, consecutive ones are edges
        road_nodes, self.road_edges, self.road_links = set(), set(), {}
        for line in lines:
            ids = [self.node_of(p) for p in line]
            road_nodes.update(ids)
            self.road_edges.update(zip(ids[:-1], ids[1:]))
            for a, b in zip(ids[:-1], ids[1:]):
                self.road_links.setdefault(a, set()).add(b)
                self.road_links.setdefault(b, set()).add(a)
        self.road_nodes = road_nodes
        self.comp = walkers.sphere_components(self.graph)
        labels, counts = np.unique(self.comp[self.comp >= 0], return_counts=True)
        self.main = int(labels[np.argmax(counts)])
        self.towns = [i for i, node in enumerate(self.site_nodes) if self.comp[node] == self.main]
        self.towns_a, self.towns_b = self.towns[0::2], self.towns[1::2]
        self.lairs = self._lairs(4)
        self.slope = self._slopes()
        self.mat = self._materials(sorted(road_nodes))
        self.speed = walk_speed(1.0, self.mat, self.slope)[0]
        self.edge = self.edge_times(self.speed)
        self.goals = [[self.site_nodes[i] for i in self.towns_a], [self.site_nodes[i] for i in self.towns_b],
                      [self.site_nodes[i] for i in self.towns], list(self.lairs)]
        self.fields = [*self.goal_fields(self.edge), walkers.SphereField.clearance(self.graph, RADIUS)]

    # -- the surface
    def _slopes(self) -> np.ndarray:
        """Each node's gradient: the steepest rise over run to a neighbour, from the terrain law's heights, in
        hundredths (so the page's table and the law's column hold the very same numbers); zero in water."""
        nodes, height = self.nodes, self.height
        nb = self.graph.nearest
        arc = RADIUS * np.arccos(np.clip(np.einsum("ij,ikj->ik", nodes, nodes[nb]), -1.0, 1.0))
        steep = (np.abs(height[:, None] - height[nb]) / np.maximum(arc, 1e-9)).max(axis=1)
        steep = np.where(self.dry, steep, 0.0)
        return np.round(np.minimum(steep, 2.55) * 100.0) / 100.0

    def _materials(self, road: list[int]) -> np.ndarray:
        mat = np.full(len(self.nodes), GRASS)
        mat[self.height < SAND_HEIGHT] = SAND
        mat[(self.height > ROCK_HEIGHT) | (self.slope > road_plan.MAX_SLOPE)] = ROCK
        mat[road] = ROAD
        mat[~self.dry] = WATER
        return mat

    def edge_times(self, speed: np.ndarray) -> list[dict]:
        """Per node, each neighbour's edge cost: the time to cross at the two nodes' speeds (their harmonic
        mean over the arc), infinite where either bars (a wall)."""
        out = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for a, row in enumerate(self.graph.adjacent):
                nb = np.asarray(row)
                arc = RADIUS * np.arccos(np.clip(self.nodes[nb] @ self.nodes[a], -1.0, 1.0))
                va, vb = speed[a], speed[nb]
                time = np.where((va > 0.0) & (vb > 0.0), arc * 0.5 * (1.0 / va + 1.0 / vb), np.inf)
                # (an edge between two land nodes whose straight walk clips a water cell is not offered: the
                # walker would crawl through the cell -- the graph's "no corner cutting")
                cut = [(min(a, b), max(a, b)) in self.graph.cut for b in row]
                out.append(dict(zip(row, np.where(cut, np.inf, time).tolist())))
        return out

    def goal_fields(self, edge: list[dict]) -> list:
        def cost(a, b):
            return edge[a][b]
        return [walkers.SphereField.build(self.graph, goals, cost=cost) for goals in self.goals]

    def ground(self) -> "Ground":
        return Ground(self)

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


class Ground:
    """One game's surface: the world's materials and edge costs plus what the player has built on them.

    A wall bars its node: its material becomes ``WALL`` (speed 0), every edge through it costs infinity, and the
    flow fields are rebuilt over the new costs (``refit``) -- settlers, raiders and the click path all route
    around it.  Everything is priced by ``walk_speed``; there is no second table."""

    def __init__(self, world: World):
        self.world = world
        self.mat = world.mat.copy()
        self.speed = world.speed.copy()
        self.edge = [dict(row) for row in world.edge]
        self.fields = list(world.fields)
        self.taken: dict[int, float] = {}          # node -> the kind of tile standing on it
        self.version = 0                           # bumped whenever a material changes (the page's table)

    def cost(self, a: int, b: int) -> float:
        return self.edge[a][b]

    def path(self, start: int, goal: int) -> list[int] | None:
        """The cheapest node path, by the walk-speed law's times (Dijkstra, as the flow fields are built)."""
        return walkers.sphere_path(self.world.graph.adjacent, self.cost, int(start), int(goal))

    def block(self, node: int) -> None:
        self.mat[node] = WALL
        self.speed[node] = 0.0
        for b in self.world.graph.adjacent[node]:
            self.edge[node][b] = np.inf
            self.edge[b][node] = np.inf
        self.version += 1

    def refit(self) -> None:
        self.fields = [*self.world.goal_fields(self.edge), self.world.fields[F_AVOID]]

    def buildable(self, node: int) -> str | None:
        """Why a tile cannot stand on ``node`` (None: it can).  A tile ADHERES to a road: it stands on a node
        beside a road node (never on the road, in the water or where something stands)."""
        w = self.world
        if not (isinstance(node, (int, np.integer)) and 0 <= node < len(w.nodes)):
            return "no such node"
        if self.mat[node] == WATER:
            return "no building in water"
        if self.mat[node] == WALL or node in self.taken:
            return "something stands there"
        if node in w.site_nodes:
            return "a town stands there"
        if node in w.road_nodes:
            return "tiles stand beside a road, not on it"
        if not any(b in w.road_nodes for b in w.graph.adjacent[node]):
            return "not beside a road"
        return None

    def candidates(self, point: np.ndarray, reach: float) -> list[int]:
        """Nodes within ``reach`` units of arc of ``point``, nearest first."""
        w = self.world
        arc = RADIUS * np.arccos(np.clip(w.nodes @ np.asarray(point), -1.0, 1.0))
        near = np.flatnonzero(arc <= reach)
        return [int(i) for i in near[np.argsort(arc[near])]]

    def road_heading(self, node: int) -> np.ndarray:
        """The unit tangent at ``node`` along the road it stands beside (a tile is oriented along its road)."""
        w = self.world
        u = w.nodes[node]
        beside = [b for b in w.graph.adjacent[node] if b in w.road_nodes]
        r = max(beside, key=lambda b: float(w.nodes[b] @ u))                   # the nearest road node
        links = sorted(w.road_links.get(r, ()))
        if len(links) >= 2:
            v = w.nodes[links[-1]] - w.nodes[links[0]]
        elif links:
            v = w.nodes[links[0]] - w.nodes[r]
        else:
            v = np.cross(u, w.nodes[r] - u)
        v = v - (v @ u) * u
        return v / (np.linalg.norm(v) + 1e-12)


_WORLD: World | None = None


def world() -> World:
    global _WORLD
    if _WORLD is None:
        _WORLD = World()
    return _WORLD


def initial_columns(width: float = sc.WIDTH, seed: int = 0, ground: Ground | None = None) -> dict[str, np.ndarray]:
    w = world()
    ground = ground or w.ground()
    rng = np.random.default_rng(int(seed) + 7919)
    kind = np.full(LANES, float(MAP))
    for lanes, kid in (((0, 1), PLAYER), ((V0, B0), SETTLER), ((B0, R0), BUILDER), ((R0, M0), RAIDER), ((M0, T0), BEAST),
                       ((T0, A0), TOWER), ((A0, L0), ARROW), ((L0, S0), LIGHT), ((S0, ENT), SLOT)):
        kind[lanes[0]:lanes[1]] = kid
    columns = {"lane": np.arange(LANES, dtype=np.float64), "kind": kind}
    columns.update(walkers.kind_columns(KINDS, kind))
    columns["respawn_t"][T0 + N_SITES:A0] = 0.0       # a tower the player raised does not come back by itself
    # where each lane lives: commuters at the A towns, raiders and beasts at the lairs, a tower at each site
    home = np.zeros(LANES)
    owner = np.full(LANES, -1.0)
    a_nodes = [w.site_nodes[i] for i in w.towns_a]
    for lane in range(V0, R0):
        home[lane] = MAP0 + a_nodes[(lane - V0) % len(a_nodes)]
    for lane in range(R0, T0):
        home[lane] = MAP0 + w.lairs[(lane - R0) % len(w.lairs)]
    for i in range(N_SITES):
        home[T0 + i] = MAP0 + w.site_nodes[i]
    for i in range(N_TOWERS):
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
    for lane in range(L0, ENT):                      # lights and build slots wait, dark, at the pole
        ux[lane], uy[lane], uz[lane] = 0.0, 0.0, 1.0
        hx[lane], hy[lane], hz[lane] = 0.0, 1.0, 0.0
    columns.update({"owner": owner, "home_node": home})
    mat = np.zeros(LANES)
    mat[MAP0:] = ground.mat
    slope = np.zeros(LANES)
    slope[MAP0:] = w.slope
    columns["mat"], columns["slope"] = mat, slope
    columns.update({name: np.zeros(LANES) for name in PATH_COLUMNS})
    columns.update(walkers.sphere_node_columns(w.graph, w.comp, MAP0, LANES))
    columns.update(walkers.sphere_field_columns(ground.fields, w.nodes, MAP0, LANES))
    alive = np.zeros(LANES)
    alive[0:A0] = 1.0
    alive[T0 + N_SITES:A0] = 0.0                     # the towers to be built
    timer = np.zeros(LANES)
    timer[V0:R0] = rng.uniform(0.0, 4.0, R0 - V0)
    cool = np.zeros(LANES)
    cool[B0:R0] = rng.uniform(2.0, BUILD_EVERY, N_BUILDERS)
    yaw = np.zeros(LANES)
    yaw[0] = sc.YAW0
    columns.update({
        "alive": alive, "x": np.full(LANES, width / 2.0), "y": np.full(LANES, PLAYER_Y), "vx": np.zeros(LANES),
        "vy": np.zeros(LANES), "ux": ux, "uy": uy, "uz": uz, "hx": hx, "hy": hy, "hz": hz, "spd": np.zeros(LANES),
        "yaw": yaw, "vyaw": np.zeros(LANES), "vspd": np.zeros(LANES), "bank": np.full(LANES, sc.BANK_MAX),
        "mdx": hx.copy(), "mdy": hy.copy(), "mdz": hz.copy(), "gspd": np.zeros(LANES),
        "vbank": np.zeros(LANES), "hp": columns["hpmax"].copy(), "hurt": np.zeros(LANES), "cool": cool,
        "timer": timer, "need": np.zeros(LANES), "phase": np.zeros(LANES), "mot": np.zeros(LANES),
        "think": rng.uniform(0.0, 0.25, LANES), "born": np.zeros(LANES), "node": node,
        "map_u": np.zeros(LANES), "map_v": np.zeros(LANES), "map_gx": np.zeros(LANES), "map_gy": np.full(LANES, -1.0),
        "score": np.zeros(LANES), "kills": np.zeros(LANES), "trips": np.zeros(LANES), "losses": np.zeros(LANES),
        "built": np.zeros(LANES), "lives": np.full(LANES, PLAYER_HP), "clock": np.zeros(LANES),
        "route": np.zeros(LANES), "wp": np.zeros(LANES),
        "field_w": np.full(LANES, float(width)),
        **{name: np.zeros(LANES) for name in CONTROLS}, "steer_x": np.full(LANES, -1.0), "steer_y": np.full(LANES, -1.0),
    })
    return columns


# ------------------------------------------------------------------ the game

#: Mesh vocabulary this game adds to ``web_export``'s shared tables: name -> emoji.
EMOJI = {"builder": "\U0001F477", "market": "\U0001F3EA"}
EMOJI_SUBDIVISIONS: dict[str, int] = {}

#: kind -> (mesh, colour, size)
KIND_LOOK = {PLAYER: ("wizard", "wizard", 0.42), SETTLER: ("villager", "villager", 0.36),
             BUILDER: ("builder", "builder", 0.38), RAIDER: ("goblin", "goblin", 0.4),
             BEAST: ("wolf", "wolf", 0.42), TOWER: ("tower", "tower", 0.5), ARROW: ("bolt", "bolt", 0.2),
             HOUSE: ("home", "house", 0.55), PALISADE: ("brick", "wall", 0.55), MARKET: ("market", "market", 0.55)}
#: Kinds that stand or fly without walking (no bob, no lean).
STILL = (ARROW, LIGHT, HOUSE, PALISADE, MARKET)
#: What the build control can raise: tile -> its kind.
TILES = {"house": HOUSE, "wall": PALISADE, "tower": TOWER, "market": MARKET}


class WorldWalk(EntityGame):
    """World Walk.  Actions (``/api/action``, all ``worldwalk_*``):

    ``worldwalk_goto``   ``u`` (a unit point on the globe, the tap) or ``node``: lay the cheapest path from the
                         wizard to it (``Ground.path``, the walk-speed law's costs) as waypoints on the first
                         map lanes, and start it; any stick input cancels it
    ``worldwalk_stop``   cancel the path
    ``worldwalk_tool``   ``tool``: walk | house | wall | tower | market (what a tap does)
    ``worldwalk_place``  ``tool`` (default: the chosen one) at ``node`` (exactly: it must be buildable) or at the
                         buildable node nearest the point ``u`` (within ``SNAP_REACH``)
    ``worldwalk_here``   the same at the buildable node nearest the wizard

    A refusal is a ``ValueError`` (the server answers with the message)."""

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
    #: The player is a walker, not a craft: the page draws no marker lights, strobe, headlights, beam or shield on it.
    AVATAR = "walker"

    def __init__(self, board_id: str = "", aspect: float | None = None, seed: int = 0):
        self.width = sc.field_width(aspect)
        self.seed = int(seed)
        self.tool = "walk"
        self.route_id = 0
        self.route: dict[str, Any] | None = None
        self.houses = 0
        self.ground = world().ground()
        super().__init__(board_id)

    def make_piece(self):
        return WorldWalkPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        self.ground = world().ground()
        self.route, self.houses = None, 0
        self._surface = None
        return initial_columns(self.width, self.seed, self.ground)

    def reset(self) -> None:
        super().reset()
        self.route_id += 1

    def adopt_axes(self) -> list[str]:
        return ["yaw", "spd"]

    def motion(self) -> dict[str, Any]:
        """The wizard's walk as the law runs it (``walk`` holds the constants of ``walk_speed``'s caller, ``walk_drive``
        and ``walk_route``), the patch the page draws in, and the globe's land."""
        return {"speed": WALK_TOP, "tau": PLAYER_TAU, "reach": ROUTE_REACH,
                "yaw": [-1.0e9, 1.0e9], "spd": [-WALK_TOP, WALK_TOP],
                "walk": {"base": WALK_BASE, "tau": PLAYER_TAU, "turn": TURN_RATE, "route_turn": ROUTE_TURN,
                         "reach": ROUTE_REACH, "dead": INPUT_DEAD, "radius": RADIUS, "wall": WALL, "path_max": PATH_MAX,
                         "materials": {name: float(factor) for name, (_id, factor) in WALK_MATERIALS.items()}},
                "patch": {"cx": self.width / 2.0, "cy": PLAYER_Y, "radius": RADIUS},
                "travel": ["map_u", "map_v", "map_gx", "map_gy", "ux", "uy", "uz", "hx", "hy", "hz", "wp", "route"],
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

    # ------------------------------------------------------------- actions
    def do(self, payload: dict) -> None:
        kind = str(payload.get("type"))
        if self.session.state is None:
            raise ValueError("the world has not started")
        if kind == "worldwalk_goto":
            self.goto(node=payload.get("node"), point=payload.get("u"))
        elif kind == "worldwalk_stop":
            self.stop()
        elif kind == "worldwalk_tool":
            tool = str(payload.get("tool"))
            if tool != "walk" and tool not in TILES:
                raise ValueError(f"unknown tool {tool!r}")
            self.tool = tool
        elif kind == "worldwalk_place":
            self.place(str(payload.get("tool") or self.tool), node=payload.get("node"), point=payload.get("u"))
        elif kind == "worldwalk_here":
            self.place(self.tool, point=self.session_point(0))
        else:
            raise ValueError(f"unknown worldwalk action {kind!r}")

    def session_point(self, lane: int) -> np.ndarray:
        c = self.session.column
        return np.array([c("ux")[lane], c("uy")[lane], c("uz")[lane]])

    def _point(self, value) -> np.ndarray:
        p = np.asarray(value, dtype=float).reshape(-1)
        if p.shape != (3,) or not np.isfinite(p).all() or np.linalg.norm(p) < 1e-9:
            raise ValueError("a point on the globe is three numbers")
        return p / np.linalg.norm(p)

    def goto(self, *, node=None, point=None) -> list[int]:
        """Walk to a node (or the node nearest a point): the cheapest path from the wizard's node, laid as
        waypoints; returns the node path (start first)."""
        s, w = self.session, world()
        if s.column("alive")[0] <= 0.0 or s.column("lives")[0] <= 0.0:
            raise ValueError("the wizard is down")
        goal = int(node) if node is not None else w.node_of(self._point(point))
        if not 0 <= goal < len(w.nodes):
            raise ValueError("no such node")
        if self.ground.mat[goal] == WALL:
            raise ValueError("a wall bars that node")
        start = int(s.column("node")[0]) - MAP0
        path = self.ground.path(start, goal)
        if path is None:
            raise ValueError("no way there")
        way = path[1:]
        if not way:
            self.stop()
            return path
        if len(way) > PATH_MAX:
            raise ValueError(f"too far to walk in one go ({len(way)} nodes, {PATH_MAX} at most)")
        for name, k in (("rp_x", 0), ("rp_y", 1), ("rp_z", 2)):
            s.column(name)[MAP0:MAP0 + len(way)] = w.nodes[way, k]
        s.column("rp_len")[0] = float(len(way))
        s.column("wp")[0] = 0.0
        s.column("route")[0] = 1.0
        self.route_id += 1
        self.route = {"id": self.route_id, "nodes": [int(n) for n in path], "goal": goal}
        return path

    def stop(self) -> None:
        self.session.column("route")[0] = 0.0
        self.route = None

    def place(self, tool: str, *, node=None, point=None) -> int:
        """Raise a tile.  ``node``: exactly there, or refused.  ``point``: the buildable node nearest it within
        ``SNAP_REACH`` (the tap snaps to the road), or refused.  Returns the build lane."""
        if tool not in TILES:
            raise ValueError("choose a house, wall, tower or market to build" if tool == "walk" else f"unknown tile {tool!r}")
        if node is not None:
            at = int(node)
            why = self.ground.buildable(at)
            if why:
                raise ValueError(f"cannot build a {tool} there: {why}")
        else:
            p = self._point(point)
            near = self.ground.candidates(p, SNAP_REACH)
            reasons = [self.ground.buildable(n) for n in near]
            ok = [n for n, why in zip(near, reasons) if why is None]
            if not ok:
                raise ValueError(f"cannot build a {tool} there: {reasons[0] if reasons else 'nothing near'}")
            at = ok[0]
        lane = self._free_lane(tool)
        self._build(lane, tool, at)
        return lane

    def _free_lane(self, tool: str) -> int:
        alive = self.session.column("alive")
        lanes = range(T0 + N_SITES, A0) if tool == "tower" else range(S0, ENT)
        for lane in lanes:
            if alive[lane] <= 0.0:
                return lane
        raise ValueError(f"no free slot for a {tool}")

    def _build(self, lane: int, tool: str, node: int) -> None:
        s, w, g = self.session, world(), self.ground
        kid = TILES[tool]
        c = s.column
        for name, values in walkers.kind_columns(KINDS, np.asarray([float(kid)])).items():
            c(name)[lane] = values[0]
        if kid == TOWER:
            c("respawn_t")[lane] = 0.0
        p, h = w.nodes[node], g.road_heading(node)
        for name, v in (("ux", p[0]), ("uy", p[1]), ("uz", p[2]), ("hx", h[0]), ("hy", h[1]), ("hz", h[2]),
                        ("mdx", h[0]), ("mdy", h[1]), ("mdz", h[2])):
            c(name)[lane] = v
        c("kind")[lane] = float(kid)
        c("node")[lane] = c("home_node")[lane] = float(MAP0 + node)
        c("hp")[lane] = c("hpmax")[lane]
        for name in ("hurt", "timer", "need", "phase", "mot", "think", "spd", "gspd"):
            c(name)[lane] = 0.0
        c("cool")[lane] = 0.5
        c("born")[lane] = float(c("clock")[0])
        c("alive")[lane] = 1.0
        g.taken[node] = float(kid)
        if kid == HOUSE:                           # a settler's home: it comes back here
            c("home_node")[V0 + self.houses % N_SETTLERS] = float(MAP0 + node)
            self.houses += 1
        if kid == PALISADE:
            g.block(node)
            g.refit()
            c("mat")[MAP0 + node] = WALL
            for name, column in walkers.sphere_field_columns(g.fields, w.nodes, MAP0, LANES).items():
                c(name)[...] = column
            self._surface = None
            if self.route is not None:             # a path through the new wall is laid again
                try:
                    self.goto(node=self.route["goal"])
                except ValueError:
                    self.stop()

    # ---------------------------------------------------------------- the page
    @property
    def emissive(self) -> dict[str, list[float]]:
        return {"city": [3.0, 2.4, 1.4]}

    def palette(self) -> dict[str, list[float]]:
        return {"wizard": [0.52, 0.42, 0.96], "villager": [0.96, 0.82, 0.62], "builder": [0.95, 0.62, 0.2],
                "goblin": [0.46, 0.78, 0.26], "wolf": [0.66, 0.66, 0.72], "tower": [0.88, 0.88, 0.94],
                "bolt": [1.0, 0.92, 0.25], "city": [1.0, 0.86, 0.55], "house": [0.86, 0.5, 0.36],
                "wall": [0.62, 0.6, 0.58], "market": [0.95, 0.74, 0.3]}

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
                        "lean": 0.0 if kid in STILL else 0.45, "face": False, "translucent": False,
                        "ribbon": False, "ground": True, "glow": kid == LIGHT, "spin": 0.0, "owner": -1,
                        "age": clock - float(col["born"][k]),
                        "bob": 1.0 if moving and kid not in STILL else 0.0})
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]),
                "detail": f"{int(s.column('kills')[0])} raiders, {int(s.column('trips')[0])} commutes, "
                          f"{int(s.column('built')[0])} lights"}

    def controls_ui(self) -> list[dict[str, Any]]:
        """The shared controls bar's entries (``game.controls_ui``): what a tap does, and two buttons."""
        return [
            {"id": "tool", "kind": "select", "label": "tap", "value": self.tool, "action": "worldwalk_tool", "field": "tool",
             "options": [["walk", "walk there"], ["house", "build house"], ["wall", "build wall"],
                         ["tower", "build tower"], ["market", "build market"]]},
            {"id": "here", "kind": "button", "label": "Build here", "action": "worldwalk_here", "enabled": self.tool != "walk"},
            {"id": "stop", "kind": "button", "label": "Stop", "action": "worldwalk_stop"},
        ]

    def surface(self) -> dict[str, Any]:
        """The page's copy of the surface (cached until a material changes): the node count, the materials as one
        digit a node, the slopes in hundredths as base64 bytes -- the numbers the law's columns hold."""
        key = self.ground.version
        if getattr(self, "_surface", None) is None or self._surface[0] != key:
            import base64
            w = world()
            mat = "".join(str(int(m)) for m in self.ground.mat)
            slope = base64.b64encode(np.round(w.slope * 100.0).astype(np.uint8).tobytes()).decode("ascii")
            self._surface = (key, {"nodes": len(w.nodes), "mat": mat, "slope": slope, "version": key})
        return self._surface[1]

    def state(self) -> dict[str, Any]:
        st = super().state()
        st["avatar"] = self.AVATAR
        st["tool"] = self.tool
        st["controls_ui"] = self.controls_ui()
        if self.session.state is not None:
            c = self.session.column
            if not hasattr(self, "_roads_published"):
                _s, lines, _p = road_plan.plan(WORLD_SEED, RADIUS, sites=N_SITES, road_points=160)
                self._roads_published = [[[round(float(v), 5) for v in pt] for pt in line] for line in lines]
            st["roads"] = self._roads_published
            st["globe"] = {"u": [float(c("ux")[0]), float(c("uy")[0]), float(c("uz")[0])],
                           "h": [float(c("hx")[0]), float(c("hy")[0]), float(c("hz")[0])],
                           "yaw": float(c("yaw")[0]), "sun": list(self.SUN), "radius": RADIUS}
            # the walk: the surface (changes with a wall), the click path (changes with a click) and, apart,
            # its progress (changes every tick, so it rides in a key of its own)
            route = self.route
            st["walk"] = {**self.surface(), "route": {"id": route["id"], "path": route["nodes"]} if route else None}
            st["walking"] = {"id": route["id"] if route else 0, "wp": int(c("wp")[0]), "active": bool(c("route")[0] > 0.5)}
            alive = c("alive")
            st["builds"] = {"house": int((alive[S0:ENT] * (c("kind")[S0:ENT] == HOUSE)).sum()),
                            "wall": int((alive[S0:ENT] * (c("kind")[S0:ENT] == PALISADE)).sum()),
                            "market": int((alive[S0:ENT] * (c("kind")[S0:ENT] == MARKET)).sum()),
                            "tower": int(alive[T0 + N_SITES:A0].sum())}
        return st

    def describe(self) -> str:
        s = self.session
        seconds, calls = s.last_round
        alive = s.column("alive")[:ENT]
        head = "GAME OVER · R to restart" if s.column("lives")[0] <= 0 else f"hp {s.column('lives')[0]:.1f}"
        on = "walking" if s.column("route")[0] > 0.5 else "standing"
        return (f"{head} · {on} · kills {int(s.column('kills')[0])} · commutes {int(s.column('trips')[0])} · "
                f"lights {int(alive[L0:S0].sum())} · built {int(alive[S0:ENT].sum() + alive[T0 + N_SITES:A0].sum())} · "
                f"settlers {int(alive[V0:B0].sum())}/{N_SETTLERS} · "
                f"raiders+beasts {int(alive[R0:T0].sum())} · tick {seconds * 1000:.0f} ms = {calls} law steps")
