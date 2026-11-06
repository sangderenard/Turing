"""Boardwalk: a turn-taking race round a looping, forking track, one dt-system entity law.

An original slow-tick board game.  A looping track of 44 tiles (a ring of 34
with two shortcuts through its middle) is walked by KINDS of entity that take
TURNS: the pilgrim (the player), peddlers who shuttle between markets and
treasure, brigands who lurk at dens and rob whoever they catch, a knight who
patrols dens and camps, and rival racers.  First to three laps wins; coins,
duels and the road decide who gets there.

THE TURN.  One entity is ACTIVE.  When its hold has run out (and, for the
player, ``fire`` confirms) its turn resolves in ONE law step:

    roll        a die (1..6) from the lane's own noise at this turn, scaled by
                the kind's stride
    choice      the kind's choice network (``walkers.choose_motivation``) picks
                a motivation; the player's pad (``move_axis_y``: up runs, down
                rests) overrides it, and ``move_axis`` picks the fork
    spend       how much of the roll the choice spends: ``wander`` (stroll)
                half, ``follow`` (route) and ``seek`` (hunt) all of it,
                ``attack`` (fight) one more, ``flee`` (run) three more (two
                again with a kit), ``rest`` none
    allowance   spend, plus 1 when an ally walks within two tiles ahead (the
                draft), minus 1 when tired; at most STEPS
    walk        STEPS gathers along the path tables: every step needs
                allowance and energy for the next tile's cost (a bog costs
                two), takes the branch at a fork by the choice (route: lower
                goal distance; hunt: nearer a foe; run: farther from the
                pursuers; stroll: the kind's boldness; the player: the pad),
                and ENDS the move on a market, treasure, gate, den or a tile a
                foe holds
    landing     the tile applies (treasure pays its pile, the gate and dens
                take a toll, a market trades coin for energy and a kit when
                the lander rests, a camp mends and refills); landing on a foe
                is a duel unless the lander ran or rested (dice plus damage; a
                kit adds two when fighting; the loser is hurt and loses half
                its purse to the winner; at no hit points it is carried to the
                camp and misses two rounds)
    pass        the next entity that is up becomes active (a round ends when
                the order wraps), and the hold restarts

Every lane computes its own hypothetical turn together; the ACTIVE-TURN MASK
selects whose move lands.  The path is data on the map's lanes (``na``/``nb``
successors, tile type and cost, ``dmat`` forward distances between tiles and
five goal distance fields), exactly as Tile Quest's flow fields are: a new
board is new column values, never a recompile.

Lanes: 0 the pilgrim, 1..2 peddlers, 3..5 brigands, 6 the knight, 7..8 rivals,
then the map: ``N * N`` lanes (tile ``k`` on lane ``MAP0 + k``; the distance
matrix entry ``(a, b)`` on lane ``MAP0 + a * N + b``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from meowdoku import walkers
from meowdoku.entities import EntityGame, LawPiece, as_float, first_of, noise
from meowdoku.walkers import ATTACK, FLEE, FOLLOW, REST, SEEK, WANDER, Kind
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

# ---------------------------------------------------------------- the world
WIDTH, HEIGHT = 16, 11
PLAIN, MARKET, CAMP, TREASURE, BOG, GATE, DEN, START = range(8)
STOPPING = (MARKET, TREASURE, GATE, DEN)
#: Goal distance fields, in column order.
F_MARKET, F_TREASURE, F_CAMP, F_DEN, F_FINISH = range(5)
NF = 5
N = 44                      # tiles: ring 0..33, shortcut x=5 34..38, shortcut x=10 39..43
STEPS = 10                  # the most tiles one turn can walk
BIG = 999.0
LAPS_TO_WIN = 3
MAX_ROUNDS = 60
REGEN = 2.0                 # energy back at the start of every own turn

PILGRIM, PEDDLER, BRIGAND, KNIGHT, RIVAL, MAP = range(6)
N_PEDDLERS, N_BRIGANDS, N_KNIGHTS, N_RIVALS = 2, 3, 1, 2
P0 = 1
B0 = P0 + N_PEDDLERS
K0 = B0 + N_BRIGANDS
R0 = K0 + N_KNIGHTS
ENT = R0 + N_RIVALS
MAP0 = ENT
LANES = ENT + N * N

HOLD_PLAYER, HOLD_NPC, HOLD_START = 0.45, 0.8, 0.6
PLAYER_HP = 8.0

# ----------------------------------------------------------- the kind rows
# ``speed`` is the stride (the roll's scale), ``sense`` how many tiles away a
# foe is felt, ``mends`` hit points healed by a rest (three times at a camp).
# Features: foe_prox / foe_close / foe_ahead are measured along the path
# (nearest foe either way; within 2.5 tiles; positive when it is AHEAD and
# negative when it is behind: a pursuer); need = 1 - energy / capacity;
# busy = purse / 10; goal_far = goal distance / 20.
KINDS: dict[int, Kind] = {
    PILGRIM: Kind("pilgrim", PILGRIM, team=0, speed=1.0, sense=6.0, hp=PLAYER_HP, damage=1.0, mends=1.0, goal=(F_FINISH, F_FINISH),
                  refuge=F_CAMP, net={
        "hidden": {
            "tired": ({"need": 6.0}, -3.0),
            "hurt": ({"health": -6.0}, 3.0),
            "prey": ({"foe_close": 5.0, "foe_ahead": 1.0}, -2.0),
        },
        "out": {
            "follow": ({"tired": -2.0, "hurt": -2.0, "prey": -1.0}, 1.2),
            "attack": ({"prey": 3.0, "hurt": -3.0}, -1.6),
            "rest": ({"tired": 3.0, "hurt": 2.5}, -1.5),
            "flee": ({"hurt": 3.0, "prey": 0.5}, -2.5),
        }}),
    PEDDLER: Kind("peddler", PEDDLER, team=0, speed=0.8, sense=6.0, hp=4.0, damage=0.0, mends=1.0,
                  goal=(F_MARKET, F_TREASURE), refuge=F_CAMP, net={
        "hidden": {
            "fear": ({"foe_prox": 5.0, "ally_prox": -1.0}, -1.6),
            "tired": ({"need": 6.0}, -3.2),
            "hurt": ({"health": -6.0}, 3.0),
            "there": ({"goal_far": -30.0}, 1.0),
        },
        "out": {
            "follow": ({"fear": -3.0, "tired": -2.5, "hurt": -2.5, "there": -3.0}, 1.4),
            "wander": ({"fear": -2.0}, 0.2),
            "flee": ({"fear": 4.0}, -1.6),
            "rest": ({"tired": 3.0, "hurt": 2.5, "there": 3.0, "fear": -2.0}, -1.0),
        }}),
    BRIGAND: Kind("brigand", BRIGAND, team=1, speed=1.0, sense=8.0, hp=6.0, damage=1.0, mends=1.0,
                  goal=(F_DEN, F_DEN), refuge=F_DEN, net={
        "hidden": {
            "prey": ({"foe_ahead": 4.0, "foe_prox": 2.0}, -2.2),
            "near": ({"foe_close": 6.0}, -2.0),
            "hurt": ({"health": -6.0}, 2.6),
            "home": ({"refuge_near": 5.0}, -3.0),
        },
        "out": {
            "wander": ({"prey": -2.0, "hurt": -3.0}, 0.8),
            "seek": ({"prey": 3.2, "hurt": -3.0}, -1.0),
            "attack": ({"near": 4.5, "prey": 0.5, "hurt": -3.0}, -1.5),
            "flee": ({"hurt": 4.5, "home": -2.5}, -2.0),
            "rest": ({"hurt": 2.5, "home": 3.0}, -2.4),
        }}),
    KNIGHT: Kind("knight", KNIGHT, team=0, speed=0.8, sense=8.0, hp=10.0, damage=2.0, mends=1.5,
                 goal=(F_DEN, F_CAMP), refuge=F_CAMP, net={
        "hidden": {
            "prey": ({"foe_ahead": 4.0, "foe_prox": 2.0}, -2.2),
            "near": ({"foe_close": 6.0}, -2.0),
            "hurt": ({"health": -6.0}, 2.6),
            "tired": ({"need": 6.0}, -3.2),
            "there": ({"goal_far": -30.0}, 1.0),
        },
        "out": {
            "follow": ({"prey": -3.0, "hurt": -3.0, "tired": -2.5, "there": -3.0}, 1.2),
            "seek": ({"prey": 3.2, "hurt": -3.0}, -0.8),
            "attack": ({"near": 4.5, "hurt": -3.0}, -1.5),
            "flee": ({"hurt": 4.5}, -2.2),
            "rest": ({"hurt": 2.5, "tired": 3.0, "there": 3.0}, -1.4),
        }}),
    RIVAL: Kind("rival", RIVAL, team=2, speed=1.2, sense=5.0, hp=6.0, damage=1.0, mends=1.0,
                goal=(F_TREASURE, F_FINISH), refuge=F_CAMP, net={
        "hidden": {
            "pursued": ({"foe_prox": 4.0, "foe_ahead": -4.0}, -2.0),
            "tired": ({"need": 6.0}, -3.0),
        },
        "out": {
            "follow": ({"tired": -2.5, "pursued": -1.0}, 1.5),
            "wander": ({}, 0.3),
            "flee": ({"pursued": 3.5}, -1.2),
            "rest": ({"tired": 3.0}, -1.2),
        }}),
}
#: Per kind: energy capacity, boldness (the chance a stroller takes the
#: shortcut), profit (coins a peddler earns by trading).
STATS = {PILGRIM: (12.0, 0.0, 0.0), PEDDLER: (10.0, 0.1, 2.0), BRIGAND: (12.0, 0.5, 0.0), KNIGHT: (12.0, 0.5, 0.0),
         RIVAL: (14.0, 0.4, 0.0)}

# ------------------------------------------------------------- the columns
CONTROLS = {"move_axis": "move-axis-x", "move_axis_y": "move-axis-y", "fire": "primary-action"}
KIND_COLUMNS = ("team", "speed", "sense", "hpmax", "dmg", "goalf0", "goalf1", "refugef", "mends")
PATH_COLUMNS = ("nodeidx", "nx", "ny", "na", "nb", "ntype", "ncost", "nstop", "dmat",
                *(f"pgd{i}" for i in range(NF)))
STATIC = ("lane", "kind", "alive", "vx", "vy", *KIND_COLUMNS, "emax", "bold", "profit", "jx", "jy", "camp0", *PATH_COLUMNS)
STATE = ("x", "y", "node", "energy", "hp", "coin", "kit", "down", "laps", "wins", "phase", "mot", "roll", "allow",
         "moved", "forkb", "forka", "lastnode", "born", "pile",
         "active", "hold", "rnd", "turns", "over", "score", "duels", "trades", "clock", "lives")
ARGUMENTS = (*STATIC, *STATE, *CONTROLS, "dt")


def boardwalk_step(*columns):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span (``dt`` a scalar)."""
    c = dict(zip(ARGUMENTS, columns))
    f = as_float
    n, m0, nn = ENT, MAP0, N
    dt = (c["x"] * 0.0 + c["dt"])[0:n]

    def e(name):                     # the entity lanes of a column
        return c[name][0:n]

    lane, kind, team, speed, sense = e("lane"), e("kind"), e("team"), e("speed"), e("sense")
    hpmax, dmg, mends, emax, bold, profit = e("hpmax"), e("dmg"), e("mends"), e("emax"), e("bold"), e("profit")
    jx, jy, camp0 = e("jx"), e("jy"), e("camp0")
    node, energy, hp, coin, kit, down = e("node"), e("energy"), e("hp"), e("coin"), e("kit"), e("down")
    laps, wins, phase, mot = e("laps"), e("wins"), e("phase"), e("mot")
    roll_o, allow_o, moved_o, forkb, forka = e("roll"), e("allow"), e("moved"), e("forkb"), e("forka")
    lastnode, born = e("lastnode"), e("born")
    active, hold, rnd, turns, over = e("active"), e("hold"), e("rnd"), e("turns"), e("over")
    zero = e("x") * 0.0
    lanes_total = int(c["nodeidx"].shape[0])

    is_player = kind == PILGRIM
    is_brigand = kind == BRIGAND
    up = down <= 0.0

    # --------------------------------------------------- the path as tables
    na_t, nb_t, nx_t, ny_t = c["na"], c["nb"], c["nx"], c["ny"]
    ntype_t, ncost_t, nstop_t, dmat_t = c["ntype"], c["ncost"], c["nstop"], c["dmat"]
    pgd = walkers.stack_columns([c[f"pgd{i}"] for i in range(NF)])

    def at(table, tile):             # a table's value at each lane's tile
        return walkers.gather(table, tile + float(m0))

    def row(v):
        return v.reshape((n, 1))

    def col(v):
        return v.reshape((1, n))

    def dfrom(a, b):                 # forward path length from tile a to tile b (matrices)
        return walkers.gather(dmat_t, a * float(nn) + b + float(m0))

    def field(field_id, tile):       # a goal field's distance at each lane's tile
        return walkers.field_at(pgd, field_id, tile + float(m0), lanes_total)

    other = row(lane) != col(lane)
    foe_mat = row(up) & col(up) & (row(team) != col(team))
    ally_mat = row(up) & col(up) & (row(team) == col(team)) & other

    # ------------------------------------------------ circumstances, felt
    d_fwd_all = dfrom(row(node), col(node))        # lane i's path length to lane j
    d_back_all = dfrom(col(node), row(node))       # lane j's path length to lane i
    dfwd = where(foe_mat, d_fwd_all, BIG).min(dim=1)
    dback = where(foe_mat, d_back_all, BIG).min(dim=1)
    dnear = where(dfwd < dback, dfwd, dback)
    afwd = where(ally_mat, d_fwd_all, BIG).min(dim=1)
    aback = where(ally_mat, d_back_all, BIG).min(dim=1)
    anear = where(afwd < aback, afwd, aback)
    crowd = (f(other & (col(node) == row(node))).sum(dim=1) * (1.0 / 3.0)).clip(0.0, 1.0)
    gfield = where(phase > 0.5, e("goalf1"), e("goalf0"))
    refuge = e("refugef")
    gdist = field(gfield, node)
    rdist = field(refuge, node)

    prox = (1.0 - dnear * (1.0 / (sense + 1e-6))).clip(0.0, 1.0)
    feats = {
        "bias": zero + 1.0,
        "foe_prox": prox,
        "foe_close": (1.0 - dnear * (1.0 / 2.5)).clip(0.0, 1.0),
        "foe_ahead": where(dfwd <= dback, prox, -prox),
        "ally_prox": (1.0 - anear * (1.0 / 4.0)).clip(0.0, 1.0),
        "goal_far": (gdist * (1.0 / 20.0)).clip(0.0, 1.0),
        "health": (hp * (1.0 / (hpmax + 1e-6))).clip(0.0, 1.0),
        "need": (1.0 - energy * (1.0 / (emax + 1e-6))).clip(0.0, 1.0),
        "crowd": crowd,
        "phase": phase,
        "busy": (coin * 0.1).clip(0.0, 1.0),
        "refuge_near": (1.0 - rdist * (1.0 / 10.0)).clip(0.0, 1.0),
    }
    choice = walkers.choose_motivation(KINDS, kind, feats, zero)

    # --------------------------------------------- whose turn it is, now
    hold_n = (hold - dt).clip(0.0, 1.0e6)
    go = (hold_n <= 0.0) & (over < 0.5) & (~is_player | (e("fire") > 0.0))
    act = (lane == active) & go & up
    any_act = f(act).sum()

    # The player's pad overrides its network: up runs, down rests.
    ax, ay = e("move_axis"), e("move_axis_y")
    m_eff = where(is_player & (ay > 0.5), zero + float(REST), where(is_player & (ay < -0.5), zero + float(FLEE), choice))

    def is_m(k):
        return m_eff == float(k)

    # ---------------------------------------- the roll and the allowance
    roll = (noise(lane * 5.3 + turns * 1.7 + 0.31) * 6.0).floor() + 1.0
    stride = (roll * speed + 0.5).floor().clip(1.0, 9.0)
    kit_run = f(is_m(FLEE) & (kit > 0.5))
    spend = where(is_m(WANDER), (stride * 0.5).floor() + 1.0,
            where(is_m(ATTACK), stride + 1.0,
            where(is_m(FLEE), stride + 3.0 + 2.0 * kit_run, stride)))
    draft = f((afwd >= 1.0) & (afwd <= 2.0))
    tired = f(energy < 3.0)
    allowance = where(is_m(REST), zero, (spend + draft - tired).clip(1.0, float(STEPS)))
    en0 = (energy + REGEN).clip(0.0, 1.0e6)
    en0 = where(en0 > emax, emax, en0)

    # ---------------------------------------------- the walk along the path
    is_pursuit = is_m(SEEK) | is_m(ATTACK)
    cur, rem, en = node, allowance, en0
    going = up & ~is_m(REST)
    steps, lapped, fk_b, fk_a = zero, zero, zero, zero
    for s in range(STEPS):
        tile_a, tile_b = at(na_t, cur), at(nb_t, cur)
        is_fork = tile_a != tile_b
        toward_a = where(foe_mat, dfrom(row(tile_a), col(node)), BIG).min(dim=1)
        toward_b = where(foe_mat, dfrom(row(tile_b), col(node)), BIG).min(dim=1)
        away_a = where(foe_mat, dfrom(col(node), row(tile_a)), BIG).min(dim=1)
        away_b = where(foe_mat, dfrom(col(node), row(tile_b)), BIG).min(dim=1)
        route_b = field(gfield, tile_b) < field(gfield, tile_a)
        hunt_b = toward_b < toward_a
        run_b = (away_b > away_a) | ((away_b == away_a) & (field(refuge, tile_b) < field(refuge, tile_a)))
        stroll_b = noise(lane * 3.1 + turns * 1.3 + (s + 1.0) * 7.7) < bold
        pick_b = where(is_m(WANDER), stroll_b, where(is_pursuit, hunt_b, where(is_m(FLEE), run_b, route_b)))
        pick_b = where(is_player & (ax.abs() > 0.5), ax > 0.5, pick_b)
        nxt = where(pick_b, tile_b, tile_a)
        cost = at(ncost_t, nxt)
        foe_there = f(foe_mat & (col(node) == row(nxt))).sum(dim=1) > 0.0
        can = going & (rem >= cost) & (en >= cost)
        cur = where(can, nxt, cur)
        rem = rem - f(can) * cost
        en = en - f(can) * cost
        steps = steps + f(can)
        lapped = lapped + f(can & (nxt == 0.0))
        fk_b = fk_b + f(can & is_fork & pick_b)
        fk_a = fk_a + f(can & is_fork & ~pick_b)
        going = can & (at(nstop_t, nxt) < 0.5) & ~foe_there
    fn = cur

    # --------------------------------------------------- landing: the tile
    t = at(ntype_t, fn)
    landed = steps > 0.5
    resting = is_m(REST)
    # treasure: the pile on the tile, and a kit when it is rich
    pile_here = at(c["pile"], fn)
    picks = (t == float(TREASURE)) & up & (landed | resting) & (pile_here > 0.5)
    gain = where(picks, pile_here, zero)
    kit1 = kit + f(picks & (pile_here >= 3.0) & (kit < 1.5))
    coin1 = coin + gain
    # the gate and the dens take a toll
    gate_toll = f((t == float(GATE)) & landed)
    paid = gate_toll * f(coin1 >= 1.0)
    coin1 = coin1 - paid
    en = (en - gate_toll * (1.0 - paid) * 2.0).clip(0.0, 1.0e6)       # no toll paid: two energy, never below none
    den_toll = f((t == float(DEN)) & landed & ~is_brigand) * (coin1.clip(0.0, 2.0))
    coin1 = coin1 - den_toll
    # a market trades for the lander that rests: coin for energy, then a kit
    at_market = resting & (t == float(MARKET))
    buys = at_market & (coin1 >= 3.0)
    coin1 = coin1 - 3.0 * f(buys)
    en = en + 5.0 * f(buys)
    coin1 = coin1 + profit * f(buys)
    kit_buy = buys & (coin1 >= 4.0) & (kit1 < 1.5)
    coin1 = coin1 - 4.0 * f(kit_buy)
    kit1 = kit1 + f(kit_buy)
    # rest mends: a camp refills and mends threefold
    at_camp = resting & (t == float(CAMP))
    en = where(at_camp, emax, en + 4.0 * f(resting))
    en = where(en > emax, emax, en)
    hp1 = hp + f(resting) * mends * (1.0 + 2.0 * f(at_camp))
    hp1 = where(hp1 > hpmax, hpmax, hp1)

    # ----------------------------------------------------------- the duel
    foe_here = foe_mat & (col(node) == row(fn))
    duelist = act & ~is_m(REST) & ~is_m(FLEE) & (f(foe_here).sum(dim=1) > 0.0)
    attacked = (f(duelist).reshape((n, 1)) * f(foe_here)).sum(dim=0) > 0.0
    target = first_of(attacked, lane)
    did = (f(duelist).sum() * f(target).sum()).clip(0.0, 1.0)
    die = (noise(lane * 9.7 + turns * 2.3 + 0.37) * 6.0).floor() + 1.0 + dmg
    kit_edge = f(is_m(ATTACK) & (kit1 > 0.5)) * 2.0
    mover_total = (f(act) * (die + 1.0 * f(is_m(ATTACK)) + kit_edge)).sum()
    target_total = (f(target) * die).sum()
    mover_loses = f(mover_total < target_total) + zero
    hit = 1.0 + ((mover_total - target_total).abs() * 0.5).floor()
    loser = where(mover_loses > 0.5, act, target)
    winner = where(mover_loses > 0.5, target, act)

    # ---------------------------------------------- bookkeeping of the lane
    did_f = did > 0.5
    at_goal = field(gfield, fn) < 0.5
    flip = at_goal & (resting | (t == float(TREASURE))) & (landed | resting)
    phase_n = where(act & flip, 1.0 - phase, phase)
    laps_n = laps + f(act) * lapped
    en_f = where(act, en, energy)
    hp_a, coin_a, kit_a, node_a = where(act, hp1, hp), where(act, coin1, coin), where(act, kit1, kit), where(act, fn, node)
    taken = ((f(loser) * coin_a).sum() * 0.5).floor() * did
    coin_n = coin_a - f(loser) * taken + f(winner) * taken
    hp_d = hp_a - f(loser) * hit * did
    knocked = did_f & loser & (hp_d <= 0.0)
    hp_n = where(knocked, hpmax * 0.5, hp_d)
    kit_n = kit_a - f(act & is_m(ATTACK) & (kit_a > 0.5)) * did
    node_n = where(knocked, camp0, node_a)
    down_n = where(knocked, zero + 2.0, down)
    wins_f = wins + f(winner) * did
    roll_n = where(act, roll, roll_o)
    allow_n = where(act, allowance, allow_o)
    moved_n = where(act, steps, moved_o)
    mot_n = where(act, m_eff, mot)
    forkb_n = forkb + f(act) * fk_b
    forka_n = forka + f(act) * fk_a
    last_n = where(act, node, lastnode)

    # ----------------------------------------- pass the turn to the next up
    cand = (lane > active) & (down_n <= 0.0)
    nxt_lane = where(cand, lane, 1.0e9).min()
    has_next = f(nxt_lane < 1.0e8)
    decayed = (down_n - 1.0).clip(0.0, 9.0)
    first_lane = where(decayed <= 0.0, lane, 1.0e9).min()
    next_active = has_next * nxt_lane + (1.0 - has_next) * first_lane
    new_round = (1.0 - has_next) * any_act
    down_final = where(new_round > 0.5, decayed, down_n)
    rnd_s = rnd.max() + new_round
    hold_after = where(is_player, zero + HOLD_PLAYER, zero + HOLD_NPC)
    hold_s = any_act * (f(act) * hold_after).sum() + (1.0 - any_act) * hold_n.max()
    active_s = any_act * next_active + (1.0 - any_act) * active.max()
    turns_s = turns.max() + any_act

    # ------------------------------------------- the treasure piles, mapped
    cells = c["pile"][m0:m0 + nn]
    tile_ids = c["nodeidx"][m0:m0 + nn]
    tile_type = ntype_t[m0:m0 + nn]
    picked = (tile_ids.reshape((nn, 1)) == col(fn)) & col(picks) & col(act)
    swept = f(picked).sum(dim=1) > 0.0
    cells_n = where(swept, cells * 0.0, (cells + new_round * f(tile_type == float(TREASURE))).clip(0.0, 5.0))
    pile_full = AbstractTensor.cat([c["pile"][0:m0], cells_n, c["pile"][m0 + nn:]], dim=0)

    racer = f(is_player | (kind == RIVAL))
    laps_max = (laps_n * racer).max()
    over_s = (over.max() + f(laps_max >= float(LAPS_TO_WIN)) + f(rnd_s >= float(MAX_ROUNDS))).clip(0.0, 1.0)
    cash = (f(is_player) * (coin_n + 15.0 * laps_n + 3.0 * wins_f)).sum()
    lead = f((f(is_player) * laps_n).sum() >= float(LAPS_TO_WIN)) * 30.0
    hero_hp = (f(is_player) * hp_n).sum()
    trades_d = f(buys & act).sum()

    # the board position of every lane: the tile's centre, spread so lanes on one tile show
    x_n = at(nx_t, node_n) + jx
    y_n = at(ny_t, node_n) + jy

    def wide(name, head):            # entity lanes' new values, then the map lanes (unchanged)
        return walkers.widen(head, c[name][n:])

    full = c["x"]

    def whole(name, value):          # a game-wide column: the same value on every lane
        return full * 0.0 + value

    return (
        wide("x", x_n), wide("y", y_n), wide("node", node_n), wide("energy", en_f), wide("hp", hp_n),
        wide("coin", coin_n), wide("kit", kit_n), wide("down", down_final), wide("laps", laps_n), wide("wins", wins_f),
        wide("phase", phase_n), wide("mot", mot_n), wide("roll", roll_n), wide("allow", allow_n), wide("moved", moved_n),
        wide("forkb", forkb_n), wide("forka", forka_n), wide("lastnode", last_n), wide("born", born),
        pile_full,
        whole("active", active_s), whole("hold", hold_s), whole("rnd", rnd_s), whole("turns", turns_s),
        whole("over", over_s), whole("score", cash + lead), c["duels"] + did, c["trades"] + trades_d,
        c["clock"] + c["dt"], whole("lives", hero_hp),
        full * 0.0 + f(down_final <= 0.0).sum(),
        full * 0.0 + 0.1,
    )


class BoardwalkPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "boardwalk_step"
    law = staticmethod(boardwalk_step)
    argument_names = ARGUMENTS
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")
    source_modules = ("meowdoku.walkers",)

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


# ------------------------------------------------------------ board building

@dataclass
class Board:
    """A generated track and everything precomputed on it."""

    seed: int
    pos: list[tuple[int, int]]            # cell of each tile
    na: np.ndarray
    nb: np.ndarray
    kinds: np.ndarray                     # tile type
    cost: np.ndarray
    dist: np.ndarray                      # (N, N) forward path length
    fields: list[np.ndarray]              # NF goal distance fields over the tiles
    camp: int
    starts: list[int]                     # starting tile of each entity lane


def _forward_distances(na: np.ndarray, nb: np.ndarray) -> np.ndarray:
    count = len(na)
    dist = np.full((count, count), BIG)
    for a in range(count):
        dist[a, a] = 0.0
        frontier, step = [a], 0
        while frontier:
            step += 1
            nxt = []
            for tile in frontier:
                for succ in {int(na[tile]), int(nb[tile])}:
                    if dist[a, succ] > step:
                        dist[a, succ] = step
                        nxt.append(succ)
            frontier = nxt
    return dist


def make_board(seed: int) -> Board:
    rng = np.random.default_rng(int(seed))
    ring = ([(x, 2) for x in range(2, 14)] + [(13, y) for y in range(3, 8)]
            + [(x, 8) for x in range(13, 1, -1)] + [(2, y) for y in range(7, 2, -1)])
    pos = ring + [(5, y) for y in range(3, 8)] + [(10, y) for y in range(3, 8)]
    na = np.array(list(range(1, 34)) + [0] + [35, 36, 37, 38, 25] + [40, 41, 42, 43, 20])
    nb = na.copy()
    na[3], nb[3] = 4, 34                 # the fork at the top of the x=5 shortcut
    na[8], nb[8] = 9, 39                 # the fork at the top of the x=10 shortcut
    kinds = np.zeros(N, dtype=int)
    kinds[0] = START
    market_east = int(rng.choice([14, 15]))
    treasure_east = int(rng.choice([12, 13]))
    den_top = int(rng.choice([5, 6, 7]))
    den_bottom = int(rng.choice([26, 27, 28]))
    camp_bottom = int(rng.choice([21, 22, 23]))
    for tile, kind in ((market_east, MARKET), (30 + int(rng.integers(0, 2)), MARKET), (treasure_east, TREASURE),
                       (36, TREASURE), (35, BOG), (37, BOG), (41, GATE), (den_top, DEN), (den_bottom, DEN),
                       (camp_bottom, CAMP), (32, CAMP)):
        kinds[tile] = kind
    cost = np.where(kinds == BOG, 2.0, 1.0)
    dist = _forward_distances(na, nb)
    fields = []
    for goal_kinds in ((MARKET,), (TREASURE,), (CAMP,), (DEN,), (START,)):
        targets = np.flatnonzero(np.isin(kinds, goal_kinds))
        fields.append(dist[:, targets].min(axis=1))
    starts = [0, market_east, 31, den_top, den_bottom, den_top + 1, 20, 1, 2]
    return Board(int(seed), pos, na, nb, kinds, cost, dist, fields, 32, starts)


def initial_columns(board: Board) -> dict[str, np.ndarray]:
    kind = np.full(LANES, float(MAP))
    for lanes, kid in (((0, P0), PILGRIM), ((P0, B0), PEDDLER), ((B0, K0), BRIGAND), ((K0, R0), KNIGHT), ((R0, ENT), RIVAL)):
        kind[lanes[0]:lanes[1]] = kid
    columns = {"lane": np.arange(LANES, dtype=np.float64), "kind": kind}
    columns.update({k: v for k, v in walkers.kind_columns(KINDS, kind).items() if k in KIND_COLUMNS})
    emax, bold, profit = np.zeros(LANES), np.zeros(LANES), np.zeros(LANES)
    for lane in range(ENT):
        emax[lane], bold[lane], profit[lane] = STATS[int(kind[lane])]
    jx, jy = np.zeros(LANES), np.zeros(LANES)
    for lane in range(ENT):
        jx[lane], jy[lane] = ((lane % 3) - 1) * 0.14, (((lane // 3) % 3) - 1) * 0.12
    node = np.zeros(LANES)
    node[:ENT] = board.starts
    px = np.zeros(LANES)
    py = np.zeros(LANES)
    nodeidx = np.zeros(LANES)
    ntype, ncost, nstop = np.zeros(LANES), np.zeros(LANES), np.zeros(LANES)
    na, nb, nx, ny = np.zeros(LANES), np.zeros(LANES), np.zeros(LANES), np.zeros(LANES)
    nodeidx[MAP0:MAP0 + N] = np.arange(N)
    nx[MAP0:MAP0 + N] = [p[0] + 0.5 for p in board.pos]
    ny[MAP0:MAP0 + N] = [p[1] + 0.5 for p in board.pos]
    na[MAP0:MAP0 + N], nb[MAP0:MAP0 + N] = board.na, board.nb
    ntype[MAP0:MAP0 + N], ncost[MAP0:MAP0 + N] = board.kinds, board.cost
    nstop[MAP0:MAP0 + N] = np.isin(board.kinds, STOPPING)
    dmat = np.zeros(LANES)
    dmat[MAP0:MAP0 + N * N] = board.dist.reshape(-1)
    columns.update({"emax": emax, "bold": bold, "profit": profit, "jx": jx, "jy": jy,
                    "camp0": np.full(LANES, float(board.camp)), "nodeidx": nodeidx, "nx": nx, "ny": ny, "na": na,
                    "nb": nb, "ntype": ntype, "ncost": ncost, "nstop": nstop, "dmat": dmat})
    for i, fld in enumerate(board.fields):
        col = np.zeros(LANES)
        col[MAP0:MAP0 + N] = fld
        columns[f"pgd{i}"] = col
    alive = np.zeros(LANES)
    alive[:ENT] = 1.0
    pile = np.zeros(LANES)
    pile[MAP0:MAP0 + N] = np.where(board.kinds == TREASURE, 3.0, 0.0)
    coin = np.zeros(LANES)
    coin[:ENT] = [{PILGRIM: 3.0, PEDDLER: 6.0}.get(int(k), 1.0) for k in kind[:ENT]]
    hp = columns["hpmax"].copy()
    energy = emax.copy()
    columns.update({
        "alive": alive, "vx": np.zeros(LANES), "vy": np.zeros(LANES),
        "x": px, "y": py, "node": node, "energy": energy, "hp": hp, "coin": coin, "kit": np.zeros(LANES),
        "down": np.zeros(LANES), "laps": np.zeros(LANES), "wins": np.zeros(LANES), "phase": np.zeros(LANES),
        "mot": np.zeros(LANES), "roll": np.zeros(LANES), "allow": np.zeros(LANES), "moved": np.zeros(LANES),
        "forkb": np.zeros(LANES), "forka": np.zeros(LANES), "lastnode": node.copy(), "born": np.zeros(LANES), "pile": pile,
        "active": np.zeros(LANES), "hold": np.full(LANES, HOLD_START), "rnd": np.zeros(LANES), "turns": np.zeros(LANES),
        "over": np.zeros(LANES), "score": np.zeros(LANES), "duels": np.zeros(LANES), "trades": np.zeros(LANES),
        "clock": np.zeros(LANES), "lives": np.full(LANES, PLAYER_HP),
        "move_axis": np.zeros(LANES), "move_axis_y": np.zeros(LANES), "fire": np.zeros(LANES),
    })
    for lane in range(ENT):
        columns["x"][lane] = board.pos[int(node[lane])][0] + 0.5 + jx[lane]
        columns["y"][lane] = board.pos[int(node[lane])][1] + 0.5 + jy[lane]
    return columns


# ------------------------------------------------------------------ the game

#: Mesh vocabulary beyond the shared tables (door, lair, ground come from Tile Quest
#: and the side scroller): name -> emoji, exported by ``web_export``.
EMOJI = {"stall": "\U0001F3EA", "chest": "\U0001F4B0", "camp": "⛺", "bog": "\U0001F7EB", "flag": "\U0001F3C1",
         "walker": "\U0001F6B6", "peddler": "\U0001F9F3", "brigand": "\U0001F977", "warden": "\U0001F482",
         "runner": "\U0001F3C3"}
EMOJI_SUBDIVISIONS: dict[str, int] = {}
#: tile type -> (mesh, colour); plain tiles are the path itself.
TILE_LOOK = {PLAIN: ("ground", "path"), MARKET: ("stall", "stall"), CAMP: ("camp", "camp"), TREASURE: ("chest", "chest"),
             BOG: ("bog", "bog"), GATE: ("door", "gate"), DEN: ("lair", "den"), START: ("flag", "flag")}
KIND_LOOK = {PILGRIM: ("walker", "walker", 0.46), PEDDLER: ("peddler", "peddler", 0.38), BRIGAND: ("brigand", "brigand", 0.4),
             KNIGHT: ("warden", "warden", 0.44), RIVAL: ("runner", "runner", 0.4)}
MOTIVE_WORDS = {WANDER: "strolled", FOLLOW: "followed the route", SEEK: "hunted", ATTACK: "fought", FLEE: "ran", REST: "rested"}


class Boardwalk(EntityGame):
    game = "boardwalk"
    title = "Boardwalk (turn race)"
    width, height = float(WIDTH), float(HEIGHT)
    CONTROLS = tuple(CONTROLS)
    HEALTH_MAX = PLAYER_HP
    SCENE = {"floor": True, "glass": False, "arcade": False}

    def __init__(self, board_id: str = "", aspect: float | None = None, seed: int | None = None):
        self.seed = int(time.time_ns() % 1000) if seed is None else int(seed)
        self.board = make_board(self.seed)
        self._tiles = None
        super().__init__(board_id)

    def make_piece(self):
        return BoardwalkPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.board)

    def reset(self) -> None:
        self.seed = (self.seed + 1) % 1000          # a new track each run
        self.board = make_board(self.seed)
        self._tiles = None
        super().reset()

    def palette(self) -> dict[str, list[float]]:
        return {"floor": [0.30, 0.52, 0.30], "path": [0.78, 0.68, 0.46], "stall": [0.92, 0.62, 0.30], "camp": [0.40, 0.70, 0.46],
                "chest": [0.98, 0.84, 0.25], "bog": [0.40, 0.30, 0.20], "gate": [0.56, 0.36, 0.20], "den": [0.58, 0.22, 0.44],
                "flag": [0.95, 0.95, 0.95], "walker": [0.52, 0.42, 0.96], "peddler": [0.96, 0.82, 0.62],
                "brigand": [0.85, 0.25, 0.25], "warden": [0.80, 0.82, 0.90], "runner": [0.30, 0.78, 0.90]}

    def tiles(self) -> list[dict[str, Any]]:
        """The track's tiles as flat entities (ids past the law's lanes)."""
        if self._tiles is None:
            out = []
            for k, (cx, cy) in enumerate(self.board.pos):
                mesh, color = TILE_LOOK[int(self.board.kinds[k])]
                out.append({"id": f"{10000 + k}:0", "x": cx + 0.5, "y": cy + 0.5, "vx": 0.0, "vy": 0.0, "mesh": mesh,
                            "color": color, "size": 0.42, "roll": 0.0, "lean": 0.0, "owner": -1, "age": 99.0, "spin": 0.0,
                            "bob": 0.0})
            self._tiles = out
        return self._tiles

    def entities(self) -> list[dict[str, Any]]:
        names = ("kind", "alive", "x", "y", "born", "down", "active")
        col = {name: self.session.column(name)[:ENT] for name in names}
        clock = float(self.session.column("clock")[0])
        active = int(self.session.column("active")[0])
        out = list(self.tiles())
        for k in np.flatnonzero(col["alive"] > 0.0):
            mesh, color, size = KIND_LOOK[int(col["kind"][k])]
            out_cold = col["down"][k] > 0.0
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": 0.0, "vy": 0.0, "mesh": mesh, "color": color, "size": size * (0.6 if out_cold else 1.0),
                        "roll": 0.0, "lean": 0.35, "owner": -1, "age": clock - float(col["born"][k]), "spin": 0.0,
                        "bob": 1.0 if k == active else 0.0})
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("over")[0] < 0.5:
            return None
        return {"score": float(s.column("score")[0]),
                "detail": f"{int(s.column('laps')[0])} laps, {int(s.column('coin')[0])} coins, {int(s.column('rnd')[0])} rounds"}

    def describe(self) -> str:
        s = self.session
        if s.state is None:
            return "Boardwalk"
        col = s.column
        you = int(col("active")[0]) == 0
        laps = col("laps")[:ENT]
        head = "FINISHED · R for a new track" if col("over")[0] > 0.5 else (
            "YOUR TURN: pad left/right picks the fork, up runs, down rests, space goes" if you else "waiting")
        last = f"last roll {int(col('roll')[0])} walked {int(col('moved')[0])}"
        return (f"{head} · round {int(col('rnd')[0])}/{MAX_ROUNDS} · laps {int(laps[0])}/{LAPS_TO_WIN} "
                f"(best rival {int(laps[R0:ENT].max())}) · hp {col('hp')[0]:.0f} energy {col('energy')[0]:.0f} "
                f"coin {int(col('coin')[0])} kit {int(col('kit')[0])} · {last} · duels {int(col('duels')[0])}")
