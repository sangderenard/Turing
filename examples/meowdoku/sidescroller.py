"""A procedural side-scrolling platformer ("runner") as one dt-system entity law.

The same engine as the climber (``platformer.py``), Space Invaders and the
bullet-hell scroller (``entities.py``): entity lanes with a fixed slot each, a
branch-free step law compiled to a NumPy product, the dt controller
subdividing each tick, and the player moved sideways by
``entity_motion.damped_axis`` (the law the page predicts with).  Gravity pulls
toward +y, the bottom of the screen; the world scrolls toward -x.

Lanes (contiguous blocks):
    0              the player (a frog); feet at ``y + PLAYER_RY``
    1..26          ground tiles: ONE LANE PER WORLD COLUMN of a sliding window.
                   A column that is a pit is a lane that is simply not alive.
    27..52         bricks, lane for lane above the ground lanes (the same
                   world column).  A brick is plain, or CONTAINS something
                   (``holds``: 1 coin, 2 mushroom, 3 star) until it is spent.
    53..56         items: a coin that pops, a mushroom or star that emerges
                   from a struck brick and slides along the ground
    57..58         walkers: reverse at edges, defeated by landing on them
    59..62         booms: the debris flash of a broken brick or a defeated walker

The player is integrated, never placed: semi-implicit gravity each step; when
the feet cross a tile's top from above the player stands (``vy = 0``); when
the head crosses a brick's underside while rising the brick is struck (the
nearest one under the head) and the player is bumped down.  A plain brick
breaks; a containing brick releases its content once and becomes a spent
block (``spent = 1``); a spent block only bumps.  ``fire`` is the jump
button: held on the ground it sets the upward velocity, released early it cuts
the rise (variable jump height).

The world scrolls: when the player runs past ``SCROLL_FRACTION`` of the field
width every lane's x shifts left by the overshoot (``distance`` and the score
grow by it).  Tiles that fall off the left edge are recycled ahead of the
right-most one; WHAT a world column holds (a pit, a brick, a brick's content)
is a pure function of its column number and the run's ``seed`` (``world_at``),
so the level is the same however the steps were cut.  Pits are 2-3 columns,
never twice in a row (at least six ground tiles between pits).  Falling below
the field ends the run.

Columns: lane, slot, rx, ry (static); alive, x, y, vx, vy, walk, born, holds,
spent, col, energy, score, lives, coins, distance, clock (state; ``vx`` of a
non-player lane is its DRAWN velocity, scroll plus ``walk``, its own);
field_w, seed (settings); move_axis, steer_x, fire (controls).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from meowdoku.entities import EntityGame, LawPiece, as_float, first_of, noise
from meowdoku.entity_motion import damped_axis
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

HEIGHT = 10.0
WIDTH = 9.0
#: The field follows the screen's shape within these width / height limits (a
#: phone held upright still gets a usable portrait view, a desktop a wide one).
ASPECT_MIN, ASPECT_MAX = 0.62, 1.8

PLAYER, GROUND, BRICK, ITEM, WALKER, BOOM_SLOT = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
GROUND_LANES = (1, 27)
BRICK_LANES = (27, 53)
TILES = (1, 53)
ITEM_LANES = (53, 57)
WALKER_LANES = (57, 59)
BOOM_LANES = (59, 63)
BODIES = (53, 59)
LANES = BOOM_LANES[1]
COLUMNS = GROUND_LANES[1] - GROUND_LANES[0]      # world columns in the window
FIRST_COLUMN = -2                                 # the leftmost column at round zero

#: Content of a brick (``holds``) and of an item lane (``holds`` = its kind).
NOTHING, COIN, MUSHROOM, STAR = 0.0, 1.0, 2.0, 3.0

GRAVITY = 28.0
JUMP = 15.0                                       # apex JUMP**2 / (2 GRAVITY) = 4.0 above the feet
JUMP_CUT = 4.0                                    # rising faster than this, fire released: cut to it
FALL_MAX = 20.0
BUMP = 2.0                                        # downward speed after the head strikes a brick
STOMP_BOUNCE = 9.0
SPEED = 8.0                                       # run: a jump covers 2 * JUMP / GRAVITY * SPEED = 8.6 units
TAU = 0.06
REACH = 0.2
MARGIN = 0.4
PLAYER_RX, PLAYER_RY = 0.3, 0.4
FOOT = 0.2                                        # half width of the feet that land (beyond a tile's own)
HEAD = 0.3                                        # half width of the head that strikes
SCROLL_FRACTION = 0.45
GONE_X = -1.6                                     # a tile centred left of this is off the screen
GROUND_TOP = HEIGHT - 1.0
GROUND_Y = GROUND_TOP + 0.5
LOW_Y = GROUND_TOP - 3.0                          # underside 2.5 above the floor, top 3.5: one jump
HIGH_Y = GROUND_TOP - 5.5                         # underside 5, top 6: reached from a low brick
PIT_CLEAR = 4                                     # no low brick this many columns either side of a pit
PIT_CHANCE, PIT_BLOCK = 0.55, 10.0
ROW_CHANCE, ROW_BLOCK = 0.8, 9.0
HIGH_CHANCE = 0.3
COIN_CHANCE, MUSHROOM_CHANCE, STAR_CHANCE = 0.18, 0.28, 0.33      # cumulative thresholds of one noise draw
ITEM_RX, ITEM_RY = 0.3, 0.3
ITEM_POP = 8.0
ITEM_SPEED = 2.5
COIN_LIFE = 0.5
BOOM = 0.35
WALKER_RX, WALKER_RY = 0.35, 0.35
WALKER_SPEED = 1.8
WALKER_CHANCE = 0.1
WALKER_FROM = 12.0

SHIELD_COST = 0.5
START_ENERGY = 0.5

POINTS_PER_UNIT = 10.0
COIN_POINTS, BREAK_POINTS, STOMP_POINTS = 100.0, 50.0, 200.0
MUSHROOM_POINTS, STAR_POINTS = 500.0, 1000.0

CONTROLS = {"move_axis": "move-axis-x", "steer_x": "pointer-board-x", "fire": "primary-action"}

#: Meshes this game needs beyond ``invaders.EMOJI`` (which has the frog, the
#: brick, the star and the boom): name -> emoji, exported by ``web_export``
#: through the same glyph builder.
EMOJI = {"ground": "🟫", "coin": "🟡", "mushroom": "🍄", "snail": "🐌"}
EMOJI_SUBDIVISIONS: dict[str, int] = {}


def _pit(col, seed):
    """Is world column ``col`` a pit (a function of the column and the seed)."""
    pb = (col * (1.0 / PIT_BLOCK)).floor()
    pu = col - pb * PIT_BLOCK
    start = 3.0 + (noise(pb * 3.7 + seed + 5.0) * 2.0).floor()
    width = 2.0 + (noise(pb * 2.9 + seed + 8.0) * 2.0).floor()
    return (noise(pb * 5.3 + seed + 1.0) < PIT_CHANCE) & (pb >= 1.0) & (pu >= start) & (pu < start + width)


def world_at(col, seed):
    """What world column ``col`` holds, as pure functions of the column number
    and the seed: ``(pit, brick, brick_y, holds)`` spans of ``col``'s shape.

    A pit block (``PIT_BLOCK`` columns) may carry one pit 2-3 columns wide that
    starts at column 3 or 4 of the block and ends by column 6, so two pits are
    always at least six ground tiles apart; the first block is solid.  A brick
    block (``ROW_BLOCK``) may carry one short row of 3-5 bricks, low or high;
    the first block is empty, and a low row keeps ``PIT_CLEAR`` columns from
    any pit.  A brick holds a coin, a mushroom or a star by
    one noise draw."""
    pit = _pit(col, seed)
    rb = (col * (1.0 / ROW_BLOCK)).floor()
    ru = col - rb * ROW_BLOCK
    first = 1.0 + (noise(rb * 4.3 + seed + 17.0) * 3.0).floor()
    length = 3.0 + (noise(rb * 2.7 + seed + 19.0) * 3.0).floor()
    row = ((noise(rb * 6.1 + seed + 13.0) < ROW_CHANCE) & (rb >= 1.0) & (ru >= first) & (ru < first + length))
    high = noise(rb * 1.9 + seed + 23.0) < HIGH_CHANCE
    # A low row would block the jump over a pit (the head strikes it on the
    # way up): none within PIT_CLEAR columns of one.  A high row clears the apex.
    pit_near = pit
    for d in range(-PIT_CLEAR, PIT_CLEAR + 1):
        pit_near = pit_near | _pit(col + float(d), seed)
    brick = row & (high | ~pit_near)
    brick_y = where(high, HIGH_Y, LOW_Y) + col * 0.0
    draw = noise(col * 4.7 + seed + 29.0)
    holds = where(draw < COIN_CHANCE, COIN, where(draw < MUSHROOM_CHANCE, MUSHROOM,
                                                  where(draw < STAR_CHANCE, STAR, NOTHING)))
    return pit, brick, brick_y, holds * as_float(brick)


def sidescroller_step(lane, slot, rx, ry,
                      alive, x, y, vx, vy, walk, born, holds, spent, col, energy,
                      score, lives, coins, distance, clock, field_w, seed,
                      move_axis, steer_x, fire, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    f = as_float
    some = lambda mask: f(mask).sum() > 0.0  # noqa: E731  (a scalar: does anything hold)
    n = int(lane.shape[0])
    t0, t1 = TILES
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_ground, is_brick = slot == PLAYER, slot == GROUND, slot == BRICK
    is_item, is_walker, is_boom = slot == ITEM, slot == WALKER, slot == BOOM_SLOT
    is_tile = is_ground | is_brick
    zero = x * 0.0
    clock_next = clock + dt

    # ------------------------------------------------------ the player, sideways
    ship_x, ship_vx = damped_axis(x, vx, move_axis, steer_x, zero + MARGIN, field_w - MARGIN,
                                  SPEED, TAU, REACH, dt)
    me = is_player & live & playing
    px, py, pvy = (f(me) * ship_x).sum(), (f(me) * y).sum(), (f(me) * vy).sum()
    has_me = some(me)
    held = (f(me) * fire).sum() > 0.0
    shield = (f(me) * energy).sum()

    # ------------------------------------------------------ the player, vertically
    # A rise cut short by releasing fire, then gravity.
    cut = where((pvy < -JUMP_CUT) & ~held, -JUMP_CUT, pvy)
    fall = (cut + GRAVITY * dt).clip(-FALL_MAX, FALL_MAX)
    py_new = py + fall * dt
    top, bottom = y - ry, y + ry
    near = (x - px).abs()
    # Standing: the feet cross a tile's top from above over the tile's width.
    support = (is_tile & live & has_me & (fall > 0.0) & (py + PLAYER_RY <= top + 1e-6)
               & (py_new + PLAYER_RY >= top) & (near < rx + FOOT))
    landed = some(support)
    stand = where(support, top, 1.0e9).min()
    # Striking: the head crosses a brick's underside while rising; the nearest
    # brick under the head takes it (the lowest lane breaks a tie).
    under = (is_brick & live & has_me & (fall < 0.0) & (py - PLAYER_RY >= bottom - 1e-6)
             & (py_new - PLAYER_RY <= bottom) & (near < rx + HEAD))
    rank = where(under, near + lane * 1e-6, 1.0e9)
    struck = under & (rank == rank.min())
    bumped = some(struck)
    under_y = (f(struck) * bottom).sum()
    py1 = where(landed, stand - PLAYER_RY, where(bumped, under_y + PLAYER_RY, py_new))
    pvy1 = where(landed, 0.0, where(bumped, BUMP, fall))
    pvy1 = where(landed & held, -JUMP, pvy1)

    # ------------------------------------------------------ what the strike does
    spent_hit = struck & (spent > 0.0)
    breaks = struck & (spent <= 0.0) & (holds <= 0.0)
    releases = struck & (spent <= 0.0) & (holds > 0.0)
    free_item = first_of(is_item & ~live, lane)
    can_release = some(free_item)
    release = releases & can_release
    sx, sy, sh = (f(struck) * x).sum(), (f(struck) * y).sum(), (f(struck) * holds).sum()
    spawn_item = free_item & some(release)
    coin_out = spawn_item & (sh == COIN)

    # ------------------------------------------------------ bodies: items and walkers
    # Gravity, then standing on a tile top; a coin is a pop that flies free.
    body = (is_item | is_walker) & live & playing
    solid = body & ~(is_item & (holds == COIN))
    vb = (vy + GRAVITY * dt).clip(-FALL_MAX, FALL_MAX)
    bx = x + walk * dt
    by = y + vb * dt
    tx = x[t0:t1].reshape((1, t1 - t0))
    ttop = top[t0:t1].reshape((1, t1 - t0))
    trx = rx[t0:t1].reshape((1, t1 - t0))
    tlive = live[t0:t1].reshape((1, t1 - t0))
    ttop_full = ttop + zero.reshape((n, 1))
    crossing = (tlive & (solid & (vb > 0.0)).reshape((n, 1))
                & ((y + ry).reshape((n, 1)) <= ttop + 1e-6) & ((by + ry).reshape((n, 1)) >= ttop)
                & ((bx.reshape((n, 1)) - tx).abs() < trx + rx.reshape((n, 1)) * 0.3))
    body_top = where(crossing, ttop_full, 1.0e9).min(dim=1)
    body_on = body_top < 1.0e8
    by1 = where(body_on, body_top - ry, by)
    vb1 = where(body_on, 0.0, vb)
    # A walker turns at the edge: no tile under the point just ahead of its feet.
    probe_x = (bx + where(walk >= 0.0, 1.0, -1.0) * (rx + 0.15)).reshape((n, 1))
    probe_y = (by1 + ry + 0.1).reshape((n, 1))
    ty = y[t0:t1].reshape((1, t1 - t0))
    try_ = ry[t0:t1].reshape((1, t1 - t0))
    ahead = f(tlive & ((probe_x - tx).abs() < trx) & ((probe_y - ty).abs() < try_)).sum(dim=1) > 0.0
    turn = is_walker & body & body_on & ~ahead
    walk1 = where(turn, -walk, walk)

    # ------------------------------------------------------ walkers meet the player
    w_live = is_walker & live & playing & has_me
    w_top = y - ry
    stomped = (w_live & (fall > 0.0) & (py + PLAYER_RY <= w_top + 0.2) & (py_new + PLAYER_RY >= w_top)
               & ((x - px).abs() < rx + PLAYER_RX))
    touching = (w_live & ~stomped & ((x - px).abs() < rx + 0.9 * PLAYER_RX)
                & ((y - py1).abs() < ry + 0.9 * PLAYER_RY))
    hurt = some(touching)
    absorbed = hurt & (shield >= SHIELD_COST)
    dies_hit = hurt & ~absorbed
    stomp = some(stomped)
    pvy1 = where(stomp, -STOMP_BOUNCE, pvy1)
    walker_gone = stomped | (touching & absorbed)

    # ------------------------------------------------------ pickups
    reach = (is_item & live & playing & has_me & (holds > COIN) & ((x - px).abs() < rx + PLAYER_RX)
             & ((y - py1).abs() < ry + PLAYER_RY))
    got_mushroom = some(reach & (holds == MUSHROOM))
    got_star = some(reach & (holds == STAR))
    energy_player = where(absorbed, shield - SHIELD_COST, where(got_mushroom, 1.0, shield))

    # ------------------------------------------------------ scroll
    overshoot = (px - SCROLL_FRACTION * field_w).clip(0.0, 100.0) * f(playing & has_me)
    scroll_v = overshoot / dt

    # ------------------------------------------------------ positions after the step
    x1 = where(is_player, where(me, px, x), where(body, bx, x)) - overshoot
    y1 = where(is_player, where(me, py1, y), where(body, by1, y))
    vy1 = where(is_player, where(me, pvy1, vy), where(body, vb1, 0.0))

    # ------------------------------------------------------ tiles: recycle ahead
    gone = is_tile & (x1 < GONE_X) & playing
    gone_ground, gone_brick = gone & is_ground, gone & is_brick

    def recycled(kind_mask, gone_kind):
        anchor_x = where(kind_mask & ~gone_kind, x1, -1.0e9).max()
        anchor_c = where(kind_mask & ~gone_kind, col, -1.0e9).max()
        step = gone_kind.to_dtype("float64").cumsum(dim=0)
        return anchor_x + step, anchor_c + step

    gx, gc = recycled(is_ground, gone_ground)
    bx_new, bc = recycled(is_brick, gone_brick)
    new_x = where(is_ground, gx, bx_new)
    new_col = where(is_ground, gc, bc)
    pit, brick, brick_y, brick_holds = world_at(new_col, seed)
    new_alive = where(is_ground, f(~pit), f(brick))
    new_y = where(is_ground, GROUND_Y, brick_y)

    # A walker may stand on the first fresh ground tile that is not a pit.
    candidate = (gone_ground & ~pit & (new_col >= WALKER_FROM)
                 & (noise(new_col * 2.3 + seed + 31.0) < WALKER_CHANCE))
    first_candidate = first_of(candidate, lane)
    free_walker = first_of(is_walker & ~live, lane)
    spawn_walker = free_walker & some(first_candidate)
    wx = (f(first_candidate) * new_x).sum()

    # ------------------------------------------------------ booms
    free_boom = is_boom & ~live
    boom_rank = free_boom.to_dtype("float64").cumsum(dim=0)
    boom_a, boom_b = free_boom & (boom_rank == 1.0), free_boom & (boom_rank == 2.0)
    broke = some(breaks)
    walked = some(walker_gone)
    wgx, wgy = (f(first_of(walker_gone, lane)) * x).sum(), (f(first_of(walker_gone, lane)) * y).sum()
    broke_x, broke_y = sx, sy
    # The first boom is the brick's, the second the walker's; with no brick
    # broken this step the walker takes the first.
    boom_walker = ((boom_b & broke) | (boom_a & ~broke)) & walked
    boom_brick = boom_a & broke

    # ------------------------------------------------------ assemble
    spawn_any = spawn_item | spawn_walker | boom_brick | boom_walker
    live1 = live & ~walker_gone & ~breaks & ~reach
    live1 = live1 & ~(is_item & ((holds == COIN) & ((clock_next - born) > COIN_LIFE)))
    live1 = live1 & ~(is_boom & ((clock_next - born) > BOOM))
    live1 = live1 & ~((is_item | is_walker) & ((y1 > HEIGHT + 1.0) | (x1 < GONE_X)))
    live2 = where(gone, new_alive > 0.0, live1)
    live2 = live2 | spawn_any
    x2 = where(gone, new_x, where(spawn_item, sx, where(spawn_walker, wx,
               where(boom_brick, broke_x, where(boom_walker, wgx, x1)))))
    y2 = where(gone, new_y, where(spawn_item, sy, where(spawn_walker, GROUND_TOP - WALKER_RY,
               where(boom_brick, broke_y, where(boom_walker, wgy, y1)))))
    col2 = where(gone, new_col, col)
    holds2 = where(gone, brick_holds, where(spawn_item, sh, holds))
    spent2 = where(gone, 0.0, where(release, 1.0, spent))
    vy2 = where(spawn_item, -ITEM_POP, where(spawn_any, 0.0, vy1))
    walk2 = where(spawn_item, where(sh > COIN, ITEM_SPEED, 0.0),
                  where(spawn_walker, -WALKER_SPEED, where(spawn_any, 0.0, walk1)))
    vx2 = where(is_player, ship_vx, walk2 - scroll_v)
    born2 = where(gone | spawn_any, clock, born)

    # ------------------------------------------------------ the run
    dead_fall = me & (y2 > HEIGHT + PLAYER_RY)
    dead = (is_player & live & playing) & (dead_fall | dies_hit)
    live3 = live2 & ~dead
    lives_next = lives - f(dead).sum()
    earned = (POINTS_PER_UNIT * overshoot + BREAK_POINTS * f(breaks).sum() + COIN_POINTS * f(coin_out).sum()
              + STOMP_POINTS * f(stomped).sum() + MUSHROOM_POINTS * f(got_mushroom) + STAR_POINTS * f(got_star))
    score_next = score + earned
    coins_next = coins + f(coin_out).sum()
    distance_next = distance + overshoot
    energy_next = where(is_player, energy_player, energy)

    # No step may carry anything through a tile's thickness, nor outrun the
    # sideways law's damping.
    window = 2.0 * where(is_ground, ry, 1.0e9).min()
    fastest = (where(live3, vy2.abs(), 0.0)).max() + 1e-6
    dt_limit = (0.9 * window / fastest).clip(0.0, 0.5 * TAU) + zero
    residual = f(live3 & is_tile).sum() + zero
    return (f(live3), x2, y2, vx2, vy2, walk2, born2, holds2, spent2, col2, energy_next,
            score_next, lives_next, coins_next, distance_next, clock_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "vx", "vy", "walk", "born", "holds", "spent", "col", "energy",
         "score", "lives", "coins", "distance", "clock")
SETTINGS = ("field_w", "seed")


class SidescrollerPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "sidescroller_step"
    law = staticmethod(sidescroller_step)
    argument_names = ("lane", "slot", "rx", "ry", *STATE, *SETTINGS, *CONTROLS, "dt")
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


def field_width(aspect: float | None) -> float:
    """The field's width for a screen of ``aspect`` (width / height): the
    height is fixed, the width follows the screen within ``[0.62, 1.8]``."""
    if not aspect or aspect <= 0.0:
        return WIDTH
    return float(np.clip(round(HEIGHT * float(np.clip(aspect, ASPECT_MIN, ASPECT_MAX)) * 2.0) / 2.0,
                         HEIGHT * ASPECT_MIN, HEIGHT * ASPECT_MAX))


def initial_columns(width: float = WIDTH, seed: int = 0) -> dict[str, np.ndarray]:
    seed_value = float(int(seed) % 1000)
    lane = np.arange(LANES, dtype=np.float64)
    slot = np.zeros(LANES)
    rx, ry = np.full(LANES, PLAYER_RX), np.full(LANES, PLAYER_RY)
    for (a, b), kind in ((GROUND_LANES, GROUND), (BRICK_LANES, BRICK)):
        slot[a:b], rx[a:b], ry[a:b] = kind, 0.5, 0.5
    slot[ITEM_LANES[0]:ITEM_LANES[1]] = ITEM
    rx[ITEM_LANES[0]:ITEM_LANES[1]], ry[ITEM_LANES[0]:ITEM_LANES[1]] = ITEM_RX, ITEM_RY
    slot[WALKER_LANES[0]:WALKER_LANES[1]] = WALKER
    rx[WALKER_LANES[0]:WALKER_LANES[1]], ry[WALKER_LANES[0]:WALKER_LANES[1]] = WALKER_RX, WALKER_RY
    slot[BOOM_LANES[0]:BOOM_LANES[1]] = BOOM_SLOT
    rx[BOOM_LANES[0]:BOOM_LANES[1]], ry[BOOM_LANES[0]:BOOM_LANES[1]] = 0.4, 0.4

    # The window of world columns: the same lane order for ground and bricks,
    # and what each column holds is the law's own ``world_at``.
    column = np.arange(FIRST_COLUMN, FIRST_COLUMN + COLUMNS, dtype=np.float64)
    pit, brick, brick_y, brick_holds = (np.asarray(v.numpy(), dtype=np.float64) for v in world_at(
        AbstractTensor.get_tensor(column), AbstractTensor.get_tensor(np.full(COLUMNS, seed_value))))
    x, y, alive = np.zeros(LANES), np.zeros(LANES), np.zeros(LANES)
    col, holds = np.zeros(LANES), np.zeros(LANES)
    g0, b0 = GROUND_LANES[0], BRICK_LANES[0]
    x[g0:g0 + COLUMNS], x[b0:b0 + COLUMNS] = column + 0.5, column + 0.5
    col[g0:g0 + COLUMNS], col[b0:b0 + COLUMNS] = column, column
    y[g0:g0 + COLUMNS], y[b0:b0 + COLUMNS] = GROUND_Y, brick_y
    alive[g0:g0 + COLUMNS] = 1.0 - pit
    alive[b0:b0 + COLUMNS] = brick
    holds[b0:b0 + COLUMNS] = brick_holds
    x[0], y[0], alive[0] = width * 0.25, GROUND_TOP - PLAYER_RY - 0.02, 1.0
    energy = np.zeros(LANES)
    energy[0] = START_ENERGY
    return {
        "lane": lane, "slot": slot, "rx": rx, "ry": ry,
        "alive": alive, "x": x, "y": y, "vx": np.zeros(LANES), "vy": np.zeros(LANES), "walk": np.zeros(LANES),
        "born": np.zeros(LANES), "holds": holds, "spent": np.zeros(LANES), "col": col, "energy": energy,
        "score": np.zeros(LANES), "lives": np.ones(LANES), "coins": np.zeros(LANES),
        "distance": np.zeros(LANES), "clock": np.zeros(LANES),
        "field_w": np.full(LANES, float(width)), "seed": np.full(LANES, seed_value),
        "move_axis": np.zeros(LANES), "steer_x": np.full(LANES, -1.0), "fire": np.zeros(LANES),
    }


class SideScroller(EntityGame):
    game = "side_scroller"
    title = "Runner (free board)"
    height = HEIGHT
    CONTROLS = tuple(CONTROLS)
    # A side view: the world stands on its own ground tiles, so no slab or
    # glass plate behind it.
    SCENE = {"floor": False, "glass": False, "arcade": True}
    # The scroll shifts the player in x with the world: never adopt x from the page.
    SCROLL_AXES = ("x",)
    PLAYER_LANE = 0
    HOSTILE_LANES = (WALKER_LANES,)

    def __init__(self, board_id: str = "", aspect: float | None = None, seed: int | None = None):
        self.width = field_width(aspect)
        self.seed = int(time.time_ns() % 1000) if seed is None else int(seed)
        super().__init__(board_id)

    def make_piece(self):
        return SidescrollerPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.width, self.seed)

    def motion(self) -> dict[str, Any]:
        return {"speed": SPEED, "tau": TAU, "reach": REACH, "x": [MARGIN, self.width - MARGIN]}

    def reset(self) -> None:
        self.seed = (self.seed + 1) % 1000          # a new level each run
        super().reset()

    def layout(self, aspect: float) -> None:
        """Fit the field to a new screen shape mid-game; the player's scroll
        line follows the width, the tiles keep their world positions."""
        width = field_width(aspect)
        if width == self.width:
            return
        self.session.column("field_w")[...] = width
        self.epoch += 1
        self.width = width

    def palette(self) -> dict[str, list[float]]:
        return {"ground": [0.35, 0.62, 0.25], "frog": [0.45, 0.9, 0.35], "brick": [0.75, 0.42, 0.25],
                "brick_full": [0.95, 0.72, 0.2], "brick_spent": [0.42, 0.36, 0.33],
                "coin": [1.0, 0.85, 0.2], "mushroom": [0.9, 0.25, 0.2], "star": [1.0, 0.92, 0.3],
                "walker": [0.55, 0.45, 0.7], "boom": [1.0, 0.55, 0.1]}

    def entities(self) -> list[dict[str, Any]]:
        names = ("slot", "alive", "x", "y", "vx", "vy", "born", "rx", "holds", "spent")
        col = {name: self.session.column(name) for name in names}
        clock = float(self.session.column("clock")[0])
        out = []
        for k in np.flatnonzero(col["alive"] > 0.0):
            slot, holds, spent = col["slot"][k], col["holds"][k], col["spent"][k]
            if slot == PLAYER:
                mesh, color, size = "frog", "frog", PLAYER_RY
            elif slot == GROUND:
                mesh, color, size = "ground", "ground", 0.5
            elif slot == BRICK:
                mesh, size = "brick", 0.5
                color = "brick_spent" if spent > 0.0 else "brick_full" if holds > 0.0 else "brick"
            elif slot == ITEM:
                mesh = {COIN: "coin", MUSHROOM: "mushroom"}.get(float(holds), "star")
                color, size = mesh, ITEM_RX
            elif slot == WALKER:
                mesh, color, size = "snail", "walker", WALKER_RX
            else:
                mesh, color, size = "boom", "boom", 0.4
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": float(col["vx"][k]),
                        "vy": float(col["vy"][k]) if slot != PLAYER else 0.0,
                        "mesh": mesh, "color": color, "size": size, "roll": 0.0,
                        "lean": 0.25 if slot == PLAYER else 0.55, "owner": -1,
                        "age": clock - float(col["born"][k]),
                        "spin": 1.5 if (slot == ITEM and holds in (COIN, STAR)) else 0.0, "bob": 0.0})
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]),
                "detail": f"ran {s.column('distance')[0]:.0f} m, {int(s.column('coins')[0])} coins"}

    def describe(self) -> str:
        s = self.session
        seconds, calls = s.last_round
        head = "GAME OVER · R to restart" if s.column("lives")[0] <= 0 else "run!"
        return (f"{head} · distance {s.column('distance')[0]:.0f} · coins {int(s.column('coins')[0])} · "
                f"score {int(s.column('score')[0])} · tick {seconds * 1000:.0f} ms = {calls} law steps")
