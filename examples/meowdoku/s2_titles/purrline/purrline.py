"""Purrline: a swap-and-match puzzle, one dt-system entity law over the grid cells.

An original match-three.  An 8 x 8 board of cats, fish, yarn, mice, bulbs and
stars: swap two neighbouring tiles; a swap that lines up three or more of a
kind in a row or column clears them, the tiles above fall, new tiles drop in
from above, and matches the fall makes clear again (a CASCADE; each cascade
step scores more).  A swap that matches nothing is refused (the two tiles nudge
toward each other and back, and no move is spent).

Three POWER PIECES form where a match is made (on the tile you swapped, else in
the middle of the run), keep their kind, and fire when they are next cleared:

    pounce    a 4-run: a paw; clears its whole row and column (a cross)
    hairball  a crossing of two runs (an L, T or +): bursts the 3 x 3 around it
    catnip    a 5-run: a sprig; swap it with any neighbour and every tile of
              that kind goes (swap two sprigs and the board goes); caught in
              another blast it clears the commonest kind

Blasts chain: a power piece caught in a blast fires too.

THE LAW.  Lane ``k`` is cell ``k`` (row-major, 8 columns).  One law call lands
``dt`` of animation time and takes AT MOST one board step, when nothing is
moving and no pause is running:

    mode 0  IDLE      the pointer (``fire`` + ``steer_x/y``) selects a tile; a
                      second tile beside it (a tap, or a drag) swaps them.  The
                      swap is scanned for a match: refused, or applied (the
                      tiles slide) and mode 1
    mode 1  RESOLVE   scan the board: run lengths by doubling gathers along the
                      four directions, power pieces made, blasts chained, points
                      awarded (10 per tile, 25 per power piece fired, x the
                      cascade depth); the cleared tiles flash (mode 2).  With no
                      match left: a level is won (score target), the game is
                      lost (no moves), the board is reshuffled (no legal move
                      left) or the player is handed the board again (mode 0)
    mode 2  POP       the flashing tiles go; every column compacts (a survivor
                      falls by the cleared cells below it: a suffix scan, then
                      a one-hot cell-by-cell match), new tiles are dealt above
                      (deterministic from the seed and a counter) and fall

Tiles are entities: a tile keeps its id (``tid``) while it falls and slides, so
the page animates it with the velocity the law publishes; a cleared tile leaves
(the page shrinks it); a new tile is a new id.

Columns: per cell ``kind spec tid ox oy flag act`` (the tile in a cell, its
power piece, its identity, the offset it still has to travel, whether it is
flashing, whether it is an armed catnip); GAME-WIDE (the same value on every
lane) ``mode wait score lscore moves target level combo counter seed sel
prevfire swa swb tkind quiet over clock fired cleared shuffles bestcombo``;
the CONTROLS ``steer_x steer_y fire`` and ``dt``.

AUTO SHUFFLE (``autoshuffle`` = 1, default; a page toggle): with no legal swap the occupied cells' kinds are permuted
(deterministically from ``seed``, ``shuffles`` and the attempt ``retry``), and a candidate with a ready-made match or
no legal swap is permuted again, up to MAX_SHUFFLE_TRIES; then, or with the toggle off, the game is over.

SAND MODE (``finite`` = 1; the page's "Finite sky stock" toggle).  A fixed STOCK of replenishment items
(``stock``, GAME-WIDE) replaces the endless refill.  At POP the survivors fall as ever (a hole counts as a
cleared cell), but a new item is only dealt while stock remains: each one consumes one, the lowest cells first,
and the cells left over stay EMPTY (``kind`` = -1; a hole).  A new item does not join its cell at once: it
is born ABOVE the board on its own plane (per cell ``sky_on sky_kind sky_tid sky_y sky_vy``: the slot of its
target cell, so the item is a column, a position ``sky_y`` and a velocity ``sky_vy`` in board units), falls at
``SKY_G`` and only when it reaches its cell does it LAND: it becomes the cell's tile (and so a scan,
a swap, a solid).  The board is busy while anything is in the air.  Empty cells match nothing, cannot be
swapped and are never counted.  There are no levels or move limit; the game is OVER when the board settles
with no legal swap left (out of stock, or out of tiles), and its result is the score.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from meowdoku.entities import EntityGame, LawPiece, as_float, noise
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

# ---------------------------------------------------------------- the board
W = H = 8
N = W * H
LANES = N
KINDS = 6                          # cat fish yarn mouse bulb star
NAMES = ("cat", "fish", "yarn", "mouse", "bulb", "star")
PAW, BALL, NIP = 1, 2, 3           # pounce, hairball, catnip (spec ids)
IDLE, RESOLVE, POP = 0, 1, 2
MOVE_V = 8.0                       # cells a second tiles slide and fall at
POP_T = 0.26                       # how long cleared tiles flash
SHUFFLE_T = 0.45
LEVEL_T = 0.7
POOL = 8000                        # tile identities wrap here (lanes of the socket are u16)
MOVES_PER_LEVEL = 20
TARGET0, TARGET_STEP = 400.0, 300.0
POINTS_TILE, POINTS_FIRED, POINTS_MADE, POINTS_LEFT = 10.0, 25.0, 20.0, 25.0
NUDGE = 0.32
EMPTY = -1.0                       # the kind of a hole
STOCK0 = 48                        # sand mode: replenishment items in a new game
SKY_G = 40.0                       # sand mode: board cells / s^2 a new item falls with
SKY_TOP = 1.0                      # sand mode: a new item is born this far above the board's top edge
SKY_GAP = 1.2                      # sand mode: spacing of the items stacked above one column (> 1 cell)
PLANE_Z = 0.9                      # sand mode: the sky plane's height above the board (toward the camera)
SAND_MOVES = 9999.0                # sand mode has no move limit
MAX_SHUFFLE_TRIES = 24             # auto shuffle: permutations tried before the game is declared over

CONTROLS = {"steer_x": "pointer-board-x", "steer_y": "pointer-board-y", "fire": "primary-action"}
STATIC = ("lane",)
CELL_STATE = ("kind", "spec", "tid", "ox", "oy", "flag", "act", "sky_on", "sky_kind", "sky_tid", "sky_y", "sky_vy")
GAME_STATE = ("mode", "wait", "score", "lscore", "moves", "target", "level", "combo", "counter", "seed", "sel",
              "prevfire", "swa", "swb", "tkind", "quiet", "over", "clock", "fired", "cleared", "shuffles",
              "bestcombo", "finite", "stock", "autoshuffle", "retry")
STATE = (*CELL_STATE, *GAME_STATE)
ARGUMENTS = (*STATIC, *STATE, *CONTROLS, "dt")


def purrline_step(*columns):
    """Advance the board by ``dt``.  Every operand is a (LANES,) span (``dt`` a scalar)."""
    c = dict(zip(ARGUMENTS, columns))
    f = as_float
    lane = c["lane"]
    zero = lane * 0.0
    dt = zero + c["dt"]
    kind, spec, tid, ox, oy, flag, act = (c[n] for n in CELL_STATE[:7])
    sky_on, sky_kind, sky_tid, sky_y, sky_vy = (c[n] for n in CELL_STATE[7:])
    mode, wait, score, lscore, moves, target, level = (c[n] for n in GAME_STATE[:7])
    combo, counter, seed, sel, prevfire, swa, swb = (c[n] for n in GAME_STATE[7:14])
    tkind, quiet, over, clock, fired_t, cleared_t, shuffles, bestcombo = (c[n] for n in GAME_STATE[14:22])
    finite, stock, autoshuffle, retry = c["finite"], c["stock"], c["autoshuffle"], c["retry"]
    steer_x, steer_y, fire = c["steer_x"], c["steer_y"], c["fire"]
    FALSE = zero > 0.5
    fin = finite > 0.5
    empty = kind < 0.0
    empty_f = f(empty)

    def pick(cond, a, b):                 # where() with plain numbers allowed on either side
        a = zero + a if isinstance(a, (int, float)) else a
        b = zero + b if isinstance(b, (int, float)) else b
        return where(cond, a, b)

    def g(source, index):                 # source[index]
        return source.index_select(0, index.long())

    row = ((lane + 0.5) * (1.0 / W)).floor()
    col = lane - row * W

    def nbr(direction, s):
        """The lane ``s`` cells away in a direction (itself where off the board) and whether it is on it."""
        if direction == "L":
            ok = col >= float(s)
            return where(ok, lane - float(s), lane), f(ok)
        if direction == "R":
            ok = col <= float(W - 1 - s)
            return where(ok, lane + float(s), lane), f(ok)
        if direction == "U":
            ok = row >= float(s)
            return where(ok, lane - float(s * W), lane), f(ok)
        ok = row <= float(H - 1 - s)
        return where(ok, lane + float(s * W), lane), f(ok)

    # ------------------------------------------------------------ animation
    # Offsets (a tile still travelling to its cell) shrink at MOVE_V; pauses run down.
    big = 1.0e6
    ox_d = pick(ox > 0.0, (ox - MOVE_V * dt).clip(0.0, big), pick(ox < 0.0, (ox + MOVE_V * dt).clip(-big, 0.0), zero))
    oy_d = pick(oy > 0.0, (oy - MOVE_V * dt).clip(0.0, big), pick(oy < 0.0, (oy + MOVE_V * dt).clip(-big, 0.0), zero))
    wait_d = (wait - dt).clip(0.0, big)
    busy = ((zero + (ox_d.abs() + oy_d.abs()).max()) > 1.0e-9) | ((zero + wait_d.max()) > 1.0e-9) \
        | ((zero + sky_on.max()) > 0.5)
    settled = ~busy
    playing = over < 0.5
    can_pick = (mode == float(IDLE)) & playing
    do_input = can_pick & settled
    do_resolve = (mode == float(RESOLVE)) & settled & playing
    do_pop = (mode == float(POP)) & settled & playing

    # ---------------------------------------------------- the pointer's cell
    inb = (fire > 0.5) & (steer_x >= 0.0) & (steer_y >= 0.0) & (steer_x < float(W)) & (steer_y < float(H))
    pcell = steer_y.floor().clip(0.0, float(H - 1)) * float(W) + steer_x.floor().clip(0.0, float(W - 1))
    press = inb & (prevfire < 0.5)
    sel_row = ((sel + 0.5) * (1.0 / W)).floor()
    sel_col = sel - sel_row * W
    p_row = ((pcell + 0.5) * (1.0 / W)).floor()
    p_col = pcell - p_row * W
    adjacent = ((sel_row - p_row).abs() + (sel_col - p_col).abs()) == 1.0
    a_cell, b_cell = sel, pcell
    is_a, is_b = (lane == a_cell), (lane == b_cell)
    empty_a = zero + (f(is_a) * empty_f).sum()                # a hole is never picked or swapped
    empty_b = zero + (f(is_b) * empty_f).sum()
    attempt = do_input & inb & (sel >= 0.0) & adjacent & (moves > 0.0) & ((empty_a + empty_b) < 0.5)
    swap_index = pick(is_a, b_cell, pick(is_b, a_cell, lane))

    # ------------------------------------------------- the swapped board
    wild = spec == float(NIP)
    km = pick(wild | empty, -1.0 - lane - 10.0, kind)    # wild tiles and holes are unique: they match nothing
    km_sw = g(km, swap_index)
    kind_sw, spec_sw, tid_sw = g(kind, swap_index), g(spec, swap_index), g(tid, swap_index)
    km_scan = pick(attempt, km_sw, km)
    sw_cells = f(attempt & (is_a | is_b)) + f((mode == float(RESOLVE)) & ((lane == swa) | (lane == swb)))

    # ------------------------------------------------------------ the scan
    def side(direction, board, flags):
        """(run length behind each cell along a direction, whether a flagged cell is in that stretch)."""
        i1, v1 = nbr(direction, 1)
        e1 = f(board == g(board, i1)) * v1
        n = 1.0 + e1
        flood = (flags + e1 * g(flags, i1)).clip(0.0, 1.0)
        reach = []
        for s in (2, 4):
            i, v = nbr(direction, s)
            e = f(board == g(board, i)) * v
            n = n + f(n == float(s)) * e * g(n, i) * v
            reach.append((s, i, v))
        for s, i, v in reach:
            flood = (flood + f(n > float(s)) * g(flood, i) * v).clip(0.0, 1.0)
        return n, flood

    left, s_left = side("L", km_scan, sw_cells)
    right, s_right = side("R", km_scan, sw_cells)
    up, s_up = side("U", km_scan, sw_cells)
    down, s_down = side("D", km_scan, sw_cells)
    h_len, v_len = left + right - 1.0, up + down - 1.0
    in_h, in_v = h_len >= 3.0, v_len >= 3.0
    matched = in_h | in_v
    swapped_here = sw_cells > 0.5
    h_has_swap, v_has_swap = (s_left + s_right) > 0.5, (s_up + s_down) > 0.5
    # The tile a power piece forms on: the swapped one if the run holds it, else the run's middle.
    mid_h = left == ((h_len + 1.0) * 0.5).floor()
    mid_v = up == ((v_len + 1.0) * 0.5).floor()
    make_h = (h_has_swap & swapped_here) | (~h_has_swap & mid_h)
    make_v = (v_has_swap & swapped_here) | (~v_has_swap & mid_v)
    makes_nip = (in_h & (h_len >= 5.0) & make_h) | (in_v & (v_len >= 5.0) & make_v)
    makes_ball = in_h & in_v & ~makes_nip
    makes_paw = ((in_h & (h_len >= 4.0) & (h_len < 5.0) & make_h) | (in_v & (v_len >= 4.0) & (v_len < 5.0) & make_v)) \
        & ~makes_nip & ~makes_ball
    makes = makes_nip | makes_ball | makes_paw

    # ----------------------------------------------------- blasts, chained
    kreal = kind
    counts = [f(kreal == float(k)).sum() for k in range(KINDS)]
    common, best = zero, zero + counts[0]
    for k in range(1, KINDS):
        better = (zero + counts[k]) > best
        common = pick(better, float(k), common)
        best = pick(better, zero + counts[k], best)
    target_kind = pick(tkind >= 0.0, tkind, common)
    t_all = f(matched) + act
    t_all = t_all.clip(0.0, 1.0)
    for _ in range(4):
        tp, th, tn = t_all * f(spec == float(PAW)), t_all * f(spec == float(BALL)), t_all * f(spec == float(NIP))
        grid = tp.reshape((H, W))
        hit_p = f((g(grid.sum(dim=1), row) + g(grid.sum(dim=0), col)) > 0.5)
        il, vl = nbr("L", 1)
        ir, vr = nbr("R", 1)
        iu, vu = nbr("U", 1)
        idn, vd = nbr("D", 1)
        wide = (th + g(th, il) * vl + g(th, ir) * vr).clip(0.0, 1.0)
        hit_b = (wide + g(wide, iu) * vu + g(wide, idn) * vd).clip(0.0, 1.0)
        hit_n = f(((zero + tn.sum()) > 0.5) & ((kreal == target_kind) | (target_kind > 90.0)))
        t_all = (t_all + hit_p + hit_b + hit_n).clip(0.0, 1.0)
    cleared_now = t_all * (1.0 - f(makes)) * (1.0 - empty_f)    # the power pieces being made stay; a hole is not a tile
    n_clear = cleared_now.sum()
    n_fired = (cleared_now * f(spec > 0.5)).sum()
    n_made = f(makes).sum()
    any_clear = (zero + n_clear) > 0.5

    # ----------------------------------------------- is there a legal swap?
    kq = pick(wild, -777.0, pick(empty, -999.0, kind))
    nb_real = zero
    for d in ("L", "R", "U", "D"):
        i, v = nbr(d, 1)
        nb_real = nb_real + v * (1.0 - g(empty_f, i))
    kn = {}
    for d in ("L", "R", "U", "D"):
        for s in (1, 2):
            i, v = nbr(d, s)
            kn[(d, s)] = pick(v > 0.5, g(kq, i), -888.0)
    any_move = FALSE
    for donor in ("L", "R", "U", "D"):
        k = kn[(donor, 1)]
        place = (k >= 0.0) & (kq != k) & (kq > -998.0)

        def same(d, s, donor=donor, k=k):
            return FALSE if d == donor else (kn[(d, s)] == k)

        line_h = (same("L", 1) & same("L", 2)) | (same("L", 1) & same("R", 1)) | (same("R", 1) & same("R", 2))
        line_v = (same("U", 1) & same("U", 2)) | (same("U", 1) & same("D", 1)) | (same("D", 1) & same("D", 2))
        any_move = any_move | (place & (line_h | line_v))
    legal = ((zero + f(any_move).sum()) > 0.5) | ((zero + (f(wild) * f(nb_real > 0.5)).sum()) > 0.5)

    # --------------------------------------------- IDLE: the swap, applied
    nip_a = (zero + (f(is_a) * f(wild)).sum()) > 0.5
    nip_b = (zero + (f(is_b) * f(wild)).sum()) > 0.5
    kind_a, kind_b = (f(is_a) * kind).sum(), (f(is_b) * kind).sum()
    valid = ((zero + f(matched).sum()) > 0.5) | nip_a | nip_b
    apply_swap = attempt & valid
    refuse = attempt & ~valid
    new_target = pick(nip_a & nip_b, 99.0, pick(nip_a, zero + kind_b, zero + kind_a))
    ox_s = g(col, swap_index) - col
    oy_s = g(row, swap_index) - row
    act_s = f(spec_sw == float(NIP)) * f(is_a | is_b)
    sel_i = pick(can_pick, pick(attempt, -1.0, pick(press & (empty_b < 0.5), pcell, sel)), sel)

    # --------------------------------------------- RESOLVE: mark the match
    trial = retry > 0.5                                    # a shuffle's candidate board is being checked
    matches_present = (zero + f(matched).sum()) > 0.5
    marking = do_resolve & any_clear & ~trial
    stage = do_resolve & ~any_clear
    pts = (n_clear * POINTS_TILE + n_fired * POINTS_FIRED + n_made * POINTS_MADE) * (combo + 1.0) * (1.0 - quiet)
    spec_m = pick(makes_nip, float(NIP), pick(makes_ball, float(BALL), pick(makes_paw, float(PAW), spec)))
    tid_m_new = counter + lane
    tid_m_new = tid_m_new - (tid_m_new * (1.0 / POOL)).floor() * POOL
    tid_m = pick(makes, tid_m_new, tid)
    flag_m = cleared_now

    # --------------------------------------------- RESOLVE: nothing to clear
    level_up = stage & (lscore >= target) & ~fin
    lost = stage & ~level_up & (moves <= 0.5)
    # AUTO SHUFFLE: no legal swap permutes the occupied cells' kinds (seed, shuffle count and attempt decide the
    # permutation); a candidate with a ready-made match or no legal swap is permuted again, at most
    # MAX_SHUFFLE_TRIES times, then the game is over.  Switched off, or with no way out: over.
    first_bad = stage & ~level_up & ~lost & ~legal & ~trial
    trial_bad = do_resolve & trial & (matches_present | ~legal) & ~lost
    want = (first_bad & (autoshuffle > 0.5)) | trial_bad
    in_bound = retry < float(MAX_SHUFFLE_TRIES)
    shuffle = want & in_bound
    dead = (first_bad & ~(autoshuffle > 0.5)) | (want & ~in_bound)
    hand_back = stage & ~level_up & ~lost & legal
    level_n = level + f(level_up)
    target_next = TARGET0 + TARGET_STEP * (level_n - 1.0)
    cnt_mod = counter - (counter * (1.0 / 4096.0)).floor() * 4096.0
    tid_r = counter + lane
    tid_r = tid_r - (tid_r * (1.0 / POOL)).floor() * POOL

    # ----------------------------------------- POP: gravity and refill
    gone = flag + empty_f                                  # flashing cells and holes (sand mode) are not survivors
    below = gone
    for s in (1, 2, 4):
        i, v = nbr("D", s)
        below = below + g(below, i) * v
    drop = below - gone                                    # cleared cells under each cell, same column
    survive = 1.0 - gone
    as_target = lambda v: v.reshape((N, 1))
    as_source = lambda v: v.reshape((1, N))
    lands = f((as_target(col) == as_source(col)) & ((as_source(row) + as_source(drop)) == as_target(row))) * as_source(survive)

    def take(v):
        return (lands * as_source(v)).sum(dim=1)

    live = 1.0 - empty_f
    occupied_before = (f(as_source(lane) < as_target(lane)) * as_source(live)).sum(dim=1)      # index among occupied cells
    shuffle_key = noise((lane + 1.0) * 0.6180339 + seed * 7.123 + shuffles * 13.37 + retry * 3.71 + 0.5)
    shuffle_rank = ((f(as_source(shuffle_key) < as_target(shuffle_key))
                     + f(as_source(shuffle_key) == as_target(shuffle_key)) * f(as_source(lane) < as_target(lane)))
                    * as_source(live)).sum(dim=1)
    gives = f(as_source(shuffle_rank) == as_target(occupied_before)) * as_source(live) * as_target(live)
    kind_r = pick(empty, kind, (gives * as_source(kind)).sum(dim=1))   # the permuted kinds; holes stay holes

    has_src = lands.sum(dim=1) > 0.5
    fresh_cell = ~has_src
    rank = (f(as_source(lane) < as_target(lane)) * as_source(f(fresh_cell))).sum(dim=1)
    new_id = counter + rank
    cnt_p = new_id - (new_id * (1.0 / 4096.0)).floor() * 4096.0
    kind_new = (noise(cnt_p * 0.6180339 + seed * 7.123) * float(KINDS)).floor().clip(0.0, float(KINDS - 1))
    tid_new = new_id - (new_id * (1.0 / POOL)).floor() * POOL
    column_total = g(below, col)                           # flashing cells in the column (the sum seen from row 0)
    fresh_f = f(fresh_cell)
    # Sand mode: the fresh cells take the stock lowest-first (priority: lower row, then lower column); an item
    # is born in the sky above its column (stacked SKY_GAP apart, the lowest first) and the cell stays a hole
    # until it lands.  A cell with no item left stays a hole for good.
    priority = ((f(as_source(row) > as_target(row)) + f(as_source(row) == as_target(row))
                 * f(as_source(lane) < as_target(lane))) * as_source(fresh_f)).sum(dim=1)
    alloc = fresh_cell & fin & (priority < stock)
    depth = (f(as_target(col) == as_source(col)) * f(as_source(row) > as_target(row)) * as_source(fresh_f)).sum(dim=1)
    n_alloc = f(alloc).sum()
    kind_p = pick(has_src, take(kind), pick(fin, EMPTY, kind_new))
    spec_p = pick(has_src, take(spec), zero)
    tid_p = pick(has_src, take(tid), tid_new)
    oy_p = pick(has_src, -take(drop), pick(fin, 0.0, -column_total))
    n_new = f(fresh_cell).sum()
    sky_on_p = f(alloc)
    sky_kind_p = pick(alloc, kind_new, sky_kind)
    sky_tid_p = pick(alloc, tid_new, sky_tid)
    sky_y_p = pick(alloc, -SKY_TOP - SKY_GAP * depth, sky_y)

    # ---------------------------------------- the sky: items falling to their cells
    air = sky_on > 0.5
    sky_vy1 = sky_vy + SKY_G * dt
    sky_y1 = sky_y + sky_vy1 * dt
    cell_y = row + 0.5
    land = air & (sky_y1 >= cell_y)

    # ------------------------------------------------------- the next state
    kind_n = pick(land, sky_kind, pick(do_pop, kind_p, pick(apply_swap, kind_sw, pick(shuffle, kind_r, kind))))
    spec_n = pick(do_pop, spec_p, pick(apply_swap, spec_sw, pick(marking, spec_m, spec)))
    tid_n = pick(land, sky_tid, pick(do_pop, tid_p, pick(apply_swap, tid_sw, pick(marking, tid_m, pick(shuffle, tid_r, tid)))))
    ox_n = pick(apply_swap, ox_s, pick(refuse, ox_s * NUDGE, pick(do_pop | shuffle, zero, ox_d)))
    oy_n = pick(apply_swap, oy_s, pick(refuse, oy_s * NUDGE, pick(do_pop, oy_p, pick(shuffle, zero, oy_d))))
    flag_n = pick(do_pop, zero, pick(marking, flag_m, flag))
    act_n = pick(do_pop, zero, pick(apply_swap, act_s, act))

    mode_n = pick(apply_swap, float(RESOLVE), pick(marking, float(POP), pick(do_pop, float(RESOLVE),
             pick(shuffle | level_up, float(RESOLVE), pick(lost | hand_back, float(IDLE), mode)))))
    wait_n = pick(marking, POP_T, pick(shuffle & ~trial, SHUFFLE_T, pick(level_up, LEVEL_T, wait_d)))
    combo_n = pick(apply_swap, zero, pick(marking, combo + 1.0, pick(hand_back, zero, combo)))
    quiet_n = pick(apply_swap, zero, pick(shuffle, 1.0, pick(hand_back, zero, quiet)))
    bonus = moves * POINTS_LEFT * f(level_up)
    score_n = score + pick(marking, zero + pts, zero) + bonus
    lscore_n = pick(level_up, zero, lscore + pick(marking, zero + pts, zero))
    moves_n = pick(level_up, float(MOVES_PER_LEVEL), moves - f(apply_swap))
    target_n = pick(level_up, target_next, target)
    counter_n = pick(do_pop, counter + n_new, pick(shuffle, counter + float(N), pick(marking, counter + float(N), counter)))
    sel_n = pick(hand_back | level_up, -1.0, sel_i)
    prevfire_n = f(fire > 0.5)
    swa_n = pick(apply_swap, a_cell, pick(marking, -1.0, swa))
    swb_n = pick(apply_swap, b_cell, pick(marking, -1.0, swb))
    tkind_n = pick(apply_swap, new_target, pick(marking, -1.0, tkind))
    over_n = pick(lost | dead, 1.0, over)
    sky_on_n = pick(do_pop, sky_on_p, pick(land, 0.0, sky_on))
    sky_kind_n = pick(do_pop, sky_kind_p, sky_kind)
    sky_tid_n = pick(do_pop, sky_tid_p, sky_tid)
    sky_y_n = pick(do_pop, sky_y_p, pick(land, cell_y, pick(air, sky_y1, sky_y)))
    sky_vy_n = pick(do_pop, zero, pick(land, 0.0, pick(air, sky_vy1, sky_vy)))
    stock_n = stock - pick(do_pop, zero + n_alloc, zero)
    fired_n = fired_t + pick(marking, zero + n_fired, zero)
    cleared_n = cleared_t + pick(marking, zero + n_clear, zero)
    shuffles_n = shuffles + f(shuffle & ~trial)
    retry_n = pick(shuffle, retry + 1.0, pick(do_resolve & trial & ~trial_bad, zero, retry))
    best_n = pick(marking & ((combo + 1.0) > bestcombo), combo + 1.0, bestcombo)

    return (
        kind_n, spec_n, tid_n, ox_n, oy_n, flag_n, act_n, sky_on_n, sky_kind_n, sky_tid_n, sky_y_n, sky_vy_n,
        mode_n, wait_n, score_n, lscore_n, moves_n, target_n, level_n, combo_n, counter_n, seed, sel_n,
        prevfire_n, swa_n, swb_n, tkind_n, quiet_n, over_n, clock + dt, fired_n, cleared_n, shuffles_n, best_n,
        finite, stock_n, autoshuffle, retry_n,
        zero,
        zero + 0.05,
    )


class PurrlinePiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "purrline_step"
    law = staticmethod(purrline_step)
    argument_names = ARGUMENTS
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


# ------------------------------------------------------------ board dealing

def deal(seed: int) -> np.ndarray:
    """A settled board (no run of three) with at least one legal swap, from a seed."""
    rng = np.random.default_rng(int(seed))
    while True:
        kind = np.zeros((H, W), dtype=int)
        for r in range(H):
            for c in range(W):
                choices = [k for k in range(KINDS)
                           if not (c >= 2 and kind[r, c - 1] == k and kind[r, c - 2] == k)
                           and not (r >= 2 and kind[r - 1, c] == k and kind[r - 2, c] == k)]
                kind[r, c] = int(rng.choice(choices))
        if legal_swaps(kind):
            return kind.reshape(-1)


def runs_of(kind: np.ndarray, spec: np.ndarray | None = None) -> np.ndarray:
    """Cells in a run of three or more (the plain reference the law is checked against)."""
    grid = np.asarray(kind).reshape(H, W).astype(int).copy()
    if spec is not None:
        wild = np.asarray(spec).reshape(H, W) == NIP
        grid[wild] = -1 - np.arange(N).reshape(H, W)[wild] - 10
    hole = np.asarray(kind).reshape(H, W) < 0                  # a hole (sand mode) matches nothing
    grid[hole] = -1000 - np.arange(N).reshape(H, W)[hole]
    hit = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W - 2):
            if grid[r, c] == grid[r, c + 1] == grid[r, c + 2]:
                hit[r, c:c + 3] = True
    for r in range(H - 2):
        for c in range(W):
            if grid[r, c] == grid[r + 1, c] == grid[r + 2, c]:
                hit[r:r + 3, c] = True
    return hit.reshape(-1)


def legal_swaps(kind: np.ndarray, spec: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Every neighbouring pair whose swap makes a match (or holds a catnip), by brute force."""
    flat = np.asarray(kind).reshape(-1).copy()
    sp = np.zeros(N, dtype=int) if spec is None else np.asarray(spec).reshape(-1).astype(int)
    out = []
    for a in range(N):
        r, c = divmod(a, W)
        for b in ((a + 1) if c + 1 < W else None, (a + W) if r + 1 < H else None):
            if b is None:
                continue
            if flat[a] < 0 or flat[b] < 0:                    # a hole is never swapped
                continue
            if sp[a] == NIP or sp[b] == NIP:
                out.append((a, b))
                continue
            flat[a], flat[b] = flat[b], flat[a]
            sp[a], sp[b] = sp[b], sp[a]
            if runs_of(flat, sp).any():
                out.append((a, b))
            flat[a], flat[b] = flat[b], flat[a]
            sp[a], sp[b] = sp[b], sp[a]
    return out


def initial_columns(seed: int, sand: bool = False, stock: int = STOCK0, autoshuffle: bool = True) -> dict[str, np.ndarray]:
    kind = deal(seed).astype(np.float64)

    def whole(value: float) -> np.ndarray:
        return np.full(LANES, float(value))

    columns = {"lane": np.arange(LANES, dtype=np.float64), "kind": kind, "spec": np.zeros(LANES),
               "tid": 100.0 + np.arange(LANES, dtype=np.float64), "ox": np.zeros(LANES), "oy": np.zeros(LANES),
               "flag": np.zeros(LANES), "act": np.zeros(LANES), "sky_on": np.zeros(LANES), "sky_kind": np.zeros(LANES),
               "sky_tid": np.zeros(LANES), "sky_y": np.zeros(LANES), "sky_vy": np.zeros(LANES)}
    columns.update({"mode": whole(RESOLVE), "wait": whole(0.0), "score": whole(0.0), "lscore": whole(0.0),
                    "moves": whole(SAND_MOVES if sand else MOVES_PER_LEVEL), "target": whole(TARGET0), "level": whole(1.0), "combo": whole(0.0),
                    "counter": whole(1000.0), "seed": whole(float(int(seed) % 997)), "sel": whole(-1.0),
                    "prevfire": whole(0.0), "swa": whole(-1.0), "swb": whole(-1.0), "tkind": whole(-1.0),
                    "quiet": whole(0.0), "over": whole(0.0), "clock": whole(0.0), "fired": whole(0.0),
                    "cleared": whole(0.0), "shuffles": whole(0.0), "bestcombo": whole(0.0),
                    "autoshuffle": whole(1.0 if autoshuffle else 0.0), "retry": whole(0.0),
                    "finite": whole(1.0 if sand else 0.0), "stock": whole(float(stock) if sand else 0.0),
                    "steer_x": whole(-1.0), "steer_y": whole(-1.0), "fire": whole(0.0)})
    return columns


# ------------------------------------------------------------------ the game

#: Mesh vocabulary: name -> emoji, exported by ``web_export``.
EMOJI = {"pl_cat": "\U0001F431", "pl_fish": "\U0001F41F", "pl_yarn": "\U0001F9F6", "pl_mouse": "\U0001F42D",
         "pl_bulb": "\U0001F4A1", "pl_star": "⭐", "pl_paw": "\U0001F43E", "pl_hairball": "\U0001F300",
         "pl_catnip": "\U0001F33F", "pl_cursor": "⭕"}
EMOJI_SUBDIVISIONS: dict[str, int] = {}
KIND_MESH = tuple(f"pl_{name}" for name in NAMES)
SPEC_MESH = {PAW: "pl_paw", BALL: "pl_hairball", NIP: "pl_catnip"}
TILE_SIZE, POWER_SIZE = 0.36, 0.45


class Purrline(EntityGame):
    game = "purrline"
    title = "Purrline (match three)"
    width, height = float(W), float(H)
    CONTROLS = tuple(CONTROLS)
    SCENE = {"floor": True, "glass": False, "arcade": False}

    #: Selectable modes (the page shows a toggle): SAND is the finite sky stock (see the module docstring).
    MODES = {"autoshuffle": "Auto shuffle", "sand": "Finite sky stock"}

    def __init__(self, board_id: str = "", aspect: float | None = None, seed: int | None = None,
                 sand: bool = False, stock: int = STOCK0):
        self.seed = int(time.time_ns() % 997) if seed is None else int(seed)
        self.sand, self.stock0 = bool(sand), int(stock)
        self.autoshuffle = True
        self._cells = None
        super().__init__(board_id)

    def modes(self) -> dict[str, bool]:
        return {"autoshuffle": self.autoshuffle, "sand": self.sand}

    def set_mode(self, name: str, on: bool) -> None:
        """A host write: the mode is a new game (a new deal with the stock, or without it)."""
        if name == "autoshuffle":              # read by the law as a setting; the game goes on
            self.autoshuffle = bool(on)
            if self.session.state is not None:
                self.session.column("autoshuffle")[...] = 1.0 if self.autoshuffle else 0.0
            self.epoch += 1
            return
        if name != "sand":
            raise ValueError(f"no mode {name!r}")
        self.sand = bool(on)
        self.reset()

    def make_piece(self):
        return PurrlinePiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.seed, self.sand, self.stock0, self.autoshuffle)

    def reset(self) -> None:
        self.seed = (self.seed + 1) % 997           # a new deal each run
        super().reset()

    def palette(self) -> dict[str, list[float]]:
        return {"floor": [0.12, 0.10, 0.18], "pl_cell_a": [0.20, 0.17, 0.30], "pl_cell_b": [0.16, 0.14, 0.25],
                "pl_cat": [0.98, 0.62, 0.25], "pl_fish": [0.30, 0.62, 0.95], "pl_yarn": [0.95, 0.30, 0.50],
                "pl_mouse": [0.72, 0.74, 0.80], "pl_bulb": [1.0, 0.90, 0.30], "pl_star": [0.70, 0.45, 0.95],
                "pl_catnip": [0.35, 0.90, 0.40], "pl_pop": [1.0, 1.0, 1.0], "pl_cursor": [1.0, 0.95, 0.55]}

    def backdrop(self) -> list[dict[str, Any]]:
        """The checkerboard under the tiles (flat entities, ids past the tiles')."""
        if self._cells is None:
            self._cells = [{"id": f"{20000 + k}:0", "x": k % W + 0.5, "y": k // W + 0.5, "vx": 0.0, "vy": 0.0,
                            "mesh": "ground", "color": "pl_cell_a" if (k // W + k % W) % 2 == 0 else "pl_cell_b",
                            "size": 0.5, "roll": 0.0, "lean": 0.0, "owner": -1, "age": 99.0, "spin": 0.0, "bob": 0.0}
                           for k in range(N)]
        return self._cells

    def entities(self) -> list[dict[str, Any]]:
        col = {name: self.session.column(name) for name in (*CELL_STATE, "sel", "clock")}
        clock = float(col["clock"][0])
        out = list(self.backdrop())
        for k in range(N):
            if col["kind"][k] < 0.0:                    # a hole: nothing there
                continue
            spec = int(col["spec"][k])
            flashing = col["flag"][k] > 0.5
            ox, oy = float(col["ox"][k]), float(col["oy"][k])
            mesh = SPEC_MESH.get(spec) or KIND_MESH[int(col["kind"][k])]
            color = "pl_pop" if flashing else "pl_catnip" if spec == NIP else KIND_MESH[int(col["kind"][k])]
            vx = -MOVE_V * (1.0 if ox > 0.0 else -1.0) if abs(ox) > 1e-9 else 0.0
            vy = -MOVE_V * (1.0 if oy > 0.0 else -1.0) if abs(oy) > 1e-9 else 0.0
            out.append({"id": f"{int(col['tid'][k])}:0", "x": k % W + 0.5 + ox, "y": k // W + 0.5 + oy, "vx": vx, "vy": vy,
                        "mesh": mesh, "color": color, "size": POWER_SIZE if spec else TILE_SIZE, "roll": 0.0, "lean": 0.7,
                        "owner": -1, "age": 99.0, "spin": 0.0, "bob": 0.0})
        for item in self.falling(col):                   # the sky plane: in front of the board, toward the camera
            kind = int(col["sky_kind"][item["cell"]])
            out.append({"id": item["id"], "x": item["x"], "y": item["y"], "vx": item["vx"], "vy": item["vy"],
                        "z": item["plane_z"], "vz": 0.0, "mesh": KIND_MESH[kind], "color": KIND_MESH[kind],
                        "size": item["size"], "roll": 0.0, "lean": 0.7, "owner": -1, "age": 99.0, "spin": 0.0, "bob": 0.0})
        sel = int(col["sel"][0])
        if sel >= 0:
            out.append({"id": "9000:0", "x": sel % W + 0.5, "y": sel // W + 0.5, "vx": 0.0, "vy": 0.0, "mesh": "pl_cursor",
                        "color": "pl_cursor", "size": 0.5, "roll": 0.0, "lean": 0.7, "owner": -1, "age": clock,
                        "spin": 0.0, "bob": 1.0})
        return out

    def falling(self, col: dict[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
        """The items in the sky, from the law's state: board units, y down, on the plane ``PLANE_Z`` in front."""
        col = col or {name: self.session.column(name) for name in CELL_STATE}
        out = []
        for k in range(N):
            if col["sky_on"][k] > 0.5:
                out.append({"id": f"{int(col['sky_tid'][k])}:0", "cell": k, "col": k % W, "row": k // W,
                            "x": k % W + 0.5, "y": float(col["sky_y"][k]), "vx": 0.0, "vy": float(col["sky_vy"][k]),
                            "size": TILE_SIZE, "plane_z": PLANE_Z})
        return out

    def occupancy(self, col: dict[str, np.ndarray] | None = None) -> list[int]:
        """Row-major bits, one per cell: 1 where the cell holds a tile (a flashing tile is still there)."""
        kind = (col or {"kind": self.session.column("kind")})["kind"]
        return [1 if kind[k] >= 0.0 else 0 for k in range(N)]

    def state(self) -> dict[str, Any]:
        st = super().state()
        if self.session.state is None:
            return st
        s = self.session
        # For the fountain around the board (WORKLIST.md, "Purrline sand mode"): all in board units (a cell is
        # ``cell`` wide, the board's top-left corner is ``origin``, x right, y down).
        st["purrline"] = {"sand": self.sand, "stock": int(s.column("stock")[0]), "stock0": self.stock0,
                          "rows": H, "cols": W, "cell": 1.0, "origin": [0.0, 0.0],
                          "occupancy": self.occupancy(), "falling": self.falling(),
                          "over": bool(s.column("over")[0] > 0.5)}
        return st

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("over")[0] < 0.5:
            return None
        if self.sand:
            return {"score": float(s.column("score")[0]),
                    "detail": f"sand: stock {int(s.column('stock')[0])} left, best cascade x{int(s.column('bestcombo')[0])}"}
        return {"score": float(s.column("score")[0]),
                "detail": f"level {int(s.column('level')[0])}, best cascade x{int(s.column('bestcombo')[0])}"}

    def describe(self) -> str:
        s = self.session
        if s.state is None:
            return "Purrline"
        col = s.column
        if self.sand:
            if col("over")[0] > 0.5:
                head = "NO MOVES LEFT · R for a new deal"
            elif int(col("mode")[0]) == IDLE:
                head = "swap two neighbouring tiles to line up three"
            else:
                head = f"cascade x{max(1, int(col('combo')[0]))}"
            return f"{head} · score {int(col('score')[0])} · stock {int(col('stock')[0])}/{self.stock0}"
        if col("over")[0] > 0.5:
            head = "OUT OF MOVES · R for a new deal"
        elif int(col("mode")[0]) == IDLE:
            head = "swap two neighbouring tiles to line up three"
        else:
            head = f"cascade x{max(1, int(col('combo')[0]))}"
        return (f"{head} · level {int(col('level')[0])} · score {int(col('score')[0])} "
                f"(level {int(col('lscore')[0])}/{int(col('target')[0])}) · moves {int(col('moves')[0])}")
