"""Space Invaders as one dt-system entity law on a free board.

The state is a span of entity LANES.  Every lane has a fixed SLOT (what may
live there) and static geometry; whether something lives there, and where,
is state.  The step law is masked span arithmetic with no branch, the same
form as ``meowdoku.program.meowdoku_step``: spawning is "the first free lane
of the right slot", collision is a pairwise overlap matrix over lanes masked
by which slots may touch, and dying moves a lane from ``alive`` to ``boom``.

Lanes:
    0            the player (the rocket)
    1            the player's shot (one in flight, as in the arcade)
    2..5         invader bombs
    6            the saucer
    7..61        the fleet: 5 rows x 11 columns
    62..85       four shields of 3 x 2 blocks, 3 hp each
    86..109      FREE FIRE shots: with the ``freefire`` setting on, every
                 volley sprays SPRAY_N shots into these lanes on a short
                 reload, fanned and jittered by the law's dice (lane 1, the
                 arcade's one shot, rests); off, these rest and lane 1 fires
    110..127     ATTACK WAVES: 3 formations of WAVE_SIZE ships.  Each ship's place is
                 a parametric law of its wave time (``wave_path``: enter from the
                 top-left / top-right / top-centre along an arc or an S-curve, hold
                 on a small loop while the firing pattern runs, then fly off)
    128..175     WAVE BULLETS: the patterns' shots (slow, modest in number), pooled

Columns (one value per lane; game-wide values are broadcast to every lane):
    lane, slot, home_x, home_y, points, rx, ry, home_hp      static
    alive, x, y, vx, vy, hp, boom, born, owner, energy        entity state (owner:
                                                              the lane that fired it)
    score, lives, wave, fleet_dir, clock, ufo_clock           game state
    field_w                                                   the field's width
    move_axis, steer_x, fire                                  the CONTROLS

Positions are board units, ``x`` across and ``y`` down the board; there are
no cells.  ``dt_limit`` is the time the fastest live entity takes to cross
the thinnest overlap window, so the dt controller (at cfl 0.5) never steps a
shot over an invader: tileless collision is the dt system's subdivision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meowdoku.entities import EntityGame, LawPiece
from meowdoku.entity_motion import damped_axis
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

WIDTH, HEIGHT = 12.5, 20.0
#: The field is portrait (phone-shaped) whatever the screen: its width follows
#: the screen's shape only up to this width-to-height ratio.
PORTRAIT = 0.62
PLAYER, SHOT, BOMB, UFO, INVADER, SHIELD = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
BOMB_LANES = range(2, 6)
UFO_LANE = 6
FLEET_ROWS, FLEET_COLS = 5, 11
FLEET0 = 7
SHIELD0 = FLEET0 + FLEET_ROWS * FLEET_COLS
SHIELDS, SHIELD_COLS, SHIELD_ROWS = 4, 3, 2
SPRAY_LANES = (SHIELD0 + SHIELDS * SHIELD_COLS * SHIELD_ROWS, SHIELD0 + SHIELDS * SHIELD_COLS * SHIELD_ROWS + 24)
#: Attack waves: formations of ships on parametric paths, firing patterns.
WAVER, WBULLET = 6.0, 7.0
WAVE_GROUPS, WAVE_SIZE = 3, 6
WAVE0 = SPRAY_LANES[1]
WBUL0 = WAVE0 + WAVE_GROUPS * WAVE_SIZE
WBULLETS = 48
WAVE_LANES = (WAVE0, WBUL0)
WBULLET_LANES = (WBUL0, WBUL0 + WBULLETS)
LANES = WBULLET_LANES[1]
#: The wave timeline (seconds of a ship's own wave clock): enter, hold (the
#: patterns fire here, a volley every VOLLEY_EVERY after VOLLEY_FIRST, each one
#: telegraphed for TELEGRAPH before it leaves), exit.  A formation's ships trail
#: one another by WAVE_LAG along the same path.
WAVE_ENTER, WAVE_EXIT, VOLLEY_FIRST, VOLLEY_EVERY, TELEGRAPH, WAVE_LAG = 2.0, 1.8, 0.8, 0.8, 0.45, 0.3
WAVE_FIRST_DELAY = 3.0
WAVE_POINTS = 40.0
#: Pattern kinds (``w_pat``).
PAT_AIM, PAT_FAN, PAT_RING, PAT_SPIRAL, PAT_ALT = 0.0, 1.0, 2.0, 3.0, 4.0
FAN_STEP, SPIRAL_STEP, RING_SLOW = 0.28, 0.55, 0.8
#: The simplicity limits the probe holds the law to.
MAX_BULLET_SPEED = 5.0
MAX_SHOTS_PER_EVENT = 10
#: The difficulty curve by level (the classic ``wave`` counter; the last row
#: holds for every later level), declared as data: ``rows`` classic fleet rows;
#: ``gap`` seconds between wave launches (0: no waves); ``groups`` formations
#: at once; ``patterns`` how many pattern kinds may be drawn (in PAT_ order);
#: ``ships`` per formation; ``volleys`` per ship; ``speed`` bullet speed;
#: ``extra`` the density step (wider fans, bigger rings, more spiral arms).
LEVEL_COLUMNS = ("rows", "gap", "groups", "patterns", "ships", "volleys", "speed", "extra")
LEVELS = (
    (5, 0.0, 0, 0, 0, 0, 0.0, 0),      # 1: the classic block alone
    (5, 16.0, 1, 1, 3, 2, 3.2, 0),     # 2: a few ships now and then, aimed shots
    (4, 12.0, 1, 2, 4, 3, 3.4, 0),     # 3: + fans
    (4, 10.0, 2, 3, 4, 3, 3.6, 0),     # 4: + rings, two formations
    (3, 8.5, 2, 4, 5, 4, 3.8, 0),      # 5: + spirals
    (3, 7.0, 2, 5, 6, 4, 4.0, 1),      # 6: + alternating volleys, denser
    (2, 6.0, 3, 5, 6, 4, 4.3, 1),      # 7: three formations
    (1, 5.0, 3, 5, 6, 5, 4.6, 1),      # 8+: a single classic row, mostly waves
)
#: Free fire: shots a volley, the fan between them, the dice's spread, the reload.
SPRAY_N, SPRAY_FAN, SPRAY_JITTER, FREE_RELOAD = 3.0, 0.16, 0.22, 0.09
INVADERS = float(FLEET_ROWS * FLEET_COLS)

LIVES = 3.0
PLAYER_Y = HEIGHT - 0.9
PLAYER_SPEED = 9.0
#: The craft's regenerative energy shield (the ``energy`` column, player lane):
#: a hit that finds at least SHIELD_COST of charge is absorbed -- the
#: projectile still dies, the ship keeps its life -- and drains that much; the
#: charge regenerates continuously at SHIELD_REGEN per second up to 1.
SHIELD_COST = 0.5
SHIELD_REGEN = 0.25
#: The rocket as drawn: its half size, its body box (the hitbox) and the
#: muzzle (where a shot leaves, the nose) -- one geometry for the eye and the
#: law.
SHIP_SIZE = 0.6            # body only (exhaust puff dropped at export): ~1.6 units long
SHIP_RX, SHIP_RY = 0.3, 0.7
MUZZLE = 0.76             # the nose: 1.31 drawn half-sizes ahead, foreshortened by the 0.25 lean
#: The rocket is integrated, not placed: its velocity relaxes toward the
#: commanded velocity with time constant SHIP_TAU (the damping), and a held
#: pointer commands the velocity that would close the gap in SHIP_REACH.
#: Together a damped second-order approach (damping ratio ~0.95): no
#: overshoot to speak of, and no network delay inside the loop, since the
#: pointer's position -- not a speed derived from a stale ship -- is sent.
SHIP_TAU = 0.05
SHIP_REACH = 0.18
SHOT_SPEED = 18.0
BOMB_SPEED = 7.0
UFO_SPEED = 4.0
FLEET_PACE = 0.4
UFO_Y = 0.75
FLEET_MARGIN = 0.55
FLEET_DROP = 0.42
INVASION_Y = HEIGHT - 1.9
PLAYER_BOOM = 1.6
BOOM = 0.35

#: The control record: state column -> the semantic action a shell fills it
#: from.  ``move_axis`` is -1..1 (keys); ``steer_x`` is the board x a held
#: pointer asks the rocket to go to (-1: no pointer, the keys rule);
#: ``fire`` is held, and a shot leaves whenever the shot lane is free.
CONTROLS = {"move_axis": "move-axis-x", "steer_x": "pointer-board-x", "fire": "primary-action"}

#: Mesh vocabulary: name -> emoji, extruded by ``web_export`` through the
#: same glyph builder as every other piece (Segoe UI Emoji outlines).
EMOJI = {
    "rocket": "\U0001F680", "alien": "\U0001F47D", "invader": "\U0001F47E", "crab": "\U0001F980",
    "ufo": "\U0001F6F8", "bolt": "⚡", "bomb": "\U0001F4A3", "brick": "\U0001F9F1", "boom": "\U0001F4A5",
    "frog": "\U0001F438", "star": "⭐",
}
#: Outline subdivisions per emoji (the rocket's tail collapses at 1).
EMOJI_SUBDIVISIONS = {"rocket": 2}
#: Glyphs whose outline carries detached decoration (the rocket's exhaust puff
#: floats beside its body): only the largest connected part is the piece.
EMOJI_BODY_ONLY = {"rocket"}


def _frac(value):
    return value - value.floor()


def _hash(value):
    """A deterministic per-lane noise in [0, 1): the law's dice."""
    return _frac((value * 12.9898 + 78.233).sin() * 43758.5453)


def _level(wave, name):
    """One column of the level curve at level ``wave`` (a span): a sum over the
    declared rows picked by equality, so it is a table lookup in span form."""
    column = LEVEL_COLUMNS.index(name)
    level = wave.clip(0.0, len(LEVELS) - 1.0)
    out = wave * 0.0
    for i, row in enumerate(LEVELS):
        out = out + (level == float(i)).to_dtype("float64") * float(row[column])
    return out


def wave_path(s, kind, hold, rad, bx, by, field_w):
    """Where a wave ship is at wave time ``s`` (a span): the parametric law.

    ``kind`` = 2 * side + shape: side 0 enters from the top-left, 1 from the
    top-right, 2 from the top centre; shape 0 bows along an arc, 1 along an
    S-curve.  It flies in to the anchor (``bx`` of the field across, ``by``
    down), holds ``hold`` seconds on one loop of radius ``rad`` around it (the
    time the firing pattern runs), then flies off the top corner.  The pieces
    meet with zero speed at the anchor, so the path is continuous."""
    side = (kind * 0.5).floor()
    shape = kind - side * 2.0
    left, right = side < 0.5, side < 1.5
    ax = where(left, -1.2, where(right, field_w + 1.2, field_w * 0.5))
    ay = where(right, -1.2, -1.4)
    bpx, bpy = bx * field_w, by
    exit_right = where(left, 1.0, where(right, 0.0, where(bx < 0.5, 1.0, 0.0)))
    cx = where(exit_right > 0.5, field_w + 1.4, -1.4)
    cy = ay * 0.0 - 1.2
    turn = where(left, 1.0, where(right, -1.0, 1.0))
    arc = shape < 0.5
    two_pi = 6.283185307179586
    # enter: eased to rest at the anchor, bowed by an arc or an S
    u = (s / WAVE_ENTER).clip(0.0, 1.0)
    ease_in = 1.0 - (1.0 - u) * (1.0 - u)
    dx, dy = bpx - ax, bpy - ay
    length = (dx * dx + dy * dy).sqrt() + 1e-6
    bow = where(arc, (u * 3.141592653589793).sin(), (u * two_pi).sin()) * (1.0 - u) * 0.22 * length * turn
    # hold: one loop, smoothstepped so the speed is zero at both ends
    w = ((s - WAVE_ENTER) / hold).clip(0.0, 1.0)
    theta = two_pi * w * w * (3.0 - 2.0 * w)
    # exit: accelerate away, bowed the other way
    v = ((s - WAVE_ENTER - hold) / WAVE_EXIT).clip(0.0, 1.0)
    ex, ey = cx - bpx, cy - bpy
    elen = (ex * ex + ey * ey).sqrt() + 1e-6
    leave = where(arc, (v * 3.141592653589793).sin(), (v * two_pi).sin()) * v * 0.18 * elen * (0.0 - turn)
    x = ax + dx * ease_in + (0.0 - dy) / length * bow + rad * theta.sin() + ex * v * v + (0.0 - ey) / elen * leave
    y = ay + dy * ease_in + dx / length * bow + rad * (1.0 - theta.cos()) + ey * v * v + ex / elen * leave
    return x, y


def invaders_step(lane, slot, home_x, home_y, points, rx, ry, home_hp, w_group, w_member, w_lag,
                  alive, x, y, vx, vy, hp, boom, born, owner, energy,
                  score, lives, wave, fleet_dir, clock, ufo_clock, reload,
                  w_t, w_kind, w_pat, w_hold, w_rad, w_bx, w_by, tele, w_clock, w_count,
                  field_w, freefire, seed,
                  move_axis, steer_x, fire, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    n = int(lane.shape[0])
    f = lambda mask: mask.to_dtype("float64")  # noqa: E731
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_shot, is_bomb = slot == PLAYER, slot == SHOT, slot == BOMB
    is_ufo, is_inv, is_shield = slot == UFO, slot == INVADER, slot == SHIELD
    is_wv, is_wb = slot == WAVER, slot == WBULLET
    clock_next = clock + dt

    # The fleet: one heading for every invader; fewer invaders, faster fleet.
    fleet = is_inv & live
    count = f(fleet).sum()
    rows_now = _level(wave, "rows")
    speed = (FLEET_PACE + 0.08 * wave) * (1.0 + 4.0 * (1.0 - count / (FLEET_COLS * rows_now)))
    step_x = fleet_dir * speed * dt
    beyond = fleet & ((((x + step_x) > (field_w - FLEET_MARGIN)) & (fleet_dir > 0.0))
                      | (((x + step_x) < FLEET_MARGIN) & (fleet_dir < 0.0)))
    edge = (f(beyond).sum() > 0.0) & playing
    fleet_dir_next = where(edge, -fleet_dir, fleet_dir)
    fleet_dx = where(edge, 0.0, step_x)
    fleet_dy = where(edge, FLEET_DROP, 0.0)

    # The player's lane: steer from the control, read where it is.
    player_live = is_player & live
    player_x = (f(player_live) * x).sum()
    player_on = f(player_live).sum() > 0.0

    # Spawns.  Shot: whenever fire is held and the one shot lane is free --
    # or, in free fire, a sprayed volley whenever the reload has run out.
    spraying = freefire > 0.0
    classic_shot = is_shot & (lane == 1.0) & ~live & (fire > 0.0) & player_on & playing & ~spraying
    reload_now = (f(is_player) * reload).sum()
    volley = (fire > 0.0) & player_on & playing & spraying & (reload_now <= 0.0)
    free_spray = is_shot & ~live & (lane >= SPRAY_LANES[0])
    spray_rank = f(free_spray).cumsum(dim=0) - 1.0
    spray_shot = free_spray & (spray_rank < SPRAY_N) & volley
    spray_angle = (spray_rank - 0.5 * (SPRAY_N - 1.0)) * SPRAY_FAN + (_hash(lane * 1.7 + clock * 13.1) - 0.5) * SPRAY_JITTER
    shot_spawn = classic_shot | spray_shot
    # Bombs: the dice pick a front invader (nothing alive below it in its
    # column) and the first free bomb lane carries its bomb.
    hx1, hx2 = home_x.reshape((n, 1)), home_x.reshape((1, n))
    hy1, hy2 = home_y.reshape((n, 1)), home_y.reshape((1, n))
    below = (hx1 == hx2) & (hy2 > hy1) & fleet.reshape((1, n))
    front = fleet & ~(f(below).sum(dim=1) > 0.0)
    dice = _hash(lane + clock * 7.31)
    ranked = where(front, dice, -1.0)
    shooter = front & (ranked == ranked.max())
    rate = 0.45 + 0.15 * wave
    trigger = (_hash(clock * 3.17 + 0.5) < rate * dt) & (f(front).sum() > 0.0) & playing
    free_bomb = is_bomb & ~live
    first_bomb = free_bomb & (lane == where(free_bomb, lane, 1.0e9).min())
    bomb_spawn = first_bomb & trigger
    shooter_x = (f(shooter) * x).sum()
    shooter_lane = (f(shooter) * lane).sum()
    shooter_y = (f(shooter) * y).sum()
    # The saucer: when its clock runs out, from a side the dice pick.
    ufo_spawn = is_ufo & ~live & (ufo_clock <= 0.0) & playing
    from_left = _hash(clock * 1.91 + 2.0) > 0.5
    ufo_x0 = where(from_left, -0.9, field_w + 0.9)
    ufo_vx0 = where(from_left, UFO_SPEED, -UFO_SPEED)
    ufo_launched = f(ufo_spawn).sum() > 0.0
    ufo_clock_next = where(ufo_launched, 14.0 + 10.0 * _hash(clock * 5.3), ufo_clock - dt)

    # ATTACK WAVES.  Launch: when the wave clock runs out, the lowest idle
    # formation starts (its parameters drawn from the seed and the launch
    # count); each ship then flies ``wave_path`` of its own wave time.
    gap, groups, pats = _level(wave, "gap"), _level(wave, "groups"), _level(wave, "patterns")
    ships, bspeed, extra = _level(wave, "ships"), _level(wave, "speed"), _level(wave, "extra")
    vols_level = _level(wave, "volleys")
    wavers = is_wv & live
    n_busy = x * 0.0
    g_star = x * 0.0 + 99.0
    for g in reversed(range(WAVE_GROUPS)):
        busy_g = ((f(wavers & (w_group == float(g))).sum() + x * 0.0) > 0.0)
        n_busy = n_busy + f(busy_g)
        g_star = where(busy_g, g_star, float(g))
    waves_on = gap > 0.0
    launch = (w_clock <= 0.0) & waves_on & playing & (n_busy < groups) & (g_star < 90.0)

    def dice(k):
        return _hash(seed * 0.173 + w_count * 3.7 + k * 1.37 + 0.31)

    wave_spawn = is_wv & ~live & launch & (w_group == g_star) & (w_member < ships)
    kind_new = (dice(1.0) * 3.0).floor().clip(0.0, 2.0) * 2.0 + (dice(2.0) * 2.0).floor().clip(0.0, 1.0)
    pat_new = (dice(3.0) * pats).floor().clip(0.0, 4.0)
    hold_new = VOLLEY_FIRST + vols_level * VOLLEY_EVERY + 0.6
    w_t1 = where(wave_spawn, 0.0, w_t)
    w_kind1 = where(wave_spawn, kind_new, w_kind)
    w_pat1 = where(wave_spawn, pat_new, w_pat)
    w_hold1 = where(wave_spawn, hold_new, w_hold)
    w_rad1 = where(wave_spawn, 0.35 + dice(4.0), w_rad)
    w_bx1 = where(wave_spawn, 0.22 + 0.56 * dice(5.0), w_bx)
    w_by1 = where(wave_spawn, 3.2 + 3.0 * dice(6.0), w_by)
    w_count_next = where(launch, w_count + 1.0, w_count)
    w_clock_next = where(launch, gap * (0.75 + 0.5 * dice(7.0)), where(waves_on, w_clock - dt, WAVE_FIRST_DELAY))
    wv_move = is_wv & (live | wave_spawn) & playing
    s_now = w_t1 - w_lag
    s_nxt = s_now + where(wv_move, dt, 0.0)
    path_args = (w_kind1, w_hold1, w_rad1, w_bx1, w_by1, field_w)
    wp_x, wp_y = wave_path(s_nxt.clip(0.0, 1.0e6), *path_args)
    eps = 0.01
    ahead_s, behind_s = (s_nxt + eps).clip(0.0, 1.0e6), (s_nxt - eps).clip(0.0, 1.0e6)
    ahead_x, ahead_y = wave_path(ahead_s, *path_args)
    behind_x, behind_y = wave_path(behind_s, *path_args)
    span_s = ahead_s - behind_s + 1.0e-9
    wp_vx, wp_vy = (ahead_x - behind_x) / span_s, (ahead_y - behind_y) / span_s
    # Firing: a ship fires when its clock crosses a volley time; the pattern
    # decides how many shots, and the free bullet lanes (ranked, as the
    # spray's are) take them in order.
    q_now = (s_now - (WAVE_ENTER + VOLLEY_FIRST)) / VOLLEY_EVERY
    q_nxt = (s_nxt - (WAVE_ENTER + VOLLEY_FIRST)) / VOLLEY_EVERY
    k_now, k_nxt = q_now.floor(), q_nxt.floor()
    vols_ship = ((w_hold1 - 0.6 - VOLLEY_FIRST) / VOLLEY_EVERY + 0.5).floor()
    fires = wv_move & (q_nxt >= 0.0) & (k_nxt > k_now) & (k_nxt < vols_ship)
    upcoming = 0.0 - (0.0 - q_nxt).floor()
    telegraphing = (wv_move & (upcoming >= 0.0) & (upcoming < vols_ship)
                    & ((upcoming - q_nxt) * VOLLEY_EVERY <= TELEGRAPH) & ~fires)
    leader = ((w_member - (w_member / 3.0).floor() * 3.0) < 0.5)
    parity = w_member - (w_member * 0.5).floor() * 2.0
    k_par = k_nxt - (k_nxt * 0.5).floor() * 2.0
    n_fan, n_ring, n_arms = 3.0 + 2.0 * extra, 8.0 + 2.0 * extra, 2.0 + extra
    n_ship = where(w_pat1 < 0.5, 1.0, where(w_pat1 < 1.5, n_fan,
                   where(w_pat1 < 2.5, where(leader, n_ring, 0.0),
                         where(w_pat1 < 3.5, where(leader, n_arms, 0.0), where(parity == k_par, n_fan, 0.0)))))
    n_ship = where(fires, n_ship, 0.0)
    offset = n_ship.cumsum(dim=0) - n_ship
    free_wb = is_wb & ~live
    wb_rank = f(free_wb).cumsum(dim=0) - 1.0
    take = (free_wb.reshape((n, 1)) & (wb_rank.reshape((n, 1)) >= offset.reshape((1, n)))
            & (wb_rank.reshape((n, 1)) < (offset + n_ship).reshape((1, n))) & (n_ship.reshape((1, n)) > 0.0))
    take_f = f(take)

    def pick(value):
        return (take_f * value.reshape((1, n))).sum(dim=1)

    wb_spawn = pick(x * 0.0 + 1.0) > 0.5
    aim_dx, aim_dy = player_x - wp_x, PLAYER_Y - wp_y
    aim_len = (aim_dx * aim_dx + aim_dy * aim_dy).sqrt() + 1.0e-6
    aim_x, aim_y = where(player_on, aim_dx / aim_len, 0.0), where(player_on, aim_dy / aim_len, 1.0)
    pat_p, n_p, k_p, shot_j = pick(w_pat1), pick(n_ship).clip(1.0, 99.0), pick(k_nxt), wb_rank - pick(offset)
    aimed = (pat_p < 1.5) | (pat_p > 3.5)
    base_x, base_y = where(aimed, pick(aim_x), 0.0), where(aimed, pick(aim_y), 1.0)
    turn_fan = (shot_j - 0.5 * (n_p - 1.0)) * FAN_STEP
    turn_ring = shot_j * (6.283185307179586 / n_p) + k_p * (3.141592653589793 / n_p)
    turn_spiral = shot_j * (6.283185307179586 / n_p) + k_p * SPIRAL_STEP
    turn = where(pat_p < 0.5, 0.0, where(aimed, turn_fan, where(pat_p < 2.5, turn_ring, turn_spiral)))
    speed_b = bspeed * where(pat_p == PAT_RING, RING_SLOW, 1.0)
    wb_ux, wb_uy = base_x * turn.cos() + base_y * turn.sin(), base_y * turn.cos() - base_x * turn.sin()
    wb_vx, wb_vy = speed_b * wb_ux, speed_b * wb_uy
    wb_x, wb_y, wb_owner = pick(wp_x), pick(wp_y), pick(lane)

    spawn = shot_spawn | bomb_spawn | ufo_spawn | wave_spawn | wb_spawn
    x1 = where(shot_spawn, player_x + where(spray_shot, (spray_rank - 0.5 * (SPRAY_N - 1.0)) * 0.15, 0.0),
               where(bomb_spawn, shooter_x, where(ufo_spawn, ufo_x0, where(wb_spawn, wb_x, x))))
    y1 = where(shot_spawn, PLAYER_Y - MUZZLE, where(bomb_spawn, shooter_y + 0.5, where(ufo_spawn, UFO_Y, where(wb_spawn, wb_y, y))))
    # The rocket: the shared damped motion law (entity_motion.damped_axis),
    # the same text the page runs as WebAssembly to predict its own ship.
    ship_x, ship_v = damped_axis(x, vx, move_axis, steer_x, x * 0.0 + 0.7, field_w - 0.7,
                                 PLAYER_SPEED, SHIP_TAU, SHIP_REACH, dt)
    vx1 = where(is_player, ship_v,
                where(is_inv, fleet_dir_next * speed, where(ufo_spawn, ufo_vx0, where(wb_spawn, wb_vx,
                      where(spray_shot, SHOT_SPEED * spray_angle.sin(), where(spawn, 0.0, vx))))))
    vx1 = where(wv_move, wp_vx, vx1)
    vy1 = where(wb_spawn, wb_vy, where(spray_shot, -SHOT_SPEED * spray_angle.cos(),
                where(shot_spawn, -SHOT_SPEED, where(bomb_spawn, BOMB_SPEED, where(spawn, 0.0, vy)))))
    vy1 = where(wv_move, wp_vy, vy1)
    reload_next = where(is_player, where(volley, FREE_RELOAD, (reload - dt).clip(0.0, 1.0)), reload)
    live1 = live | spawn
    # Born at the start of the step: a spawn leaves the muzzle at x1/y1 and flies
    # the whole step, so position = birth point + velocity x (clock - born).
    born1 = where(spawn, clock, born)
    owner1 = where(shot_spawn, 0.0, where(bomb_spawn, shooter_lane, where(ufo_spawn, -1.0, where(wb_spawn, wb_owner, owner))))

    # Motion: the fleet steps as one; everything else flies its velocity.
    moving = live1 & playing
    x2 = where(moving & is_inv, x1 + fleet_dx, where(moving, x1 + vx1 * dt, x1))
    y2 = where(moving & is_inv, y1 + fleet_dy, where(moving, y1 + vy1 * dt, y1))
    x2 = where(is_player & moving, ship_x, x2)
    # (a wave ship is placed by its path, not flown: position is a law of its clock)
    x2 = where(wv_move, wp_x, x2)
    y2 = where(wv_move, wp_y, y2)
    vx_out = where(is_player & ~moving, 0.0, vx1)
    gone = (((is_shot | is_bomb | is_wb) & ((y2 < -0.5) | (y2 > HEIGHT + 0.5) | (x2 < -0.5) | (x2 > field_w + 0.5)))
            | (is_ufo & ((x2 < -1.5) | (x2 > field_w + 1.5)))
            | (is_wv & (s_nxt > WAVE_ENTER + w_hold1 + WAVE_EXIT)))
    live2 = live1 & ~gone

    # Collisions: overlapping boxes between slots that may touch.  Invaders
    # erase shields and are not hurt by them; every other touch harms both.
    X1, X2 = x2.reshape((n, 1)), x2.reshape((1, n))
    Y1, Y2 = y2.reshape((n, 1)), y2.reshape((1, n))
    S1, S2 = slot.reshape((n, 1)), slot.reshape((1, n))
    touch = (((X1 - X2).abs() < (rx.reshape((n, 1)) + rx.reshape((1, n))))
             & ((Y1 - Y2).abs() < (ry.reshape((n, 1)) + ry.reshape((1, n))))
             & live2.reshape((n, 1)) & live2.reshape((1, n)))

    def pair(p, q):
        return ((S1 == p) & (S2 == q)) | ((S1 == q) & (S2 == p))

    may_touch = (pair(SHOT, INVADER) | pair(SHOT, UFO) | pair(SHOT, SHIELD) | pair(SHOT, BOMB)
                 | pair(SHOT, WAVER) | pair(WAVER, PLAYER) | pair(WBULLET, PLAYER) | pair(WBULLET, SHIELD)
                 | pair(BOMB, PLAYER) | pair(BOMB, SHIELD) | pair(INVADER, SHIELD) | pair(INVADER, PLAYER))
    harmless = (S1 == INVADER) & ((S2 == SHIELD) | (S2 == PLAYER))
    hits = f(touch & may_touch & ~harmless).sum(dim=1)
    hp2 = where(is_shield, hp - hits, hp)
    # The craft's shield takes a hit while it holds enough charge.
    absorbed = is_player & live2 & (hits > 0.0) & (energy >= SHIELD_COST)
    dies = live2 & (hits > 0.0) & (~is_shield | (hp2 <= 0.0)) & ~absorbed
    live3 = live2 & ~dies
    boom3 = where(dies & (is_inv | is_ufo | is_wv), BOOM,
                  where(dies & is_player, PLAYER_BOOM, (boom - dt).clip(0.0, PLAYER_BOOM)))
    score_next = score + (f(dies & (is_inv | is_ufo | is_wv)) * points).sum()
    invaded = f(fleet & live3 & (y2 > INVASION_Y)).sum() > 0.0
    lives_after = lives - f(dies & is_player).sum()
    lives_next = where(invaded, 0.0, lives_after).clip(0.0, LIVES)

    # The rocket comes back when its explosion has burned out.
    respawn = is_player & ~live3 & (boom > 0.0) & (boom3 <= 0.0) & (lives_next > 0.0)
    # A cleared wave: the fleet re-forms (a row lower each wave, up to four),
    # bombs in flight vanish, and the fleet is faster and shoots more.
    cleared = (f(is_inv & live3).sum() == 0.0) & playing
    fleet_row = ((lane - float(FLEET0)) / float(FLEET_COLS)).floor()
    reform = is_inv & cleared & (fleet_row < _level(wave + 1.0, "rows"))
    live4 = (live3 | respawn | reform) & ~((is_bomb | is_wb) & cleared)
    x4 = where(respawn, field_w / 2.0, where(reform, home_x, x2))
    vx4 = where(respawn, 0.0, vx_out)
    y4 = where(reform, home_y + FLEET_DROP * (wave + 1.0).clip(0.0, 4.0), y2)
    born4 = where(respawn | reform, clock_next, born1)
    energy_next = where(is_player, where(respawn, 1.0, where(absorbed, energy - SHIELD_COST,
                                                              (energy + SHIELD_REGEN * dt).clip(0.0, 1.0))), energy)
    wave_next = where(cleared, wave + 1.0, wave)
    fleet_dir_out = where(cleared, 1.0, fleet_dir_next)

    # The fastest live mover against the thinnest overlap window (a shot's
    # length plus a brick's): the most a step may move anything.
    pace = where(live4, vx1.abs() + vy1.abs(), 0.0) + where(is_inv & live4, speed, 0.0)
    # ... and no step longer than half the rocket's damping time.
    dt_limit = (0.5 / (pace.max() + 1e-6)).clip(0.0, 0.5 * SHIP_TAU) + x * 0.0
    residual = f(is_inv & live4).sum() + x * 0.0
    tele_next = where(is_wv & live4, f(telegraphing), 0.0)
    w_t_next = where(wv_move, w_t1 + dt, w_t1)
    return (f(live4), x4, y4, vx4, vy1, hp2, boom3, born4, owner1, energy_next,
            score_next, lives_next, wave_next, fleet_dir_out, clock_next, ufo_clock_next, reload_next,
            w_t_next, w_kind1, w_pat1, w_hold1, w_rad1, w_bx1, w_by1, tele_next, w_clock_next, w_count_next,
            residual, dt_limit)


STATE = ("alive", "x", "y", "vx", "vy", "hp", "boom", "born", "owner", "energy",
         "score", "lives", "wave", "fleet_dir", "clock", "ufo_clock", "reload",
         "w_t", "w_kind", "w_pat", "w_hold", "w_rad", "w_bx", "w_by", "tele", "w_clock", "w_count")
#: Read by the law, written by the host only (a layout change), never stepped.
SETTINGS = ("field_w", "freefire", "seed")


class InvadersPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "invaders_step"
    law = staticmethod(invaders_step)
    argument_names = ("lane", "slot", "home_x", "home_y", "points", "rx", "ry", "home_hp", "w_group", "w_member", "w_lag",
                      *STATE, *SETTINGS, *CONTROLS, "dt")
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


def field_width(aspect: float | None) -> float:
    """The field's width for a screen of ``aspect`` (width / height): the
    height is fixed, the width follows the screen within arcade limits."""
    if not aspect or aspect <= 0.0:
        return WIDTH
    return float(np.clip(round(HEIGHT * min(aspect, PORTRAIT) * 2.0) / 2.0, 12.5, HEIGHT * PORTRAIT + 0.5))


def layout(width: float = WIDTH) -> dict[str, np.ndarray]:
    """The static columns: each lane's slot, home and size, for a field ``width`` wide."""
    WIDTH = width  # noqa: N806  (the layout below is written against the field width)
    slot = np.zeros(LANES)
    home_x = np.full(LANES, WIDTH / 2.0)
    home_y = np.full(LANES, PLAYER_Y)
    points = np.zeros(LANES)
    rx = np.full(LANES, 0.5)
    ry = np.full(LANES, 0.38)
    home_hp = np.ones(LANES)
    rx[0], ry[0] = SHIP_RX, SHIP_RY                 # the rocket's drawn body
    slot[1], rx[1], ry[1] = SHOT, 0.08, 0.28
    for k in range(*SPRAY_LANES):
        slot[k], rx[k], ry[k] = SHOT, 0.08, 0.28
    for k in BOMB_LANES:
        slot[k], rx[k], ry[k] = BOMB, 0.16, 0.24
    w_group, w_member, w_lag = np.zeros(LANES), np.zeros(LANES), np.zeros(LANES)
    for k in range(*WAVE_LANES):
        slot[k], rx[k], ry[k], points[k] = WAVER, 0.4, 0.34, WAVE_POINTS
        home_y[k] = 0.0
        w_group[k], w_member[k] = (k - WAVE0) // WAVE_SIZE, (k - WAVE0) % WAVE_SIZE
        w_lag[k] = w_member[k] * WAVE_LAG
    for k in range(*WBULLET_LANES):
        slot[k], rx[k], ry[k] = WBULLET, 0.14, 0.14
    slot[UFO_LANE], rx[UFO_LANE], ry[UFO_LANE], points[UFO_LANE], home_y[UFO_LANE] = UFO, 0.6, 0.35, 100.0, UFO_Y
    spacing_x, spacing_y = 1.05, 0.92
    left = (WIDTH - spacing_x * (FLEET_COLS - 1)) / 2.0
    for r in range(FLEET_ROWS):
        for c in range(FLEET_COLS):
            k = FLEET0 + r * FLEET_COLS + c
            slot[k], rx[k], ry[k] = INVADER, 0.4, 0.34
            home_x[k], home_y[k] = left + c * spacing_x, 1.9 + r * spacing_y
            points[k] = 30.0 if r == 0 else 20.0 if r < 3 else 10.0
    k = SHIELD0
    for s in range(SHIELDS):
        centre = WIDTH * (s + 0.5) / SHIELDS
        for r in range(SHIELD_ROWS):
            for c in range(SHIELD_COLS):
                slot[k], rx[k], ry[k], home_hp[k] = SHIELD, 0.27, 0.22, 3.0
                home_x[k], home_y[k] = centre + (c - 1) * 0.55, PLAYER_Y - 1.75 + r * 0.45
                k += 1
    return {"lane": np.arange(LANES, dtype=np.float64), "slot": slot, "home_x": home_x, "home_y": home_y,
            "points": points, "rx": rx, "ry": ry, "home_hp": home_hp,
            "w_group": w_group, "w_member": w_member, "w_lag": w_lag}


def initial_columns(width: float = WIDTH, freefire: bool = False, seed: float = 0.0) -> dict[str, np.ndarray]:
    columns = layout(width)
    slot = columns["slot"]
    starts = (slot == PLAYER) | (slot == INVADER) | (slot == SHIELD)
    columns.update({
        "alive": starts.astype(np.float64), "x": columns["home_x"].copy(), "y": columns["home_y"].copy(),
        "vx": np.zeros(LANES), "vy": np.zeros(LANES), "hp": columns["home_hp"].copy(),
        "boom": np.zeros(LANES), "born": np.zeros(LANES), "owner": np.full(LANES, -1.0),
        "energy": (slot == PLAYER).astype(np.float64),
        "score": np.zeros(LANES), "lives": np.full(LANES, LIVES), "wave": np.zeros(LANES),
        "fleet_dir": np.ones(LANES), "clock": np.zeros(LANES), "ufo_clock": np.full(LANES, 12.0),
        "reload": np.zeros(LANES),
        "w_t": np.zeros(LANES), "w_kind": np.zeros(LANES), "w_pat": np.zeros(LANES), "w_hold": np.full(LANES, 4.0),
        "w_rad": np.full(LANES, 0.5), "w_bx": np.full(LANES, 0.5), "w_by": np.full(LANES, 4.0), "tele": np.zeros(LANES),
        "w_clock": np.full(LANES, WAVE_FIRST_DELAY), "w_count": np.zeros(LANES), "seed": np.full(LANES, float(seed)),
        "field_w": np.full(LANES, float(width)), "freefire": np.full(LANES, 1.0 if freefire else 0.0),
        "move_axis": np.zeros(LANES), "steer_x": np.full(LANES, -1.0), "fire": np.zeros(LANES),
    })
    return columns


class SpaceInvaders(EntityGame):
    game = "space_invaders"
    title = "Space Invaders (free board)"
    width, height = WIDTH, HEIGHT
    CONTROLS = tuple(CONTROLS)

    def __init__(self, board_id: str = "", aspect: float | None = None, freefire: bool = False, seed: float | None = None):
        self.width = field_width(aspect)
        self.freefire = bool(freefire)
        # The waves' parameters are a function of this seed and the launch count.
        import hashlib
        self.seed = float(seed) if seed is not None else int(hashlib.sha256(board_id.encode("utf-8")).hexdigest()[:6], 16) % 9973 / 97.0
        super().__init__(board_id)

    def make_piece(self):
        return InvadersPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(self.width, self.freefire, self.seed)

    #: Selectable modes (the page shows a toggle for each): FREE FIRE sprays
    #: volleys instead of the arcade's one shot on the screen at a time.
    MODES = {"freefire": "Free fire"}

    def modes(self) -> dict[str, bool]:
        return {"freefire": self.freefire}

    def set_mode(self, name: str, on: bool) -> None:
        """A host write, like the layout: the law reads it as a setting."""
        if name != "freefire":
            raise ValueError(f"no mode {name!r}")
        self.freefire = bool(on)
        if self.session.state is not None:
            self.session.column("freefire")[...] = 1.0 if self.freefire else 0.0
        self.epoch += 1

    PLAYER_LANE = 0
    SPEED_CONTROL = ZOOM_CONTROL = True              # (no world beyond its field: the zoom only goes in)

    def motion(self) -> dict[str, Any]:
        return {"speed": PLAYER_SPEED, "tau": SHIP_TAU, "reach": SHIP_REACH,
                "x": [0.7, self.width - 0.7],
                # The weapon, as the law fires it: one shot in flight, from
                # the nose, straight up -- so the page can show its own shot
                # leaving the muzzle the moment fire is held.
                # (In free fire the spray's angles are the law's dice: the page
                # waits for the server's shots instead of guessing them.)
                **({} if self.freefire else {
                    "weapon": {"muzzles": [[0.0, -MUZZLE]], "velocity": [0.0, -SHOT_SPEED], "reload": 0.0,
                               "max_in_flight": 1, "mesh": "bolt", "color": "bolt", "size": 0.26}})}

    def layout(self, aspect: float) -> None:
        """Fit the field to a new screen shape mid-game: the walls move and
        everything on the field shifts by half the change, so the play stays
        centred.  A host write between rounds, like the controls."""
        width = field_width(aspect)
        if width == self.width:
            return
        shift = (width - self.width) / 2.0
        s = self.session
        for name in ("x", "home_x"):
            s.column(name)[...] += shift
        s.column("field_w")[...] = width
        self.epoch += 1
        self.width = width

    def palette(self) -> dict[str, list[float]]:
        return {"rocket": [0.85, 0.88, 0.95], "alien": [0.35, 0.95, 0.35], "invader": [0.75, 0.35, 0.95],
                "crab": [0.95, 0.35, 0.25], "ufo": [0.95, 0.15, 0.55], "bolt": [1.0, 0.92, 0.25],
                "bomb": [0.2, 0.2, 0.24], "brick": [0.75, 0.42, 0.25], "brick_hurt": [0.45, 0.25, 0.15],
                "boom": [1.0, 0.55, 0.1], "orb": [1.0, 0.3, 0.35]}

    def entities(self) -> list[dict[str, Any]]:
        col = {name: self.session.column(name) for name in ("slot", "alive", "x", "y", "vx", "vy", "hp", "boom", "born", "owner", "home_y", "rx", "tele", "w_group")}
        clock = float(self.session.column("clock")[0])
        out = []
        for k in range(LANES):
            slot, alive, boom = col["slot"][k], col["alive"][k] > 0.0, col["boom"][k]
            exploding = not alive and boom > 0.0 and (slot != PLAYER or boom > PLAYER_BOOM - 0.5)
            if not (alive or exploding):
                continue
            if exploding:
                mesh, color, size = "boom", "boom", 0.42
            elif slot == PLAYER:
                mesh, color, size = "rocket", "rocket", SHIP_SIZE
            elif slot == SHOT:
                mesh, color, size = "bolt", "bolt", 0.26
            elif slot == BOMB:
                mesh, color, size = "bomb", "bomb", 0.22
            elif slot == UFO:
                mesh, color, size = "ufo", "ufo", 0.55
            elif slot == WAVER:
                mesh = ("invader", "crab", "alien")[int(col["w_group"][k]) % 3]
                telegraphing = col["tele"][k] > 0.5
                color, size = ("boom" if telegraphing else mesh), (0.5 if telegraphing else 0.4)
            elif slot == WBULLET:
                mesh, color, size = "star", "orb", 0.2
            elif slot == INVADER:
                row = (k - FLEET0) // FLEET_COLS
                mesh = "alien" if row == 0 else "invader" if row < 3 else "crab"
                color, size = mesh, 0.38
            else:
                mesh, size = "brick", 0.26
                color = "brick" if col["hp"][k] >= 3.0 else "brick_hurt"
            out.append({
                "id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                "vx": 0.0 if exploding else float(col["vx"][k]), "vy": 0.0 if exploding else float(col["vy"][k]),
                "mesh": mesh, "color": color, "size": size,
                "roll": 0.785 if mesh == "rocket" else 0.0, "lean": 0.25 if slot == PLAYER else 0.55,
                "owner": int(col["owner"][k]) if slot in (SHOT, BOMB) and not exploding else -1,
                "age": clock - float(col["born"][k]),
                "spin": 2.5 if mesh == "ufo" else 6.0 if slot == WAVER and col["tele"][k] > 0.5 and not exploding else 0.0,
                "bob": 1.0 if slot in (INVADER, WAVER) and not exploding else 0.0,
            })
        return out

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]), "detail": f"reached wave {int(s.column('wave')[0]) + 1}"}

    HOSTILE_LANES = ((BOMB_LANES.start, BOMB_LANES.stop), (UFO_LANE, UFO_LANE + 1), (FLEET0, SHIELD0),
                     WAVE_LANES, WBULLET_LANES)

    def describe(self) -> str:
        s = self.session
        lives, score, wave = int(s.column("lives")[0]), int(s.column("score")[0]), int(s.column("wave")[0])
        fleet = int(s.column("alive")[FLEET0:SHIELD0].sum())
        seconds, calls = s.last_round
        head = "GAME OVER · R to restart" if lives <= 0 else f"lives {lives}"
        attackers = int(s.column("alive")[WAVE0:WBUL0].sum())
        return (f"{head} · score {score} · wave {wave + 1} · {fleet} invaders · {attackers} attack ships · "
                f"tick {seconds * 1000:.0f} ms = {calls} law steps")
