"""Mobroad: a road-defence mob shooter.  An original game.

A MOB comes down a long road toward your shooters.  The mob is the 2-D fountain law (``fountain.make_law_mobroad``, a product VARIANT of it): every grain
is a mob member, the road's features (two pillars, a narrowing, a funnel, a peg field) are SDF slots the crowd is pushed through by the law's own contact
forces, and the pressure at a narrowing is the grains' contact count (the colour goes red, ``n_crush`` counts them).  You keep it off you: bullets kill
grains, a grain that gets within the attack radius of you has reached you (every grain is walked toward you by its own tiny neural pather; gravity is orthogonal to the scene), and when ``LOSE_AT`` have, you lose.

WHERE THE AUTHORITY LIVES (one law each, no copy of a law anywhere):
    the MOB     the fountain law, run by the PAGE (GPU compute / wasm: the server has no 4096 grains to look at).  The law itself kills: the page writes
                the bullets' swept segments into it each batch (one way, ``fountain.mobroad_inputs``), and a killed or arrived grain is PARKED in the law's
                own state (counted by ``n_dead`` / ``n_goal``).  The server never sees a grain.
    the GAME    this file's lane law (``mobroad_step``, a dt-system entity law on the server, like invaders): the shooters, the bullets and their flight,
                the panels, the multipliers, the score and the lose condition.  Bullets are the server's, so the page has the exact paths the mob law is fed.
    the LINK    the page reports the mob's cumulative ``dead`` and ``goal`` counts (the law's diagnostics) in its ticks (``mob``); the server writes them into
                its law as host settings (they only ever grow, are bounded by the id count and carry the run's ``epoch``).  Score and lives are functions of them.
    Bullets PIERCE: the server cannot see which grain a bullet hit, so a bullet kills everything its path touches until it expires (its range).

PANELS stand across the road in front of the shooters.  They are glass: a bullet that flies through one HITS it, and every hit raises the panel's value
(shown on it).  Four kinds, two rules:
    PASS-THROUGH (``split``, ``damage``)  the panel keeps its value and every bullet that crosses it takes it at the value it had as the bullet
                  arrived:  split  -> the bullet leaves as ``floor(value)`` bullets, fanned;  damage -> the bullet's kill reach grows with its value (x value damage,
                  reach x sqrt(value)).  Bullets that cross several panels take each.
    COMPLETION    (``rate``, ``shooters``)  the value rises to its target; when it gets there the panel is SPENT and the player is upgraded for good:
                  rate -> the fire rate x ``amount``;  shooters -> ``amount`` more escort shooters (up to ``MAX_SHOOTERS``).
Hits count 1 point, a spent panel 25, a killed grain 10.

THE LAW.  Lane ``k`` of one dt-system entity span:  0 the player;  1..4 escort shooters (alive when ``shooters`` > their rank, flanking the player);
5..8 panels;  9..40 bullets (a pool: a volley that finds no free lane is cut short).  Columns: static ``lane slot home_x home_y rx ry pkind pstep ptarget
pamount pcap eoff erank``; state ``alive x y vx vy born dmg reload pval phits flash`` and game-wide ``score lives clock rate_mult shooters ncomp``;
host settings ``field_w dead goal autofire``; controls ``move_axis steer_x fire``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from meowdoku import fountain as F
from meowdoku.entities import EntityGame, LawPiece
from meowdoku.entity_motion import damped_axis
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

WIDTH, HEIGHT = F.MOB_BOARD
PLAYER, ESCORT, PANEL, BULLET, TOWER = 0.0, 1.0, 2.0, 3.0, 4.0
SPLIT, DAMAGE, RATE, SHOOTERS, WEAPON, UNIT = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
KIND_NAMES = {SPLIT: "split", DAMAGE: "damage", RATE: "rate", SHOOTERS: "shooters", WEAPON: "weapon", UNIT: "unit"}
STABLE_PANELS, PICKUPS, TOWERS = 4, 4, 6
ESCORTS, PANELS, BULLETS = 4, STABLE_PANELS + PICKUPS, 32      # the panel lanes: the level's stable multipliers, then the descending pickups
ESCORT0 = 1
PANEL0 = ESCORT0 + ESCORTS
BULLET0 = PANEL0 + PANELS
TOWER0 = BULLET0 + BULLETS
LANES = TOWER0 + TOWERS
ESCORT_OFFSETS = (-1.3, 1.3, -2.6, 2.6)
MAX_SHOOTERS = 1 + ESCORTS
PLAYER_Y = HEIGHT - 0.9
PLAYER_SPEED, SHIP_TAU, SHIP_REACH = 9.0, 0.05, 0.18
MUZZLE = 0.7
BULLET_SPEED = 12.0                   # board units / s (straight up the road)
BULLET_LIFE = 0.9                     # s: a bullet's range is 10.8 board units: from the shooters up through the peg field to the funnel's mouth
BASE_RATE = 3.0                       # shots per second per shooter at rate x1
BOLT_SIZE = 0.26                      # the bolt's drawn size at damage 1 ...
KILL_REACH = 0.45                     # ... and its kill reach (board units): reach = KILL_REACH * sqrt(damage), size = BOLT_SIZE * sqrt(damage)
REACH_PER_SIZE = KILL_REACH / BOLT_SIZE
FAN = 0.12                            # radians between the bullets a split leaves as
MAX_COPIES = 6.0
POINTS_KILL, POINTS_PANEL, POINTS_HIT = 10.0, 25.0, 1.0
LOSE_AT = 40.0                        # grains that reach the player that end the run
PANEL_THICK = 0.18

#: The level's panels: kind, centre (board units), half width, start value, rise per hit, completion target, effect amount, value cap.
LEVEL_PANELS = (
    {"kind": SPLIT, "x": 3.4, "y": 15.4, "hw": 2.3, "v0": 2.0, "step": 0.1, "target": 0.0, "amount": 0.0, "cap": 5.0},
    {"kind": RATE, "x": 9.1, "y": 15.4, "hw": 2.3, "v0": 1.0, "step": 0.05, "target": 2.0, "amount": 2.0, "cap": 2.0},
    {"kind": SPLIT, "x": 6.25, "y": 13.6, "hw": 2.0, "v0": 2.0, "step": 0.1, "target": 0.0, "amount": 0.0, "cap": 5.0},
    {"kind": RATE, "x": 9.4, "y": 12.8, "hw": 2.0, "v0": 1.0, "step": 0.05, "target": 2.0, "amount": 2.0, "cap": 2.0},
)

#: The DESCENDING PICKUPS (the panels' lanes after the stable ones; ``t0`` the clock second each appears at the top, ``eoff`` in the law): they drift down the
#: road at ``PICK_SPEED`` and are placed in the side lanes, so collecting one takes the guns OFF the mob.  UNIT gates start negative (value in 'units / UNIT_PER'
#: steps), every bullet that hits one raises it, and it counts only if it CROSSES THE PLAYER on its way down (it then adds ``floor(value / UNIT_PER)``
#: shooters, or takes them while still negative).  WEAPON pickups take ``target`` hits and apply at once (the fire rate x ``amount``).
PICK_SPEED = 0.9
UNIT_PER = 4.0
PICK_Y0 = -0.5
LEVEL_PICKUPS = (
    {"kind": UNIT, "x": 1.7, "hw": 1.5, "v0": -8.0, "step": 1.0, "target": 0.0, "amount": 0.0, "cap": 40.0, "t0": 6.0},
    {"kind": WEAPON, "x": 11.0, "hw": 1.1, "v0": 0.0, "step": 1.0, "target": 3.0, "amount": 1.5, "cap": 3.0, "t0": 14.0},
    {"kind": UNIT, "x": 10.8, "hw": 1.5, "v0": -12.0, "step": 1.0, "target": 0.0, "amount": 0.0, "cap": 40.0, "t0": 28.0},
    {"kind": WEAPON, "x": 1.4, "hw": 1.1, "v0": 0.0, "step": 1.0, "target": 3.0, "amount": 1.5, "cap": 3.0, "t0": 42.0},
)
ALL_PANELS = LEVEL_PANELS + LEVEL_PICKUPS

#: TOWERS (placed by the player on the field, server-authoritative): kind -> cost (score), law kind code (in ``pval``), fire interval (s).
TOWER_KINDS = {"gun": {"cost": 60.0, "code": 0.0, "interval": 0.7}, "rapid": {"cost": 120.0, "code": 1.0, "interval": 0.35}}
TOWER_R = 0.5                                                      # board units: a tower's footprint
TOWER_ZONE_Y = (3.0, 17.5)                      # board rows a tower may stand in (between the funnel and the shooters)

CONTROLS = {"move_axis": "move-axis-x", "steer_x": "pointer-board-x", "fire": "primary-action"}
STATE = ("alive", "x", "y", "vx", "vy", "born", "dmg", "reload", "pval", "phits", "flash",
         "score", "lives", "clock", "rate_mult", "shooters", "ncomp")
STATIC = ("lane", "slot", "home_x", "home_y", "rx", "ry", "pkind", "pstep", "ptarget", "pamount", "pcap", "eoff", "erank")
SETTINGS = ("field_w", "dead", "goal", "autofire", "spent")


def panel_effect(kind: float, value: float) -> dict[str, float]:
    """What a bullet takes crossing a pass-through panel at ``value`` (the rule, plain: the law does the same on its spans)."""
    if kind == SPLIT:
        return {"copies": float(min(max(math.floor(value), 1.0), MAX_COPIES)), "damage": 1.0}
    if kind == DAMAGE:
        return {"copies": 1.0, "damage": float(value)}
    return {"copies": 1.0, "damage": 1.0}


def mobroad_step(lane, slot, home_x, home_y, rx, ry, pkind, pstep, ptarget, pamount, pcap, eoff, erank,
                 alive, x, y, vx, vy, born, dmg, reload, pval, phits, flash,
                 score, lives, clock, rate_mult, shooters, ncomp,
                 field_w, dead, goal, autofire, spent,
                 move_axis, steer_x, fire, dt):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span."""
    n = int(lane.shape[0])
    f = lambda mask: mask.to_dtype("float64")  # noqa: E731
    live = alive > 0.0
    playing = lives > 0.0
    is_player, is_escort, is_panel, is_bullet, is_tower = slot == PLAYER, slot == ESCORT, slot == PANEL, slot == BULLET, slot == TOWER
    clk = (f(is_player) * clock).sum()
    playing_s = (f(is_player) * lives).sum() > 0.0
    clock_next = clock + dt
    fw = (f(is_player) * field_w).sum()
    rate_now, shooters_now = (f(is_player) * rate_mult).sum(), (f(is_player) * shooters).sum()
    ncomp_now = (f(is_player) * ncomp).sum()

    # The shooters: the player by the shared damped motion law (entity_motion.damped_axis, the text the page predicts its ship with), the escorts
    # flanking it; an escort lives while the shooter count is above its rank.
    ship_x, ship_v = damped_axis(x, vx, move_axis, steer_x, x * 0.0 + 0.7, field_w - 0.7, PLAYER_SPEED, SHIP_TAU, SHIP_REACH, dt)
    player_x = (f(is_player) * ship_x).sum()
    want = player_x + eoff
    esc_x = where(want < 0.7, 0.7, where(want > fw - 0.7, fw - 0.7, want))
    esc_alive = is_escort & (shooters_now > erank)
    shooter = playing & ((is_player & live) | esc_alive | (is_tower & live))
    sx = where(is_player, ship_x, where(is_escort, esc_x, x))

    # Fire: a held button (or AUTOFIRE) and a ready shooter loose one bullet each, into the first free bullet lanes (ranked).
    firing = shooter & (((fire > 0.0) | (autofire > 0.0)) | is_tower) & (reload <= 0.0)
    fire_rank = f(firing).cumsum(dim=0) - 1.0
    n_fire = f(firing).sum()
    interval = where(is_tower, where(pval > 0.5, TOWER_KINDS["rapid"]["interval"], TOWER_KINDS["gun"]["interval"]), 1.0 / (BASE_RATE * rate_now))
    reload_next = where(firing, interval, (reload - dt).clip(0.0, 10.0))
    free_b = is_bullet & ~live
    b_rank = f(free_b).cumsum(dim=0) - 1.0
    spawn_s = free_b & (b_rank < n_fire)
    take_s = f(spawn_s.reshape((n, 1)) & firing.reshape((1, n)) & (b_rank.reshape((n, 1)) == fire_rank.reshape((1, n))))
    spawn_x = (take_s * sx.reshape((1, n))).sum(dim=1)
    spawn_y = (take_s * where(is_tower, y, y * 0.0 + (PLAYER_Y - MUZZLE)).reshape((1, n))).sum(dim=1)

    # Bullets fly straight; PANELS: a bullet that crosses a panel's line within its width HITS it (once: the crossing is a sign change).
    bl = is_bullet & live
    xb, yb = x + vx * dt, y + vy * dt
    dmg_mult = x * 0.0 + 1.0
    copies = x * 0.0 + 1.0
    pval_n, alive_n, phits_n, born_n, y_n = pval, alive, phits, born, y
    flash_n = (flash - dt * 4.0).clip(0.0, 1.0)
    rate_n, shoot_n, ncomp_n = rate_now, shooters_now, ncomp_now
    for k in range(PANEL0, PANEL0 + PANELS):
        sel = lane == float(k)
        pick = lambda col: (f(sel) * col).sum()  # noqa: E731
        px, py_home, prx = pick(home_x), pick(home_y), pick(rx)
        kind, step, target, amount, cap = pick(pkind), pick(pstep), pick(ptarget), pick(pamount), pick(pcap)
        pv, pa_raw = pick(pval), pick(alive) > 0.0
        # a PICKUP (kind >= WEAPON) is on the road from its spawn second (``eoff``) until it is spent or past the field, and its place is a function of the clock
        is_pick = kind >= WEAPON
        spawn_t = pick(eoff)
        py = where(is_pick, py_home + PICK_SPEED * (clk + dt - spawn_t).clip(0.0, 1.0e6), py_home)
        pa = where(is_pick, (clk >= spawn_t) & (pick(born) < 0.5) & (py < HEIGHT), pa_raw)
        cross = bl & pa & playing & (y > py) & (yb <= py) & ((xb - px).abs() <= prx)
        hits = f(cross).sum()
        dmg_mult = dmg_mult * where(cross & (kind == DAMAGE), pv, 1.0)
        copies = copies * where(cross & (kind == SPLIT), pv.floor().clip(1.0, MAX_COPIES), 1.0)
        new = where(pv + step * hits > cap, cap, pv + step * hits)
        completes = pa & (kind >= RATE) & (kind <= WEAPON) & (hits > 0.0) & (new >= target - 1.0e-9)
        # a UNIT gate counts only when it CROSSES THE PLAYER: it reaches the shooters' row within the player's reach on its way down
        crossed = pa & (kind == UNIT) & playing_s & (py >= PLAYER_Y - 0.4) & ((px - player_x).abs() <= prx + 0.6)
        pval_n = where(sel, new, pval_n)
        phits_n = where(sel, pick(phits) + hits, phits_n)
        flash_n = where(sel & ((hits > 0.0) | crossed), 1.0, flash_n)
        alive_val = where(is_pick, f(pa & ~completes & ~crossed), where(completes, 0.0, pick(alive)))
        alive_n = where(sel, alive_val, alive_n)
        born_n = where(sel & (completes | crossed), 1.0, born_n)
        y_n = where(sel & is_pick, py, y_n)
        rate_n = rate_n * where(completes & ((kind == RATE) | (kind == WEAPON)), amount, 1.0)
        shoot_n = shoot_n + where(completes & (kind == SHOOTERS), amount, 0.0) + where(crossed, (new * (1.0 / UNIT_PER)).floor(), 0.0)
        ncomp_n = ncomp_n + f(completes) + f(crossed)
    shoot_n = where(shoot_n > float(MAX_SHOOTERS), float(MAX_SHOOTERS), where(shoot_n < 1.0, 1.0, shoot_n))

    # A SPLIT leaves a crossing bullet as ``copies`` bullets fanned about its heading: the first keeps the bullet's lane, the others take the first free
    # lanes left (ranked, after the volley's): a pool that is full cuts the split short.
    extras = where(bl & (copies > 1.0), copies - 1.0, 0.0)
    free_c = free_b & ~spawn_s
    c_rank = f(free_c).cumsum(dim=0) - 1.0
    offset = extras.cumsum(dim=0) - extras
    take_c = (free_c.reshape((n, 1)) & (extras.reshape((1, n)) > 0.0) & (c_rank.reshape((n, 1)) >= offset.reshape((1, n)))
              & (c_rank.reshape((n, 1)) < (offset + extras).reshape((1, n))))
    take_cf = f(take_c)
    child = take_cf.sum(dim=1) > 0.0

    def from_source(value):
        return (take_cf * value.reshape((1, n))).sum(dim=1)

    angle_src = vx.atan2(0.0 - vy)
    m_idx = c_rank - from_source(offset)
    c_angle = from_source(angle_src) + FAN * (m_idx + 1.0 - 0.5 * from_source(extras))
    o_angle = angle_src - FAN * 0.5 * extras
    own_turn = bl & (extras > 0.0)

    x1 = where(spawn_s, spawn_x, where(child, from_source(xb), where(bl, xb, x)))
    y1 = where(spawn_s, spawn_y, where(child, from_source(yb), where(bl, yb, y)))
    vx1 = where(spawn_s, 0.0, where(child, BULLET_SPEED * c_angle.sin(), where(own_turn, BULLET_SPEED * o_angle.sin(), vx)))
    vy1 = where(spawn_s, 0.0 - BULLET_SPEED, where(child, 0.0 - BULLET_SPEED * c_angle.cos(), where(own_turn, 0.0 - BULLET_SPEED * o_angle.cos(), vy)))
    dmg1 = where(spawn_s, 1.0, where(child, from_source(dmg * dmg_mult), where(bl, dmg * dmg_mult, dmg)))
    born1 = where(spawn_s | child, clock, born)
    bullet_live = (bl | spawn_s | child) & ~((y1 < -0.5) | (x1 < -0.5) | (x1 > fw + 0.5) | ((clock_next - born1) > BULLET_LIFE))

    x_out = where(is_player, ship_x, where(is_escort, esc_x, where(is_bullet, x1, x)))
    y_out = where(is_bullet, y1, where(is_panel, y_n, y))
    vx_out = where(is_player | is_escort, ship_v, where(is_bullet, vx1, vx))
    vy_out = where(is_bullet, vy1, vy)
    alive_out = where(is_panel, alive_n, where(is_escort, f(esc_alive), where(is_bullet, f(bullet_live), alive)))
    dmg_out = where(is_bullet, dmg1, dmg)
    born_out = where(is_bullet, born1, where(is_panel, born_n, born))

    # Score and lives are functions of the mob's reported counts (host settings) and the panels' own.
    dead_now, goal_now = dead.max(), goal.max()
    score_next = POINTS_KILL * dead_now + POINTS_PANEL * ncomp_n + phits_n.sum() - spent.max()
    lives_next = (LOSE_AT - goal_now).clip(0.0, LOSE_AT)
    residual = x * 0.0
    dt_limit = x * 0.0 + 0.05
    return (alive_out, x_out, y_out, vx_out, vy_out, born_out, dmg_out, reload_next, pval_n, phits_n, flash_n,
            score_next, lives_next, clock_next, rate_n, shoot_n, ncomp_n, residual, dt_limit)


class MobroadPiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "mobroad_step"
    law = staticmethod(mobroad_step)
    argument_names = (*STATIC, *STATE, *SETTINGS, *CONTROLS, "dt")
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


def layout(width: float = WIDTH) -> dict[str, np.ndarray]:
    """The static columns: each lane's slot, home and size, and each panel's rule."""
    z = lambda: np.zeros(LANES)  # noqa: E731
    slot, home_x, home_y, rx, ry = z(), np.full(LANES, width / 2.0), np.full(LANES, PLAYER_Y), np.full(LANES, 0.4), np.full(LANES, 0.4)
    pkind, pstep, ptarget, pamount, pcap, eoff, erank = z(), z(), z(), z(), z(), z(), z()
    rx[0], ry[0] = 0.3, 0.6
    for i in range(ESCORTS):
        k = ESCORT0 + i
        slot[k], eoff[k], erank[k], rx[k], ry[k] = ESCORT, ESCORT_OFFSETS[i], i + 1, 0.25, 0.5
    for i, p in enumerate(ALL_PANELS[:PANELS]):
        k = PANEL0 + i
        slot[k], home_x[k], home_y[k], rx[k], ry[k] = PANEL, p["x"], p.get("y", PICK_Y0), p["hw"], PANEL_THICK
        pkind[k], pstep[k], ptarget[k], pamount[k], pcap[k] = p["kind"], p["step"], p["target"], p["amount"], p["cap"]
        eoff[k] = p.get("t0", 0.0)                                  # (a pickup's spawn second)
    for k in range(BULLET0, TOWER0):
        slot[k], rx[k], ry[k] = BULLET, 0.1, 0.3
    for k in range(TOWER0, LANES):
        slot[k], rx[k], ry[k] = TOWER, TOWER_R, TOWER_R
    return {"lane": np.arange(LANES, dtype=np.float64), "slot": slot, "home_x": home_x, "home_y": home_y, "rx": rx, "ry": ry, "pkind": pkind,
            "pstep": pstep, "ptarget": ptarget, "pamount": pamount, "pcap": pcap, "eoff": eoff, "erank": erank}


def initial_columns(width: float = WIDTH, autofire: bool = True) -> dict[str, np.ndarray]:
    columns = layout(width)
    slot = columns["slot"]
    alive = (slot == PLAYER).astype(np.float64)
    pval = np.zeros(LANES)
    for i, p in enumerate(ALL_PANELS[:PANELS]):
        alive[PANEL0 + i], pval[PANEL0 + i] = (1.0 if p["kind"] < WEAPON else 0.0), p["v0"]
    columns.update({
        "alive": alive, "x": columns["home_x"].copy(), "y": columns["home_y"].copy(), "vx": np.zeros(LANES), "vy": np.zeros(LANES),
        "born": np.zeros(LANES), "dmg": np.ones(LANES), "reload": np.zeros(LANES), "pval": pval, "phits": np.zeros(LANES), "flash": np.zeros(LANES),
        "score": np.zeros(LANES), "lives": np.full(LANES, LOSE_AT), "clock": np.zeros(LANES), "rate_mult": np.ones(LANES),
        "shooters": np.ones(LANES), "ncomp": np.zeros(LANES),
        "field_w": np.full(LANES, float(width)), "dead": np.zeros(LANES), "goal": np.zeros(LANES), "autofire": np.full(LANES, 1.0 if autofire else 0.0), "spent": np.zeros(LANES),
        "move_axis": np.zeros(LANES), "steer_x": np.full(LANES, -1.0), "fire": np.zeros(LANES),
    })
    return columns


_SPECS: dict[str, dict[str, Any]] = {}
_ROADU: dict[str, float] = {}
_GRIDS: dict[int, dict[str, Any]] = {}


def _ROAD_UNIFORMS() -> dict[str, float]:
    """The road's SDF slot uniforms (the law's own chamber), built once."""
    if not _ROADU:
        _ROADU.update(F.mobroad_uniforms(256))
    return _ROADU


def _grid_for(game) -> dict[str, Any]:
    if 0 not in _GRIDS:
        _GRIDS[0] = game.tower_grid()
    return _GRIDS[0]


class Mobroad(EntityGame):
    game = "mobroad"
    title = "Mobroad (road-defence mob shooter)"
    width, height = WIDTH, HEIGHT
    CONTROLS = tuple(CONTROLS)
    PLAYER_LANE = 0
    HEALTH_MAX = LOSE_AT
    MODES = {"autofire": "Auto fire"}
    SCENE = {"floor": True, "glass": False, "arcade": False}
    SCENE_PREFER = {"floor": False}

    def __init__(self, board_id: str = "", aspect: float | None = None, size: str = "4k", autofire: bool = True):
        if size not in ("small", "4k"):
            raise ValueError(f"mobroad: the road is built for the sizes small and 4k, not {size!r}")
        self.size, self.autofire = size, bool(autofire)
        self.active = F.SIZES[size][0]
        self.mob_epoch = 1                 # the page restarts the mob (the compute run) when this changes
        self.mob = {"dead": 0, "goal": 0, "crush": 0, "live": 0}
        self.selected, self.last_place = "gun", None          # the tower kind the side menu has chosen; the outcome of the last drop
        super().__init__(board_id)

    # -------------------------------------------------------------- the law
    def make_piece(self):
        return MobroadPiece()

    def initial_columns(self) -> dict[str, np.ndarray]:
        return initial_columns(WIDTH, self.autofire)

    def modes(self) -> dict[str, bool]:
        return {"autofire": self.autofire}

    def set_mode(self, name: str, on: bool) -> None:
        if name != "autofire":
            raise ValueError(f"no mode {name!r}")
        self.autofire = bool(on)
        self._set("autofire", 1.0 if self.autofire else 0.0)
        self.epoch += 1

    def layout(self, aspect: float) -> None:
        """The board is fixed (the road's window maps onto it): the camera fits it to the screen."""

    def motion(self) -> dict[str, Any]:
        return {"speed": PLAYER_SPEED, "tau": SHIP_TAU, "reach": SHIP_REACH, "x": [0.7, self.width - 0.7]}

    # ------------------------------------------------------------- towers (server-authoritative)
    def set_hero(self, key) -> None:
        """The side menu's cube tap (the character tile set's ``avatar`` setting) chooses the TOWER kind here."""
        if key in TOWER_KINDS:
            self.selected = str(key)

    def _lane_set(self, name: str, lane: int, value: float) -> None:
        target = self.session.columns[name] if self.session.state is None else getattr(self.session.state, name)
        target[lane] = float(value)

    def tower_check(self, kind: str, bx: float, by: float) -> str | None:
        """Why a ``kind`` tower cannot stand at board ``(bx, by)`` (None: it can): unknown kind, outside the field or the zone, on a wall or a solid of the road,
        overlapping another tower, no free lane, or too dear."""
        spec = TOWER_KINDS.get(kind)
        if spec is None:
            return "unknown kind"
        if not (math.isfinite(bx) and math.isfinite(by)):
            return "not a place"
        if not (TOWER_R <= bx <= self.width - TOWER_R) or not (TOWER_ZONE_Y[0] <= by <= TOWER_ZONE_Y[1]):
            return "outside the zone"
        k = F.mobroad_map()["k"]
        fx, fy = bx / k - F.MOB_ROAD_HALF, F.MOB_ROAD_TOP - by / k
        if float(F.chamber_distance(_ROAD_UNIFORMS(), np.array([fx]), np.array([fy]))[0][0]) < TOWER_R / k:
            return "in a wall"
        alive, x, y = (self.session.column(n) for n in ("alive", "x", "y"))
        free = False
        for lane in range(TOWER0, LANES):
            if alive[lane] > 0.0:
                if math.hypot(float(x[lane]) - bx, float(y[lane]) - by) < 2.0 * TOWER_R:
                    return "overlaps a tower"
            else:
                free = True
        if not free:
            return "no free lane"
        if float(self.session.column("score")[0]) < spec["cost"]:
            return "too dear"
        return None

    def place(self, payload: dict) -> dict:
        """A drop from the page: ``{x, y, kind?}`` in board units.  Validity and cost are decided HERE; a valid drop takes a free tower lane (it fires from the
        next round, into the same bullet lanes as the shooters) and its cost is added to ``spent`` (the score is a function of it)."""
        try:
            kind, bx, by = str(payload.get("kind") or self.selected), float(payload["x"]), float(payload["y"])
        except (KeyError, TypeError, ValueError):
            self.last_place = {"ok": False, "reason": "bad request"}
            return self.last_place
        why = self.tower_check(kind, bx, by)
        if why is not None:
            self.last_place = {"ok": False, "reason": why, "kind": kind}
            return self.last_place
        alive = self.session.column("alive")
        lane = next(k for k in range(TOWER0, LANES) if alive[k] <= 0.0)
        for name, value in (("alive", 1.0), ("x", bx), ("y", by), ("vx", 0.0), ("vy", 0.0), ("pval", TOWER_KINDS[kind]["code"]), ("reload", 0.0)):
            self._lane_set(name, lane, value)
        self._set("spent", float(self.session.column("spent")[0]) + TOWER_KINDS[kind]["cost"])
        self.last_place = {"ok": True, "lane": lane, "kind": kind, "cost": TOWER_KINDS[kind]["cost"]}
        return self.last_place

    def tower_grid(self) -> dict[str, Any]:
        """The page's ghost tint: a coarse grid of the cells where a tower may stand as far as the ROAD is concerned (the server still decides at the drop)."""
        step = 0.5
        k = F.mobroad_map()["k"]
        cols, rows = int(self.width / step), int(self.height / step)
        xs = (np.arange(cols) + 0.5) * step
        ys = (np.arange(rows) + 0.5) * step
        gx, gy = np.meshgrid(xs, ys)
        d = F.chamber_distance(_ROAD_UNIFORMS(), (gx / k - F.MOB_ROAD_HALF).ravel(), (F.MOB_ROAD_TOP - gy / k).ravel())[0].reshape(gx.shape)
        ok = (d >= TOWER_R / k) & (gx >= TOWER_R) & (gx <= self.width - TOWER_R) & (gy >= TOWER_ZONE_Y[0]) & (gy <= TOWER_ZONE_Y[1])
        return {"step": step, "cols": cols, "rows": rows, "ok": "".join("1" if v else "0" for v in ok.ravel())}

    def reset(self) -> None:
        super().reset()
        self.last_place = None
        self.mob_epoch += 1
        self.mob = {"dead": 0, "goal": 0, "crush": 0, "live": 0}

    def _set(self, name: str, value: float) -> None:
        target = self.session.columns[name] if self.session.state is None else getattr(self.session.state, name)
        target[...] = float(value)

    # ------------------------------------------------------------ the link
    def control(self, payload: dict) -> None:
        super().control(payload)
        mob = payload.get("mob")
        if isinstance(mob, dict):
            self.report(mob)

    def report(self, mob: dict) -> bool:
        """The page's report of the mob law's cumulative counts (``n_dead``, ``n_goal``, and the live ``n_crush`` / ``n_live``).  Counted only for the
        run it was made in (``epoch``), only ever upward, and never past the id count."""
        try:
            if int(mob.get("epoch", -1)) != self.mob_epoch:
                return False
            dead = min(max(int(mob.get("dead", 0)), self.mob["dead"]), self.active)
            goal = min(max(int(mob.get("goal", 0)), self.mob["goal"]), self.active - dead)
        except (TypeError, ValueError):
            return False
        self.mob = {"dead": dead, "goal": goal, "crush": max(int(mob.get("crush", 0)), 0), "live": max(int(mob.get("live", 0)), 0)}
        self._set("dead", dead)
        self._set("goal", goal)
        return True

    # ---------------------------------------------------------- the picture
    def palette(self) -> dict[str, list[float]]:
        from meowdoku import characters
        return {"floor": [0.07, 0.08, 0.11], "rocket": [0.85, 0.88, 0.95], "escort": [0.55, 0.85, 1.0], "bolt": [1.0, 0.92, 0.25],
                "bolt_hot": [1.0, 0.55, 0.15], "tower": [0.9, 0.75, 0.3], "tower_rapid": [0.95, 0.45, 0.25], **characters.PALETTE}

    def entities(self) -> list[dict[str, Any]]:
        col = {name: self.session.column(name) for name in ("slot", "alive", "x", "y", "vx", "vy", "born", "dmg", "clock")}
        out = []
        for k in range(LANES):
            slot = col["slot"][k]
            if col["alive"][k] <= 0.0 or slot == PANEL:
                continue
            if slot == TOWER:
                rapid = float(self.session.column("pval")[k]) > 0.5
                out.append({"id": f"{k}:tower", "x": float(col["x"][k]), "y": float(col["y"][k]), "vx": 0.0, "vy": 0.0, "mesh": "rocket",
                            "color": "tower_rapid" if rapid else "tower", "size": 0.8, "roll": 0.0, "lean": 0.0, "owner": -1, "age": 0.0, "spin": 0.0, "bob": 0.0})
                continue
            if slot == PLAYER:
                mesh, color, size = "rocket", "rocket", 0.6
            elif slot == ESCORT:
                mesh, color, size = "rocket", "escort", 0.45
            else:
                d = float(col["dmg"][k])
                mesh, color, size = "bolt", ("bolt_hot" if d > 1.5 else "bolt"), BOLT_SIZE * math.sqrt(d)
            out.append({"id": f"{k}:{col['born'][k]:.4f}", "x": float(col["x"][k]), "y": float(col["y"][k]), "vx": float(col["vx"][k]),
                        "vy": float(col["vy"][k]), "mesh": mesh, "color": color, "size": size, "roll": 0.785 if mesh == "rocket" else 0.0,
                        "lean": 0.25 if slot != BULLET else 0.55, "owner": -1, "age": float(col["clock"][k] - col["born"][k]), "spin": 0.0, "bob": 0.0})
        return out

    def panels(self) -> list[dict[str, Any]]:
        col = {name: self.session.column(name) for name in ("alive", "pval", "phits", "flash", "pkind", "home_x", "home_y", "rx", "ptarget", "pamount", "pcap", "y")}
        out = []
        for i in range(PANELS):
            k = PANEL0 + i
            kind = float(col["pkind"][k])
            y = float(col["y"][k]) if kind >= WEAPON else float(col["home_y"][k])
            if kind >= WEAPON and col["alive"][k] <= 0.0:
                continue                                                  # a pickup not yet on the road, or spent
            out.append({"id": k, "kind": KIND_NAMES[kind], "x": float(col["home_x"][k]), "y": y, "hw": float(col["rx"][k]), "descending": kind >= WEAPON,
                        "thick": PANEL_THICK, "value": round(float(col["pval"][k]), 4), "target": float(col["ptarget"][k]), "amount": float(col["pamount"][k]),
                        "hits": int(col["phits"][k]), "flash": round(float(col["flash"][k]), 3), "alive": bool(col["alive"][k] > 0.0)})
        return out

    def compute_spec(self) -> dict[str, Any]:
        """The mob's compute spec (the fountain's, the road variant): the page runs the law from this and restarts it when ``epoch`` moves."""
        spec = _SPECS.get(self.size) or _SPECS.setdefault(self.size, F.mobroad_spec(self.size))         # (built once: the outline is a marching-squares trace)
        return {"manifest": "fountain_mobroad.manifest.json", "size": self.size, "epoch": self.mob_epoch, "uniforms": spec["uniforms"], "view": spec["view"],
                "aspect": None, "chamber_outline": spec["chamber_outline"], "body": None, "speed": 1, "named": ["ir", "ig", "ib"], "run": "continuous",
                "tilt": spec["tilt"], "mobroad": spec["map"], "pather": spec["pather"]}

    def state(self) -> dict[str, Any]:
        st = super().state()
        col = self.session.column
        st["compute"] = self.compute_spec()
        st["pieces"] = [{"id": f"char:{key}", "cell": f"char:{i}", "mesh": f"hero_{m}", "color": f"hero_{m}", "character": key, "lit": key == self.selected}
                        for i, (key, m) in enumerate(zip(TOWER_KINDS, ("owl", "fox")))]
        st["characters"] = {"chosen": self.selected, "keys": list(TOWER_KINDS)}
        st["mobroad"] = {
            "size": self.size, "active": self.active, "epoch": self.mob_epoch, "panels": self.panels(), "map": F.mobroad_map(), "reach_per_size": REACH_PER_SIZE,
            "rate": float(col("rate_mult")[0]), "shooters": int(col("shooters")[0]), "dead": self.mob["dead"], "goal": self.mob["goal"], "crush": self.mob["crush"],
            "live": self.mob["live"], "lose_at": LOSE_AT,
            "towers": {"kinds": {k: v["cost"] for k, v in TOWER_KINDS.items()}, "selected": self.selected, "radius": TOWER_R, "last": self.last_place,
                       "grid": _grid_for(self)}, "bullet_life": BULLET_LIFE, "over": self.result() is not None}
        return st

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None:
            return None
        won = self.mob["dead"] + self.mob["goal"] >= self.active and s.column("lives")[0] > 0.0
        if s.column("lives")[0] > 0.0 and not won:
            return None
        return {"score": float(s.column("score")[0]), "won": bool(won), "level": 1,
                "detail": f"{'held the road' if won else 'overrun'}: {self.mob['dead']} of {self.active} down, {self.mob['goal']} got through"}

    def describe(self) -> str:
        s = self.session
        if s.state is None:
            return "Mobroad"
        col = s.column
        res = self.result()
        head = ("ROAD HELD" if res and res["won"] else "OVERRUN · R to restart") if res else f"{int(col('lives')[0])} to go"
        return (f"{head} · score {int(col('score')[0])} · down {self.mob['dead']}/{self.active} · crush {self.mob['crush']} · "
                f"shooters {int(col('shooters')[0])} · rate x{float(col('rate_mult')[0]):g}")
