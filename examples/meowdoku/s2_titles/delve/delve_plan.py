"""Gravity-aware traversal of a room: the generator's proof that a door can be reached.

``Mirror`` is the player's body physics of ``delve.delve_step`` (the same
arithmetic, over a batch of hypothetical players, every constant from
``delve_rules`` and the sideways motion the engine's own ``damped_axis``);
``tests/test_meowdoku_delve.py`` steps the real law and the mirror side by side
and requires them to agree.  ``RoomPlan`` runs the mirror from every place a
player can stand or hang (standing places every half tile, ladder cells) under
a fixed repertoire of control scripts -- walk off an edge, jump with the
button held for various times, with or without a running start, with a second
press for the double jump -- and records where each one lands.  The result is
a graph over standing/ladder places; a door is reachable when its place is.
Nothing here is a guess about arcs: an edge exists because the mirror's body
did it.  Each edge keeps its control script, so the probe can replay it
through the real law.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from meowdoku.s2_titles.delve.delve_rules import (
    BUFFER, CLIMB, COYOTE, EMPTY, FALL_MAX, FOOT, GRAVITY, HANG_SCALE, HANG_V, JUMP, JUMP2, JUMP_CUT, LADDER, ONEWAY, PLAYER_RX, PLAYER_RY,
    REACH, SOLID, SPEED, SPIKE, TAU)
from meowdoku.s1_engine.entity_motion import damped_axis, vertical_axis

PLAN_DT = 1.0 / 60.0                 # a step of the game's own tick; the law's ceiling is ``0.5 * TAU`` = 0.03
PLAN_STEPS = 130
Q = 2                                # standing places per tile (every half tile)
# The repertoire's times are derived from the law, never typed: the rise of a held jump takes RISE_T (ordinary
# gravity down to the hang band, scaled gravity through it), and the partial holds / the second press are fractions of it.
RISE_T = (JUMP - HANG_V) / GRAVITY + HANG_V / (GRAVITY * HANG_SCALE)
DJ_HOLD = 1.12 * RISE_T              # the second press is held to the top of the rise


class Mirror:
    """The player's body physics over a batch, against one room's tile grid (H, W)."""

    def __init__(self, grid: np.ndarray):
        self.grid = np.asarray(grid)
        self.h, self.w = self.grid.shape

    def tile(self, px, py):
        cx = np.clip(np.floor(px), 0, self.w - 1).astype(int)
        cy = np.clip(np.floor(py), 0, self.h - 1).astype(int)
        return self.grid[cy, cx]

    def step(self, s: dict, axis, fire, axy, ability, dt):
        """One law step of the player for every lane of ``s`` (x y vx vy buf coy air pf climb), in place.
        Returns ``(landed, spiked, double_jumped)`` masks."""
        t, rx, ry = self.tile, PLAYER_RX, PLAYER_RY
        x, y, vx, vy = s["x"], s["y"], s["vx"], s["vy"]
        feet = y + ry
        sl, sr = x - rx * FOOT, x + rx * FOOT
        below = feet + 0.05
        row_b = np.floor(below)
        on_solid = (t(sl, below) == SOLID) | (t(sr, below) == SOLID)
        on_plank = ((t(sl, below) == ONEWAY) | (t(sr, below) == ONEWAY)) & (feet <= row_b + 0.06)
        grounded = (on_solid | on_plank) & (vy >= -0.05)
        in_ladder = t(x, y) == LADDER
        press = (fire > 0.0) | ((axy < -0.5) & ~in_ladder)
        press_edge = press & (s["pf"] <= 0.0)
        want_climb = in_ladder & ((np.abs(axy) > 0.2) | (s["climb"] > 0.5)) & (fire <= 0.0)
        stands = grounded | (s["climb"] > 0.5)
        dbl = press_edge & ~(stands | (s["coy"] > 0.0)) & (s["air"] > 0.5) & ability
        y_free, v_free, coy_f, buf_f, jumped = vertical_axis(
            y, np.where(dbl, -JUMP2, vy), press.astype(float), s["pf"], stands.astype(float), s["coy"], s["buf"], dt,
            GRAVITY, JUMP, JUMP_CUT, FALL_MAX, HANG_V, HANG_SCALE, COYOTE, BUFFER)
        climbing = want_climb & ~(jumped > 0.5)
        s["air"] = np.where(stands, 1.0, np.where(dbl, 0.0, s["air"]))
        s["coy"] = np.where(dbl, 0.0, coy_f)
        s["buf"] = buf_f
        fall = np.where(climbing, np.clip(axy, -1.0, 1.0) * CLIMB, v_free)
        nx, ship_vx = damped_axis(x, vx, axis, -1.0, -3.0, self.w + 3.0, SPEED, TAU, REACH, dt)
        lead = nx + np.where(ship_vx >= 0.0, rx, -rx)
        hit_x = (t(lead, y - ry + 0.08) == SOLID) | (t(lead, y) == SOLID) | (t(lead, y + ry - 0.08) == SOLID)
        x1 = np.where(hit_x, x, nx)
        vx1 = np.where(hit_x, 0.0, ship_vx)
        ny = np.where(climbing, y + fall * dt, y_free)
        bottom = ny + ry
        row_land = np.floor(bottom)
        s_hit = (t(x1 - rx * FOOT, bottom) == SOLID) | (t(x1 + rx * FOOT, bottom) == SOLID)
        o_hit = (((t(x1 - rx * FOOT, bottom) == ONEWAY) | (t(x1 + rx * FOOT, bottom) == ONEWAY))
                 & (feet <= row_land + 0.02) & ~climbing)
        land = (fall > 0.0) & (s_hit | o_hit)
        top = ny - ry
        bump = (fall < 0.0) & ((t(x1 - rx * FOOT, top) == SOLID) | (t(x1 + rx * FOOT, top) == SOLID))
        y1 = np.where(land, row_land - ry - 1.0e-3, np.where(bump, np.floor(top) + 1.0 + ry + 1.0e-3, ny))
        vy1 = np.where(land | bump, 0.0, fall)
        spiked = ((t(x1, y1 + ry - 0.12) == SPIKE) | (t(x1 - rx * FOOT, y1 + ry - 0.12) == SPIKE)
                  | (t(x1 + rx * FOOT, y1 + ry - 0.12) == SPIKE) | (t(x1, y1) == SPIKE))
        s["x"], s["y"], s["vx"], s["vy"] = x1, y1, vx1, vy1
        s["pf"] = press.astype(float)
        s["climb"] = climbing.astype(float)
        return land, spiked, dbl


# ---------------------------------------------------------------- the repertoire
@dataclass(frozen=True)
class Variant:
    """A control script: the stick (``axis``) held throughout; ``run``: the player
    arrives at speed; ``jump``: press at t=0 and hold for ``hold`` seconds; ``dj``:
    a second press at that time (the double jump, needs the ability)."""

    axis: int
    run: bool
    jump: bool
    hold: float
    dj: float | None = None

    def pressed(self, tt: float) -> bool:
        if self.jump and tt < self.hold:
            return True
        return self.dj is not None and self.dj - 1e-9 <= tt < self.dj + DJ_HOLD


def repertoire() -> list[Variant]:
    out = []
    starts = [(0, False), (-1, False), (1, False), (-1, True), (1, True)]
    for axis, run in starts:
        for hold in (0.05, 0.1, 0.2, 9.0):
            out.append(Variant(axis, run, True, hold))
    for axis, run in starts:
        for hold in (0.59 * RISE_T, 0.81 * RISE_T):
            for gap in (0.03, 0.09, 0.18, 0.30):          # released for a moment, pressed again
                out.append(Variant(axis, run, True, hold, hold + 0.005 + gap))
    out += [Variant(-1, True, False, 0.0), Variant(1, True, False, 0.0), Variant(0, False, False, 0.0)]
    return out


# ---------------------------------------------------------------- places and edges
Place = tuple  # ('s', qx, feet_row) a standing place at x = qx / Q; ('l', cx, row) a ladder cell


@dataclass
class Edge:
    dst: Place
    needs_double: bool
    variant: Variant | None          # None: a plain walk or ladder move
    time: float = 0.0


@dataclass
class RoomPlan:
    grid: np.ndarray
    places: list[Place] = field(default_factory=list)
    edges: dict[Place, dict[Place, Edge]] = field(default_factory=dict)

    # -------------------------------------------------------------- build
    @classmethod
    def build(cls, grid: np.ndarray, doubles: bool = True) -> "RoomPlan":
        plan = cls(np.asarray(grid))
        plan._places()
        plan._walks()
        plan._arcs(doubles)
        return plan

    def standing_ok(self, qx: int, row: int) -> bool:
        g, (h, w) = self.grid, self.grid.shape
        x = qx / Q
        if not (1 <= row < h) or not (PLAYER_RX <= x <= w - PLAYER_RX):
            return False
        c0, c1 = int(np.floor(x - PLAYER_RX + 1e-9)), int(np.floor(x + PLAYER_RX - 1e-9))
        body = g[row - 1, c0:c1 + 1]
        if (body == SOLID).any() or (body == SPIKE).any():
            return False
        sl, sr = int(np.floor(x - PLAYER_RX * FOOT)), int(np.floor(x + PLAYER_RX * FOOT))
        return bool(g[row, sl] in (SOLID, ONEWAY) or g[row, sr] in (SOLID, ONEWAY))

    def _places(self) -> None:
        h, w = self.grid.shape
        for row in range(1, h):
            for qx in range(1, Q * w):
                if self.standing_ok(qx, row):
                    self.places.append(("s", qx, row))
        for cy, cx in zip(*np.nonzero(self.grid == LADDER)):
            self.places.append(("l", int(cx), int(cy)))
        self.edges = {p: {} for p in self.places}
        self.index = {p: i for i, p in enumerate(self.places)}

    def link(self, a: Place, b: Place, needs_double=False, variant=None, time=0.0) -> None:
        if a == b or a not in self.edges or b not in self.edges:
            return
        old = self.edges[a].get(b)
        if old is None or (old.needs_double and not needs_double) or (old.needs_double == needs_double and time < old.time):
            self.edges[a][b] = Edge(b, needs_double, variant, time)

    def _walks(self) -> None:
        for p in self.places:
            if p[0] == "s":
                _, qx, row = p
                for d in (-1, 1):
                    self.link(p, ("s", qx + d, row), time=0.5 / SPEED)
                cx = qx / Q - 0.5
                if abs(cx - round(cx)) < 1e-9 and ("l", int(round(cx)), row - 1) in self.edges:
                    # standing at a ladder cell's centre: step onto the ladder, and off it
                    self.link(p, ("l", int(round(cx)), row - 1), time=0.3)
                    self.link(("l", int(round(cx)), row - 1), p, time=0.3)
            else:
                _, cx, row = p
                for d in (-1, 1):
                    self.link(p, ("l", cx, row + d), time=1.0 / CLIMB)

    def _arcs(self, doubles: bool) -> None:
        """Run every variant from every place through the mirror; record where each lands."""
        variants = [v for v in repertoire() if doubles or v.dj is None]
        sources = [p for p in self.places]
        if not sources:
            return
        mirror = Mirror(self.grid)
        S, V = len(sources), len(variants)
        B = S * V
        src_kind = np.repeat(np.array([p[0] == "l" for p in sources]), V)
        sx = np.repeat(np.array([p[1] / Q if p[0] == "s" else p[1] + 0.5 for p in sources]), V)
        sy = np.repeat(np.array([p[2] - PLAYER_RY - 1.0e-3 if p[0] == "s" else p[2] + 0.5 for p in sources]), V)
        axis = np.tile(np.array([float(v.axis) for v in variants]), S)
        run = np.tile(np.array([v.run for v in variants]), S)
        jump = np.tile(np.array([v.jump for v in variants]), S)
        hold = np.tile(np.array([v.hold for v in variants]), S)
        dj = np.tile(np.array([v.dj if v.dj is not None else np.inf for v in variants]), S)
        s = {"x": sx.copy(), "y": sy.copy(), "vx": np.where(run & ~src_kind, axis * SPEED, 0.0), "vy": np.zeros(B),
             "buf": np.zeros(B), "coy": np.where(src_kind, 0.0, COYOTE), "air": np.ones(B), "pf": np.zeros(B),
             "climb": src_kind.astype(float)}
        airborne = np.zeros(B, bool)
        armed = np.zeros(B, bool)                       # has left every ladder cell (so re-entering is a grab)
        done = np.zeros(B, bool)
        used_dj = np.zeros(B, bool)
        result = np.full(B, -1, dtype=np.int64)         # index into a list of landing records
        landings: list[tuple[int, Place, float]] = []
        h, w = self.grid.shape
        for k in range(PLAN_STEPS):
            tt = k * PLAN_DT
            fire = ((jump & (tt < hold)) | ((dj - 1e-9 <= tt) & (tt < dj + DJ_HOLD))).astype(float)
            land, spiked, dbl = mirror.step(s, axis, fire, np.zeros(B), np.ones(B, bool), PLAN_DT)
            used_dj |= dbl & ~done
            in_lad = mirror.tile(s["x"], s["y"]) == LADDER
            grab = ~done & armed & in_lad
            armed |= ~in_lad
            landed = ~done & land & airborne
            airborne |= ~done & ~land
            bad = ~done & (spiked | (s["x"] < 0) | (s["x"] > w) | (s["y"] < 0) | (s["y"] > h))
            for i in np.nonzero(grab & ~bad)[0]:
                landings.append((int(i), ("l", int(np.floor(s["x"][i])), int(np.floor(s["y"][i]))), tt))
            for i in np.nonzero(landed & ~bad)[0]:
                landings.append((int(i), ("s", int(round(s["x"][i] * Q)), int(round(s["y"][i] + PLAYER_RY))), tt))
            done |= grab | landed | bad
            if done.all():
                break
        for i, dst, tt in landings:
            si, vi = divmod(i, V)
            if dst in self.edges:
                self.link(sources[si], dst, needs_double=bool(used_dj[i]), variant=variants[vi], time=tt)

    # -------------------------------------------------------------- queries
    def reach(self, starts, ability: bool) -> set:
        """Every place reachable from ``starts`` (single jumps; with ``ability`` also double jumps)."""
        seen = {p for p in starts if p in self.edges}
        stack = list(seen)
        while stack:
            p = stack.pop()
            for q, e in self.edges[p].items():
                if q not in seen and (ability or not e.needs_double):
                    seen.add(q)
                    stack.append(q)
        return seen

    def path(self, a: Place, b: Place, ability: bool) -> list[Edge] | None:
        """The quickest route from ``a`` to ``b`` (a list of edges), or None."""
        import heapq
        prev, best, queue = {a: None}, {a: 0.0}, [(0.0, 0, a)]
        tick = 0
        while queue:
            cost, _, p = heapq.heappop(queue)
            if p == b:
                break
            if cost > best.get(p, 1e18):
                continue
            for q, e in self.edges[p].items():
                if ability or not e.needs_double:
                    c = cost + max(e.time, 0.05)
                    if c < best.get(q, 1e18):
                        best[q], prev[q] = c, (p, e)
                        tick += 1
                        heapq.heappush(queue, (c, tick, q))
        if b not in prev:
            return None
        out, cur = [], b
        while prev[cur] is not None:
            cur, e = prev[cur][0], prev[cur][1]
            out.append(e)
        return out[::-1]
