"""Scripted teachers for the declared autopilot games (``pilot_decl.py``).

A teacher is built for one headless game and answers ``act(game) -> outputs`` (the policy's
outputs, in the declaration's order, axes in [-1, 1] and buttons at +-0.9): what a competent
player does from THIS state.  They label states for imitation (``distill``) and are the
baseline row of the comparison.  They read the game's own columns and maps; nothing here is a
second physics.

* Delve: ``delve_plan.RoomPlan`` (the law's own body physics) plans the route inside a room and
  ``delve_probe.world_route`` the rooms; the repertoire's control scripts (walk, ladder, jump
  with a hold time, double jump) are replayed closed loop from wherever the player stands.
* Climber: steer to the highest platform the bounce can still land on.
* Runner: run right; jump pits and walkers (the button held through the rise).
* Tile Quest: breadth-first path over the map to the nearest monster and swing when in reach;
  retreat at low health.
"""
from __future__ import annotations

import collections
import math

import numpy as np

BUTTON = 0.9


def col_all(game):
    from meowdoku.delve_rules import ENT
    c = game.session.column
    return c("x")[:ENT], c("y")[:ENT], c("alive")[:ENT]


class DelveTeacher:
    def __init__(self, game):
        self.script = None            # (variant, t0, airborne)
        self.room = None

    def act(self, game) -> np.ndarray:
        from meowdoku import delve_probe as probe
        from meowdoku.delve_rules import LADDER
        from meowdoku.pilot_decl import _delve_goal
        col = probe.col
        out = np.array([0.0, 0.0, -BUTTON])
        goal = _delve_goal(game)
        if goal is None:
            return out
        kind, leg, gx, gy = goal
        room = game.world.rooms[game.room]
        if game.room != self.room:
            self.room, self.script = game.room, None
        ability = col(game, "ability") > 0.5
        x, y, clock = col(game, "x"), col(game, "y"), col(game, "clock")
        supported = probe.supported(game)
        on_ladder = probe.tile_at(game, x, y) == LADDER
        if self.script is not None:
            v, t0, airborne = self.script
            tt = clock - t0
            airborne = airborne or not supported
            self.script = (v, t0, airborne)
            landed = airborne and (supported or col(game, "climb") > 0.5)
            if landed or tt > 2.2 or (tt > 0.3 and not airborne):
                self.script = None
            else:
                grab = airborne and on_ladder
                return np.array([float(v.axis), -1.0 if grab else 0.0, BUTTON if v.pressed(tt) else -BUTTON])
        target = room.exit_place(leg["exit"]) if kind == "door" else ("s", room.orb[0] * 2 + 1, room.orb[1] + 1)
        here = probe.place_of(game)
        if kind == 'door':
            d = room.doors[leg['exit']]
            in_mouth = abs(y - (d.floor - 0.4)) < 0.7 and ((d.side == 'L' and x <= 1.7) or (d.side == 'R' and x >= room.w - 1.7))
            if in_mouth:                                            # in the corridor: walk out through the door
                return np.array([-1.0 if d.side == 'L' else 1.0, 0.0, -BUTTON])
            if d.side == 'U' and d.ladder and abs(x - (d.a + 1.0)) < 1.3 and y <= 1.6:
                return np.array([0.0, -1.0, -BUTTON])               # at the top of the ladder: climb out
            if d.side == 'D' and d.ladder and abs(x - (d.a + 1.0)) < 1.3 and y >= room.h - 2.0:
                return np.array([0.0, 1.0, -BUTTON])
        if here is None:
            out[0] = float(np.clip((target[1] / 2.0 - x) * 2.0, -1.0, 1.0)) if target[0] == "s" else 0.0
            return out
        if kind == "door" and (here == target or (here[0] == "s" and target[0] == "s" and here[2] == target[2] and abs(here[1] - target[1]) <= 1)):
            d = room.doors[leg["exit"]]
            axis, axy = {"L": (-1.0, 0.0), "R": (1.0, 0.0), "U": (0.0, -1.0)}.get(d.side, (0.0, 1.0))
            if d.side == "D" and d.oneway:
                axis, axy = 1.0, 0.0
            return np.array([axis, axy, -BUTTON])
        path = room.plan.path(here, target, ability)
        if not path:
            out[0] = float(np.clip((target[1] / 2.0 - x) * 2.0, -1.0, 1.0)) if target[0] == "s" else 0.0
            return out
        cur = here
        for e in path:
            if e.variant is not None:
                self.script = (e.variant, clock, False) if cur == here else None
                break
            cur = e.dst
        if self.script is not None:
            v = self.script[0]
            return np.array([float(v.axis), 0.0, BUTTON if v.pressed(0.0) else -BUTTON])
        dst = cur
        if dst[0] == "s":
            out[0] = float(np.clip((dst[1] / 2.0 - x) * 3.0, -1.0, 1.0))
            ahead = np.sign(out[0]) or 1.0                       # an enemy close ahead on the floor: hop it (a stomp kills)
            ex, ey, ea = col_all(game)
            for k in range(1, len(ea)):
                if ea[k] > 0.0 and 0.2 < (ex[k] - x) * ahead < 1.9 and abs(ey[k] - y) < 0.9 and supported:
                    out[2] = BUTTON
        else:
            ty = dst[2] + 0.5
            out[0] = float(np.clip((dst[1] + 0.5 - x) * 2.0, -1.0, 1.0)) if not on_ladder else 0.0
            out[1] = 1.0 if ty > y else -1.0
        return out


class PlatformerTeacher:
    def __init__(self, game):
        pass

    def act(self, game) -> np.ndarray:
        from meowdoku import platformer as p
        col = game.session.column
        x, y, vy, alive, slot = col("x"), col("y"), col("vy"), col("alive"), col("slot")
        ry = col("ry")
        px, feet = float(x[0]), float(y[0]) + p.PLAYER_RY
        apex = feet - (float(vy[0]) ** 2) / (2.0 * p.GRAVITY) if vy[0] < 0 else feet
        best, best_cost = None, 1e9
        G, v = p.GRAVITY, float(vy[0])
        for k in range(*p.PLATFORMS):
            if alive[k] <= 0.0:
                continue
            top = float(y[k] - ry[k])
            if top < apex - 0.05 or (v >= 0 and top < feet - 0.05):
                continue                                           # above what the bounce can reach / behind
            # time left to be over it when the feet come down to its top, and how far the sticks can carry us
            t = (-v + math.sqrt(v * v + 2.0 * G * max(top - feet, 0.0))) / G if v >= 0 else -v / G + math.sqrt(2.0 * max(top - apex, 0.0) / G)
            gap = abs(float(x[k]) - px)
            cost = top + 0.05 * gap
            if gap < 8.0 * t and cost < best_cost:
                best, best_cost = k, cost
        if best is None:
            return np.zeros(1)
        return np.array([float(np.clip((float(x[best]) - px) * 3.0, -1.0, 1.0))])


class SideScrollerTeacher:
    def __init__(self, game):
        self.jumping = 0

    def act(self, game) -> np.ndarray:
        from meowdoku import sidescroller as s
        col = game.session.column
        x, y, vy, alive, slot = col("x"), col("y"), col("vy"), col("alive"), col("slot")
        px = float(x[0])
        g0, g1 = s.GROUND_LANES

        def ground(xw):
            c = int(math.floor(xw))
            return any(alive[k] > 0.0 and int(math.floor(x[k])) == c for k in range(g0, g1))

        supported = abs(float(vy[0])) < 0.2
        walker = any(alive[k] > 0.0 and 0.4 < x[k] - px < 2.4 and abs(y[k] - y[0]) < 1.2
                     for k in range(*s.WALKER_LANES))
        press = False
        if self.jumping and not supported:
            press = float(vy[0]) < 0.0 or self.jumping < 3          # held through the rise, released at the apex
            self.jumping += 1
        else:
            self.jumping = 0
            if supported and (not ground(px + 0.9) or walker):
                press, self.jumping = True, 1
        return np.array([1.0, BUTTON if press else -BUTTON])


class TileQuestTeacher:
    def __init__(self, game):
        pass

    def act(self, game) -> np.ndarray:
        from meowdoku import tilequest as t
        col = game.session.column
        x, y, alive = col("x")[:t.ENT], col("y")[:t.ENT], col("alive")[:t.ENT]
        px, py = float(x[0]), float(y[0])
        foes = [k for k in range(t.S0, t.T0) if alive[k] > 0.0]
        if not foes:
            return np.array([0.0, 0.0, -BUTTON])
        foes.sort(key=lambda k: (x[k] - px) ** 2 + (y[k] - py) ** 2)
        k = foes[0]
        dx, dy = float(x[k]) - px, float(y[k]) - py
        dist = math.hypot(dx, dy)
        hp = float(col("hp")[0])
        if hp <= 2.0 and dist < 3.0:                                # hurt: back away
            return np.array([float(np.clip(-dx, -1, 1)), float(np.clip(-dy, -1, 1)), -BUTTON])
        if dist < t.MELEE_REACH * 0.85:
            return np.array([float(np.clip(dx * 2, -1, 1)), float(np.clip(dy * 2, -1, 1)), BUTTON])
        if dist > 11.0:
            return np.array([0.0, 0.0, -BUTTON])
        step = self._step(game, (int(px), int(py)), (int(x[k]), int(y[k])))
        if step is None:
            return np.array([float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1)), -BUTTON])
        cx, cy = step[0] + 0.5 - px, step[1] + 0.5 - py
        n = max(math.hypot(cx, cy), 1e-6)
        return np.array([cx / n, cy / n, BUTTON if dist < t.MELEE_REACH * 1.05 else -BUTTON])

    @staticmethod
    def _step(game, start, goal):
        """The first cell of a shortest path over the passable tiles (None: no path)."""
        from meowdoku import tilequest as t
        grid = game.map.grid
        blocked = np.isin(grid, t.BLOCKED)
        prev, queue = {start: None}, collections.deque([start])
        while queue:
            c = queue.popleft()
            if c == goal:
                break
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if 0 <= n[0] < t.WIDTH and 0 <= n[1] < t.HEIGHT and n not in prev and not blocked[n[1], n[0]]:
                    prev[n] = c
                    queue.append(n)
        if goal not in prev or goal == start:
            return None
        c = goal
        while prev[c] != start:
            c = prev[c]
        return c
