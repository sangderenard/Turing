"""The autopilot declaration: how ANY ``EntityGame`` opts in to the autopilot.

A game declares, once, in a ``PilotDecl``:

* its CONTROLS: the policy's outputs, in order, each an axis (a stick: the output
  in [-1, 1] is the control's value) or a button (pressed while the output is
  positive), by the name of the control column they drive;
* what it SEES, as a ``View`` read from the session columns by ``read``:
    - ``own``       the player's own state (``OWN_N`` values, scaled to ~[-1, 1]),
    - ``entities``  every body in reach as (class, sub-kind, dx, dy, dvx, dvy, flag),
                    positions and velocities relative to the player and scaled by
                    the declaration's ``scale``; classes are the CLASSES below,
    - ``patch``     the local MAP PATCH: ``NCH`` channels (solid, platform, ladder,
                    hazard, goal/pickup) of 0/1 over a ``(patch_h, patch_w)`` window of
                    tiles about the player,
    - ``goal``      an optional compass (dx, dy) to where the game wants the player,
    - ``danger``    nearness (0..1) of the closest hostile body;
* its decision step (seconds one decision is held), its token network variant and
  its teacher.

From the view the generic builders make BOTH networks' observations, so one
recorded row serves either: the TOKEN network (``pilot_law.token_policy_forward``,
the bullet hell's architecture and its trimmed variants) reads ``own`` + the
flattened map patch + goal as its own-state block and every entity as a token; the
GRAPH network (``pilot_law.graph_policy_forward``, in the style of the scroller's
``enemy_brain`` and the walkers' ``Kind`` nets) reads a compact feature set:
the own state, the goal, the nearest body of each class and the eight tiles
about the player.  A recorded row is the two joined (``joint``); ``slice`` cuts
one network's out of it.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from meowdoku.s3_train import pilot_law as law

OWN_N = 10
NCH = 5
CH_SOLID, CH_PLATFORM, CH_LADDER, CH_HAZARD, CH_GOAL = range(NCH)
CLASSES = ("hostile", "friendly", "pickup", "goal", "other")
HOSTILE, FRIENDLY, PICKUP, GOAL, OTHER = range(len(CLASSES))
SUBS = 5
GOAL_F = 3
RING = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))     # tile offsets about the player
RING_CH = (CH_SOLID, CH_PLATFORM, CH_HAZARD, CH_GOAL)
GRAPH_HIDDEN = 12
GRAPH_DENSITY = 0.4


@dataclass
class Control:
    """One output of the policy: the control column it drives and how."""
    name: str
    kind: str = "axis"            # "axis": value in [-1, 1]; "button": pressed while the output is positive


@dataclass
class View:
    own: np.ndarray
    entities: list = field(default_factory=list)       # (class, sub, dx, dy, dvx, dvy, flag)
    patch: np.ndarray | None = None                    # (NCH, ph, pw)
    goal: tuple[float, float] | None = None
    danger: float = 0.0
    counters: dict = field(default_factory=dict)       # score / lives / pickups / rooms / kills (for outcome tags)


@dataclass
class PilotDecl:
    game: str
    controls: tuple[Control, ...]
    read: Callable                                      # (game) -> View
    patch: tuple[int, int] = (9, 7)                     # (width, height) in tiles
    patch_anchor: tuple[int, int] = (4, 3)              # the player's cell in the patch
    step: float = 1.0 / 20.0                            # seconds one decision is held
    variant: str = "t16.h32.q2.g48"                     # the token network's variant id
    teacher: str = ""                                   # "module:Class" of the scripted teacher
    progress: Callable | None = None                    # (game) -> dict of headless progress numbers
    seconds: float = 40.0                               # an evaluation episode's game seconds
    default_arch: str = "token"
    make: Callable | None = None                        # (seed) -> a headless game

    # ------------------------------------------------------------ widths
    @property
    def outputs(self) -> int:
        return len(self.controls)

    @property
    def tokens(self) -> int:
        return law.spec_of(self.variant)["tokens"]

    @property
    def ship(self) -> int:
        return OWN_N + GOAL_F + NCH * self.patch[0] * self.patch[1]

    @property
    def graph_features(self) -> int:
        return OWN_N + GOAL_F + len(CLASSES) * 5 + len(RING) * len(RING_CH) + 1

    def widths(self, arch: str) -> tuple[int, int]:
        """(features, outputs) of the network ``arch`` (``token`` / ``graph`` / ``joint``)."""
        tok = law.features_of(self.tokens, self.ship)
        if arch == "joint":
            return tok + self.graph_features, self.outputs
        return (tok if arch == "token" else self.graph_features), self.outputs

    # ------------------------------------------------------------ observations
    def token_obs(self, v: View) -> np.ndarray:
        T = self.tokens
        ship = np.zeros(self.ship)
        ship[:OWN_N] = v.own
        if v.goal is not None:
            ship[OWN_N:OWN_N + GOAL_F] = (1.0, *v.goal)
        if v.patch is not None:
            ship[OWN_N + GOAL_F:] = v.patch.reshape(-1)
        rows = list(v.entities)
        if v.goal is not None:                                  # the compass is a token too
            rows.append((GOAL, 0, v.goal[0], v.goal[1], 0.0, 0.0, 1.0))
        tokens = np.zeros((T, law.TOKEN_F))
        mask = np.zeros(T)
        rows.sort(key=lambda r: r[2] * r[2] + r[3] * r[3])
        for i, (cls, sub, dx, dy, dvx, dvy, flag) in enumerate(rows[:T]):
            t = tokens[i]
            t[0:4] = dx, dy, dvx, dvy
            t[4 + cls] = 1.0
            t[9] = flag
            t[10] = min(1.0, math.hypot(dx, dy))
            t[11 + min(sub, SUBS - 1)] = 1.0
            mask[i] = 1.0
        return np.concatenate([ship, tokens.reshape(-1), mask])

    def graph_obs(self, v: View) -> np.ndarray:
        out = np.zeros(self.graph_features)
        out[:OWN_N] = v.own
        if v.goal is not None:
            out[OWN_N:OWN_N + GOAL_F] = (1.0, *v.goal)
        at = OWN_N + GOAL_F
        for cls in range(len(CLASSES)):                         # the nearest body of each class
            best = None
            for r in v.entities:
                if r[0] == cls and (best is None or r[2] * r[2] + r[3] * r[3] < best[2] * best[2] + best[3] * best[3]):
                    best = r
            if best is not None:
                out[at:at + 5] = (1.0, best[2], best[3], best[4], best[5])
            at += 5
        if v.patch is not None:
            ax, ay = self.patch_anchor
            ph, pw = v.patch.shape[1:]
            for dx, dy in RING:
                for ch in RING_CH:
                    x, y = ax + dx, ay + dy
                    out[at] = v.patch[ch, y, x] if 0 <= x < pw and 0 <= y < ph else 1.0 if ch == CH_SOLID else 0.0
                    at += 1
        out[at] = v.danger
        return out

    def observe(self, game, arch: str = "joint") -> np.ndarray:
        v = self.read(game)
        if arch == "token":
            return self.token_obs(v)
        if arch == "graph":
            return self.graph_obs(v)
        return np.concatenate([self.token_obs(v), self.graph_obs(v)])

    def slice(self, arch: str, rows: np.ndarray) -> np.ndarray:
        """One network's observation out of recorded ``joint`` rows (already that width: unchanged)."""
        if arch == "joint":
            return rows
        tok = self.widths("token")[0]
        if rows.shape[-1] == self.widths(arch)[0]:
            return rows
        return rows[..., :tok] if arch == "token" else rows[..., tok:]

    # ------------------------------------------------------------ controls
    def controls_from(self, game, out) -> dict[str, float]:
        """The policy's outputs as the game's control columns; controls it does not drive rest."""
        rest = getattr(game, "_control_rest", None)
        if rest is None:
            columns = game.initial_columns()
            rest = game._control_rest = {n: float(columns[n][0]) for n in game.CONTROLS if n in columns}
        values = dict(rest)
        for c, o in zip(self.controls, out):
            values[c.name] = float(max(-1.0, min(1.0, o))) if c.kind == "axis" else (1.0 if o > 0.0 else 0.0)
        return {n: values.get(n, 0.0) for n in game.CONTROLS}

    def target_from(self, game, controls: dict) -> np.ndarray:
        """The controls a decision used (a human's, a teacher's) as the policy's outputs.  A held pointer
        is its axis equivalent under the motion law, as ``autopilot.target_from``."""
        motion = game.motion() or {}
        k = game.PLAYER_LANE
        out = np.zeros(self.outputs)
        steer_of = {"move_axis": ("steer_x", "x"), "move_axis_y": ("steer_y", "y")}
        for i, c in enumerate(self.controls):
            if c.kind == "button":
                out[i] = 0.9 if float(controls.get(c.name, 0.0)) > 0.0 else -0.9
                continue
            steer_name, pos = steer_of.get(c.name, (None, None))
            steer = float(controls.get(steer_name, -1.0)) if steer_name else -1.0
            if steer >= 0.0 and motion.get(pos) and k is not None:
                gap = (steer - float(game.session.column(pos)[k])) / float(motion.get("reach", 1.0))
                out[i] = max(-1.0, min(1.0, gap / float(motion.get("speed", 1.0))))
            else:
                out[i] = max(-1.0, min(1.0, float(controls.get(c.name, 0.0))))
        return out

    def make_teacher(self, game):
        import importlib
        module, _, name = self.teacher.partition(":")
        return getattr(importlib.import_module(module), name)(game)

    # ------------------------------------------------------------ fresh networks
    def init_params(self, arch: str, seed: int = 0) -> dict[str, np.ndarray]:
        from meowdoku.s3_train import autopilot as ap
        rng = np.random.default_rng(seed)
        if arch == "token":
            params = ap.init_token_params(seed=seed, outputs=self.outputs, variant=self.variant, ship=self.ship)
            params["variant"] = np.array(self.variant)
            return params
        f, o = self.graph_features, self.outputs
        shapes = law.graph_param_shapes(f, o, GRAPH_HIDDEN)
        m1 = (rng.random(shapes["GM1"]) < GRAPH_DENSITY).astype(np.float64)
        m1[:, m1.sum(axis=0) == 0] = 1.0                        # no hidden unit sees nothing
        for row in np.flatnonzero(m1.sum(axis=1) == 0):         # every feature feeds some unit
            m1[row, rng.integers(0, GRAPH_HIDDEN)] = 1.0
        return {"GW1": rng.normal(0.0, math.sqrt(1.0 / max(1.0, f * GRAPH_DENSITY)), shapes["GW1"]),
                "Gb1": np.zeros(shapes["Gb1"]), "GM1": m1,
                "GW2": rng.normal(0.0, math.sqrt(1.0 / GRAPH_HIDDEN), shapes["GW2"]), "Gb2": np.zeros(shapes["Gb2"]),
                "GM2": np.ones(shapes["GM2"])}


# ----------------------------------------------------------------------- the registry
DECLS: dict[str, PilotDecl] = {}


def declare(decl: PilotDecl) -> PilotDecl:
    DECLS[decl.game] = decl
    return decl


ARCHS = ("token", "graph")
_CHOICE = None


def choice_path() -> Path:
    from meowdoku.s3_train import autopilot as ap
    return ap.BUILD / "arch_choice.json"


def arch_for(name: str) -> str:
    """The network a declared game's autopilot flies: MEOWDOKU_PILOT_ARCH_<GAME>, else the saved choice
    (``set_arch``), else the declaration's default."""
    env = os.environ.get(f"MEOWDOKU_PILOT_ARCH_{name.upper()}")
    if env in ARCHS:
        return env
    try:
        saved = json.loads(choice_path().read_text(encoding="utf-8")).get(name)
    except (OSError, ValueError):
        saved = None
    return saved if saved in ARCHS else DECLS[name].default_arch


def set_arch(name: str, arch: str) -> None:
    if arch not in ARCHS:
        raise ValueError(f"unknown autopilot network {arch!r} (token or graph)")
    try:
        saved = json.loads(choice_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    saved[name] = arch
    choice_path().parent.mkdir(parents=True, exist_ok=True)
    choice_path().write_text(json.dumps(saved), encoding="utf-8")


# ----------------------------------------------------------------------- reading helpers
def _patch(grid_of: Callable, cx: int, cy: int, decl_patch: tuple[int, int], anchor: tuple[int, int]) -> np.ndarray:
    """A (NCH, ph, pw) patch of the tile cells about (cx, cy): ``grid_of(x, y)`` -> the channel (or None)."""
    pw, ph = decl_patch
    out = np.zeros((NCH, ph, pw))
    for j in range(ph):
        for i in range(pw):
            ch = grid_of(cx + i - anchor[0], cy + j - anchor[1])
            if ch is None:
                continue
            for c in (ch if isinstance(ch, tuple) else (ch,)):
                out[c, j, i] = 1.0
    return out


def _near(rows: list) -> float:
    d = [math.hypot(r[2], r[3]) for r in rows if r[0] == HOSTILE]
    return max(0.0, 1.0 - min(d) * 2.0) if d else 0.0


# ----------------------------------------------------------------------- delve
def _delve_goal(game):
    """The compass: the orb while the way is gated and it is not taken, else the door that leads along the
    world's route to the goal room.  Returns (kind, room, x, y) in the room's tile frame (cached per room)."""
    from meowdoku.s7_tools import delve_probe as probe
    key = (game.room, getattr(game, "arrival_door", -1), game.session.column("ability")[0] > 0.5)
    cache = getattr(game, "_goal_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    world, room = game.world, game.world.rooms[game.room]
    legs = probe.world_route(world, start=(game.room, getattr(game, "arrival_door", -1), key[2]))
    out = None
    if legs:
        leg = legs[0]
        if leg["orb"] and room.orb is not None:
            out = ("orb", leg, room.orb[0] + 0.5, room.orb[1] + 0.5)
        else:
            d = room.doors[leg["exit"]]
            x, y = {"L": (0.0, d.a + 1.5), "R": (float(room.w), d.a + 1.5), "U": (d.a + 1.0, 0.0), "D": (d.a + 1.0, float(room.h))}[d.side]
            out = ("door", leg, x, y)
    game._goal_cache = (key, out)
    return out


def read_delve(game) -> View:
    from meowdoku.s2_titles.delve import delve_rules as r
    col = game.session.column
    room = game.world.rooms[game.room]
    px, py = float(col("x")[0]), float(col("y")[0])
    grid = np.asarray(col("tile")[r.MAP0:]).reshape(r.MAXH, r.MAXW)[:room.h, :room.w]
    vx, vy = float(col("vx")[0]), float(col("vy")[0])
    cx, cy = int(math.floor(px)), int(math.floor(py))
    feet = py + r.PLAYER_RY

    def tile(x, y):
        return int(grid[y, x]) if 0 <= x < room.w and 0 <= y < room.h else r.SOLID

    below = [tile(int(math.floor(px + d)), int(math.floor(feet + 0.05))) for d in (-0.27, 0.27)]
    supported = abs(vy) < 0.5 and (r.SOLID in below or r.ONEWAY in below)
    own = np.array([px - cx - 0.5, py - cy - 0.5, vx / r.SPEED, vy / 12.0, float(supported), float(col("climb")[0]),
                    float(col("air")[0]) * float(col("ability")[0]), float(col("hurt")[0] > 0.0),
                    float(col("lives")[0]) / r.PLAYER_HP, float(col("ability")[0])])
    chans = {r.SOLID: CH_SOLID, r.ONEWAY: CH_PLATFORM, r.LADDER: CH_LADDER, r.SPIKE: CH_HAZARD, r.GEM: CH_GOAL, r.ORB: CH_GOAL}

    def grid_of(x, y):
        return chans.get(tile(x, y))

    ents = []
    alive, ex, ey = col("alive")[:r.ENT], col("x")[:r.ENT], col("y")[:r.ENT]
    evx, evy, slot = col("vx")[:r.ENT], col("vy")[:r.ENT], col("slot")[:r.ENT]
    for k in range(1, r.ENT):
        if alive[k] > 0.0:
            ents.append((HOSTILE, 0 if slot[k] == r.WALKER else 1, (ex[k] - px) / 8.0, (ey[k] - py) / 6.0,
                         (evx[k] - vx) / 6.0, evy[k] / 6.0, 1.0))
    goal = _delve_goal(game)
    g = ((goal[2] - px) / 8.0, (goal[3] - py) / 6.0) if goal is not None else None
    counters = {"px": px, "py": py, "score": float(col("score")[0]), "lives": float(col("lives")[0]), "pickups": float(col("gems")[0]),
                "rooms": float(len(game.visited)), "kills": float(col("kills")[0])}
    return View(own, ents, _patch(grid_of, cx, cy, DELVE_PATCH, DELVE_ANCHOR), g, _near(ents), counters)


DELVE_PATCH, DELVE_ANCHOR = (11, 7), (5, 3)


def progress_delve(game) -> dict:
    s = game.session
    return {"rooms": float(len(game.visited)), "score": float(s.column("score")[0]), "gems": float(s.column("gems")[0]),
            "survival": float(s.column("clock")[0]), "goal": float(game.room == game.world.goal_room)}


# ----------------------------------------------------------------------- platformer
def read_platformer(game) -> View:
    from meowdoku.s2_titles.platformer import platformer as p
    col = game.session.column
    x, y, vx, vy, alive, slot = col("x"), col("y"), col("vx"), col("vy"), col("alive"), col("slot")
    px, py, fw = float(x[0]), float(y[0]), float(col("field_w")[0])
    own = np.zeros(OWN_N)
    own[:6] = (2.0 * px / fw - 1.0, 2.0 * py / p.HEIGHT - 1.0, float(vx[0]) / p.SPEED, float(vy[0]) / p.FALL_MAX,
               float(vy[0] < 0.0), float(col("lives")[0] > 0.0))
    ents, cells = [], {}
    for k in range(1, p.LANES):
        if alive[k] <= 0.0:
            continue
        dx, dy = (x[k] - px) / 4.5, (y[k] - py) / 8.0
        if slot[k] == p.PICKUP:
            ents.append((PICKUP, 0, dx, dy, 0.0, float(vy[k]) / 8.0, 1.0))
        else:
            ents.append((OTHER, 0, dx, dy, 0.0, 0.0, float(col("rx")[k]) / 0.4))
            top = int(math.floor(y[k] - col("ry")[k]))
            for cxx in range(int(math.floor(x[k] - col("rx")[k])), int(math.floor(x[k] + col("rx")[k])) + 1):
                cells[(cxx, top)] = CH_PLATFORM
    cx, cy = int(math.floor(px)), int(math.floor(py))
    counters = {"px": px, "py": py, "score": float(col("score")[0]), "lives": float(col("lives")[0]), "pickups": float(col("score")[0] // 100.0)}
    return View(own, ents, _patch(lambda a, b: cells.get((a, b)), cx, cy, PLATFORMER_PATCH, PLATFORMER_ANCHOR), None, 0.0, counters)


PLATFORMER_PATCH, PLATFORMER_ANCHOR = (9, 9), (4, 4)


def progress_platformer(game) -> dict:
    s = game.session
    return {"score": float(s.column("score")[0]), "climbed": float(s.column("climbed")[0]),
            "survival": float(s.column("clock")[0])}


# ----------------------------------------------------------------------- side scroller
def read_side_scroller(game) -> View:
    from meowdoku.s2_titles.sidescroller import sidescroller as s
    col = game.session.column
    x, y, vx, vy, alive, slot = col("x"), col("y"), col("vx"), col("vy"), col("alive"), col("slot")
    holds, spent = col("holds"), col("spent")
    px, py, view_x, fw = float(x[0]), float(y[0]), float(col("distance")[0]), float(col("field_w")[0])
    supported = abs(float(vy[0])) < 0.2
    own = np.zeros(OWN_N)
    own[:7] = (float(vx[0]) / s.SPEED, float(vy[0]) / s.JUMP, float(supported), float(col("energy")[0]),
               (px - view_x) / fw * 2.0 - 1.0, 2.0 * py / s.HEIGHT - 1.0, float(col("lives")[0] > 0.0))
    cells, ents = {}, []
    for k in range(s.TILES[0], s.TILES[1]):
        if alive[k] > 0.0:
            cell = (int(math.floor(x[k])), int(math.floor(y[k])))
            if slot[k] == s.GROUND:
                cells[cell] = CH_SOLID
            else:
                cells[cell] = (CH_PLATFORM, CH_GOAL) if holds[k] > 0.0 and spent[k] <= 0.0 else CH_PLATFORM
    for k in range(s.ITEM_LANES[0], s.ITEM_LANES[1]):
        if alive[k] > 0.0:
            ents.append((PICKUP, int(holds[k]) % SUBS, (x[k] - px) / 6.0, (y[k] - py) / 5.0, (vx[k] - vx[0]) / 8.0, 0.0, 1.0))
    for k in range(s.WALKER_LANES[0], s.WALKER_LANES[1]):
        if alive[k] > 0.0:
            ents.append((HOSTILE, 0, (x[k] - px) / 6.0, (y[k] - py) / 5.0, (vx[k] - vx[0]) / 8.0, 0.0, 1.0))
    cx, cy = int(math.floor(px)), int(math.floor(py))
    counters = {"px": px, "py": py, "score": float(col("score")[0]), "lives": float(col("lives")[0]), "pickups": float(col("coins")[0])}
    return View(own, ents, _patch(lambda a, b: cells.get((a, b)), cx, cy, SIDE_PATCH, SIDE_ANCHOR), None, _near(ents), counters)


SIDE_PATCH, SIDE_ANCHOR = (13, 8), (3, 5)


def progress_side_scroller(game) -> dict:
    s = game.session
    return {"distance": float(s.column("distance")[0]), "score": float(s.column("score")[0]),
            "coins": float(s.column("coins")[0]), "survival": float(s.column("clock")[0])}


# ----------------------------------------------------------------------- tilequest
def read_tilequest(game) -> View:
    from meowdoku.s2_titles.tilequest import tilequest as t
    col = game.session.column
    x, y, vx, vy, alive, kind = col("x")[:t.ENT], col("y")[:t.ENT], col("vx")[:t.ENT], col("vy")[:t.ENT], col("alive")[:t.ENT], col("kind")[:t.ENT]
    px, py = float(x[0]), float(y[0])
    own = np.zeros(OWN_N)
    own[:8] = (float(col("hp")[0]) / t.PLAYER_HP, float(vx[0]) / t.PLAYER_SPEED, float(vy[0]) / t.PLAYER_SPEED,
               float(col("cool")[0] > 0.0), px - math.floor(px) - 0.5, py - math.floor(py) - 0.5,
               2.0 * px / t.WIDTH - 1.0, 2.0 * py / t.HEIGHT - 1.0)
    subs = {t.SLIME: 0, t.WOLF: 1, t.GOBLIN: 2}
    ents = []
    for k in range(1, t.ENT):
        if alive[k] <= 0.0:
            continue
        kd = int(kind[k])
        cls = HOSTILE if kd in subs else FRIENDLY if kd in (t.VILLAGER, t.TOWER) else OTHER
        sub = subs.get(kd, 3 if kd == t.TOWER else 0)
        ents.append((cls, sub, (x[k] - px) / 7.0, (y[k] - py) / 7.0, (vx[k] - vx[0]) / 5.0, (vy[k] - vy[0]) / 5.0, 1.0))
    grid = game.map.grid
    chans = {t.WALL: CH_SOLID, t.WATER: CH_SOLID, t.TREE: CH_SOLID, t.LAIR: CH_HAZARD, t.HOME: CH_GOAL, t.WORK: CH_GOAL, t.DOOR: CH_PLATFORM}

    def grid_of(a, b):
        return chans.get(int(grid[b, a])) if 0 <= a < t.WIDTH and 0 <= b < t.HEIGHT else CH_SOLID

    cx, cy = int(math.floor(px)), int(math.floor(py))
    counters = {"px": px, "py": py, "score": float(col("score")[0]), "lives": float(col("lives")[0]), "kills": float(col("kills")[0]),
                "pickups": float(col("trips")[0])}
    return View(own, ents, _patch(grid_of, cx, cy, TILEQUEST_PATCH, TILEQUEST_ANCHOR), None, _near(ents), counters)


TILEQUEST_PATCH, TILEQUEST_ANCHOR = (11, 9), (5, 4)


def progress_tilequest(game) -> dict:
    s = game.session
    return {"score": float(s.column("score")[0]), "kills": float(s.column("kills")[0]),
            "survival": float(s.column("clock")[0]), "hp": float(s.column("lives")[0])}


def _make(cls_path: str, **kw):
    def make(seed: int):
        import importlib
        import src.common.tensors.numpy_backend  # noqa: F401 - registers the NumPy backend a law's columns are built on
        module, _, name = cls_path.partition(":")
        game = getattr(importlib.import_module(module), name)(seed=int(seed), **kw)
        game.learns = False
        return game
    return make


declare(PilotDecl("delve", (Control("move_axis"), Control("move_axis_y"), Control("fire", "button")), read_delve,
                  patch=DELVE_PATCH, patch_anchor=DELVE_ANCHOR, step=1 / 20, variant="t8.h32.q2.g48",
                  teacher="meowdoku.pilot_teachers:DelveTeacher", progress=progress_delve, seconds=60.0,
                  make=_make("meowdoku.delve:Delve", aspect=1.7)))
declare(PilotDecl("platformer", (Control("move_axis"),), read_platformer, patch=PLATFORMER_PATCH,
                  patch_anchor=PLATFORMER_ANCHOR, step=1 / 15, variant="t16.h24.q2.g32",
                  teacher="meowdoku.pilot_teachers:PlatformerTeacher", progress=progress_platformer, seconds=40.0,
                  make=_make("meowdoku.platformer:Platformer")))
declare(PilotDecl("side_scroller", (Control("move_axis"), Control("fire", "button")), read_side_scroller,
                  patch=SIDE_PATCH, patch_anchor=SIDE_ANCHOR, step=1 / 20, variant="t8.h24.q2.g32",
                  teacher="meowdoku.pilot_teachers:SideScrollerTeacher", progress=progress_side_scroller, seconds=40.0,
                  make=_make("meowdoku.sidescroller:SideScroller")))
declare(PilotDecl("tilequest", (Control("move_axis"), Control("move_axis_y"), Control("fire", "button")), read_tilequest,
                  patch=TILEQUEST_PATCH, patch_anchor=TILEQUEST_ANCHOR, step=1 / 10, variant="t16.h32.q2.g48",
                  teacher="meowdoku.pilot_teachers:TileQuestTeacher", progress=progress_tilequest, seconds=40.0,
                  make=_make("meowdoku.tilequest:TileQuest")))
