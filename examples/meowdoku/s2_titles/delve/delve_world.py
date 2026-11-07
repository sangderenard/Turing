"""Delve's world generator: a connected map of rooms, each a platforming layout
proven traversable with the platformer's own movement.

THE WORLD.  A coarse grid of screens (``SCREEN_W x SCREEN_H`` tiles each) is
tiled with rooms of one to three screens a side: a *chamber* (one screen), a
*hall* (wide: scrolls sideways), a *shaft* (tall: scrolls up and down) or a
*cavern* (both).  Rooms that share a border may be joined by a DOOR: an opening
in the side wall of both (``L``/``R``, a corridor whose floor is ``DOOR_H``
rows under the lintel), or a LADDER SHAFT through the floor of the upper room
and the ceiling of the lower (``D``/``U``).  The joins are a randomized
spanning tree (so the world is connected and has a MAIN PATH from the start
room to the deepest room, with side branches) plus extra joins (loops).  A
vertical loop is sometimes ONE-WAY: a hole in the floor above and a ceiling
that cannot be climbed below; the loops are what bring the player back.

ABILITY GATE.  On the main path one room is a gate: the door that leaves it for
the next room is a ledge four rows above the floor, which a single jump cannot
climb and the double jump can.  The double-jump ORB lies in a room on the start
side of the gate; loops that would bypass the gate are not made.

THE ROOMS.  A room is a ring of wall, door structures, a floor with spike
strips and pillars, and platform CHAINS (one-way planks, two rows apart,
overlapping, so each is a jump from the one below) from every raised door down
to the floor, plus loose planks, ladders, gems and enemies.  A room is
accepted only when ``delve_plan.RoomPlan`` (the law's own body physics) finds
every door usable from every other (the gate door only with the ability).
Rooms that fail are re-rolled; the number of tries and why are in the report.

Everything is seeded: ``generate(seed, size, gated)`` is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from meowdoku.s2_titles.delve.delve_plan import RoomPlan
from meowdoku.s2_titles.delve.delve_rules import (
    DOOR_H, DOOR_W, EMPTY, GEM, LADDER, N_FLYERS, N_WALKERS, ONEWAY, ORB, PLAYER_RY, SCREEN_H, SCREEN_W, SOLID, SPIKE)

SIZES = {"small": (5, 3), "large": (8, 4)}
SHAPES = (((1, 1), 6), ((2, 1), 3), ((3, 1), 1), ((1, 2), 2), ((1, 3), 1), ((2, 2), 1))
KINDS = {(1, 1): "chamber", (2, 1): "hall", (3, 1): "hall", (1, 2): "shaft", (1, 3): "shaft", (2, 2): "cavern"}
ADJECTIVES = ("Hollow", "Gilded", "Sunken", "Mossy", "Echoing", "Narrow", "Broken", "Amber", "Quiet", "Drifting",
              "Lantern", "Cobbled", "Verdant", "Frozen", "Ashen", "Tilted")
NOUNS = ("Gallery", "Cistern", "Stair", "Vault", "Cellar", "Atrium", "Chimney", "Bridge", "Cloister", "Nook",
         "Foundry", "Orchard", "Landing", "Well", "Passage", "Terrace")
ROOM_TRIES = 14
WORLD_TRIES = 30


class GenerationFailed(RuntimeError):
    """A world (or a room) the generator could not make valid within its tries."""


# ---------------------------------------------------------------- data
@dataclass
class Door:
    side: str                    # 'L' 'R' (wall opening) or 'U' 'D' (ladder shaft through ceiling / floor)
    a: int                       # L/R: the lintel row (the opening is rows a .. a+DOOR_H-1); U/D: the left column
    oneway: bool = False         # D: a hole with no ladder; U: a ceiling that cannot be climbed
    gate: bool = False           # an ability is needed to leave through it

    @property
    def floor(self) -> int:      # L/R: the corridor floor's row
        return self.a + DOOR_H

    @property
    def ladder(self) -> bool:
        return not self.oneway


@dataclass
class Room:
    id: int
    gx: int
    gy: int
    cw: int
    ch: int
    name: str = ""
    doors: list[Door] = field(default_factory=list)
    grid: np.ndarray | None = None
    walkers: list[tuple[float, float]] = field(default_factory=list)
    flyers: list[tuple[float, float]] = field(default_factory=list)
    spawn: tuple[float, float] | None = None
    tries: int = 1
    plan: RoomPlan | None = None
    orb: tuple[int, int] | None = None

    @property
    def w(self) -> int:
        return self.cw * SCREEN_W

    @property
    def h(self) -> int:
        return self.ch * SCREEN_H

    @property
    def ox(self) -> int:
        return self.gx * SCREEN_W

    @property
    def oy(self) -> int:
        return self.gy * SCREEN_H

    @property
    def kind(self) -> str:
        return KINDS[(self.cw, self.ch)]

    # -------------------------------------------------------------- places
    def entry_place(self, d: int):
        """Where the player stands on arriving through door ``d`` (a plan place)."""
        door = self.doors[d]
        if door.side == "L":
            return ("s", 3, door.floor)
        if door.side == "R":
            return ("s", 2 * self.w - 3, door.floor)
        if door.side == "U":
            return ("l", door.a, 1) if door.ladder else ("s", 2 * door.a + 2, 3)
        return ("l", door.a, self.h - 2) if door.ladder else ("s", 2 * door.a - 1, self.h - 1)

    def exit_place(self, d: int):
        """The place a player must reach to leave through door ``d`` (None: this door cannot be left by)."""
        door = self.doors[d]
        if door.side == "U" and door.oneway:
            return None
        if door.side == "D" and door.oneway:
            return ("s", 2 * door.a - 1, self.h - 1)
        return self.entry_place(d)

    def arrival_at(self, d: int) -> tuple[float, float, bool]:
        """Where the game puts the player arriving through door ``d``: ``(x, y, on_a_ladder)``."""
        door = self.doors[d]
        if door.side == "L":
            return 1.5, door.floor - PLAYER_RY - 1.0e-3, False
        if door.side == "R":
            return self.w - 1.5, door.floor - PLAYER_RY - 1.0e-3, False
        if door.side == "U":
            return (door.a + 0.5, 0.5, True) if door.ladder else (door.a + 1.0, 0.5, False)
        return door.a + 0.5, self.h - 0.5, True

    def spawn_arrival(self) -> tuple[float, float, bool]:
        x, y = self.spawn if self.spawn else (self.w / 2.0, self.h - PLAYER_RY - 1.0e-3)
        return x, y, False

    def door_point(self, d: int) -> tuple[float, float]:
        """The middle of door ``d`` in world tiles (for the map)."""
        door = self.doors[d]
        if door.side == "L":
            return float(self.ox), self.oy + door.a + 1.5
        if door.side == "R":
            return float(self.ox + self.w), self.oy + door.a + 1.5
        if door.side == "U":
            return self.ox + door.a + 1.0, float(self.oy)
        return self.ox + door.a + 1.0, float(self.oy + self.h)

    def spawn_place(self):
        x, y, _ = self.spawn_arrival()
        return ("s", int(round(x * 2)), int(round(y + PLAYER_RY)))

    def door_for(self, side: str, x: float, y: float) -> int | None:
        """The door of ``side`` the player left through at ``(x, y)``."""
        best, where = None, 1.0e9
        for i, door in enumerate(self.doors):
            if door.side != side:
                continue
            centre = door.a + (DOOR_H - 1) / 2.0 if side in "LR" else door.a + 1.0
            at = y if side in "LR" else x
            if abs(at - centre) < where:
                best, where = i, abs(at - centre)
        return best


@dataclass
class World:
    seed: int
    size: str
    gated: bool
    gw: int
    gh: int
    rooms: list[Room]
    links: dict[tuple[int, int], tuple[int, int]]
    start_room: int
    goal_room: int
    main_path: list[int]
    orb_room: int | None
    gate_room: int | None
    report: dict[str, Any] = field(default_factory=dict)

    def link(self, room: int, door: int) -> tuple[int, int]:
        return self.links[(room, door)]

    def overview(self, visited: set[int], current: int, show: bool) -> dict[str, Any]:
        """What the page's map overlay draws: rooms (screens on the coarse grid) and the joins between them."""
        rooms = [{"id": r.id, "gx": r.gx, "gy": r.gy, "cw": r.cw, "ch": r.ch, "seen": r.id in visited, "kind": r.kind}
                 for r in self.rooms]
        joins = []
        for (a, da), (b, db) in self.links.items():
            if a < b and (a in visited or b in visited):
                x, y = self.rooms[a].door_point(da)
                joins.append({"a": a, "b": b, "x": x, "y": y, "side": self.rooms[a].doors[da].side,
                              "oneway": bool(self.rooms[a].doors[da].oneway or self.rooms[b].doors[db].oneway)})
        return {"seed": self.seed, "grid": [self.gw, self.gh], "unit": [SCREEN_W, SCREEN_H], "show": bool(show),
                "current": current, "rooms": rooms, "joins": joins}


# ---------------------------------------------------------------- the coarse layout
def _tile_rooms(rng: np.random.Generator, gw: int, gh: int):
    owner = -np.ones((gh, gw), dtype=int)
    rooms: list[Room] = []
    for gy in range(gh):
        for gx in range(gw):
            if owner[gy, gx] >= 0:
                continue
            fits = [(shape, wt) for shape, wt in SHAPES
                    if gx + shape[0] <= gw and gy + shape[1] <= gh and (owner[gy:gy + shape[1], gx:gx + shape[0]] < 0).all()]
            weights = np.array([wt for _, wt in fits], float)
            (cw, ch), _ = fits[int(rng.choice(len(fits), p=weights / weights.sum()))]
            room = Room(len(rooms), gx, gy, cw, ch)
            owner[gy:gy + ch, gx:gx + cw] = room.id
            rooms.append(room)
    for room in rooms:
        room.name = f"{ADJECTIVES[int(rng.integers(len(ADJECTIVES)))]} {NOUNS[int(rng.integers(len(NOUNS)))]}"
    return rooms, owner


def _borders(owner: np.ndarray) -> dict[tuple[int, int], list[tuple[str, int, int]]]:
    """Adjacent room pairs (sorted ids) with their shared border units: ``('h', gy, left)`` a room ``left`` to the
    left of the other on coarse row ``gy``, or ``('v', gx, top)`` above it on coarse column ``gx``."""
    gh, gw = owner.shape
    out: dict[tuple[int, int], list[tuple[str, int, int]]] = {}
    for gy in range(gh):
        for gx in range(gw):
            a = int(owner[gy, gx])
            if gx + 1 < gw and owner[gy, gx + 1] != a:
                b = int(owner[gy, gx + 1])
                out.setdefault((min(a, b), max(a, b)), []).append(("h", gy, a))
            if gy + 1 < gh and owner[gy + 1, gx] != a:
                b = int(owner[gy + 1, gx])
                out.setdefault((min(a, b), max(a, b)), []).append(("v", gx, a))
    return out


def _span(door: Door, room: Room) -> tuple[int, int, int, int]:
    """The (x0, x1, y0, y1) tile rectangle a door's structure claims (inclusive)."""
    if door.side == "L":
        return 0, 5, door.a, door.floor
    if door.side == "R":
        return room.w - 6, room.w - 1, door.a, door.floor
    if door.side == "U":
        return door.a - 2, door.a + 3, 0, 4
    return door.a - 2, door.a + 3, room.h - 3, room.h - 1


def _clash(room: Room, door: Door) -> bool:
    x0, x1, y0, y1 = _span(door, room)
    for other in room.doors:
        a0, a1, b0, b1 = _span(other, room)
        if x0 <= a1 and a0 <= x1 and y0 <= b1 and b0 <= y1:
            return True
    return False


# ---------------------------------------------------------------- one room's interior
def _put(g: np.ndarray, row: int, c0: int, c1: int, kind: int, only_empty: bool = True) -> None:
    h, w = g.shape
    for c in range(max(1, c0), min(w - 1, c1)):
        if 0 < row < h - 1 and (not only_empty or g[row, c] == EMPTY):
            g[row, c] = kind


def build_interior(room: Room, rng: np.random.Generator) -> np.ndarray:
    """A candidate layout for ``room`` (not yet proven)."""
    w, h = room.w, room.h
    g = np.full((h, w), EMPTY, dtype=int)
    g[0, :] = g[h - 1, :] = SOLID
    g[:, 0] = g[:, w - 1] = SOLID
    protect = np.zeros((h, w), bool)
    gate_door = next((d for d in room.doors if d.gate), None)
    chains: list[tuple[int, int, int]] = []              # (start row, start x0, start x1) of platform chains to lay
    for door in room.doors:
        x0, x1, y0, y1 = _span(door, room)
        protect[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True
        if door.side in "LR":
            g[door.a:door.a + DOOR_H, 0 if door.side == "L" else w - 1] = EMPTY
            cols = range(1, 5) if door.side == "L" else range(w - 5, w - 1)
            if door.floor < h - 1:
                for c in cols:
                    g[door.floor, c] = ONEWAY
                if not door.gate:
                    chains.append((door.floor, cols[0], cols[-1] + 1))
        elif door.side == "D":
            g[h - 1, door.a:door.a + DOOR_W] = EMPTY
            if door.ladder:
                g[h - 2:h, door.a] = LADDER
        else:
            g[0, door.a:door.a + DOOR_W] = EMPTY
            for c in range(door.a - 2, door.a + 4):
                g[3, c] = ONEWAY
            if door.ladder:
                g[0:4, door.a] = LADDER
            chains.append((3, door.a - 2, door.a + 4))
    # platform chains: planks two rows apart, overlapping, from each raised door down to the floor
    for row, x0, x1 in chains:
        px0, px1, q = x0, x1, row
        while True:
            q += 2
            if q > h - 2:
                break
            length = int(rng.integers(4, 7))
            nx0 = int(rng.integers(px0 - length + 2, px1 - 1))          # overlaps the plank above by two or more tiles
            nx0 = int(np.clip(nx0, 1, w - 1 - length))
            if min(nx0 + length, px1) - max(nx0, px0) < 1:
                nx0 = int(np.clip(px0, 1, w - 1 - length))
            if gate_door is not None and abs(q - gate_door.floor) <= 2:      # nothing level with the gate's ledge near it
                nx0 = max(nx0, 10) if gate_door.side == "L" else min(nx0, w - 10 - length)
                nx0 = int(np.clip(nx0, 1, w - 1 - length))
            _put(g, q, nx0, nx0 + length, ONEWAY)
            px0, px1 = nx0, nx0 + length
    # the floor: spike strips and pillars, away from the doors
    free = ~protect[h - 2, :]
    for _ in range(int(rng.integers(0, 1 + w // 10))):
        width = int(rng.integers(1, 3))                  # one jump clears two
        c0 = int(rng.integers(2, max(3, w - 2 - width)))
        if free[max(0, c0 - 1):c0 + width + 1].all() and (g[h - 2, c0 - 1:c0 + width + 1] == EMPTY).all():
            g[h - 2, c0:c0 + width] = SPIKE
    for _ in range(int(rng.integers(0, 1 + w // 12))):
        c0 = int(rng.integers(2, w - 2))
        if free[c0 - 1:c0 + 2].all() and (g[h - 2, max(0, c0 - 2):c0 + 3] == EMPTY).all():
            g[h - 2, c0] = SOLID
    # loose planks and ladders between planks
    for _ in range(0 if gate_door is not None else int(rng.integers(1, 2 + (w * h) // 90))):
        row = int(rng.integers(4, h - 2))
        length = int(rng.integers(3, 6))
        c0 = int(rng.integers(1, max(2, w - 1 - length)))
        if not protect[row, c0:c0 + length].any() and not protect[row - 1, c0:c0 + length].any():
            _put(g, row, c0, c0 + length, ONEWAY)
    if gate_door is not None:
        # keep the gate's approach bare so only the double jump climbs it
        x0, x1, _, _ = _span(gate_door, room)
        lo, hi = (1, 12) if gate_door.side == "L" else (w - 13, w - 1)
        g[gate_door.floor + 1:h - 1, lo:hi] = np.where(g[gate_door.floor + 1:h - 1, lo:hi] == ONEWAY, EMPTY,
                                                       g[gate_door.floor + 1:h - 1, lo:hi])
    return g


def _prove(room: Room, grid: np.ndarray) -> tuple[bool, str, RoomPlan | None]:
    """Is every door of ``room`` usable from every other (the gate only with the ability)?"""
    plan = RoomPlan.build(grid)
    places = set(plan.places)
    ents = {d: room.entry_place(d) for d in range(len(room.doors))}
    exits = {d: room.exit_place(d) for d in range(len(room.doors)) if room.exit_place(d) is not None}
    for d, p in list(ents.items()) + list(exits.items()):
        if p not in places:
            return False, f"door {room.doors[d].side} place {p} not standable", None
    for d, entry in ents.items():
        free = plan.reach([entry], False)
        full = plan.reach([entry], True)
        for e, p in exits.items():
            door = room.doors[e]
            gated = door.gate and e != d
            if gated:
                if p in free:
                    return False, "gate door reachable without the ability", None
                if p not in full:
                    return False, "gate door unreachable even with the ability", None
            elif p not in free:
                return False, f"door {room.doors[e].side} unreachable from {room.doors[d].side} entry", None
    return True, "", plan


def make_interior(room: Room, rng: np.random.Generator, report: dict) -> None:
    reasons = report.setdefault("room_failures", [])
    for attempt in range(1, ROOM_TRIES + 1):
        grid = build_interior(room, rng)
        ok, why, plan = _prove(room, grid)
        if ok:
            room.grid, room.plan, room.tries = grid, plan, attempt
            return
        reasons.append((room.id, attempt, why))
    raise GenerationFailed(f"room {room.id} ({room.w}x{room.h}, doors {[d.side for d in room.doors]}) unprovable in {ROOM_TRIES} tries: {reasons[-1][2]}")


# ---------------------------------------------------------------- contents
def furnish(room: Room, rng: np.random.Generator, world_start: bool, orb: bool) -> None:
    """Gems, the orb, enemies and the spawn point, on places the plan proves reachable."""
    plan, g, w, h = room.plan, room.grid, room.w, room.h
    free_places = set()
    for d in range(len(room.doors)):
        free_places |= plan.reach([room.entry_place(d)], False)
    if not room.doors:
        free_places = set(plan.places)
    stand = sorted(p for p in free_places if p[0] == "s" and p[1] % 2 == 1)        # cell-centred places
    keep_clear = []
    for door in room.doors:
        x0, x1, y0, y1 = _span(door, room)
        keep_clear.append((x0, x1, y0, y1))

    def near_door(p):
        x, row = p[1] / 2.0, p[2]
        return any(x0 - 1 <= x <= x1 + 1 and y0 - 1 <= row <= y1 + 1 for x0, x1, y0, y1 in keep_clear)

    open_places = [p for p in stand if not near_door(p) and g[p[2] - 1, p[1] // 2] == EMPTY]
    if world_start:
        floor = [p for p in open_places if p[2] == h - 1] or open_places or stand
        centre = min(floor, key=lambda p: abs(p[1] / 2.0 - w / 2.0))
        room.spawn = (centre[1] / 2.0, centre[2] - PLAYER_RY - 1.0e-3)
        open_places = [p for p in open_places if abs(p[1] - centre[1]) > 4]
    if orb and open_places:
        top = min(open_places, key=lambda p: (p[2], abs(p[1] / 2.0 - w / 2.0)))
        g[top[2] - 1, top[1] // 2] = ORB
        room.orb = (top[1] // 2, top[2] - 1)
        open_places = [p for p in open_places if p != top]
    order = rng.permutation(len(open_places)) if open_places else []
    for i in order[:max(2, (w * h) // 80)]:
        p = open_places[int(i)]
        if g[p[2] - 1, p[1] // 2] == EMPTY:
            g[p[2] - 1, p[1] // 2] = GEM
    # walkers on a plank run at least three tiles long; flyers in the air above the reachable places
    runs: dict[int, list[int]] = {}
    for p in open_places:
        if g[p[2] - 1, p[1] // 2] == EMPTY:
            runs.setdefault(p[2], []).append(p[1] // 2)
    spots = []
    for row, cols in runs.items():
        cols = sorted(cols)
        for c in cols:
            if c - 1 in cols and c + 1 in cols:
                spots.append((c + 0.5, row - 0.35 - 1.0e-3))
    n_walk = min(N_WALKERS, (w // SCREEN_W) + (h // SCREEN_H) + int(rng.integers(0, 2)))
    room.walkers = [spots[int(i)] for i in rng.permutation(len(spots))[:n_walk]] if spots else []
    n_fly = min(N_FLYERS, (w * h) // 300)
    cand = [p for p in open_places if p[2] >= 5]
    room.flyers = [(cand[int(i)][1] / 2.0, cand[int(i)][2] - 2.2) for i in rng.permutation(len(cand))[:n_fly]] if cand else []


# ---------------------------------------------------------------- the world
def _door_pair(rng, A: Room, B: Room, unit: tuple[str, int, int], force_floor: int | None = None, ground: int | None = None):
    """A pair of doors for rooms A (left / top) and B joined at border ``unit``, or None when none fits."""
    kind, k, _ = unit
    if kind == "v" and ground is not None and B.id == ground:
        return None                  # the gate room has no way in from above that bypasses its ledge
    if kind == "h":
        lo, hi = k * SCREEN_H + 4, k * SCREEN_H + SCREEN_H - 1
        options = [force_floor] if force_floor is not None else [int(v) for v in rng.permutation(np.arange(lo, hi + 1))]
        for room in (A, B):
            if ground is not None and room.id == ground and force_floor is None:
                options = [v for v in options if v == room.oy + room.h - 1]
        if force_floor is None and rng.random() < 0.55 and hi in options:
            options.remove(hi)
            options.insert(0, hi)                        # often at the floor
    else:
        options = [int(v) for v in rng.permutation(np.arange(k * SCREEN_W + 3, k * SCREEN_W + SCREEN_W - 4))]
    for g in options:
        if kind == "h":
            if not lo <= g <= hi:
                return None
            da, db = Door("R", g - A.oy - DOOR_H), Door("L", g - B.oy - DOOR_H)
        else:
            da, db = Door("D", g - A.ox), Door("U", g - B.ox)
        if not _clash(A, da) and not _clash(B, db):
            return da, db
    return None


def generate(seed: int, size: str = "small", gated: bool = True, *, tries: int = WORLD_TRIES) -> World:
    """A deterministic world; raises ``GenerationFailed`` when ``tries`` layouts all fail."""
    failures: list[str] = []
    for attempt in range(1, tries + 1):
        rng = np.random.default_rng([int(seed), attempt])
        try:
            world = _generate_once(rng, int(seed), size, gated)
        except GenerationFailed as error:
            failures.append(str(error))
            continue
        world.report["attempt"] = attempt
        world.report["failures"] = failures
        return world
    raise GenerationFailed(f"seed {seed}: no valid world in {tries} tries: {failures[-3:]}")


def _generate_once(rng: np.random.Generator, seed: int, size: str, gated: bool) -> World:
    gw, gh = SIZES[size]
    rooms, owner = _tile_rooms(rng, gw, gh)
    borders = _borders(owner)
    pairs = sorted(borders)
    adj: dict[int, list[int]] = {r.id: [] for r in rooms}
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)
    start = int(rng.integers(len(rooms)))
    # a randomized spanning tree (depth-first, so it is long and branching)
    parent, order, stack = {start: None}, [], [start]
    while stack:
        r = stack.pop()
        order.append(r)
        nbrs = [n for n in adj[r] if n not in parent]
        for n in (nbrs[int(i)] for i in rng.permutation(len(nbrs))):
            if n not in parent:
                parent[n] = r
                stack.append(n)
    if len(parent) != len(rooms):
        raise GenerationFailed("layout not connected")
    depth = {start: 0}
    for r in order:
        if parent[r] is not None:
            depth[r] = depth[parent[r]] + 1
    goal = max(order, key=lambda r: (depth[r], -r))
    main, cur = [goal], goal
    while parent[cur] is not None:
        cur = parent[cur]
        main.append(cur)
    main.reverse()
    tree = {tuple(sorted((r, p))) for r, p in parent.items() if p is not None}
    # the gate: a horizontal join on the main path, with at least one room before it besides the start
    gate_edge, gate_floor = None, None
    if gated and len(main) >= 4:
        for i in rng.permutation(np.arange(1, len(main) - 1)):
            G, N = rooms[main[int(i)]], rooms[main[int(i) + 1]]
            key = tuple(sorted((G.id, N.id)))
            floor = G.oy + G.h - 5
            if any(u[0] == "h" and u[1] * SCREEN_H + 4 <= floor <= u[1] * SCREEN_H + SCREEN_H - 1 for u in borders[key]):
                gate_edge, gate_floor = (G.id, N.id, int(i)), floor
                break
    side_of_gate: set[int] = set()
    if gate_edge:
        G, N, i = gate_edge
        # the start side of the cut: rooms the tree reaches from the start without crossing the gate edge
        seen, st = {start}, [start]
        while st:
            r = st.pop()
            for q in adj[r]:
                if q not in seen and tuple(sorted((r, q))) in tree and tuple(sorted((r, q))) != tuple(sorted((G, N))):
                    seen.add(q)
                    st.append(q)
        side_of_gate = seen
    # the joins: tree edges, then loops that do not bypass the gate
    chosen = list(tree)
    for key in pairs:
        if key in tree:
            continue
        crosses = gate_edge and ((key[0] in side_of_gate) != (key[1] in side_of_gate))
        if not crosses and rng.random() < 0.4:
            chosen.append(key)
    links: dict[tuple[int, int], tuple[int, int]] = {}
    gate_key = tuple(sorted(gate_edge[:2])) if gate_edge else None
    for key in sorted(chosen, key=lambda k: (k != gate_key, k)):       # the gate's door first: its row is fixed
        a, b = key
        is_gate = bool(gate_edge) and key == tuple(sorted(gate_edge[:2]))
        unit_list = borders[key]
        units = [unit_list[int(i)] for i in rng.permutation(len(unit_list))]
        if is_gate:
            units = [u for u in units if u[0] == "h" and u[1] * SCREEN_H + 4 <= gate_floor <= u[1] * SCREEN_H + SCREEN_H - 1]
        made, first = None, None
        for unit in units:
            first = rooms[unit[2]]
            second = rooms[b if unit[2] == a else a]
            made = _door_pair(rng, first, second, unit, force_floor=gate_floor if is_gate else None,
                              ground=gate_edge[0] if gate_edge and not is_gate else None)
            if made:
                break
        if not made:
            if key in tree:
                raise GenerationFailed(f"no door fits between rooms {a} and {b} ({'gate' if is_gate else 'tree'} "
                                       f"{[(u[0], rooms[u[2]].w, rooms[u[2]].h, [d.side + str(d.a) for d in rooms[u[2]].doors]) for u in units]})")
            continue
        A, B = first, rooms[b if first.id == a else a]
        da, db = made
        if is_gate:
            (da if A.id == gate_edge[0] else db).gate = True
        elif key not in tree and da.side == "D" and rng.random() < 0.5:
            da.oneway = db.oneway = True                 # a drop: down only; the other joins bring the player back
        A.doors.append(da)
        B.doors.append(db)
        links[(A.id, len(A.doors) - 1)] = (B.id, len(B.doors) - 1)
        links[(B.id, len(B.doors) - 1)] = (A.id, len(A.doors) - 1)
    orb_room = None
    if gate_edge:
        cands = [r for r in side_of_gate if r != start and r != gate_edge[0]] or [r for r in side_of_gate if r != start]
        if not cands:
            raise GenerationFailed("no room for the orb before the gate")
        leaves = [r for r in cands if len([k for k in tree if r in k]) == 1]
        pool = leaves or cands
        orb_room = pool[int(rng.integers(len(pool)))]
    report: dict[str, Any] = {}
    for room in rooms:
        make_interior(room, rng, report)
    for room in rooms:
        furnish(room, rng, room.id == start, room.id == orb_room)
    world = World(seed, size, gated, gw, gh, rooms, links, start, goal, main, orb_room,
                  gate_edge[0] if gate_edge else None, report)
    world.report["gate_edge"] = gate_edge
    analysis = analyze(world)
    world.report.update(analysis)
    if not analysis["ok"]:
        raise GenerationFailed(f"world invalid: {analysis['why']}")
    return world


# ---------------------------------------------------------------- the world-level proof
def analyze(world: World) -> dict[str, Any]:
    """Reachability over the whole world: which rooms the player gets to without the ability, whether the orb is
    among them, whether every room is reached with it, and whether the player can always get back to the start."""
    rooms = world.rooms

    def closure(ability: bool):
        """Arrival states (room, door or -1 for the spawn) reached, with the edges between them."""
        states = {(world.start_room, -1)}
        edges: dict = {}
        stack = [(world.start_room, -1)]
        got_orb = False
        while stack:
            room_id, d = stack.pop()
            room = rooms[room_id]
            entry = room.spawn_place() if d < 0 else room.entry_place(d)
            reach = room.plan.reach([entry], ability)
            if room.orb is not None and ("s", room.orb[0] * 2 + 1, room.orb[1] + 1) in reach:
                got_orb = True
            for e in range(len(room.doors)):
                p = room.exit_place(e)
                if p is None or p not in reach:
                    continue
                other = world.links.get((room_id, e))
                if other is None:
                    continue
                edges.setdefault((room_id, d), set()).add(other)
                if other not in states:
                    states.add(other)
                    stack.append(other)
        return states, edges, got_orb

    states0, _, orb0 = closure(False)
    states1, edges1, _ = closure(True)
    reached0 = {s[0] for s in states0}
    reached1 = {s[0] for s in states1}
    # return: every reached state can get back to a state in the start room
    rev: dict = {}
    for src, dsts in edges1.items():
        for dst in dsts:
            rev.setdefault(dst, set()).add(src)
    back = {s for s in states1 if s[0] == world.start_room}
    stack = list(back)
    while stack:
        s = stack.pop()
        for q in rev.get(s, ()):
            if q not in back:
                back.add(q)
                stack.append(q)
    stuck = sorted(s for s in states1 if s not in back)
    gate_needed = bool(world.gate_room is not None)
    ok, why = True, ""
    if len(reached1) != len(rooms):
        ok, why = False, f"rooms {sorted(set(range(len(rooms))) - reached1)} unreachable even with the ability"
    elif world.gated and gate_needed and not orb0:
        ok, why = False, "the orb cannot be reached without the ability"
    elif stuck:
        ok, why = False, f"arrivals {stuck[:4]} cannot return to the start room"
    elif gate_needed and len(reached0) == len(rooms):
        ok, why = False, "the gate gates nothing"
    return {"ok": ok, "why": why, "rooms": len(rooms), "reached_without_ability": len(reached0),
            "reached_with_ability": len(reached1), "orb_reachable_without_ability": bool(orb0),
            "one_way_joins": sum(1 for r in rooms for d in r.doors if d.oneway and d.side == "D"),
            "joins": len(world.links) // 2, "loops": len(world.links) // 2 - (len(rooms) - 1)}
