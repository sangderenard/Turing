"""Walkers: the shared substrate for games of entities with motives.

A walker game (``tilequest``; the sphere treadmill after it) is an entity law
(``entities.LawPiece``) whose lanes are WALKERS of different KINDS.  Each kind
is a row of data: its team, speed, senses, health, damage, how long it stays
dead, and a tiny CHOICE NETWORK.  The network is a one-hidden-layer MLP with
constant weights per kind; every lane's features go through every kind's
network inside the law, batched, and each lane keeps the logits of its own
kind (``choose_motivation``).  The network does not steer.  It CHOOSES among
MOTIVATIONS, and the law turns the chosen motivation into a desired heading:

    wander   drift along a slowly turning heading, bent away from walls
    seek     direct bearing to the nearest foe (a moving target)
    flee     away from the nearest foe, toward the kind's refuge field
    follow   descend the flow field of the lane's current goal
    attack   seek, faster (the lunge)
    rest     stand still (and mend)

FEATURES (what a network sees), all computed batched by the law:
bias, foe_prox, foe_close, foe_ahead, ally_prox, goal_far, health, need,
crowd, phase, busy, refuge_near.

PATHING is by flow fields.  For each goal type the map's distance field is
built ONCE, in NumPy, at initialisation (``FlowField``: Dijkstra over the
walkable cells, eight-connected without corner cutting), together with its
unit DESCENT DIRECTION per cell.  The fields live in the dt system's columns,
one lane per map cell (``MAP`` lanes after the entity lanes), so a new map is
new column values and never a recompile.  The law looks a lane's cell up with
a gather (``field_at``).  A moving target (the player) is not a field: the
walker seeks it directly, and the clearance field (``AVOID``: the direction
away from walls, and the distance to them) bends its heading around walls.

Separation (``separation_of``) keeps walkers from stacking.  Every helper
that the law calls is written on AbstractTensor spans, batched, with no
per-lane Python and no host round trip, so the law traces through the SSA
tensor backend and compiles natively.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from src.common.tensors import AbstractTensor

where = AbstractTensor.where

MOTIVATIONS = ("wander", "seek", "flee", "follow", "attack", "rest")
WANDER, SEEK, FLEE, FOLLOW, ATTACK, REST = range(6)
FEATURES = ("bias", "foe_prox", "foe_close", "foe_ahead", "ally_prox", "goal_far", "health", "need",
            "crowd", "phase", "busy", "refuge_near")
#: Speed multiplier of each motivation, applied to the kind's speed.
PACE = (0.55, 0.9, 1.25, 1.0, 1.3, 0.0)


def as_float(mask):
    return mask.to_dtype("float64")


# ------------------------------------------------------------------- kinds

@dataclass
class Kind:
    """One kind of walker, as data.

    ``net``: ``{"hidden": {name: ({feature: weight}, bias)}, "out": {motivation:
    ({hidden: weight}, bias)}}``.  A hidden unit is the gate
    ``(tanh(w . features + bias) + 1) / 2`` in [0, 1]; a motivation's logit is
    ``bias + w . gates``.  Absent weights are zero (the net is sparse)."""

    name: str
    id: int
    team: int = 0                 # lanes of different teams are foes; -1: nobody's
    speed: float = 0.0            # 0: does not walk
    sense: float = 6.0            # how far it sees a foe
    hp: float = 1.0
    damage: float = 0.0           # per contact blow
    radius: float = 0.3
    reload: float = 0.9           # seconds between blows (or shots)
    respawn: float = 0.0          # seconds dead before it returns (0: never)
    needrate: float = 0.0         # need (hunger) per second
    goal: tuple[int, int] = (4, 4)   # flow field to follow, by phase 0 / 1
    refuge: int = 4               # flow field to flee toward
    mends: float = 0.0            # hit points per second while resting
    net: dict = field(default_factory=dict)


def kind_columns(kinds: dict[int, Kind], kind_of_lane: np.ndarray) -> dict[str, np.ndarray]:
    """The per-lane static columns a kind row fills in (the law reads these,
    never a kind table).  Lanes whose kind id is unknown get zeros."""
    out = {name: np.zeros(len(kind_of_lane)) for name in
           ("team", "speed", "sense", "hpmax", "dmg", "rad", "reload_t", "respawn_t", "needrate", "goalf0", "goalf1",
            "refugef", "mends")}
    for lane, kid in enumerate(kind_of_lane):
        row = kinds.get(int(kid))
        if row is None:
            out["team"][lane], out["goalf0"][lane], out["goalf1"][lane], out["refugef"][lane] = -1.0, 4.0, 4.0, 4.0
            continue
        for name, value in (("team", row.team), ("speed", row.speed), ("sense", row.sense), ("hpmax", row.hp),
                            ("dmg", row.damage), ("rad", row.radius), ("reload_t", row.reload),
                            ("respawn_t", row.respawn), ("needrate", row.needrate), ("goalf0", row.goal[0]),
                            ("goalf1", row.goal[1]), ("refugef", row.refuge), ("mends", row.mends)):
            out[name][lane] = float(value)
    return out


def network_logits(row: Kind, feats: dict, zero):
    """One kind's logits (a list of six spans) from the feature spans."""
    net = row.net
    if not net:
        return None
    gates = {}
    for name, (weights, bias) in net["hidden"].items():
        z = zero + float(bias)
        for feature, w in weights.items():
            if w:
                z = z + feats[feature] * float(w)
        gates[name] = z.tanh() * 0.5 + 0.5
    logits = []
    for motivation in MOTIVATIONS:
        weights, bias = net["out"].get(motivation, ({}, -9.0))
        z = zero + float(bias)
        for hidden, w in weights.items():
            if w:
                z = z + gates[hidden] * float(w)
        logits.append(z)
    return logits


def choose_motivation(kinds: dict[int, Kind], kind, feats: dict, zero):
    """The motivation index each lane's own-kind network picks (a span).

    Every kind's net is evaluated on every lane's features and the lane keeps
    its own kind's logits (the kind row selects the weights, by mask).  The
    argmax breaks ties toward the lower index."""
    chosen = [zero - 9.0 for _ in MOTIVATIONS]
    for kid, row in kinds.items():
        logits = network_logits(row, feats, zero)
        if logits is None:
            continue
        mine = as_float(kind == float(kid))
        chosen = [c + mine * (z + 9.0) for c, z in zip(chosen, logits)]
    best, index = chosen[0], zero
    for m in range(1, len(MOTIVATIONS)):
        better = chosen[m] > best + 1e-9
        best = where(better, chosen[m], best)
        index = where(better, zero + float(m), index)
    return index


def explain(row: Kind, **given) -> tuple[str, list[float]]:
    """A kind's choice for one hand-written feature vector, in plain NumPy:
    (motivation name, the six logits).  For reading a kind's character."""
    feats = {name: np.asarray(float(given.get(name, 1.0 if name == "bias" else 0.0))) for name in FEATURES}
    gates = {name: np.tanh(sum(w * feats[f] for f, w in weights.items()) + bias) * 0.5 + 0.5
             for name, (weights, bias) in row.net["hidden"].items()}
    logits = []
    for motivation in MOTIVATIONS:
        weights, bias = row.net["out"].get(motivation, ({}, -9.0))
        logits.append(float(bias + sum(w * gates[h] for h, w in weights.items())))
    return MOTIVATIONS[int(np.argmax(logits))], logits


# ------------------------------------------------------------- flow fields

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _steps(passable: np.ndarray, cx: int, cy: int):
    """The legal neighbour moves from (cx, cy): eight-connected, a diagonal
    only when both orthogonal cells are walkable (no corner cutting)."""
    h, w = passable.shape
    for ox, oy in _NEIGHBOURS:
        nx, ny = cx + ox, cy + oy
        if not (0 <= nx < w and 0 <= ny < h) or not passable[ny, nx]:
            continue
        if ox and oy and not (passable[cy, nx] and passable[ny, cx]):
            continue
        yield nx, ny, (1.41421356 if ox and oy else 1.0)


@dataclass
class FlowField:
    """A goal's distance field over a tile map and its unit descent direction.

    ``dist``: cells of path length to the nearest goal cell (``unreachable``
    where there is none).  ``dx, dy``: the unit direction to move at each cell
    (the unit step to the lowest neighbour; zero on a goal cell and on
    unreachable cells).  Built once, in
    NumPy."""

    dist: np.ndarray
    dx: np.ndarray
    dy: np.ndarray

    UNREACHABLE = 999.0

    @classmethod
    def build(cls, passable: np.ndarray, goals: list[tuple[int, int]]) -> "FlowField":
        h, w = passable.shape
        dist = np.full((h, w), cls.UNREACHABLE)
        heap = []
        for gx, gy in goals:
            dist[gy, gx] = 0.0
            heap.append((0.0, gx, gy))
        heapq.heapify(heap)
        while heap:
            d, cx, cy = heapq.heappop(heap)
            if d > dist[cy, cx]:
                continue
            for nx, ny, cost in _steps(passable, cx, cy):
                if d + cost < dist[ny, nx]:
                    dist[ny, nx] = d + cost
                    heapq.heappush(heap, (d + cost, nx, ny))
        dx, dy = np.zeros((h, w)), np.zeros((h, w))
        for cy in range(h):
            for cx in range(w):
                if not passable[cy, cx] or dist[cy, cx] >= cls.UNREACHABLE:
                    continue
                # The single steepest legal step (ties: the first in neighbour
                # order).  A SUM of the steps to every lower neighbour cancels
                # to zero between two equidistant goals, and a walker standing
                # between two homes would not move.
                best, bx, by = dist[cy, cx], 0, 0
                for nx, ny, cost in _steps(passable, cx, cy):
                    if dist[ny, nx] < best - 1e-9:
                        best, bx, by = dist[ny, nx], nx - cx, ny - cy
                norm = np.hypot(bx, by)
                if norm > 0.0:
                    dx[cy, cx], dy[cy, cx] = bx / norm, by / norm
        return cls(dist, dx, dy)

    @classmethod
    def clearance(cls, passable: np.ndarray, cap: float = 4.0) -> "FlowField":
        """The avoidance field: ``dist`` is the distance to the nearest
        blocked cell (the map's edge counts as blocked), capped; ``dx, dy``
        the unit direction of increasing clearance."""
        h, w = passable.shape
        blocked = np.argwhere(~passable)
        ys, xs = np.mgrid[0:h, 0:w]
        clr = np.full((h, w), cap)
        for by, bx in blocked:
            clr = np.minimum(clr, np.hypot(xs - bx, ys - by))
        edge = np.minimum.reduce([xs + 1.0, w - xs, ys + 1.0, h - ys])
        clr = np.minimum(clr, edge)
        clr = np.where(passable, clr, 0.0)
        pad = np.pad(clr, 1, mode="edge")
        gx = pad[1:-1, 2:] - pad[1:-1, :-2]
        gy = pad[2:, 1:-1] - pad[:-2, 1:-1]
        norm = np.hypot(gx, gy)
        safe = np.where(norm > 1e-9, norm, 1.0)
        return cls(clr, np.where(norm > 1e-9, gx / safe, 0.0), np.where(norm > 1e-9, gy / safe, 0.0))


def field_columns(fields: list[FlowField], map_lane0: int, lanes: int) -> dict[str, np.ndarray]:
    """The fields as dt-system columns: ``fdx{i}``, ``fdy{i}``, ``fdd{i}``
    (direction x, direction y, distance), one value per map cell on the map's
    lanes (cell ``cy * width + cx`` on lane ``map_lane0 + cy * width + cx``),
    zero on the entity lanes."""
    out = {}
    for i, fld in enumerate(fields):
        for name, grid in (("fdx", fld.dx), ("fdy", fld.dy), ("fdd", fld.dist)):
            col = np.zeros(lanes)
            col[map_lane0:map_lane0 + grid.size] = grid.reshape(-1)
            out[f"{name}{i}"] = col
    return out


# ------------------------------------------------------------- law helpers

def unit(vx, vy):
    """The unit vector of (vx, vy); zero stays zero."""
    norm = (vx * vx + vy * vy).sqrt()
    safe = norm + 1e-9
    return vx / safe, vy / safe


def cell_lane(x, y, width: int, height: int, map_lane0: int):
    """The map lane of the cell each (x, y) stands in (clamped to the map)."""
    cx = x.floor().clip(0.0, float(width - 1))
    cy = y.floor().clip(0.0, float(height - 1))
    return cy * float(width) + cx + float(map_lane0)


def line_clear(x, y, dx, dy, table, width: int, height: int, map_lane0: int, samples: int = 8):
    """Whether the segment from each (x, y) along (dx, dy) crosses no blocked
    cell: ``table`` is a (LANES,) column that is 1 on blocked map lanes.  The
    segment is sampled ``samples`` times (spaced under a cell for sight
    ranges up to ``samples`` cells); ``x``, ``dx`` may be matrices (every
    lane's sight to every other).  This is what lets a walker SEE a foe, so
    it seeks it directly only when nothing stands between."""
    blocked = None
    for i in range(1, samples):
        t = i / samples
        hit = gather(table, cell_lane(x + dx * t, y + dy * t, width, height, map_lane0)) > 0.5
        blocked = hit if blocked is None else (blocked | hit)
    return ~blocked


def stack_columns(columns: list):
    """Several (LANES,) columns end to end: one gather source for the fields."""
    return AbstractTensor.cat(columns, dim=0)


def gather(source, index):
    """``source[index]`` for a span of float indices (truncated)."""
    return source.index_select(0, index.long())


def field_at(stacked, field_id, cell, lanes: int):
    """A stacked column's value at ``cell`` in field ``field_id`` (a span of
    field numbers)."""
    return gather(stacked, field_id * float(lanes) + cell)


def pairwise(x, y):
    """(dx, dy, d): matrices of the vector from lane i (row) to lane j
    (column), and its length."""
    n = int(x.shape[0])
    dx = x.reshape((1, n)) - x.reshape((n, 1))
    dy = y.reshape((1, n)) - y.reshape((n, 1))
    return dx, dy, (dx * dx + dy * dy + 1e-12).sqrt()


def nearest(dx, dy, d, valid):
    """For each row lane: the distance to the nearest valid column lane
    (1e9: none), and the unit bearing to it."""
    n = int(d.shape[0])
    masked = where(valid, d, 1.0e9)
    best = masked.min(dim=1)
    pick = as_float(masked == best.reshape((n, 1))) * as_float(valid)
    pick = pick / (pick.sum(dim=1).reshape((n, 1)) + 1e-12)
    has = best < 1.0e8
    safe = best + 1e-6
    ux = where(has, (pick * dx).sum(dim=1) / safe, best * 0.0)
    uy = where(has, (pick * dy).sum(dim=1) / safe, best * 0.0)
    return best, ux, uy


def separation_of(dx, dy, d, valid, radius: float):
    """The push each row lane feels from valid neighbours closer than
    ``radius``: linear in the overlap, along the line apart."""
    near = valid & (d < radius)
    push = as_float(near) * ((radius - d) * (1.0 / radius)) / d
    return -(push * dx).sum(dim=1), -(push * dy).sum(dim=1)


def rotate(vx, vy, turn):
    """(vx, vy) turned by ``turn`` radians: the exact rotation by the
    rational (Cayley) half-angle form, as ``entity_motion.ground_travel``
    turns its heading."""
    h = turn * 0.5
    q = 1.0 + h * h
    c = (1.0 - h * h) / q
    s = (h * 2.0) / q
    return vx * c - vy * s, vx * s + vy * c


def shift_block(values, source: int, count: int, target: int):
    """``values[source:source+count]`` moved to lanes ``target..`` of a span
    as long as ``values`` (zero elsewhere): pairs a block of lanes with the
    block that belongs to it (each tower with its arrow)."""
    n = int(values.shape[0])
    parts = []
    if target > 0:
        parts.append(values[0:target] * 0.0)
    parts.append(values[source:source + count])
    if target + count < n:
        parts.append(values[target + count:n] * 0.0)
    return AbstractTensor.cat(parts, dim=0)


def widen(head, tail):
    """The entity lanes' span followed by the map lanes' (unchanged) tail."""
    return AbstractTensor.cat([head, tail], dim=0)
