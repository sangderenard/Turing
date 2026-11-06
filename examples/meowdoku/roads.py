"""Roads on the globe: gradient-limited paths between AA sites, from the land's law.

The world's land is ``entity_motion.terrain_height`` (the law the page bakes
too).  Over a near-even scatter of NODES points on the sphere (Fibonacci),
each joined to its NEIGHBOURS nearest, an edge costs its arc length raised by
its slope (``1 + SLOPE_COST * slope``); an edge steeper than MAX_SLOPE, or
touching water, cannot be built.  SITES AA sites are spread over the land by
farthest-point sampling; the roads are the minimum spanning tree of the
sites under the cheapest (Dijkstra) path between each pair, each tree edge
built along its path.  The polylines are resampled every ROAD_STEP of arc
into the road points the law reads (a fixed population).

``plan(seed)`` is computed once per world and cached.
"""

from __future__ import annotations

import heapq
import math
from functools import lru_cache

import numpy as np

from meowdoku.entity_motion import terrain_height

NODES, NEIGHBOURS = 4000, 8
SITES = 12
MAX_SLOPE, SLOPE_COST = 0.3, 6.0
ROAD_STEP = 1.0
SEA = 0.48


def fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(1.0 - z * z)
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    p = a * (1.0 - t) + b * t
    return p / np.linalg.norm(p)


@lru_cache(maxsize=4)
def plan(seed: int, radius: float, sites: int = SITES, road_points: int = 160):
    """(site positions (S, 3), road polylines [list of (k, 3)], road points (road_points, 3))."""
    nodes = fibonacci_sphere(NODES)
    land = np.array([terrain_height(*p) for p in nodes])
    height, raw = land[:, 0], land[:, 1]
    dry = raw >= SEA
    # neighbours by angle
    dots = nodes @ nodes.T
    nearest = np.argsort(-dots, axis=1)[:, 1:NEIGHBOURS + 1]
    # AA sites: farthest-point sampling over dry land (the dice pick the first)
    rng = np.random.default_rng(seed)
    dry_ids = np.flatnonzero(dry)
    chosen = [int(rng.choice(dry_ids))]
    gap = np.full(NODES, np.inf)
    for _ in range(1, sites):
        gap = np.minimum(gap, np.arccos(np.clip(nodes @ nodes[chosen[-1]], -1.0, 1.0)))
        gap_dry = np.where(dry, gap, -1.0)
        chosen.append(int(np.argmax(gap_dry)))

    def edge_cost(a: int, b: int) -> float:
        if not (dry[a] and dry[b]):
            return math.inf
        arc = radius * math.acos(max(-1.0, min(1.0, float(nodes[a] @ nodes[b]))))
        slope = abs(height[a] - height[b]) / max(arc, 1e-9)
        return math.inf if slope > MAX_SLOPE else arc * (1.0 + SLOPE_COST * slope)

    def dijkstra(src: int):
        dist = np.full(NODES, np.inf)
        prev = np.full(NODES, -1)
        dist[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d, a = heapq.heappop(heap)
            if d > dist[a]:
                continue
            for b in nearest[a]:
                c = edge_cost(a, int(b))
                if c == math.inf:
                    continue
                nd = d + c
                if nd < dist[b]:
                    dist[b], prev[b] = nd, a
                    heapq.heappush(heap, (nd, int(b)))
        return dist, prev

    runs = [dijkstra(s) for s in chosen]
    # Prim's tree over the sites under path costs (unreachable pairs skipped)
    in_tree, edges = {0}, []
    while len(in_tree) < len(chosen):
        best = None
        for i in in_tree:
            for j in range(len(chosen)):
                if j in in_tree:
                    continue
                c = runs[i][0][chosen[j]]
                if c < math.inf and (best is None or c < best[0]):
                    best = (c, i, j)
        if best is None:
            break                       # the rest are cut off by water or cliffs
        _, i, j = best
        in_tree.add(j)
        edges.append((i, j))
    polylines = []
    for i, j in edges:
        path, at, prev = [], chosen[j], runs[i][1]
        while at != -1:
            path.append(at)
            if at == chosen[i]:
                break
            at = int(prev[at])
        polylines.append(nodes[path[::-1]])
    # the road points: every ROAD_STEP of arc along the polylines
    points = []
    for line in polylines:
        for a, b in zip(line[:-1], line[1:]):
            arc = radius * math.acos(max(-1.0, min(1.0, float(a @ b))))
            steps = max(1, int(round(arc / ROAD_STEP)))
            points.extend(_slerp(a, b, s / steps) for s in range(steps))
    points = np.asarray(points) if points else nodes[chosen]
    if len(points) > road_points:
        points = points[np.linspace(0, len(points) - 1, road_points).astype(int)]
    elif len(points) < road_points:
        points = points[np.arange(road_points) % len(points)]
    return nodes[chosen], polylines, points
