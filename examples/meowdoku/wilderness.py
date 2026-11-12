"""The wilderness as a walker meets it: a place on the hypersphere and the flat patch of ground around it.

Owner's design (2026-11-10/12). The wilderness is a flat index inside an n-dimensional hypersphere
(``src.compiler.abstract_ui_wilderness``: every map has a place on it). A walker sees a flat area as far as the eye
can see; terrain is generated just out of sight and solidifies as they approach. Leaving a living map through its
passage hands the walker from the map's flat space to the wilderness with a grace period: until it runs out they can
turn back; after it they cannot go back without finding the map again, and they start from their own map's
embedding. Other maps are met only by coincidence, when the walker's position is close to theirs in cosine.

The patch is the service's standard environment, not a new one: a ``turing.studio-rigid-scene.v1`` packet with the
``native-local-features-v1`` contact model, admitted by ``studio_mesh_environment.StudioMeshEnvironment`` and stepped
by the incumbent fountain law under its managed ``dt``. That is the "local dt flat area": the room is the patch, and
the solid ground is ONE conforming open sheet (separate tiles touching would double contacts at their seams).

What is and is not allocated. The ground everywhere is a deterministic height function of the point on the sphere,
so it is the same every time anyone comes back and costs nothing until asked: nothing past the patch exists. Only the
sheet within ``solid_m`` of the walker is meshed and admitted; the view out to ``view_m`` is described (its extent and
the height function's identity), not built. Permanence -- pinning what people make -- is not built.

Geometry. Local frame: the walker at the origin, ``x`` along the tangent east, ``z`` along the tangent north, ``y`` up,
metres. A local point (x, z) is the point the exponential map reaches from the walker's place: angle
sqrt(x^2 + z^2) / ``radius_m`` along the tangent direction. A step moves the walker by that map and carries the
tangent frame along (projected and re-orthonormalised). ``radius_m`` sets how flat the wilderness feels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from src.compiler.abstract_ui_wilderness import DIMENSIONS, cosine, position

RADIUS_M = 1.0e7          # the sphere's radius in walking metres: large, so the wilderness reads as flat
SOLID_M = 48.0            # the ground within this distance is meshed and solid
VIEW_M = 2000.0           # how far the eye sees; described, never meshed
GRID = 24                 # quads per side of the solid sheet (6 * GRID^2 features, inside the 4,096 the contacts allow)
GRACE_S = 30.0            # how long after leaving a map the walker can still turn back
TERRAIN_MATERIAL = 0
# Octaves of the ground (wavelength metres, amplitude metres). A placeholder like the embedding: blank, to evolve.
OCTAVES = ((400.0, 12.0), (120.0, 4.0), (35.0, 1.2), (10.0, 0.3))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _scale(a, s):
    return tuple(x * s for x in a)


def _add(*vectors):
    return tuple(sum(parts) for parts in zip(*vectors))


def _unit(a):
    length = math.sqrt(_dot(a, a))
    if length == 0.0:
        raise ValueError("a zero direction has no place on the sphere")
    return _scale(a, 1.0 / length)


def arc(a, b) -> float:
    """The angle between two points on the unit sphere, exact at walking scale (acos of a cosine near 1 is not)."""

    return 2.0 * math.asin(min(1.0, math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / 2.0))


def _tangent(p, v):
    """``v`` with its component along ``p`` removed, made unit length."""

    return _unit(_add(v, _scale(p, -_dot(p, v))))


@dataclass(frozen=True)
class Walker:
    """Where someone stands in the wilderness and which way is east and north there."""

    map_identity: str             # the living map they came out of
    point: tuple                  # unit vector on S^(n-1)
    east: tuple                   # unit tangent
    north: tuple                  # unit tangent, orthogonal to east
    left_at: float                # when they left the map (the caller's clock)
    radius_m: float = RADIUS_M

    def can_return(self, now: float, grace_s: float = GRACE_S) -> bool:
        """Within the grace period the way back is still open; after it, the map must be found again."""

        return 0.0 <= now - self.left_at <= grace_s


def leave_map(map_identity: str, *, now: float, dimensions: int = DIMENSIONS, radius_m: float = RADIUS_M) -> Walker:
    """A walker comes out of a living map's passage: they start from the map's own place in the wilderness."""

    point = position(map_identity, dimensions)
    east = _tangent(point, position(map_identity + "/east", dimensions))
    north = _tangent(point, position(map_identity + "/north", dimensions))
    north = _unit(_add(north, _scale(east, -_dot(east, north))))
    return Walker(map_identity, point, east, north, now, radius_m)


def place(walker: Walker, x: float, z: float) -> tuple:
    """The point on the sphere that the local ground point (x, z) metres stands on."""

    r = math.hypot(x, z)
    if r == 0.0:
        return walker.point
    direction = _add(_scale(walker.east, x / r), _scale(walker.north, z / r))
    angle = r / walker.radius_m
    return _unit(_add(_scale(walker.point, math.cos(angle)), _scale(direction, math.sin(angle))))


def walk(walker: Walker, x: float, z: float) -> Walker:
    """Move the walker by (x, z) metres of local ground and carry their east and north along."""

    point = place(walker, x, z)
    east = _tangent(point, walker.east)
    north = _tangent(point, _add(walker.north, _scale(east, -_dot(east, walker.north))))
    return replace(walker, point=point, east=east, north=north)


def _octave_directions(dimensions):
    return tuple(position(f"wilderness-ground:{k}", dimensions) for k in range(len(OCTAVES)))


def ground(point: tuple, radius_m: float = RADIUS_M) -> float:
    """Height in metres of the ground at a point on the sphere: a sum of waves along fixed directions, so it is
    smooth, the same for every visitor, and defined everywhere without being stored anywhere."""

    height = 0.0
    for (wavelength, amplitude), direction in zip(OCTAVES, _octave_directions(len(point))):
        height += amplitude * math.sin(_dot(point, direction) * radius_m * 2.0 * math.pi / wavelength)
    return height


def patch(walker: Walker, *, solid_m: float = SOLID_M, view_m: float = VIEW_M, grid: int = GRID,
          headroom_m: float = 20.0, maps: tuple = (), meet_m: float | None = None) -> dict:
    """The local flat area around the walker, as the standard scene packet the studio mesh environment admits.

    ``maps`` is a sequence of (identity, position) the walker might run into; those within ``meet_m`` (default the
    view distance) of the walker, measured on the sphere, are listed under ``encounters`` with their cosine and their
    local (x, z) direction. Everything else stays where it is: somewhere on the sphere, unmet.
    """

    if not solid_m > 0 or not view_m >= solid_m or not isinstance(grid, int) or grid < 1:
        raise ValueError("the solid ground is a positive square inside the view")
    base = ground(walker.point, walker.radius_m)
    step = 2.0 * solid_m / grid
    vertices = []
    for j in range(grid + 1):
        for i in range(grid + 1):
            x, z = -solid_m + i * step, -solid_m + j * step
            vertices.append((x, ground(place(walker, x, z), walker.radius_m) - base, z))
    triangles, sources = [], []
    for j in range(grid):
        for i in range(grid):
            a = j * (grid + 1) + i
            b, c, d = a + 1, a + grid + 1, a + grid + 2
            # wound so each face's normal points up (+y): a -> c -> b and b -> c -> d with x east, z north
            triangles += [(a, c, b), (b, c, d)]
            sources += [f"ground:{i}:{j}:0", f"ground:{i}:{j}:1"]
    heights = [v[1] for v in vertices]
    meet = view_m if meet_m is None else meet_m
    encounters = []
    for identity, where in maps:
        distance = arc(walker.point, where) * walker.radius_m      # the same closeness as cosine, measured exactly
        if distance <= meet and identity != walker.map_identity:
            towards = (_dot(where, walker.east), _dot(where, walker.north))
            encounters.append({"identity": identity, "cosine": cosine(walker.point, where),
                               "distance_m": distance, "direction": list(towards)})
    return {
        "schema": "turing.studio-rigid-scene.v1",
        "contact_model": "native-local-features-v1",
        "room": {"low": [-solid_m, min(heights) - 2.0, -solid_m], "high": [solid_m, max(heights) + headroom_m, solid_m]},
        "parts": [{
            "id": "wilderness-ground", "bodyId": "wilderness-ground", "closed": False,
            "vertices": vertices, "triangles": triangles, "sourceTriangleIds": sources,
            "materialIds": [TERRAIN_MATERIAL] * len(triangles),
            "sourceProvenance": {"ground": "abstract wave octaves on the wilderness sphere", "octaves": [list(o) for o in OCTAVES]},
        }],
        "wilderness": {
            "map": walker.map_identity, "point": list(walker.point), "radius_m": walker.radius_m,
            "solid_m": solid_m, "view_m": view_m, "ground_origin_m": base,
            "view": "implied: ground(place(walker, x, z)) for |x|, |z| <= view_m; nothing past solid_m is meshed",
            "encounters": encounters,
        },
    }


__all__ = ["arc", "GRACE_S", "RADIUS_M", "SOLID_M", "VIEW_M", "Walker", "ground", "leave_map", "patch", "place", "walk"]
