"""GLYPH START (WORKLIST 4b): a text becomes a closed box of grains that already spell it, and the 3-D fountain law evolves from there.

Host side, deterministic, and apart from ``fountain3d.py`` on purpose (an edit there re-keys the compiled products): text -> glyph
outlines -> a packing of the N grains in the glyphs' enclosed volume -> the law's initial columns and its closed-box chamber.

OUTLINES.  Each token's contours are the ones the exported glyph meshes were extruded from: ``phong_tiles.token_mesh`` builds
``scene_orders._font_contours`` -> ``_normalize_contours`` -> ``_sanitize_scaled_contours`` and extrudes them by ``depth_ratio`` (the
export's per-token font, weight, ratio and outline subdivisions, ``style_of``); emoji and symbols keep only their largest component
(``web_export.largest_component``: here the contour group with the largest outer area).  "Inside" is the builder's own rule,
``scene_orders._points_inside_even_odd`` (holes count), and ``|z| < depth / 2`` (the mesh is centred in z).  A string is laid out left
to right with a gap; every token is normalised to the same height, as the export does.

SIZE.  The string is scaled so that N grains of the species mix fill the glyph volume V at the packing fraction phi:
``N (4/3) pi <r^3> = phi V`` (``phi`` 0.64 for the random start, 0.74 for ``crystal``), the species radii as the page set them.  The
grains then sit on a lattice (close-packed layers parallel to the glass, ABC stacking; a seeded jitter for the random start) whose
spacing ``d`` is the largest that has at least N sites inside the glyphs; the first N sites in (x, y, z) order are the ids (so hue by
id reads left to right).  The species radii are scaled by ``k = min(1, (1 - GAP) d / (2 r_max))`` so that no two grains overlap, mix
kept; the achieved phi is reported.  Every number is a pure function of (text, N, mix, params, crystal).

THE CHAMBER is a closed box of the law's own SDF (one FLUID ``box`` slot, so all six walls) around the string's bounding box plus a
margin of ``MARGIN_CELLS`` grid cells; the glass planes lie ``Z_MARGIN_CELLS`` cells behind it.  The run starts from this state at
rest and emits nothing: ``emit_rate`` = 0, so every id's emit time is 0 and the law reads each as already live; ids past N stay parked.

HAND-OFF.  The law's per-id state columns ``x y z`` (and ``rs`` when a species swells) are the initial state.  The server sends them
as 16-bit fixed point over the box (``GlyphStart.payload``) in ``game.compute.start``; the page decodes them (the same formula,
``decode_payload``) and writes them through the product's existing named-column feed (``reset({named})``), the same hand-off the
image colours use.  Up to ``MAX_SHIPPED`` ids are shipped; the host side has no limit of its own.
"""
from __future__ import annotations

import base64
import hashlib
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meowdoku import fountain3d as F
from meowdoku.fountain3d import _f32

HERE = Path(__file__).resolve().parent
DEFAULT_TEXT = "MEOW"
MAX_CHARS = 16
MAX_SHIPPED = 65536                    # ids the server ships in the state (6 bytes each); larger sizes are refused with the reason
HEIGHT = 0.2                           # the exported token height (build_ink_on_slate_order's height_m)
GAP = 0.01                             # share of the lattice spacing kept between the largest grains
LETTER_GAP = 0.16                      # between neighbouring glyphs, in glyph heights
SPACE = 0.5                            # a space's advance, in glyph heights
PHI_RANDOM, PHI_CRYSTAL = 0.64, 0.74
MARGIN_CELLS = 2.0                     # box margin around the string's bounding box
QUANT = 65535.0
Z_INSET = 1.0e-4                       # layers keep this share of the depth clear of the faces (the fixed point moves a grain by ~1e-6)


# ------------------------------------------------------------------ outlines
def spectral_root() -> Path:
    """The sibling spectral-analyzer checkout (``SPECTRAL_ANALYZER_ROOT`` or the nearest ``spectral-analyzer`` above this file: a worktree has it
    several levels up)."""
    env = os.environ.get("SPECTRAL_ANALYZER_ROOT")
    for cand in ([Path(env)] if env else []) + [p / "spectral-analyzer" for p in HERE.parents]:
        if (cand / "scene_orders.py").is_file():
            return cand
    raise RuntimeError("spectral-analyzer is required beside turing (set SPECTRAL_ANALYZER_ROOT)")


def _modules():
    root = str(spectral_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    import scene_orders
    from camera_software import render_assets
    return scene_orders, render_assets


@dataclass(frozen=True)
class Style:
    family: str
    weight: str
    ratio: float                       # extrusion depth / glyph height (web_export's depth_ratio for the kind)
    subdivisions: int
    largest: bool                      # keep only the largest component (emoji)


def style_of(token: str) -> Style:
    """The export's font and extrusion for a token: letters and digits are DejaVu Sans bold at 0.28 (web_export's A-Z0-9), punctuation the same,
    the symbols block DejaVu Sans normal at 0.45 (its chess pieces), anything else Segoe UI Emoji at 0.3 with its largest component."""
    code = ord(token)
    if (token.isascii() and token.isalnum()) or code < 0x2000:
        return Style("DejaVu Sans", "bold", 0.28, 1, False)
    if 0x2600 <= code < 0x2800:
        return Style("DejaVu Sans", "normal", 0.45, 1, False)
    return Style("Segoe UI Emoji", "normal", 0.3, 1, True)


def _contains(point: np.ndarray, contour: np.ndarray) -> bool:
    scene_orders, _ = _modules()
    return bool(scene_orders._points_inside_even_odd(point.reshape(1, 2), [contour])[0])


def _area(contour: np.ndarray) -> float:
    x, y = contour[:-1, 0], contour[:-1, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def largest_component(contours: list[np.ndarray]) -> list[np.ndarray]:
    """The contour group with the largest outer area: an even-depth contour with the odd-depth contours it directly contains (its holes)."""
    count = len(contours)
    depth = [sum(_contains(contours[i][0], contours[j]) for j in range(count) if j != i) for i in range(count)]
    areas = [_area(c) for c in contours]
    groups: dict[int, list[int]] = {i: [i] for i in range(count) if depth[i] % 2 == 0}
    for i in range(count):
        if depth[i] % 2 == 1:
            parents = [j for j in range(count) if depth[j] == depth[i] - 1 and _contains(contours[i][0], contours[j])]
            if parents:
                groups[min(parents, key=lambda j: areas[j])].append(i)
    best = max(groups, key=lambda j: areas[j])
    return [contours[i] for i in groups[best]]


@dataclass(frozen=True)
class Glyph:
    token: str
    contours: tuple[np.ndarray, ...]   # closed (k, 2) polylines, the bounding box centred on the origin, height ``HEIGHT``
    depth: float                       # extrusion depth at that height
    width: float
    area: float                        # even-odd area at that height

    def inside(self, xy: np.ndarray) -> np.ndarray:
        scene_orders, _ = _modules()
        return scene_orders._points_inside_even_odd(xy, list(self.contours))


_GLYPHS: dict[str, Glyph] = {}


def glyph(token: str) -> Glyph:
    """The token's outline data, from the export's builder."""
    if token in _GLYPHS:
        return _GLYPHS[token]
    scene_orders, render_assets = _modules()
    style = style_of(token)
    font = render_assets.FontAssetSpec(family=style.family, weight=style.weight).scene_order_mapping()
    contours = scene_orders._normalize_contours(scene_orders._font_contours(token, font, style.subdivisions), HEIGHT)
    contours = scene_orders._sanitize_scaled_contours(contours, HEIGHT)
    if style.largest:
        contours = largest_component(contours)
    pts = np.concatenate([c[:-1] for c in contours])
    centre = 0.5 * (pts.min(0) + pts.max(0))
    contours = [np.ascontiguousarray(c - centre) for c in contours]
    pts = np.concatenate([c[:-1] for c in contours])
    count = len(contours)
    depth = [sum(_contains(contours[i][0], contours[j]) for j in range(count) if j != i) for i in range(count)]
    area = sum(_area(c) * (1.0 if depth[i] % 2 == 0 else -1.0) for i, c in enumerate(contours))
    _GLYPHS[token] = Glyph(token, tuple(contours), HEIGHT * style.ratio, float(pts[:, 0].max() - pts[:, 0].min()), float(area))
    return _GLYPHS[token]


def normalize_text(text: str | None) -> str:
    """The string the start spells: upper case (the export has capitals), at most ``MAX_CHARS`` characters, spaces kept inside; an empty text is the default."""
    out = " ".join(str(text or "").upper().split())[:MAX_CHARS].strip()
    return out or DEFAULT_TEXT


def layout(text: str) -> list[tuple[Glyph, float]]:
    """(glyph, x of its centre) at ``HEIGHT``, left to right with a gap, the string's bounding box centred on x = 0."""
    placed: list[tuple[Glyph, float]] = []
    cursor = 0.0
    for token in text:
        if token == " ":
            cursor += SPACE * HEIGHT
            continue
        g = glyph(token)
        placed.append((g, cursor + 0.5 * g.width))
        cursor += g.width + LETTER_GAP * HEIGHT
    if not placed:
        raise ValueError("a glyph start needs at least one visible character")
    lo = placed[0][1] - 0.5 * placed[0][0].width
    hi = placed[-1][1] + 0.5 * placed[-1][0].width
    mid = 0.5 * (lo + hi)
    return [(g, x - mid) for g, x in placed]


# ----------------------------------------------------------------- the grains
def grain_radii(n: int, u: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """(species index, radius) of ids 0..n-1 exactly as the law derives them (``fountain3d.make_law.species``: the same id hash, the species
    table ``r<s>``, irregularity, size distribution and bidisperse mode of the uniforms ``u``)."""
    from src.common.tensors import AbstractTensor
    pid = AbstractTensor.get_tensor(np.arange(n, dtype=np.float64))
    mix = lambda a, c: np.asarray(F._mix(pid, a, c).numpy(), dtype=np.float64)  # noqa: E731
    w = [float(u[f"wt{s}"]) for s in range(len(F.SPECIES))]
    total = sum(w) + 1.0e-9
    c0, c1, c2 = w[0] / total, (w[0] + w[1]) / total, (w[0] + w[1] + w[2]) / total
    pick = mix(1237.0, 5.0)
    sp = (pick >= c0).astype(int) + (pick >= c1).astype(int) + (pick >= c2).astype(int)
    u1, u2 = mix(2999.0, 61.0), mix(3511.0, 43.0)
    z = np.clip((u1 + u2 - 1.0) * 2.449489742783178, -3.0, 3.0)
    spread = np.array([float(u[f"irr{s}"]) for s in range(len(F.SPECIES))])[sp]
    ln = float(u["lognormal"])
    size = (1.0 - ln) * (1.0 + spread * (u1 * 2.0 - 1.0)) + ln * np.exp(spread * z - 0.5 * spread * spread)
    big = (mix(3733.0, 71.0) < float(u["bi_frac"])).astype(float)
    sq = math.sqrt(float(u["bi_ratio"]))
    bi = 1.0 + big * (sq - 1.0) + (1.0 - big) * (1.0 / sq - 1.0)
    return sp, np.array([float(u[f"r{s}"]) for s in range(len(F.SPECIES))])[sp] * size * bi


# ------------------------------------------------------------------- lattice
def _lattice_xy(d: float, lo: np.ndarray, hi: np.ndarray, shift: int) -> np.ndarray:
    """In-plane close-packed (hexagonal) points of spacing ``d`` inside the box [lo, hi], layer shift ``shift`` (0, 1, 2: A, B, C)."""
    row = d * math.sqrt(3.0) / 2.0
    sx, sy = shift * 0.5 * d, shift * d * math.sqrt(3.0) / 6.0
    j = np.arange(int(math.floor((lo[1] - sy) / row)) - 1, int(math.ceil((hi[1] - sy) / row)) + 2)
    i = np.arange(int(math.floor((lo[0] - sx) / d)) - 2, int(math.ceil((hi[0] - sx) / d)) + 2)
    jj, ii = np.meshgrid(j, i, indexing="ij")
    x = sx + ii * d + (jj % 2) * 0.5 * d
    y = sy + jj * row
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    keep = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
    return pts[keep]


def _layers(d: float, depth: float) -> list[tuple[int, float]]:
    """(shift, z) of the layers with |z| < depth / 2: layer spacing d sqrt(2/3), layer 0 at z = 0, ABC stacking."""
    dz = d * math.sqrt(2.0 / 3.0)
    half = 0.5 * depth * (1.0 - Z_INSET)
    top = int(math.ceil(half / dz)) + 1
    return [(k % 3, k * dz) for k in range(-top, top + 1) if abs(k * dz) < half]


class _Scaled:
    """The laid-out string at a scale: per glyph the scaled contours' inside test and extent."""

    def __init__(self, placed: list[tuple[Glyph, float]], scale: float, yc: float):
        self.placed, self.scale, self.yc = placed, scale, yc
        self.cache: dict[tuple[int, int, float], np.ndarray] = {}

    def box(self, g: Glyph, x: float) -> tuple[np.ndarray, np.ndarray]:
        h = 0.5 * self.scale * np.array([g.width, HEIGHT])
        c = np.array([self.scale * x, self.yc])
        return c - h, c + h

    def inside_xy(self, index: int, xy: np.ndarray) -> np.ndarray:
        g, x = self.placed[index]
        local = (xy - np.array([self.scale * x, self.yc])) / self.scale
        return g.inside(local)

    def sites_2d(self, index: int, d: float, shift: int) -> np.ndarray:
        g, x = self.placed[index]
        lo, hi = self.box(g, x)
        pts = _lattice_xy(d, lo, hi, shift)
        return pts[self.inside_xy(index, pts)] if len(pts) else pts

    def count(self, d: float) -> int:
        total = 0
        for index, (g, _x) in enumerate(self.placed):
            layers = _layers(d, self.scale * g.depth)
            per = [len(self.sites_2d(index, d, s)) if any(l[0] == s for l in layers) else 0 for s in range(3)]
            total += sum(per[s] for s, _z in layers)
        return total

    def sites(self, d: float) -> np.ndarray:
        out = []
        for index, (g, _x) in enumerate(self.placed):
            flat = {s: self.sites_2d(index, d, s) for s in range(3)}
            for shift, z in _layers(d, self.scale * g.depth):
                xy = flat[shift]
                out.append(np.column_stack([xy, np.full(len(xy), z)]))
        pts = np.concatenate(out)
        order = np.lexsort((np.round(pts[:, 2], 9), np.round(pts[:, 1], 9), np.round(pts[:, 0], 9)))
        return pts[order]


def fit_spacing(scaled: _Scaled, n: int, d0: float) -> float:
    """The largest lattice spacing (found by bracketing and bisection) with at least ``n`` sites inside the glyphs."""
    d = d0 * 1.25
    if scaled.count(d) >= n:
        while scaled.count(d) >= n:
            d *= 1.05
        bad, ok = d, d / 1.05
    else:
        bad = d
        while True:
            d *= 0.97
            if scaled.count(d) >= n:
                break
        ok = d
        bad = d / 0.97
    for _ in range(24):
        mid = 0.5 * (ok + bad)
        if scaled.count(mid) >= n:
            ok = mid
        else:
            bad = mid
    return ok


# ------------------------------------------------------------------ the start
@dataclass
class GlyphStart:
    text: str
    n: int
    crystal: bool
    positions: np.ndarray              # (n, 3), the 16-bit fixed point values the page decodes, as float32 -> float64
    radii: np.ndarray                  # (n,) each grain's radius (the law derives the same from the id and the scaled species table)
    species: np.ndarray
    scale: float                       # glyph height H in the law's units
    spacing: float
    k: float                           # species radii scale (<= 1) that makes the lattice overlap-free
    phi: float                         # achieved packing fraction N (4/3) pi r^3 / V
    phi_target: float
    volume: float
    jitter: float
    sites: int                         # lattice sites inside the glyphs (>= n); the rest are vacancies
    box: dict[str, float]              # hx, hy, hz (half extents; y from 0), margin, cell, rc
    overrides: dict[str, float] = field(default_factory=dict)
    bounds: tuple[float, ...] = ()
    swell: bool = False
    _payload: dict | None = None

    # -- the law's uniforms: the page's run settings, then this chamber
    def uniforms(self, *, mode: int = 0, use_image: float = 0.0, mix: float = 0.5, strata: float = F.STRATA, params: dict | None = None,
                 gravity: tuple[float, float, float] = (0.0, -F.G, 0.0)) -> dict[str, np.ndarray]:
        base = F.uniforms(n=self.n, mode=mode, use_image=use_image, mix=mix, strata=strata, params=params, shape="box", inlet="pour", gravity=gravity)
        base["spread"] = _f32(np.float64(0.0))
        base.update({k: _f32(np.float64(v)) for k, v in self.overrides.items()})
        return base

    def columns(self, pad: int, **kwargs) -> dict[str, np.ndarray]:
        """Round zero for ``pad`` ids with the grains placed: state x y z (and rs), everything else the law's own."""
        cols = F.initial_columns(pad, mix=kwargs.get("mix", 0.5), mode=kwargs.get("mode", 0), use_image=kwargs.get("use_image", 0.0),
                                 active=self.n, strata=kwargs.get("strata", F.STRATA), params=kwargs.get("params"), shape="box", inlet="pour")
        cols.update(self.uniforms(**kwargs))
        for c, name in enumerate(("x", "y", "z")):
            cols[name][: self.n] = self.positions[:, c]
        if self.swell:
            cols["rs"][: self.n] = _f32(self.radii)
        return cols

    # -- the page's hand-off
    def payload(self) -> dict:
        if self._payload is None:
            self._payload = self._encode()
        return self._payload

    def _encode(self) -> dict:
        lo, hi = np.array(self.bounds[0::2]), np.array(self.bounds[1::2])
        q = np.rint((self.positions - lo) / (hi - lo) * QUANT).astype("<u2")
        out = {"n": self.n, "bounds": [float(b) for b in self.bounds], "xyz": base64.b64encode(q.tobytes()).decode("ascii")}
        if self.swell:
            rq = np.rint(self.radii / (1.001 * float(self.radii.max())) * QUANT).astype("<u2")
            out.update({"rmax": float(1.001 * float(self.radii.max())), "rs": base64.b64encode(rq.tobytes()).decode("ascii")})
        return out

    def view(self) -> tuple[float, float, float, float]:
        c = self.box["cell"]
        return (-self.box["hx"] - 2 * c, self.box["hx"] + 2 * c, -2 * c, 2 * self.box["hy"] + 2 * c)

    def chamber_mesh(self) -> dict:
        """The closed box as the glass mesh the page draws (``fountain_game.chamber_mesh`` format): 8 vertices, 12 triangles, normals out of the fluid."""
        hx, hy, hz = self.box["hx"], self.box["hy"], self.box["hz"]
        verts = np.array([(sx * hx, hy + sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        index = lambda sx, sy, sz: (sx > 0) * 4 + (sy > 0) * 2 + (sz > 0)  # noqa: E731
        tris = []
        for axis in range(3):
            for sign in (-1, 1):
                a, b = [i for i in range(3) if i != axis]
                quad = []
                for ua, ub in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    s = [0, 0, 0]
                    s[axis], s[a], s[b] = sign, ua, ub
                    quad.append(index(*s))
                t = [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
                for tri in t:                                   # wind so the normal points along +sign * axis (out of the box)
                    p = verts[list(tri)]
                    normal = np.cross(p[1] - p[0], p[2] - p[0])
                    tris.append(tri if normal[axis] * sign > 0 else tri[::-1])
        from meowdoku.s2_titles.fountain.fountain_game import _pack
        lo, hi = verts.min(0), verts.max(0)
        return _pack(verts, np.array(tris, dtype=np.int64), [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])

    def stats(self) -> dict[str, float]:
        return {"n": self.n, "height": self.scale, "spacing": self.spacing, "k": self.k, "phi": self.phi, "phi_target": self.phi_target,
                "sites": self.sites, "vacancies": self.sites - self.n, "jitter": self.jitter, "volume": self.volume}

    def inside_glyphs(self, points: np.ndarray) -> np.ndarray:
        """The even-odd + depth test of the glyph volumes at (m, 3) points (the placed grains' centres are all inside)."""
        placed = layout(self.text)
        s = self.scale / HEIGHT
        scaled = _Scaled(placed, s, self.box["margin"] + 0.5 * self.scale)
        ok = np.zeros(len(points), dtype=bool)
        for index, (g, _x) in enumerate(placed):
            ok |= scaled.inside_xy(index, points[:, :2]) & (np.abs(points[:, 2]) < 0.5 * s * g.depth)
        return ok


def decode_payload(payload: dict, pad: int) -> dict[str, np.ndarray]:
    """The page's decode (compute.js ``decodeStart``): float32 columns of length ``pad`` for x, y, z (and rs); zero past ``n``."""
    n = int(payload["n"])
    q = np.frombuffer(base64.b64decode(payload["xyz"]), dtype="<u2").reshape(n, 3).astype(np.float64)
    b = np.array(payload["bounds"])
    lo, hi = b[0::2], b[1::2]
    out = {}
    for c, name in enumerate(("x", "y", "z")):
        col = np.zeros(pad, dtype=np.float32)
        col[:n] = (lo[c] + q[:, c] / QUANT * (hi[c] - lo[c])).astype(np.float32)
        out[name] = col
    if "rs" in payload:
        col = np.zeros(pad, dtype=np.float32)
        col[:n] = (np.frombuffer(base64.b64decode(payload["rs"]), dtype="<u2").astype(np.float64) / QUANT * float(payload["rmax"])).astype(np.float32)
        out["rs"] = col
    return out


_BUILDS: dict[tuple, GlyphStart] = {}


def build(text: str | None, n: int, *, mix: float = 0.5, params: dict | None = None, crystal: bool = False) -> GlyphStart:
    """The glyph start of ``text`` for ``n`` ids: deterministic, cached."""
    text = normalize_text(text)
    p = F.resolve_params(mix, params)
    key = (text, int(n), bool(crystal), tuple(sorted(p.items())))
    if key in _BUILDS:
        return _BUILDS[key]
    u = {k: float(v) for k, v in F.uniforms(n=n, mix=mix, params=params, shape="box", inlet="pour").items()}
    species, rho = grain_radii(n, u)                              # unscaled radii of the ids
    phi = PHI_CRYSTAL if crystal else PHI_RANDOM
    placed = layout(text)
    # V(H) = H^3 v1: the extruded even-odd areas at unit height
    v1 = sum(g.area / HEIGHT ** 2 * g.depth / HEIGHT for g, _x in placed)
    mean_r3 = float(np.mean(rho ** 3))
    height = (n * (4.0 / 3.0) * math.pi * mean_r3 / (phi * v1)) ** (1.0 / 3.0)
    volume = v1 * height ** 3
    scaled = _Scaled(placed, height / HEIGHT, 0.0)
    d0 = (math.sqrt(2.0) * volume / n) ** (1.0 / 3.0)
    extra = int(math.ceil(0.002 * n)) + 3
    while True:
        spacing = fit_spacing(scaled, n + extra, d0)
        sites = scaled.sites(spacing)
        total = len(sites)
        k = min(1.0, (1.0 - GAP) * spacing / (2.0 * float(rho.max())))
        radii = k * rho
        room = spacing - 2.0 * float(radii.max())
        amp = 0.0 if crystal else 0.9 * max(0.0, room - 0.5 * GAP * spacing) / (2.0 * math.sqrt(3.0))
        # the string stands MARGIN above the floor; the box is the string's bounding box plus the margin
        cell = 2.0 * F.rmax_of({**p, **{f"r{s}": p[f"r{s}"] * k for s in range(len(F.SPECIES))}}, True)
        margin = MARGIN_CELLS * cell
        yc = margin + 0.5 * height
        tested = _Scaled(placed, height / HEIGHT, yc)
        pts = sites + np.array([0.0, yc, 0.0])
        if amp > 0.0:                                             # seeded jitter, kept only where the grain stays inside the glyphs
            seed = int.from_bytes(hashlib.sha256(f"{text}|{n}|{crystal}".encode()).digest()[:8], "little")
            moved = pts + np.random.default_rng(seed).uniform(-amp, amp, pts.shape)
            ok = np.zeros(len(pts), dtype=bool)
            for index, (g, _x) in enumerate(placed):
                ok |= tested.inside_xy(index, moved[:, :2]) & (np.abs(moved[:, 2]) < 0.5 * tested.scale * g.depth * (1.0 - Z_INSET))
            pts = np.where(ok[:, None], moved, pts)
        depth_half = max(0.5 * tested.scale * g.depth for g, _x in placed)
        half_w = max(max(abs(tested.box(g, x)[0][0]), abs(tested.box(g, x)[1][0])) for g, x in placed)
        hx, hy, hz = half_w + margin, 0.5 * height + margin, depth_half + margin
        # fixed point over the box: the positions the page decodes are the positions of the run
        bounds = (-hx, hx, 0.0, 2.0 * hy, -hz, hz)
        lo, hi = np.array(bounds[0::2]), np.array(bounds[1::2])
        pos = (lo + np.rint((pts - lo) / (hi - lo) * QUANT) / QUANT * (hi - lo)).astype(np.float32).astype(np.float64)
        keep = np.zeros(len(pos), dtype=bool)                      # a grain the fixed point carried across an outline is a vacancy instead
        for index, (g, _x) in enumerate(placed):
            keep |= tested.inside_xy(index, pos[:, :2]) & (np.abs(pos[:, 2]) < 0.5 * tested.scale * g.depth)
        pos = pos[keep]
        if len(pos) >= n:
            break
        extra *= 2
    if len(pos) > n:                                              # vacancies, evenly spaced through the ordered sites
        gone = len(pos) - n
        pos = np.delete(pos, np.minimum(np.floor((np.arange(gone) + 0.5) * (len(pos) / gone)).astype(int), len(pos) - 1), axis=0)
    rc = 0.5 * margin
    cols = int(math.ceil(2.0 * hx / cell))
    rows = int(math.ceil(2.0 * hy / cell)) + 1
    dh = hz + F.Z_MARGIN_CELLS * cell
    lays = max(1, int(math.ceil(2.0 * dh / cell - 1.0e-9)))
    over: dict[str, float] = {"half": hx, "cols": cols, "rows": rows, "lays": lays, "cell": cell, "dh": dh, "emit_rate": 0.0,
                              "nozzle_x": 0.0, "nozzle_y": 2.0 * hy - 2.0 * cell, "nozzle_z": 0.0, "bar_half": 0.0, "bar_z": 0.0, "spread": 0.0}
    slot = F._slot("box", "fluid", 0.0, hy, 0.0, 0.0, hx, hy, hz, rc)
    over.update(F.slot_uniforms([slot]))
    over.update({name: 0.0 for name in F.POLYGON_UNIFORMS})          # no polyline walls (every segment has zero length): the SDF box is the chamber
    over.update({f"r{s}": p[f"r{s}"] * k for s in range(len(F.SPECIES))})
    achieved = n * (4.0 / 3.0) * math.pi * float(np.mean(radii ** 3)) / volume
    start = GlyphStart(text=text, n=int(n), crystal=bool(crystal), positions=pos, radii=_f32(radii), species=species, scale=height, spacing=spacing,
                       k=k, phi=achieved, phi_target=phi, volume=volume, jitter=amp, sites=total,
                       box={"hx": hx, "hy": hy, "hz": hz, "margin": margin, "cell": cell, "rc": rc}, overrides=over, bounds=bounds,
                       swell=any(p[f"swell{s}"] > 0.0 for s in range(len(F.SPECIES))))
    if len(_BUILDS) >= 6:
        _BUILDS.pop(next(iter(_BUILDS)))
    _BUILDS[key] = start
    return start
