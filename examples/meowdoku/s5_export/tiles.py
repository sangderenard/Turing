"""Meowdoku tiles as ray-traced sprites from spectral-analyzer's glyph maker.

This module is deliberately outside the compiler: it is Python and GLSL
through the sibling ``spectral-analyzer`` repository, the way its own
``compile_base_material_webgl.py`` reaches into Turing.  Nothing here has to
lower; a web server can run it.

The glyph maker is spectral-analyzer's declarative scene-order pipeline
(``scene_orders.py``, ``camera_software/render_assets.py``): a token is a
font outline extruded into a glossy solid half-embedded in a matte slate,
photographed head-on by the lab's thick-lens camera under its ring light,
through the native BDPT renderer.  ``sprite_compositor`` then separates the
capture into a background-independent sprite: a premultiplied body layer, an
alpha from the authored outline, and a signed additive layer holding the
light the glossy body spills onto its neighbours.

Meowdoku needs three sprites, not one per region: the cat (``U+1F431`` from
Segoe UI Symbol, which has real outline geometry for it), the X mark
(``U+2715``) and the heart (``U+2665``).  Empty tiles are flat region
colours, the same palette the board shader derives from the region id, and
the compositor lays sprites over them.  So the board costs three renders.

Three steps, each its own call so a server can cache between them:

1. ``tile_order_package()``: the scene-order package (JSON data).
2. ``compile_tiles`` / ``render_tiles``: spectral-analyzer's CLI on it.
   Compile needs only Python; render needs the native kernels
   (``_spectral_kernels``), which are a C++ build.
3. ``load_sprites`` + ``compose_board``: sprites from the rendered linear
   frames, composed over the board state into one image.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve()
SPECTRAL_ROOT = HERE.parents[4] / "spectral-analyzer"

CAT = "\U0001F431"
X_MARK = "✕"
HEART = "♥"
ROLE_TOKENS = {"cat": CAT, "x_mark": X_MARK, "heart": HEART}
#: Token -> the font that has real outline geometry for it on this machine
#: (DejaVu Sans has no cat; Segoe UI Symbol draws it with 465 vertices).
TOKEN_FONTS = {CAT: ("Segoe UI Symbol", "normal"), X_MARK: ("DejaVu Sans", "bold"),
               HEART: ("DejaVu Sans", "bold")}
#: Glyph materials in the scene-order material vocabulary (linear albedo,
#: reflectivity, diffusion, absorption, roughness, metallic, ior).  The cat
#: is the board shader's cat colour, glossy; the mark is slate; the heart is
#: the accepted red ink.
TILE_MATERIALS = {
    "cat_orange": {"albedo_rgb": [0.96, 0.55, 0.16], "reflectivity": 0.45, "diffusion": 0.22,
                   "absorption": 0.30, "roughness": 0.08, "metallic": 0.05, "ior": 1.5},
    "mark_slate": {"albedo_rgb": [0.30, 0.30, 0.32], "reflectivity": 0.25, "diffusion": 0.55,
                   "absorption": 0.6, "roughness": 0.35, "metallic": 0.0},
    "heart_red": {"albedo_rgb": [0.78, 0.006, 0.004], "reflectivity": 0.5, "diffusion": 0.18,
                  "absorption": 0.31, "roughness": 0.065, "metallic": 0.08, "ior": 1.52},
}
ROLE_MATERIALS = {"cat": "cat_orange", "x_mark": "mark_slate", "heart": "heart_red"}


def spectral_modules():
    """Import spectral-analyzer's asset modules from the sibling checkout."""
    if not SPECTRAL_ROOT.is_dir():
        raise RuntimeError(f"spectral-analyzer is required beside turing at {SPECTRAL_ROOT}")
    if str(SPECTRAL_ROOT) not in sys.path:
        sys.path.insert(0, str(SPECTRAL_ROOT))
    import scene_orders
    from camera_software import render_assets, sprite_compositor
    return scene_orders, render_assets, sprite_compositor


def region_palette(n: int) -> np.ndarray:
    """Flat region colours: the board shader's formula, so a composed board
    and the shader-drawn board agree on what region k looks like."""
    share = np.arange(n, dtype=np.float64) / float(n)
    return np.stack([
        0.62 + 0.30 * np.sin(share * 6.2831853 + phase)
        for phase in (0.0, 2.0943951, 4.1887902)
    ], axis=-1)


def _hue_ring(count: int, saturation: float, value: float, hue_offset: float = 0.0) -> np.ndarray:
    """``count`` colours evenly spaced around the hue circle (HSV to RGB)."""
    hue = (np.arange(count) / float(count) + hue_offset) % 1.0
    h6 = hue * 6.0
    k = np.floor(h6)
    f = h6 - k
    p, q, s = value * (1 - saturation), value * (1 - saturation * f), value * (1 - saturation * (1 - f))
    v = np.full(count, value)
    table = np.stack([
        np.select([k == 0, k == 1, k == 2, k == 3, k == 4], [v, q, p, p, s], v),
        np.select([k == 0, k == 1, k == 2, k == 3, k == 4], [s, v, v, q, p], p),
        np.select([k == 0, k == 1, k == 2, k == 3, k == 4], [p, p, s, v, v], q),
    ], axis=-1)
    return table


#: Named palettes: a palette is a ring of hue slots; ``contrast_palette``
#: assigns slots to regions so that touching regions sit far apart on the
#: ring.  A custom palette is any (slots, 3) array of linear RGB.
PALETTES: dict[str, np.ndarray] = {
    "pastel": _hue_ring(12, 0.32, 0.92),
    "vivid": _hue_ring(12, 0.78, 0.95),
    "warm": _hue_ring(12, 0.45, 0.95, hue_offset=-0.08)[:7],
    "cool": _hue_ring(12, 0.40, 0.92, hue_offset=0.42)[:7],
    "ink": np.array([[0.93, 0.93, 0.90], [0.80, 0.80, 0.78], [0.66, 0.66, 0.64], [0.52, 0.52, 0.50],
                     [0.40, 0.40, 0.38]]),
}


#: Ring palettes: a hue circle with a saturation and value; ``ring_palette``
#: gives every region its own hue on it, relaxed so touching regions repel.
RINGS: dict[str, tuple[float, float]] = {
    "pastel": (0.32, 0.92), "vivid": (0.78, 0.95), "soft": (0.22, 0.97), "deep": (0.65, 0.78),
}


def _hsv_to_rgb(hue: np.ndarray, saturation: float, value: float) -> np.ndarray:
    h6 = (np.asarray(hue, np.float64) % 1.0) * 6.0
    k = np.floor(h6)
    f = h6 - k
    p, q, s = value * (1 - saturation), value * (1 - saturation * f), value * (1 - saturation * (1 - f))
    v = np.full_like(f, value)
    choose = [k == 0, k == 1, k == 2, k == 3, k == 4]
    return np.stack([np.select(choose, [v, q, p, p, s], v),
                     np.select(choose, [s, v, v, q, p], p),
                     np.select(choose, [p, p, s, v, v], q)], axis=-1)


def ring_hues(region_grid: np.ndarray, *, iterations: int = 400, step: float = 0.02,
              neighbour_weight: float = 1.0, crowd_weight: float = 0.08, seed: int = 0,
              backend: str = "numpy", device=None) -> np.ndarray:
    """One hue per region, relaxed on the colour wheel.

    ``k`` hues start evenly spaced around the ring (in a seeded random order
    so the layout is not the region index).  Each iteration every hue is
    pushed away from the hues of the regions it TOUCHES (circular
    repulsion, ``neighbour_weight``) and, weakly, from every other hue
    (``crowd_weight``), so touching regions end far apart in hue while all
    ``k`` hues stay distinct and spread; the whole set is free to drift
    around the ring.  AbstractTensor arithmetic over the (k, k) pair table,
    float masks, no boolean operators.  Returns hues in [0, 1)."""
    from src.common.tensors import AbstractTensor as AT

    grid = np.asarray(region_grid).astype(int)
    touches = region_adjacency(grid).astype(np.float64)
    k = touches.shape[0]
    order = np.random.default_rng(seed).permutation(k)
    start = np.empty(k); start[order] = np.arange(k) / float(k)
    with AT.use_backend(backend, device):
        hue = AT.tensor(start.reshape(k, 1))
        weight = AT.tensor(touches * neighbour_weight + (1.0 - np.eye(k)) * crowd_weight)
        for _ in range(int(iterations)):
            delta = hue - hue.reshape((1, k))                           # (k, k) circular difference
            delta = delta - (delta + 0.5).floor() if hasattr(delta, "floor") else delta - ((delta + 0.5) // 1.0)
            # Repulsion falls off with ring distance; sign pushes apart.
            distance = delta.abs() + 1e-6
            push = (delta / distance) * (0.5 - distance).maximum(0.0 * distance) * weight
            hue = hue + push.sum(dim=1).reshape((k, 1)) * (float(step) / float(k))
            hue = hue - hue.floor() if hasattr(hue, "floor") else hue - (hue // 1.0)
        return np.asarray(hue.numpy(), np.float64).reshape(k)


def ring_palette(region_grid: np.ndarray, ring: str | tuple[float, float] = "pastel", **options) -> np.ndarray:
    """(k, 3) linear RGB per region: ``ring_hues`` on the named ring."""
    saturation, value = RINGS[ring] if isinstance(ring, str) else ring
    return _hsv_to_rgb(ring_hues(region_grid, **options), saturation, value)


def region_adjacency(region_grid: np.ndarray) -> np.ndarray:
    """(k, k) boolean: regions sharing a 4-neighbour edge."""
    k = int(region_grid.max()) + 1
    touches = np.zeros((k, k), bool)
    a, b = region_grid[:, 1:], region_grid[:, :-1]
    touches[a[a != b], b[a != b]] = True
    a, b = region_grid[1:, :], region_grid[:-1, :]
    touches[a[a != b], b[a != b]] = True
    return touches | touches.T


def contrast_palette(region_grid: np.ndarray, palette: str | np.ndarray = "pastel", *,
                     min_separation: float = 1.0 / 3.0, seed: int = 0) -> np.ndarray:
    """Colours per region such that touching regions contrast in hue.

    The palette is a ring of slots.  Regions are coloured most-constrained
    first (DSATUR order: most distinct neighbouring slots, then most
    neighbours).  A slot is admissible when its ring distance to every
    already-coloured neighbour's slot is at least ``min_separation`` of the
    ring (a third by default, so neighbours differ by at least 120 degrees
    of hue when the ring is a hue circle); among admissible slots the
    least-used wins, with a seeded draw breaking ties, so the whole ring
    gets used instead of two opposite slots alternating.  When no slot is
    admissible the farthest one is taken.  Returns (k, 3) linear RGB
    indexed by region id."""
    ring = PALETTES[palette] if isinstance(palette, str) else np.asarray(palette, dtype=np.float64)
    slots = ring.shape[0]
    touches = region_adjacency(region_grid)
    k = touches.shape[0]
    slot_of = -np.ones(k, int)
    use = np.zeros(slots, int)
    half = slots / 2.0
    draw = np.random.default_rng(seed)
    for _ in range(k):
        uncoloured = np.flatnonzero(slot_of < 0)
        saturation = np.array([len({slot_of[j] for j in np.flatnonzero(touches[i]) if slot_of[j] >= 0}) for i in uncoloured])
        degree = touches[uncoloured].sum(1)
        i = int(uncoloured[np.lexsort((-degree, -saturation))[0]])
        neighbour_slots = np.array([slot_of[j] for j in np.flatnonzero(touches[i]) if slot_of[j] >= 0])
        if neighbour_slots.size:
            distance = np.abs(np.arange(slots)[:, None] - neighbour_slots[None, :])
            distance = np.minimum(distance, slots - distance).min(1) / half
        else:
            distance = np.ones(slots)
        admissible = distance >= min_separation * 2.0          # distance is a fraction of half the ring
        if admissible.any():
            score = np.where(admissible, -use + draw.uniform(0.0, 0.5, slots), -np.inf)
        else:
            score = distance
        slot_of[i] = int(np.argmax(score))
        use[slot_of[i]] += 1
    return ring[slot_of]


#: Per-role extrusion overrides.  The cat outline has 465 font vertices; at
#: the ink default (outline_subdivisions 2, profile_segments 10) it is 21,356
#: triangles and the native renderer's emissive-triangle submission fails
#: its allocation on this machine (measured 2026-11-05); the X and heart
#: render at the default.  A coarser profile keeps the cat under the X's
#: budget class.
#: All roles extrude deeper than the ink default (0.03 m on a 0.2 m glyph)
#: so the 3D pieces read as objects: 0.32 of the glyph height.
ROLE_GEOMETRY: dict[str, dict[str, Any]] = {
    "cat": {"outline_subdivisions": 1, "profile_segments": 2, "extrusion_depth_ratio": 0.32},
    "x_mark": {"extrusion_depth_ratio": 0.32},
    "heart": {"extrusion_depth_ratio": 0.32},
}

#: Scene-order runtime for the tiles: four times the renderer's default ray
#: budget, since a 128x128 tile at one million rays is a sparse frame
#: (34% of pixels struck; measured on the X mark).
TILE_RUNTIME: dict[str, Any] = {"total_rays": 4_000_000}


def tile_order_package(roles: tuple[str, ...] = ("cat", "x_mark", "heart")) -> dict[str, Any]:
    """The scene-order package for the tile sprites: one ink-on-slate job per
    role, each in the font that has its glyph and in its own material."""
    scene_orders, render_assets, _ = spectral_modules()
    package = None
    jobs = []
    for role in roles:
        token = ROLE_TOKENS[role]
        family, weight = TOKEN_FONTS[token]
        font = render_assets.FontAssetSpec(family=family, weight=weight)
        order = render_assets.build_ink_on_slate_order([token], font=font)
        if package is None:
            package = order
            package["defaults"]["materials"].update(TILE_MATERIALS)
        for job in order["jobs"]:
            job = dict(job)
            job["id"] = f"meowdoku_{role}"
            job["font"] = font.scene_order_mapping()
            geometry = {**job["geometry"], "material": ROLE_MATERIALS[role], **ROLE_GEOMETRY.get(role, {})}
            if "extrusion_depth_ratio" in geometry:
                geometry.pop("depth_m", None)                 # the two depth spellings are exclusive
            job["geometry"] = geometry
            jobs.append(job)
    assert package is not None
    package["jobs"] = jobs
    package["runtime"] = {**package.get("runtime", {}), **TILE_RUNTIME}
    # The ink defaults carry depth_m; every tile job states its depth as a
    # ratio of the glyph height, and the two spellings are exclusive.
    package["defaults"].get("geometry", {}).pop("depth_m", None)
    scene_orders.validate_order(package)
    return package


def write_tile_package(path: str | Path, **options) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tile_order_package(**options), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def _run_scene_orders(package_path: Path, out_dir: Path, mode: str, jobs: tuple[str, ...] = (),
                      timeout: float = 900.0) -> dict[str, Any]:
    # The renderer runs from spectral-analyzer's root (its imports are
    # relative), so the caller's paths are made absolute first.
    package_path, out_dir = Path(package_path).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(SPECTRAL_ROOT / "render_scene_order.py"), str(package_path),
               mode, "--out-dir", str(out_dir)]
    for job in jobs:
        command += ["--job", job]
    # The native lab's recursive sensor exposure takes exactly one mip step
    # per layer (``run_recursive_sensor_epoch`` refuses more; larger work is
    # more sensor epochs).  The scene-order entry reads the step count from
    # the environment with a default of 8, so it is pinned here.
    environment = dict(os.environ)
    environment.setdefault("SPECTRAL_SENSOR_STEPS_PER_LAYER", "1")
    completed = subprocess.run(command, cwd=str(SPECTRAL_ROOT), capture_output=True,
                               text=True, encoding="utf-8", timeout=timeout, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"render_scene_order {mode} failed:\n{completed.stderr[-2000:]}")
    manifests = {}
    for job_dir in sorted(Path(out_dir).iterdir()):
        for name in ("compile_manifest.json", "composition_manifest.json"):
            manifest = job_dir / name
            if manifest.exists():
                manifests[job_dir.name] = json.loads(manifest.read_text(encoding="utf-8"))
                break
    return manifests


def glyph_geometry(package: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each job's subject geometry from the scene order alone: the font
    outline resolved, extruded and triangulated by ``scene_orders``' own
    glyph builder against the job's embed plane.  This is what a compile
    proves about the authoring (font found, glyph has an outline, extrusion
    closed) without the lab scene, which needs the native kernels."""
    scene_orders, _, _ = spectral_modules()
    report = {}
    for job in scene_orders.resolved_jobs(package):
        planes = {str(plane["id"]): plane for plane in job["planes"]}
        plane = planes[str(job["geometry"].get("embed_plane", job["planes"][0]["id"]))]
        triangles = scene_orders._glyph_triangles(job, plane)
        report[str(job["id"])] = {
            "token": str(job["token"]), "font": dict(job.get("font", {})),
            "glyph_triangles": int(triangles.shape[0]),
            "bounds_min": triangles.reshape(-1, 3).min(axis=0).tolist(),
            "bounds_max": triangles.reshape(-1, 3).max(axis=0).tolist(),
            "material": str(job["geometry"].get("material")),
        }
    return report


def compile_tiles(package_path: str | Path, out_dir: str | Path, jobs: tuple[str, ...] = ()) -> dict[str, Any]:
    """Compile the tile orders into the lab scene (spectral-analyzer's
    ``--compile-only``).  That path imports the native kernels for the lab
    tracer scene, so without them this falls back to ``glyph_geometry``:
    the same glyph builder, without the retained lab geometry."""
    if native_kernels_available():
        return _run_scene_orders(Path(package_path), Path(out_dir), "--compile-only", jobs)
    package = json.loads(Path(package_path).read_text(encoding="utf-8"))
    report = glyph_geometry(package)
    if jobs:
        report = {name: item for name, item in report.items() if name in jobs}
    out = Path(out_dir)
    for name, item in report.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "glyph_geometry.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    return {name: {"compile": item, "lab_scene": "absent: native kernels not built"}
            for name, item in report.items()}


def native_kernels_available() -> bool:
    probe = subprocess.run([sys.executable, "-c", "import _spectral_kernels"],
                           cwd=str(SPECTRAL_ROOT), capture_output=True, text=True)
    return probe.returncode == 0


def render_tiles(package_path: str | Path, out_dir: str | Path, jobs: tuple[str, ...] = (),
                 passes: int = 1) -> dict[str, Any]:
    """Ray-trace the tiles through spectral-analyzer's native thick-lens
    BDPT.  Refuses by name when the native kernels are not built.

    ``passes`` > 1 renders each job that many times, into ``pass_00``,
    ``pass_01``, ... under ``out_dir``: the catalog's refinement passes, each
    an independent exposure of the job's ray budget, which ``load_sprites``
    averages.  One pass of a 128x128 tile at four million rays strikes about
    a third of the pixels (measured on the X mark); the average over passes
    is the converging estimate.
    """
    if not native_kernels_available():
        raise RuntimeError(
            "spectral-analyzer's native kernels (_spectral_kernels) are not built in "
            f"{SPECTRAL_ROOT}; the renderer needs its csrc C++ build (pybind11) before "
            "tiles can be ray-traced")
    if passes <= 1:
        return _run_scene_orders(Path(package_path), Path(out_dir), "--render", jobs)
    manifests = {}
    for index in range(int(passes)):
        manifests = _run_scene_orders(Path(package_path), Path(out_dir) / f"pass_{index:02d}", "--render", jobs)
    return manifests


# ---------------------------------------------------------------------------
# Sprites and the board compositor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileSprite:
    """A background-independent tile sprite: ``out = premultiplied +
    (1 - alpha) * destination + additive`` (sprite_compositor's contract)."""

    role: str
    premultiplied_rgb: np.ndarray   # (h, w, 3) linear
    alpha: np.ndarray               # (h, w)
    additive_rgb: np.ndarray        # (h, w, 3) linear, signed

    def resampled(self, size: int) -> "TileSprite":
        """Nearest resample of all three layers to ``size`` x ``size``."""
        h, w = self.alpha.shape
        rows = np.minimum((np.arange(size) * h / size).astype(int), h - 1)
        cols = np.minimum((np.arange(size) * w / size).astype(int), w - 1)
        return TileSprite(
            self.role, self.premultiplied_rgb[rows][:, cols], self.alpha[rows][:, cols],
            self.additive_rgb[rows][:, cols],
        )


def load_sprites(render_dir: str | Path, package: Mapping[str, Any]) -> dict[str, TileSprite]:
    """Sprites from rendered jobs: each job directory's linear ``.npy`` frame
    through ``extract_raytraced_sprite`` with the job's own asset recipe."""
    _, render_assets, sprite_compositor = spectral_modules()
    capture = render_assets.AtlasCaptureSpec()
    sprites = {}
    for job in package["jobs"]:
        job_dirs = [Path(render_dir) / str(job["id"])]
        job_dirs += sorted(Path(render_dir).glob(f"pass_*/{job['id']}"))
        # The renderer writes several ``.npy`` sidecars per frame (exposure
        # weight, priority, field captures, the running sum); the sprite
        # comes from the linear frames, averaged over every pass found.
        frames = sorted(p for job_dir in job_dirs for p in job_dir.glob("*_linear.npy")
                        if not p.name.endswith("_sum_linear.npy"))
        if not frames:
            continue
        linear = np.mean([np.load(frame)[..., :3] for frame in frames], axis=0)
        token = str(job["token"])
        font = render_assets.FontAssetSpec.from_mapping(job.get("font", package["defaults"]["font"]))
        asset = render_assets.ink_token_asset(token, font=font)
        sprite = sprite_compositor.extract_raytraced_sprite(linear, asset, capture)
        role = next(role for role, held in ROLE_TOKENS.items() if held == token)
        sprites[role] = TileSprite(role, sprite.premultiplied_rgb, sprite.alpha, sprite.additive_rgb)
    return sprites


def compose_board(cat: np.ndarray, x_mark: np.ndarray, region: np.ndarray, hearts: int,
                  sprites: Mapping[str, TileSprite], *, cell_px: int = 96, n: int | None = None,
                  grid_px: int = 2, max_hearts: int = 3,
                  palette: str | np.ndarray | None = None) -> np.ndarray:
    """One board image (linear RGB in [0, 1]) from the state columns: flat
    region tiles, the cat and X sprites composed over them, a grid, and a
    hearts row beneath."""
    cat = np.asarray(cat, dtype=np.float64).reshape(-1)
    n = int(n or round(len(cat) ** 0.5))
    x_mark = np.asarray(x_mark, dtype=np.float64).reshape(n, n)
    region = np.asarray(region, dtype=np.float64).reshape(n, n).astype(int)
    cat = cat.reshape(n, n)
    if palette is None:
        palette = region_palette(int(region.max()) + 1)      # the shader's own formula
    elif isinstance(palette, str) and palette.startswith("ring-"):
        palette = ring_palette(region, palette[5:])          # one relaxed hue per region
    else:
        palette = contrast_palette(region, palette)          # touching regions contrast in hue
    board = n * cell_px
    image = np.zeros((board + cell_px, board, 3), dtype=np.float64)
    scaled = {role: sprite.resampled(cell_px) for role, sprite in sprites.items()}

    def lay(destination, sprite):
        alpha = sprite.alpha[..., None]
        return sprite.premultiplied_rgb + (1.0 - alpha) * destination + sprite.additive_rgb

    for r in range(n):
        for c in range(n):
            y0, x0 = r * cell_px, c * cell_px
            tile = np.broadcast_to(palette[region[r, c]], (cell_px, cell_px, 3)).copy()
            if cat[r, c] > 0 and "cat" in scaled:
                tile = lay(tile, scaled["cat"])
            elif x_mark[r, c] > 0 and "x_mark" in scaled:
                tile = lay(tile, scaled["x_mark"])
            image[y0:y0 + cell_px, x0:x0 + cell_px] = tile
    for k in range(n + 1):                       # grid lines
        line = np.clip(k * cell_px - grid_px // 2, 0, board - grid_px)
        image[line:line + grid_px, :board] = 0.12
        image[:board, line:line + grid_px] = 0.12
    strip = image[board:, :]
    strip[...] = 0.09
    if "heart" in scaled:
        for k in range(max_hearts):
            x0 = k * cell_px
            if x0 + cell_px > board:
                break
            dest = strip[:, x0:x0 + cell_px]
            strip[:, x0:x0 + cell_px] = lay(dest, scaled["heart"]) if k < hearts else dest * 0.5 + 0.045
    return np.clip(image, 0.0, 1.0)


__all__ = [
    "CAT", "HEART", "ROLE_TOKENS", "SPECTRAL_ROOT", "TileSprite", "X_MARK",
    "compile_tiles", "compose_board", "load_sprites", "native_kernels_available",
    "region_palette", "contrast_palette", "ring_palette", "ring_hues", "region_adjacency", "PALETTES", "RINGS", "render_tiles", "tile_order_package", "write_tile_package",
]
