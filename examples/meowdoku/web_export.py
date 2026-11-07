"""Export everything the WebGL Meowdoku page needs, prebaked.

    python examples/meowdoku/web_export.py --boards 16

Writes ``build/meowdoku/web/``:

- ``shaders/base_material.webgl.{vert,frag}.glsl``: Pluck's base-material
  Phong shader as Turing's ``glsl_project_ir`` compiled it for WebGL 2
  (copied from spectral-analyzer's generated shaders);
- ``assets.json``: the three tile models (the glyph extrusions, flattened
  positions and face normals, material index per role) and the material
  storage tables (``pbr``, ``phong``, ``enamel``, ``texstack``) exactly as
  ``material_db.MaterialDatabase.build_tensors`` lays them out, which the
  compiled shader reads back through ``turing_ssbo_load_*``;
- ``boards.json``: the prebaked board database: per size and style a set
  of region maps, each with the leaf the game scores against (the sampled
  walk's for free boards, the constructed cat map for relaxed ones), its
  ring-relaxed hues, and its taxonomy record and level key.

The page pulls these; nothing is generated at play time.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import tiles, tree  # noqa: E402
from meowdoku.phong_tiles import PHONG_MATERIALS, LIGHTS, _spectral, glyph_mesh  # noqa: E402
from meowdoku.phong_tiles import token_mesh as _token_mesh  # noqa: E402
from meowdoku.puzzle import generate_batch  # noqa: E402
from meowdoku.tiles import SPECTRAL_ROOT  # noqa: E402

WEB_DIR = ROOT / "build" / "meowdoku" / "web"
SHADERS = ("base_material.webgl.vert.glsl", "base_material.webgl.frag.glsl")


def export_shadow_fragment(target: Path) -> None:
    """The same fragment shader with its own shadow-map path switched on
    (``#define PLUCK_SHADOW_MAP``, the opt-in it documents for WebGL
    adapters), compiled through Turing's GLSL project IR like the tracked
    generated shader.  The page renders a depth pre-pass from the key light
    into ``uShadowMaps`` and hands ``uShadowMatrices`` per draw."""
    from src.compiler.glsl_project_ir import compile_glsl_project_to_webgl

    source = (SPECTRAL_ROOT / "csrc" / "shaders" / "base_material.frag.glsl").read_text(encoding="utf-8")
    source = source.replace("#version 430 core", "#version 430 core\n#define PLUCK_SHADOW_MAP 1", 1)
    translated = compile_glsl_project_to_webgl(source, source_name="base_material_fragment_shadow", stage_hint="fragment")
    if not translated.complete:
        report = "\n".join(item.format() for item in translated.diagnostics)
        raise RuntimeError(f"shadow fragment translation is incomplete:\n{report}")
    target.write_text(translated.source, encoding="utf-8", newline="\n")


def largest_component(verts: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the largest edge-connected part of a triangle mesh, re-centred in
    its own bounding box (a glyph's detached decoration is dropped, and the
    piece is sized and centred by its body, not by body plus decoration)."""
    keys = np.round(verts.reshape(-1, 3), 4)
    _, vertex_ids = np.unique(keys, axis=0, return_inverse=True)
    vertex_ids = vertex_ids.reshape(-1, 3)
    parent = list(range(int(vertex_ids.max()) + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in vertex_ids:
        for u, w in ((a, b), (b, c)):
            ru, rw = find(int(u)), find(int(w))
            if ru != rw:
                parent[ru] = rw
    roots = np.array([find(int(a)) for a in vertex_ids[:, 0]])
    values, counts = np.unique(roots, return_counts=True)
    keep = roots == values[np.argmax(counts)]
    body = verts[keep]
    lo, hi = body.reshape(-1, 3).min(0), body.reshape(-1, 3).max(0)
    body = body - (lo + hi)[None, None, :] / 2.0
    return np.ascontiguousarray(body, np.float32), np.ascontiguousarray(normals[keep], np.float32)


#: SIMPLIFY FILTER for every exported glyph: the glyph builder's interior
#: grid for the flat front and back caps (its default is 28).  A flat cap
#: needs no interior points -- the outline carries the shape and the page's
#: rounding moves only crease vertices -- so this cuts letters ~13x, chess
#: pieces ~5x and emoji ~2.3x with the same silhouettes.
CAP_GRID = 8


def token_mesh(*args, **kwargs):
    """``phong_tiles.token_mesh`` under the export's simplify filter."""
    return _token_mesh(*args, cap_grid=CAP_GRID, **kwargs)


# ---------------------------------------------------------------- terrain
#: The scrolling ground: a tileable fractal height map raised into real relief
#: by Pluck's DepthMesh (spectral-analyzer/depth_mesh.py, the "rect" shape:
#: every vertex pushed along the base normal by depth_map x depth_scale, smooth
#: normals recomputed), and coloured from the same map.  The page slides the
#: mesh by the scroll, wrapping at one tile, so mountains pass in perspective
#: under the Phong lights.
TERRAIN_TILE = 12.0          # board units per height-map period
TERRAIN_SPAN = (16.0, 30.0)  # mesh extent (x, z): the widest field plus margins, the field's depth plus a tile
TERRAIN_STEP = 0.2           # grid spacing
TERRAIN_SHADOW_STEP = 0.6    # the decimated triangulation drawn into the shadow map
TERRAIN_RELIEF = 0.75        # tallest peak above sea level (board units)
TERRAIN_SEA = 0.42           # height-map value of sea level (below: flat water)
TERRAIN_TEXTURE = 256


def terrain_heights(n: int, seed: int = 7) -> np.ndarray:
    """A tileable n x n fractal value-noise height map in [0, 1]."""
    rng = np.random.default_rng(seed)
    height = np.zeros((n, n))
    total, amp, period = 0.0, 1.0, 4
    ys, xs = np.mgrid[0:n, 0:n] / n
    for _ in range(6):
        lattice = rng.random((period, period))
        fx, fy = xs * period, ys * period
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - x0, fy - y0
        sx, sy = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
        at = lambda a, b: lattice[b % period, a % period]  # noqa: E731
        top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * sx
        bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * sx
        height += amp * (top + (bottom - top) * sy)
        total += amp
        amp *= 0.5
        period *= 2
    return height / total


def terrain_colours(height: np.ndarray) -> np.ndarray:
    """RGBA bytes: albedo by height (water, sand, grass, rock, snow), lightly
    shaded by slope so the relief also reads from straight above."""
    ramp = [(0.00, (0.10, 0.22, 0.42)), (0.40, (0.20, 0.42, 0.62)), (0.44, (0.78, 0.72, 0.48)),
            (0.50, (0.32, 0.58, 0.26)), (0.62, (0.22, 0.44, 0.20)), (0.72, (0.46, 0.40, 0.34)),
            (0.84, (0.92, 0.93, 0.96)), (1.00, (0.97, 0.98, 1.00))]
    stops = np.array([s for s, _ in ramp])
    colours = np.array([c for _, c in ramp])
    v = height.clip(0.0, 1.0)
    rgb = np.stack([np.interp(v, stops, colours[:, k]) for k in range(3)], axis=-1)
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 18.0
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 18.0
    light = np.array([-0.55, 0.75, -0.35]) / np.linalg.norm([-0.55, 0.75, -0.35])
    normal = np.stack([-dx, np.ones_like(dx), -dy], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    shade = np.where(v < TERRAIN_SEA, 0.95, (0.55 + 0.45 * (normal @ light)).clip(0.45, 1.1))
    rgba = np.concatenate([(rgb * shade[..., None]).clip(0, 1) * 255, np.full(v.shape + (1,), 255.0)], axis=-1)
    return rgba.astype(np.uint8)


def export_terrain() -> dict:
    import base64
    if str(SPECTRAL_ROOT) not in sys.path:
        sys.path.insert(0, str(SPECTRAL_ROOT))
    from depth_mesh import DepthMesh
    n = TERRAIN_TEXTURE
    height = terrain_heights(n)
    width, depth = TERRAIN_SPAN
    relief = ((height - TERRAIN_SEA) / (1.0 - TERRAIN_SEA)).clip(0.0, 1.0) ** 1.4   # flat water, steeper peaks

    def relief_mesh(step: float) -> tuple[np.ndarray, np.ndarray]:
        """DepthMesh over the field at ``step`` spacing, the depth map read
        from the tileable map at every grid vertex; sea level on the board."""
        grid_u, grid_v = int(round(width / step)), int(round(depth / step))
        us = (np.arange(grid_u + 1) * step / TERRAIN_TILE * n).astype(int) % n
        vs = (np.arange(grid_v + 1) * step / TERRAIN_TILE * n).astype(int) % n
        mesh = DepthMesh("rect", size=(width, depth), grid_u=grid_u, grid_v=grid_v, depth_map=relief[np.ix_(vs, us)],
                         depth_scale=TERRAIN_RELIEF, origin="bottom_left", axis_u=(1.0, 0.0, 0.0),
                         axis_v=(0.0, 0.0, 1.0), flip_normals=True, smooth_normals=True)
        data = np.asarray(mesh.vertex_data, dtype=np.float64)      # (N, 6): x y z nx ny nz
        pos, nrm = data[:, :3].copy(), data[:, 3:6].copy()
        if nrm[:, 1].mean() < 0.0:                                 # point the surface up
            nrm = -nrm
        pos[:, 1] -= pos[:, 1].min()
        return pos, nrm

    positions, normals = relief_mesh(TERRAIN_STEP)
    shadow_positions, _ = relief_mesh(TERRAIN_SHADOW_STEP)
    uvs = np.stack([positions[:, 0] / TERRAIN_TILE, positions[:, 2] / TERRAIN_TILE], axis=1)
    return {"positions": np.round(positions.reshape(-1), 4).tolist(),
            "normals": np.round(normals.reshape(-1), 4).tolist(),
            "uvs": np.round(uvs.reshape(-1), 5).tolist(),
            "triangles": int(len(positions) // 3), "tile": TERRAIN_TILE, "span": [width, depth],
            # the grids and the height law the mesh was read with, so a page can
            # re-read the same map at a turned coordinate (a spherical game)
            "step": TERRAIN_STEP, "shadow_step": TERRAIN_SHADOW_STEP, "relief": TERRAIN_RELIEF, "relief_power": 1.4,
            # the decimated triangulation of the same relief, for the shadow map
            "shadow_positions": np.round(shadow_positions.reshape(-1), 3).tolist(),
            "texture": {"size": n, "rgba": base64.b64encode(terrain_colours(height).tobytes()).decode("ascii")},
            # The same height map as the Phong shader's DEPTH texture (uDepthUv), in its
            # channel layout: R depth (0..1), G thickness, B translucence mask, A scatter mask.
            "depth": {"size": n, "relief_mm": TERRAIN_RELIEF * 1000.0,
                      "rgba": base64.b64encode(terrain_depth_rgba(height).tobytes()).decode("ascii")}}


def terrain_depth_rgba(height: np.ndarray) -> np.ndarray:
    relief = ((height - TERRAIN_SEA) / (1.0 - TERRAIN_SEA)).clip(0.0, 1.0)
    rgba = np.stack([relief, relief, np.ones_like(relief), np.zeros_like(relief)], axis=-1)
    return (rgba * 255.0).round().astype(np.uint8)


def export_assets(out: Path) -> dict:
    _, _, MaterialDatabase = _spectral()
    db = MaterialDatabase()
    material_index = {role: db.register(f"meowdoku_{role}", PHONG_MATERIALS[role]) for role in PHONG_MATERIALS}
    tensors = db.build_tensors()
    tables = {
        "pbr": tensors["pbr"], "phong": tensors["phong_compat"],
        "enamel": tensors["enamel"], "texstack": tensors["texture_stack"],
    }
    meshes = {}

    def pack(name, verts, normals, material=0):
        meshes[name] = {
            "positions": np.round(verts.reshape(-1), 5).tolist(),
            "normals": np.round(np.repeat(normals, 3, axis=0).reshape(-1), 4).tolist(),
            "material": int(material),
            "triangles": int(verts.shape[0]),
            "extent": float(np.abs(verts[..., :2]).max()),
        }

    for role in PHONG_MATERIALS:
        verts, normals = glyph_mesh(role, cap_grid=CAP_GRID)
        pack(role, verts, normals, material_index[role])
    # The other games' pieces: letters A-Z, the six chess pieces (the filled
    # glyphs U+265A..U+265F), a disc for checkers; DejaVu Sans has all of
    # them with light outlines.
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+":
        pack(letter, *token_mesh(letter, "DejaVu Sans", "bold", depth_ratio=0.28, outline_subdivisions=1))
    # Wordle's rack keys: Enter (U+21B5) and Backspace (U+232B); without the glyph the key is a blank button that still works
    for name, ch in (("enter", "↵"), ("back", "⌫"), ("giveup", "✕")):
        try:
            pack(name, *token_mesh(ch, "DejaVu Sans", "bold", depth_ratio=0.28, outline_subdivisions=1))
        except Exception as error:                                   # noqa: BLE001 - a missing glyph is not an export failure
            print(f"[export] wordle key {name}: no glyph ({type(error).__name__})", flush=True)
    for name, code in (("king", 0x265A), ("queen", 0x265B), ("rook", 0x265C), ("bishop", 0x265D),
                       ("knight", 0x265E), ("pawn", 0x265F)):
        pack(name, *token_mesh(chr(code), "DejaVu Sans", "normal", depth_ratio=0.45, outline_subdivisions=1))
    # Meowdoku's mice: the mouse glyph (U+1F401) from Segoe UI Symbol, a solid
    # silhouette (Segoe UI Emoji's layers extrude to an outline and a speck);
    # it takes the X mark's material, and the page draws it in the X's place.
    # Without the font the page keeps the X mesh.
    try:
        pack("mouse", *token_mesh("🐁", "Segoe UI Symbol", "normal", depth_ratio=0.3, outline_subdivisions=2),
             material_index["x_mark"])
    except Exception as error:  # noqa: BLE001 - a missing font must not stop the export
        print(f"[export] no mouse mesh ({error}); the page keeps the X", flush=True)
    pack("disc", *token_mesh("●", "DejaVu Sans", "normal", depth_ratio=0.22, outline_subdivisions=1))
    # The free-board games' entities: emoji outlines (Segoe UI Emoji), named
    # by the game that declares them.
    # Meowdoku's lives: a goldfish (cracker).
    try:
        pack("fish", *token_mesh("\U0001F41F", "Segoe UI Emoji", "normal", depth_ratio=0.32,
                                 outline_subdivisions=2), material_index["heart"])
    except Exception as error:  # noqa: BLE001  (the page falls back to the heart)
        print(f"[export] fish mesh skipped: {error}", flush=True)
    # Meowdoku's hint: a light bulb dropped on the cell it points at, and in
    # the power-up tray.
    try:
        verts, normals = token_mesh("💡", "Segoe UI Emoji", "normal", depth_ratio=0.32, outline_subdivisions=2)
        pack("bulb", *largest_component(verts, normals))
    except Exception as error:  # noqa: BLE001
        print(f"[export] bulb mesh skipped: {error}", flush=True)
    from meowdoku.invaders import EMOJI, EMOJI_BODY_ONLY, EMOJI_SUBDIVISIONS
    for name, glyph in EMOJI.items():
        verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.3,
                                    outline_subdivisions=EMOJI_SUBDIVISIONS.get(name, 1))
        if name in EMOJI_BODY_ONLY:
            verts, normals = largest_component(verts, normals)
        pack(name, verts, normals)
    # Bullet hell's own glyphs (its clouds).
    from meowdoku.scroller import EMOJI as SCROLLER_EMOJI
    for name, glyph in SCROLLER_EMOJI.items():
        if name not in meshes:
            try:
                verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.45, outline_subdivisions=2)
                pack(name, *largest_component(verts, normals))
            except Exception as error:  # noqa: BLE001  (the page falls back to a disc)
                print(f"[export] {name} mesh skipped: {error}", flush=True)
    # The side scroller's own glyphs (the ones the shared table lacks).
    from meowdoku.sidescroller import EMOJI as SIDE_EMOJI, EMOJI_SUBDIVISIONS as SIDE_SUBDIVISIONS
    for name, glyph in SIDE_EMOJI.items():
        if name not in meshes:
            verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.3,
                                        outline_subdivisions=SIDE_SUBDIVISIONS.get(name, 1))
            pack(name, *largest_component(verts, normals))
    # Tile Quest's own glyphs (the tiles and kinds the shared tables lack).
    from meowdoku.tilequest import EMOJI as TILE_EMOJI, EMOJI_SUBDIVISIONS as TILE_SUBDIVISIONS
    for name, glyph in TILE_EMOJI.items():
        if name not in meshes:
            try:
                verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.3,
                                            outline_subdivisions=TILE_SUBDIVISIONS.get(name, 1))
                pack(name, *largest_component(verts, normals))
            except Exception as error:  # noqa: BLE001  (the page falls back to a disc)
                print(f"[export] {name} mesh skipped: {error}", flush=True)
    # Delve's own glyphs (its ladder, spikes, gem, ability orb and bat).
    from meowdoku.delve import EMOJI as DELVE_EMOJI, EMOJI_SUBDIVISIONS as DELVE_SUBDIVISIONS
    for name, glyph in DELVE_EMOJI.items():
        if name not in meshes:
            try:
                verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.3,
                                            outline_subdivisions=DELVE_SUBDIVISIONS.get(name, 1))
                pack(name, *largest_component(verts, normals))
            except Exception as error:  # noqa: BLE001  (the page falls back to a disc)
                print(f"[export] {name} mesh skipped: {error}", flush=True)
    # The character tile set of the platformer-style games (characters.py): the player's chosen sprite.
    from meowdoku.characters import EMOJI as HERO_EMOJI
    for name, glyph in HERO_EMOJI.items():
        if name not in meshes:
            try:
                verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.3, outline_subdivisions=1)
                pack(name, *largest_component(verts, normals))
            except Exception as error:  # noqa: BLE001  (the page falls back to a disc)
                print(f"[export] {name} mesh skipped: {error}", flush=True)
    # Boardwalk's own glyphs (the track's tiles and its racers).
    from meowdoku.boardwalk import EMOJI as BOARDWALK_EMOJI
    for name, glyph in BOARDWALK_EMOJI.items():
        if name not in meshes:
            try:
                verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.3, outline_subdivisions=1)
                pack(name, *largest_component(verts, normals))
            except Exception as error:  # noqa: BLE001  (the page falls back to a disc)
                print(f"[export] {name} mesh skipped: {error}", flush=True)
    # World Walk's own glyphs (the builder).
    from meowdoku.worldwalk import EMOJI as WALK_EMOJI, EMOJI_SUBDIVISIONS as WALK_SUBDIVISIONS
    for name, glyph in WALK_EMOJI.items():
        if name not in meshes:
            try:
                verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.3,
                                            outline_subdivisions=WALK_SUBDIVISIONS.get(name, 1))
                pack(name, *largest_component(verts, normals))
            except Exception as error:  # noqa: BLE001  (the page falls back to a disc)
                print(f"[export] {name} mesh skipped: {error}", flush=True)
    # Purrline's own glyphs (the tiles, power pieces and cursor).
    from meowdoku.purrline import EMOJI as PURR_EMOJI
    for name, glyph in PURR_EMOJI.items():
        if name not in meshes:
            try:
                verts, normals = token_mesh(glyph, "Segoe UI Emoji", "normal", depth_ratio=0.3, outline_subdivisions=1)
                pack(name, *largest_component(verts, normals))
            except Exception as error:  # noqa: BLE001  (the page falls back to a disc)
                print(f"[export] {name} mesh skipped: {error}", flush=True)
    # The particle field's emoji style (emoji_kinds below): one mesh per animal for the instanced variant, under its own names
    # ("emoji_<name>"; the games' hero_* meshes are sized and trimmed for a tile, and hero_owl keeps only a speck of the glyph),
    # and "emoji_<name>_lod": its flat front cap alone (a quarter of the triangles; seen from the front it is the same shape).
    for kind in emoji_kinds():
        try:
            verts, normals = token_mesh(kind["glyph"], "Segoe UI Emoji", "normal", depth_ratio=0.3, outline_subdivisions=1)
            body_verts, body_normals = largest_component(verts, normals)
            if np.abs(body_verts[..., :2]).max() >= 0.8 * np.abs(verts[..., :2]).max():      # the body is the glyph, not one speck of it
                verts, normals = body_verts, body_normals
            pack(kind["mesh"], verts, normals)
            front = normals[:, 2] > 0.99
            pack(kind["mesh"] + "_lod", verts[front], normals[front])
        except Exception as error:  # noqa: BLE001  (the page falls back to the image variant)
            print(f"[export] emoji mesh {kind['name']} skipped: {error}", flush=True)
    assets = {
        "meshes": meshes,
        "materials": {name: {"floats_per_record": int(table.shape[1]), "records": int(table.shape[0]),
                             "data": np.asarray(table, np.float32).reshape(-1).tolist()}
                      for name, table in tables.items()},
        "lights": [{"position": list(p), "color": list(c), "intensity": float(i)} for p, c, i in LIGHTS],
        "terrain": export_terrain(),
    }
    (out / "assets.json").write_text(json.dumps(assets), encoding="utf-8")
    return assets


# ---------------------------------------------------------------- emoji atlas
#: The particle field's "emoji" style (web/particle_styles.js): every grain an animal emoji.  The image variant samples one
#: atlas of the glyphs, baked here from the SAME Segoe UI Emoji source the entity meshes are extruded from (so it looks the
#: same on every device, not the device's own emoji font); the mesh variant draws the exported meshes named here.  The kinds
#: are the animal glyphs the games already declare (characters.py, sidescroller, delve, tilequest, invaders), in this order;
#: the page reads the order, the cells and the colours from emoji_atlas.json and holds no list of its own.
EMOJI_ATLAS_CELL = 128         # texels per cell edge
EMOJI_ATLAS_COLS = 4
EMOJI_ATLAS_INK = 0.875        # the glyph's longer side as a share of the cell (the rest is a margin that mip levels do not bleed across)
EMOJI_FONT = "C:/Windows/Fonts/seguiemj.ttf"


def emoji_kinds() -> list[dict]:
    """[{name, glyph, mesh, color}]: ``mesh`` is the entity mesh the glyph is exported under, ``color`` its display colour
    (the games' own palette entries: characters.PALETTE and the walker colours) for the mesh variant's native look."""
    from meowdoku.characters import CHARACTERS, PALETTE as HERO_PALETTE
    from meowdoku.sidescroller import EMOJI as SIDE_EMOJI
    from meowdoku.delve import EMOJI as DELVE_EMOJI
    from meowdoku.tilequest import EMOJI as TILE_EMOJI
    from meowdoku.invaders import EMOJI as INVADER_EMOJI
    kinds = [{"name": key, "glyph": glyph, "color": HERO_PALETTE[f"hero_{key}"]} for key, glyph in CHARACTERS.items()]
    kinds += [{"name": "snail", "glyph": SIDE_EMOJI["snail"], "color": [0.92, 0.62, 0.35]},
              {"name": "bat", "glyph": DELVE_EMOJI["bat"], "color": [0.55, 0.42, 0.7]},
              {"name": "wolf", "glyph": TILE_EMOJI["wolf"], "color": [0.62, 0.64, 0.7]},
              {"name": "fish", "glyph": "\U0001F41F", "color": [0.98, 0.62, 0.2]},
              {"name": "crab", "glyph": INVADER_EMOJI["crab"], "color": [0.9, 0.3, 0.22]}]
    for kind in kinds:
        kind["mesh"] = f"emoji_{kind['name']}"
    return kinds


def bake_emoji_atlas(kinds: list[dict], cell: int = EMOJI_ATLAS_CELL, cols: int = EMOJI_ATLAS_COLS,
                     ink: float = EMOJI_ATLAS_INK, font_path: str = EMOJI_FONT):
    """-> (RGBA uint8 (rows*cell, cols*cell, 4), meta).  Each glyph is drawn large in colour, cropped to its ink, scaled so
    its longer side is ``ink`` of the cell and centred by its bounding box (as ``largest_component`` centres the meshes).
    The colour of the ink is spread into the transparent texels around it so mip levels and linear filtering do not pull
    black into the edge."""
    from PIL import Image, ImageDraw, ImageFont
    rows = -(-len(kinds) // cols)
    atlas = np.zeros((rows * cell, cols * cell, 4), np.uint8)
    big = 4 * cell
    font = ImageFont.truetype(font_path, int(big * 0.85))
    for index, kind in enumerate(kinds):
        canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text((big // 2, big // 2), kind["glyph"], font=font, embedded_color=True, anchor="mm")
        box = canvas.getbbox()
        if box is None:
            raise RuntimeError(f"{kind['name']}: the font drew nothing for {kind['glyph']!r}")
        glyph = canvas.crop(box)
        scale = ink * cell / max(glyph.size)
        glyph = glyph.resize((max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale))), Image.LANCZOS)
        tile = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
        tile.paste(glyph, ((cell - glyph.width) // 2, (cell - glyph.height) // 2))
        pixels = np.asarray(tile, np.uint8).copy()
        # (only texels that are mostly ink keep their own colour: a faint edge texel's colour is rounding noise)
        covered = pixels[..., 3] >= 128
        solid = covered.copy()
        colour = pixels[..., :3].astype(np.float32)
        for _ in range(cell // 8):                               # spread the colour outward, one texel per pass
            if covered.all():
                break
            weight = covered.astype(np.float32)
            total = np.zeros_like(colour)
            count = np.zeros_like(weight)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                total += np.roll(colour * weight[..., None], (dy, dx), (0, 1))
                count += np.roll(weight, (dy, dx), (0, 1))
            fill = (~covered) & (count > 0)
            colour[fill] = total[fill] / count[fill][:, None]
            covered = covered | fill
        pixels[..., :3][~solid] = colour[~solid].round().astype(np.uint8)
        r, c = divmod(index, cols)
        atlas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = pixels
    meta = {"cell": cell, "cols": cols, "rows": rows, "ink": ink, "size": [cols * cell, rows * cell],
            "kinds": [{"name": k["name"], "mesh": k["mesh"], "color": k["color"], "glyph": k["glyph"]} for k in kinds]}
    return atlas, meta


def export_emoji_atlas(out: Path, meshes: dict) -> dict:
    """Writes emoji_atlas.png and emoji_atlas.json; a kind whose mesh was not exported keeps its cell and says ``mesh: null``."""
    from PIL import Image
    atlas, meta = bake_emoji_atlas(emoji_kinds())
    for kind in meta["kinds"]:
        if kind["mesh"] not in meshes:
            kind["mesh"] = None
    Image.fromarray(atlas, "RGBA").save(out / "emoji_atlas.png", optimize=True)
    (out / "emoji_atlas.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def export_boards(out: Path, sizes: tuple[int, ...], per_style: int, seed: int) -> dict:
    boards = []
    for n in sizes:
        for style in ("free", "relaxed"):
            started = time.perf_counter()
            puzzles = generate_batch(n, per_style, seed=seed + n, style=style)
            records = tree.catalog(puzzles, seed=seed, limit_leaves=256)
            leaves, decisions, _, solved = tree.sample_solutions(puzzles.region, seed=seed + 7)
            for index, record in enumerate(records):
                solution = leaves[index] if (style == "free" and solved[index]) else puzzles.solution[index]
                grid = puzzles.region[index].reshape(n, n).astype(int)
                hues = tiles.ring_hues(grid, seed=index)
                taxonomy = tree.classify(record)
                boards.append({
                    "id": f"{style}-{n}-{index}", "n": n, "style": style,
                    "region": grid.reshape(-1).tolist(),
                    "solution": np.asarray(solution, dtype=int).reshape(-1).tolist(),
                    "hues": np.round(hues, 4).tolist(),
                    "level": taxonomy["level"], "type": record["type"],
                    "decisions": int(record["decisions"]), "leaves": record.get("leaves"),
                    "walk_decisions": int(record["walk_decisions"]),
                    "largest_share": taxonomy["largest_share"], "singletons": taxonomy["singletons"],
                })
            print(f"[export] {style:7s} {n}x{n}: {per_style} boards in {time.perf_counter() - started:.1f}s", flush=True)
    levels: dict[str, int] = {}
    for board in boards:
        levels[board["level"]] = levels.get(board["level"], 0) + 1
    database = {"boards": boards, "levels": dict(sorted(levels.items()))}
    (out / "boards.json").write_text(json.dumps(database), encoding="utf-8")
    return database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", default="5,6,7,8,9,10")
    parser.add_argument("--boards", type=int, default=12, help="boards per size per style")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default=str(WEB_DIR))
    parser.add_argument("--skip-boards", action="store_true", help="keep the existing boards.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    (out / "shaders").mkdir(parents=True, exist_ok=True)
    for name in SHADERS:
        shutil.copyfile(SPECTRAL_ROOT / "csrc" / "shaders" / "generated" / name, out / "shaders" / name)
    export_shadow_fragment(out / "shaders" / "base_material.webgl.shadow.frag.glsl")
    for name in ("index.html", "app.js", "marble_mesh.js", "marble_studio.html", "marble_studio.js", "bodies.js", "pilot_timeline.js", "pilot_clock.js", "game_clock.js", "world_zoom.js", "world_curve.js", "compute.js", "recording.js", "replay_layers.js", "gravity.js", "particle_styles.js", "noise_filter.js", "share.js", "lifecycle.js", "wheel_blur.js", "tipjar.js", "qr_wall.js", "wordle_layout.js", "mobroad.js", "loading_panel.js", "loading_anim.js", "loading_record.js", "octet_codec.js", "access.js", "shader.html", "loading_params.js", "loading_view.html", "studio.html", "studio.js", "log.html", "log.js", "log_view.js", "local_log.js"):
        shutil.copyfile(HERE / "web" / name, out / name)
    assets = export_assets(out)
    atlas = export_emoji_atlas(out, assets["meshes"])
    print(f"[export] emoji atlas {atlas['size'][0]}x{atlas['size'][1]}: {[k['name'] for k in atlas['kinds']]}", flush=True)
    print("[export] assets:", {role: mesh["triangles"] for role, mesh in assets["meshes"].items()},
          {name: table["records"] for name, table in assets["materials"].items()}, flush=True)
    if args.skip_boards and (out / "boards.json").exists():
        print("[export] boards.json kept", flush=True)
    else:
        database = export_boards(out, tuple(int(s) for s in args.sizes.split(",")), args.boards, args.seed)
        print(f"[export] {len(database['boards'])} boards; levels: {database['levels']}", flush=True)
    print(f"[export] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
