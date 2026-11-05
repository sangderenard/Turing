"""Meowdoku tile sprites through spectral-analyzer's Phong renderer.

The tile models are the same glyph extrusions the scene orders describe
(``scene_orders._glyph_triangles``: the font outline extruded and capped).
Instead of the thick-lens path tracer they are drawn by Pluck's base
material Phong program (``base_gl_renderer.BaseGLRenderer`` with
``csrc/shaders/base_material.*.glsl``) into a hidden GL window, one
orthographic head-on frame per role, and read back as a sprite:
``premultiplied_rgb``, ``alpha`` (coverage) and an empty additive layer,
the same contract ``tiles.compose_board`` composes with.

    python examples/meowdoku/phong_tiles.py            # renders cat, X, heart
    -> build/meowdoku/tiles/phong/<role>.npz + .png

Seconds per sprite rather than minutes, deterministic, no convergence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import tiles  # noqa: E402
from meowdoku.tiles import SPECTRAL_ROOT, TileSprite  # noqa: E402

#: Phong materials per role (PBR dict as material_db.MaterialDatabase takes it).
PHONG_MATERIALS: dict[str, dict[str, Any]] = {
    "cat": {"albedo": [0.78, 0.34, 0.04], "roughness": 0.42, "metallic": 0.0},
    "x_mark": {"albedo": [0.10, 0.10, 0.12], "roughness": 0.30, "metallic": 0.10},
    "heart": {"albedo": [0.88, 0.10, 0.16], "roughness": 0.28, "metallic": 0.08},
}

#: View-space point lights: key up-left in front, fill right, rim below.
LIGHTS = (
    ((-0.35, 0.45, 0.9), (1.0, 0.97, 0.92), 1.4),
    ((0.55, 0.10, 0.8), (0.85, 0.9, 1.0), 0.7),
    ((0.0, -0.5, 0.6), (1.0, 1.0, 1.0), 0.35),
)

OUT_DIR = ROOT / "build" / "meowdoku" / "tiles" / "phong"


def _spectral():
    if str(SPECTRAL_ROOT) not in sys.path:
        sys.path.insert(0, str(SPECTRAL_ROOT))
    import scene_orders  # noqa: WPS433
    from base_gl_renderer import BaseGLRenderer  # noqa: WPS433
    from material_db import MaterialDatabase  # noqa: WPS433
    return scene_orders, BaseGLRenderer, MaterialDatabase


def glyph_mesh(role: str) -> tuple[np.ndarray, np.ndarray]:
    """The role's glyph as triangles in the embed plane's own basis
    (x right, y up, z toward the viewer), centred, with per-face normals.
    Returns ``(verts (T, 3, 3), normals (T, 3))``."""
    scene_orders, _, _ = _spectral()
    package = tiles.tile_order_package(roles=(role,))
    job = scene_orders.resolved_jobs(package)[0]
    planes = {str(plane["id"]): plane for plane in job["planes"]}
    plane = planes[str(job["geometry"].get("embed_plane", job["planes"][0]["id"]))]
    world = scene_orders._glyph_triangles(job, plane)                   # (T, 3, 3) world
    centre, normal, right, up = scene_orders._plane_basis(plane)
    local = np.stack([(world - centre) @ right, (world - centre) @ up, (world - centre) @ normal], axis=-1)
    normals = scene_orders._triangle_normals(local)
    # Centre the glyph in its own bounding box so the frame is tight.
    lo, hi = local.reshape(-1, 3).min(0), local.reshape(-1, 3).max(0)
    local = local - (lo + hi)[None, None, :] / 2.0
    return np.ascontiguousarray(local, np.float32), np.ascontiguousarray(normals, np.float32)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], np.float32)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], np.float32)


def _ortho(half_w: float, half_h: float, near: float, far: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 / half_w
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def render_phong_sprites(roles: tuple[str, ...] = ("cat", "x_mark", "heart"), *, size: int = 128,
                         content: float = 80.0 / 128.0, out_dir: str | Path = OUT_DIR,
                         exposure: float | None = None, tilt: tuple[float, float] = (-16.0, 14.0)) -> dict[str, TileSprite]:
    """Draw every role's glyph mesh through the Phong program into a hidden
    GL window of ``size`` x ``size`` and read it back as a sprite.  The
    glyph's longer side fills ``content`` of the frame (the atlas capture
    contract: 80 of 128).  Writes ``<role>.npz`` and ``<role>.png``."""
    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from OpenGL.GL import (GL_ARRAY_BUFFER, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                           GL_DYNAMIC_DRAW, GL_FLOAT, GL_INT, GL_RGBA, glBindBuffer, glBindVertexArray,
                           glBufferData, glClear, glClearColor, glEnable, glEnableVertexAttribArray, glFinish,
                           glGenBuffers, glGenVertexArrays, glGetUniformLocation, glReadPixels, glUniform1f,
                           glUseProgram, glVertexAttribIPointer, glVertexAttribPointer, glViewport)
    import ctypes

    _, BaseGLRenderer, MaterialDatabase = _spectral()
    pygame.display.init()
    for attribute, value in ((pygame.GL_CONTEXT_MAJOR_VERSION, 4), (pygame.GL_CONTEXT_MINOR_VERSION, 3),
                             (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE),
                             (pygame.GL_DEPTH_SIZE, 24), (pygame.GL_ALPHA_SIZE, 8)):
        pygame.display.gl_set_attribute(attribute, value)
    flags = pygame.OPENGL | pygame.DOUBLEBUF | getattr(pygame, "HIDDEN", 0)
    pygame.display.set_mode((size, size), flags)

    db = MaterialDatabase()
    material_index = {role: db.register(f"meowdoku_{role}", PHONG_MATERIALS[role]) for role in PHONG_MATERIALS}
    renderer = BaseGLRenderer(db, auto_drain=False)
    renderer.init_gl()
    positions = np.asarray([light[0] for light in LIGHTS], np.float32)
    colours = np.asarray([light[1] for light in LIGHTS], np.float32)
    intensities = np.asarray([light[2] for light in LIGHTS], np.float32)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sprites: dict[str, TileSprite] = {}
    manifest: dict[str, Any] = {}
    for role in roles:
        verts, normals = glyph_mesh(role)
        count = verts.shape[0] * 3
        extent = float(np.abs(verts[..., :2]).max()) or 1.0
        half = extent / content                                   # longer side fills `content`
        depth_extent = float(np.abs(verts[..., 2]).max()) + 1e-3
        # A slight tilt (about x then y) so the caps and the extruded sides
        # both face the light: the head-on view reads as a flat cut-out.
        mv = _rotation_x(np.radians(tilt[0])) @ _rotation_y(np.radians(tilt[1]))
        half = half * 1.08                                        # room for the tilted silhouette
        mvp = _ortho(half, half, -depth_extent * 4.0 - half, depth_extent * 4.0 + half) @ mv
        # Lights scale with the glyph so the look is size-independent.
        renderer.set_point_lights(positions * half * 2.0, colours, intensities * (half * 2.0) ** 2)
        interleaved = np.zeros((count, 8), np.float32)
        interleaved[:, 0:3] = verts.reshape(-1, 3)
        interleaved[:, 3:6] = np.repeat(normals, 3, axis=0)
        mats = np.full(count, material_index[role], np.int32)
        groups = np.zeros(count, np.int32)
        culls = np.zeros(count, np.int32)
        vao = int(glGenVertexArrays(1))
        vbo, mbo, gbo, cbo = (int(glGenBuffers(1)) for _ in range(4))
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_DYNAMIC_DRAW)
        stride = 8 * 4
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3); glVertexAttribPointer(3, 2, GL_FLOAT, False, stride, ctypes.c_void_p(24))
        for location, buffer, data in ((2, mbo, mats), (4, gbo, groups), (5, cbo, culls)):
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(location); glVertexAttribIPointer(location, 1, GL_INT, 4, ctypes.c_void_p(0))
        glBindVertexArray(0)

        glViewport(0, 0, size, size)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        if exposure is not None:
            glUseProgram(renderer._prog)
            glUniform1f(glGetUniformLocation(renderer._prog, b"uExposure"), float(exposure))
        renderer.draw_mesh(vao, count, mvp.ravel(), mv.ravel(), enable_blend=False, depth_write=True)
        glFinish()
        pixels = glReadPixels(0, 0, size, size, GL_RGBA, GL_FLOAT)
        frame = np.asarray(pixels, np.float32).reshape(size, size, 4)[::-1]
        alpha = (frame[..., 3] > 0.0).astype(np.float64)
        rgb = np.clip(frame[..., :3].astype(np.float64), 0.0, 1.0)
        sprite = TileSprite(role, rgb * alpha[..., None], alpha, np.zeros_like(rgb))
        sprites[role] = sprite
        np.savez_compressed(out / f"{role}.npz", premultiplied_rgb=sprite.premultiplied_rgb,
                            alpha=sprite.alpha, additive_rgb=sprite.additive_rgb)
        from PIL import Image
        preview = np.concatenate([sprite.premultiplied_rgb ** (1 / 2.2), alpha[..., None]], axis=-1)
        Image.fromarray((np.clip(preview, 0, 1) * 255).astype(np.uint8), "RGBA").save(out / f"{role}.png")
        manifest[role] = {"triangles": int(verts.shape[0]), "coverage": float(alpha.mean()),
                          "material": PHONG_MATERIALS[role]}
        print(f"[phong] {role}: {verts.shape[0]} triangles, coverage {alpha.mean():.3f}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pygame.display.quit()
    return sprites


def load_phong_sprites(out_dir: str | Path = OUT_DIR) -> dict[str, TileSprite]:
    """Sprites rendered earlier by ``render_phong_sprites``."""
    sprites = {}
    for path in sorted(Path(out_dir).glob("*.npz")):
        data = np.load(path)
        sprites[path.stem] = TileSprite(path.stem, data["premultiplied_rgb"], data["alpha"], data["additive_rgb"])
    return sprites


if __name__ == "__main__":
    render_phong_sprites()
