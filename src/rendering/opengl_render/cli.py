# opengl_render/cli.py
# Demo that submits a single layers dict per frame:
#   {"membrane": MeshLayer, "lines": LineLayer, "fluid": PointLayer}
# This mirrors the sim coordinator output so you can verify points render.

import math
import numpy as np
import pygame
from pygame.locals import QUIT
from .renderer import GLRenderer, MeshLayer, LineLayer, PointLayer
from .api import draw_layers  # <- use the same upload/draw path as the coordinator


def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye, center, up):
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye
    f /= (np.linalg.norm(f) + 1e-12)
    s = np.cross(f, up)
    s /= (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -np.array([np.dot(s, eye), np.dot(u, eye), np.dot(-f, eye)], dtype=np.float32)
    return m


def run():
    W, H = 1024, 640

    # Build MVP
    P = perspective(60, W / float(H), 0.1, 500.0)
    V = look_at((0, 0, 6.0), (0, 0, 0), (0, 1, 0))
    M = np.identity(4, np.float32)
    MVP = (P @ V @ M).astype(np.float32)

    # Create renderer (single-threaded demo)
    r = GLRenderer(MVP, size=(W, H))

    # --- Build one layer of each type ---------------------------------------

    # Mesh: simple tetra (as "membrane")
    verts = np.array([[1, 1, 1],
                      [-1, -1, 1],
                      [-1, 1, -1],
                      [1, -1, -1]], np.float32) * 1.2
    idx = np.array([0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2], np.uint32)
    membrane = MeshLayer(
        positions=verts,
        indices=idx,
        rgba=(0.2, 0.6, 0.9, 0.35),
        alpha=1.0
    )

    # Lines: wire cube (as "lines")
    c = 1.8
    cube = np.array([
        [-c, -c, -c], [ c, -c, -c],
        [ c, -c, -c], [ c,  c, -c],
        [ c,  c, -c], [-c,  c, -c],
        [-c,  c, -c], [-c, -c, -c],
        [-c, -c,  c], [ c, -c,  c],
        [ c, -c,  c], [ c,  c,  c],
        [ c,  c,  c], [-c,  c,  c],
        [-c,  c,  c], [-c, -c,  c],
        [-c, -c, -c], [-c, -c,  c],
        [ c, -c, -c], [ c, -c,  c],
        [ c,  c, -c], [ c,  c,  c],
        [-c,  c, -c], [-c,  c,  c]
    ], np.float32)
    line_colors = np.tile(np.array([0, 0, 0, 0.95], np.float32), (cube.shape[0], 1))
    lines = LineLayer(positions=cube, colors=line_colors, width=3.0, alpha=0.85)

    # Points: 200 dots on a sphere (as "fluid")
    npts = 200
    phi = np.random.rand(npts) * 2 * np.pi
    cost = np.random.rand(npts) * 2 - 1
    sint = np.sqrt(1 - cost * cost)
    base_pts = np.stack([sint * np.cos(phi), sint * np.sin(phi), cost], axis=1).astype(np.float32) * 2.2
    pt_colors = np.ones((npts, 4), np.float32)
    pt_colors[:, :3] = np.array([1.0, 0.2, 0.2], np.float32)
    pt_sizes = np.full((npts,), 8.0, np.float32)

    # ------------------------------------------------------------------------

    clock = pygame.time.Clock()
    angle = 0.0
    running = True
    t = 0.0

    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False

        # Spin mesh & lines for obvious motion
        angle += 0.8 * (clock.get_time() / 1000.0)
        ca, sa = math.cos(angle), math.sin(angle)
        R = np.array([[ ca, 0,  sa, 0],
                      [  0, 1,   0, 0],
                      [-sa, 0,  ca, 0],
                      [  0, 0,   0, 1]], np.float32)
        # Matrices are row-major; transpose before uploading to OpenGL
        r.set_mvp((P @ V @ R).astype(np.float32).T)

        # Slightly wobble the point cloud so "fluid" visibly updates
        t += 0.015
        wobble = base_pts.copy()
        wobble[:, 0] += 0.15 * np.sin(t + np.linspace(0, 2*np.pi, npts, endpoint=False))
        wobble[:, 1] += 0.15 * np.cos(1.7*t + np.linspace(0, 2*np.pi, npts, endpoint=False))

        fluid = PointLayer(positions=wobble, colors=pt_colors, sizes_px=pt_sizes, alpha=1.0)

        # --- Submit exactly one dict with one of each type ------------------
        layers = {
            "membrane": membrane,  # MeshLayer
            "lines":    lines,     # LineLayer
            "fluid":    fluid,     # PointLayer (preferred by draw_layers)
        }

        draw_layers(r, layers, (W, H))  # uploads and draws via the same path as the coordinator

        # If your GLRenderer.draw() does NOT swap buffers internally, uncomment:
        # pygame.display.flip()

        clock.tick(60)

    r.dispose()
    pygame.quit()


if __name__ == "__main__":
    run()
