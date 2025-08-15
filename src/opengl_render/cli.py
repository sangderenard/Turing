# general_gl_renderer/cli.py
# Tiny demo: spins a triangle mesh, a wire cube, and dotted points above.
# This is *only* to prove the renderer end-to-end. No physics; safe to delete.

import math
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from general_gl_renderer.renderer import GLRenderer, MeshLayer, LineLayer, PointLayer

def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = f / max(aspect, 1e-6)
    m[1,1] = f
    m[2,2] = (zfar + znear) / (znear - zfar)
    m[2,3] = (2 * zfar * znear) / (znear - zfar)
    m[3,2] = -1.0
    return m

def look_at(eye, center, up):
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye; f /= (np.linalg.norm(f) + 1e-12)
    s = np.cross(f, up); s /= (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0,:3] = s; m[1,:3] = u; m[2,:3] = -f
    m[:3,3] = -np.array([np.dot(s, eye), np.dot(u, eye), np.dot(-f, eye)], dtype=np.float32)
    return m

def run():
    pygame.init()
    W, H = 1024, 640
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 8)
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)

    # Build MVP
    P = perspective(60, W/float(H), 0.1, 500.0)
    V = look_at((0, 0, 6.0), (0, 0, 0), (0,1,0))
    M = np.identity(4, np.float32)
    MVP = (P @ V @ M).astype(np.float32)

    r = GLRenderer(MVP)

    # Mesh: simple tetra
    verts = np.array([[1,1,1], [-1,-1,1], [-1,1,-1], [1,-1,-1]], np.float32) * 1.2
    idx   = np.array([0,1,2,  0,3,1,  0,2,3,  1,3,2], np.uint32)
    r.set_mesh(MeshLayer(positions=verts, indices=idx, rgba=(0.2,0.6,0.9,0.35), alpha=1.0))

    # Lines: wire cube
    c = 1.8
    cube = np.array([
        [-c,-c,-c],[ c,-c,-c],
        [ c,-c,-c],[ c, c,-c],
        [ c, c,-c],[-c, c,-c],
        [-c, c,-c],[-c,-c,-c],
        [-c,-c, c],[ c,-c, c],
        [ c,-c, c],[ c, c, c],
        [ c, c, c],[-c, c, c],
        [-c, c, c],[-c,-c, c],
        [-c,-c,-c],[-c,-c, c],
        [ c,-c,-c],[ c,-c, c],
        [ c, c,-c],[ c, c, c],
        [-c, c,-c],[-c, c, c]
    ], np.float32)
    col = np.tile(np.array([0,0,0,0.95], np.float32), (cube.shape[0],1))
    r.set_lines(LineLayer(positions=cube, colors=col, width=3.0, alpha=0.85))

    # Points: 200 dots on a sphere
    phi = np.random.rand(200)*2*np.pi
    cost = np.random.rand(200)*2-1
    sint = np.sqrt(1-cost*cost)
    pts = np.stack([sint*np.cos(phi), sint*np.sin(phi), cost], axis=1).astype(np.float32)*2.2
    cols = np.ones((200,4), np.float32); cols[:,:3] = np.array([1.0,0.2,0.2])
    sizes = np.full((200,), 8.0, np.float32)
    r.set_points(PointLayer(positions=pts, colors=cols, sizes_px=sizes, alpha=1.0))

    clock = pygame.time.Clock()
    angle = 0.0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False

        # spin the model matrix a little for visual proof
        angle += 0.8 * (clock.get_time()/1000.0)
        ca, sa = math.cos(angle), math.sin(angle)
        R = np.array([[ ca,0, sa,0],
                      [  0,1,  0,0],
                      [-sa,0, ca,0],
                      [  0,0,  0,1]], np.float32)
        MVP = (P @ V @ R).astype(np.float32)
        r.set_mvp(MVP)

        r.draw((W,H))
        pygame.display.flip()
        clock.tick(60)

    r.dispose()
    pygame.quit()

if __name__ == "__main__":
    run()
