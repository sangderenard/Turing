"""Minimal pygame/OpenGL window that runs the dye solver until you close it.

Everything shown is live solver state. Geometry is uploaded once; every frame
steps the compute solver and then draws tubes and junction beads that read that
state directly. Nothing in the picture is a stored colour, so what moves is
transport rather than a scrolling texture.

Run it::

    python -m src.rendering.influence_flow_app

``--frames N`` exits after N frames and writes a screenshot, which is how the
window gets verified without a human watching it.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import math
import time
from pathlib import Path

import numpy as np

# PyOpenGL binds lazily -- it needs a context when a call is made, not when it
# is imported -- so the flat GL name surface can live at module level, which is
# the only place a star import is legal.
from OpenGL.GL import *  # noqa: F403

DEFAULT_EXPRESSION = (
    "(a + b)*sin(a) - 2*c**2 + cos(b*3) + sin(c)**2/(1 + a**2)"
    " + d*cos(a*d) + e**2/(1 + sin(e)**2) + f*sin(b + f)"
    " + g/(1 + cos(g)**2) + h*sin(h*2)*cos(c)"
)


def _orthographic(width: float, height: float) -> np.ndarray:
    """Map layout pixels to clip space, y downward as screen coordinates."""

    return np.asarray([
        [2.0 / width, 0.0, 0.0, -1.0],
        [0.0, -2.0 / height, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def build_scene(expression: str, width: int, height: int):
    """Graph, field, layout, and pipe lengths for one expression."""

    import sympy

    from ..compiler.influence_field import (
        InfluenceContract, field_from_process_graph,
    )
    from ..transmogrifier.graph.graph_express2 import ProcessGraph
    from .influence_field_shader import reduce_crossings

    graph = ProcessGraph(materialize_memory=False, source_language="sympy")
    graph.build_from_expression(sympy.sympify(expression, evaluate=False))
    graph.compute_levels(method="asap", order="dependency")

    field = field_from_process_graph(graph, InfluenceContract(enabled=True))
    field.propagate()

    edges = list(graph.G.edges())
    layers: dict[int, list] = collections.defaultdict(list)
    for node in graph.G.nodes():
        layers[graph.levels[node]].append(node)
    ordered = reduce_crossings(layers, edges)

    deepest = max(ordered)
    widest = max(len(members) for members in ordered.values())
    margin = 80.0
    positions = {}
    for level, members in ordered.items():
        for index, node in enumerate(members):
            positions[node] = (
                margin + (width - 2 * margin) * level / max(1, deepest),
                height / 2.0
                + (index - (len(members) - 1) / 2.0)
                * (height - 2 * margin) / max(6, widest),
            )
    lengths = {
        (source, target): math.hypot(
            positions[target][0] - positions[source][0],
            positions[target][1] - positions[source][1],
        )
        for source, target in edges
    }
    return field, positions, lengths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", default=DEFAULT_EXPRESSION)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--pipe-width", type=float, default=6.0)
    parser.add_argument("--flow-speed", type=float, default=120.0,
                        help="cells/second; 1/dt gives diffusion-free drops")
    parser.add_argument("--emission-period", type=float, default=1.1)
    parser.add_argument("--emission-duty", type=float, default=0.07,
                        help="short duty releases discrete drops of dye")
    parser.add_argument("--bead-radius", type=float, default=2.6,
                        help="junction bead radius, in pipe widths")
    parser.add_argument("--blend", type=float, default=7.0,
                        help="smooth-union radius joining tubes into beads")
    parser.add_argument("--dye-strength", type=float, default=3.2,
                        help="Beer-Lambert absorption of dye into the water")
    parser.add_argument("--steps-per-frame", type=int, default=2)
    parser.add_argument("--prefill", type=float, default=10.0,
                        help="seconds of solver time before the window opens")
    parser.add_argument("--frames", type=int, default=0,
                        help="exit after N frames instead of running until closed")
    parser.add_argument("--screenshot", default="")
    args = parser.parse_args(argv)

    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE

    pygame.init()
    for attribute, value in (
        (pygame.GL_CONTEXT_MAJOR_VERSION, 4),
        (pygame.GL_CONTEXT_MINOR_VERSION, 6),
        (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE),
    ):
        pygame.display.gl_set_attribute(attribute, value)
    pygame.display.set_mode((args.width, args.height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("influence field - live dye transport")

    from .influence_field_image import bake_palette
    from .influence_flow import FlowSettings, InfluenceFlow
    from .influence_flow_gpu import (
        B_GEOMETRY, B_INCIDENT_ENTRY, B_INCIDENT_OFFSET, B_PIPE_IN,
        BULB_FRAGMENT, BULB_VERTEX, GPUInfluenceFlow, PIPE_FRAGMENT,
        PIPE_VERTEX, TransparencyLists, draw_program, storage_buffer,
    )

    field, positions, lengths = build_scene(
        args.expression, args.width, args.height
    )
    flow = InfluenceFlow(field, lengths=lengths, settings=FlowSettings(
        emission_period=args.emission_period,
        emission_duty=args.emission_duty,
        flow_speed=args.flow_speed,
        cell_density=0.10,
        max_cells=128,
    ))
    gpu = GPUInfluenceFlow(flow)

    # Prefill so the network already carries dye when the window opens; an
    # empty network takes a while to reach its far end and looks broken.
    dt = 1.0 / 120.0
    for _ in range(int(args.prefill / dt)):
        gpu.step(dt)

    def locations(program, names):
        return {name: glGetUniformLocation(program, name) for name in names}

    shared = (
        "uMVP", "uPalette", "uWaterColour", "uDyeStrength", "uPeakWeight",
        "uBakedCategory", "uCells", "uCategories", "uMaxFragments",
    )
    pipe_draw = draw_program(PIPE_VERTEX, PIPE_FRAGMENT)
    pipe_uniforms = locations(pipe_draw, shared + ("uPipeWidth", "uPipeDepth"))
    bulb_draw = draw_program(BULB_VERTEX, BULB_FRAGMENT)
    bulb_uniforms = locations(
        bulb_draw, shared + ("uBeadRadius", "uBlend", "uBulbDepth")
    )
    lists = TransparencyLists(args.width, args.height)

    # Tube geometry: two triangles per edge. aCorner carries position along and
    # across the pipe and the vertex stage extrudes it.
    corners = ((0.0, -1.0), (1.0, -1.0), (1.0, 1.0),
               (0.0, -1.0), (1.0, 1.0), (0.0, 1.0))
    vertices = []
    for index, (source, target, _role) in enumerate(flow.edges):
        if (source, target) not in lengths:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        for corner_x, corner_y in corners:
            vertices.append((corner_x, corner_y, x0, y0, x1, y1, float(index)))
    mesh = np.asarray(vertices, dtype=np.float32)

    def upload(data, layout):
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        stride = data.shape[1] * 4
        for location, size, offset in layout:
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE,
                                  stride, ctypes.c_void_p(offset))
        glBindVertexArray(0)
        return vao

    pipe_vao = upload(mesh, ((0, 2, 0), (1, 2, 8), (2, 2, 16), (3, 1, 24)))

    # Scene description the bead shader evaluates: one capsule per tube, and
    # for each node the tubes meeting there plus which end is near. The bead
    # needs both -- the capsules to smooth-union its surface against, and the
    # near cells to read the fluid genuinely present at that junction.
    capsules = np.zeros(len(flow.edges), dtype=np.dtype([
        ("from", np.float32, 2), ("to", np.float32, 2),
        ("radius", np.float32), ("edge", np.uint32),
    ]))
    incident: dict[int, list[tuple[int, int]]] = {}
    for index, (source, target, _role) in enumerate(flow.edges):
        if (source, target) not in lengths:
            continue
        capsules[index] = (
            positions[source], positions[target], args.pipe_width, index
        )
        # end 0 = head cell (fluid leaving here), 1 = tail cell (arriving here)
        incident.setdefault(flow._node_index[source], []).append((index, 0))
        incident.setdefault(flow._node_index[target], []).append((index, 1))

    incident_offsets = np.zeros(len(flow.nodes) + 1, dtype=np.uint32)
    incident_entries: list[tuple[int, int]] = []
    for node in range(len(flow.nodes)):
        incident_offsets[node] = len(incident_entries)
        incident_entries.extend(incident.get(node, ()))
    incident_offsets[len(flow.nodes)] = len(incident_entries)
    entry_array = np.asarray(
        incident_entries or [(0, 0)], dtype=np.uint32
    ).reshape(-1, 2)

    storage_buffer(B_GEOMETRY, capsules)
    storage_buffer(B_INCIDENT_OFFSET, incident_offsets)
    storage_buffer(B_INCIDENT_ENTRY, entry_array)

    # Bead quads, oversized so the smooth union with the tubes has room to
    # spread beyond the sphere itself.
    bead_radius = args.pipe_width * args.bead_radius
    quad_radius = bead_radius + args.blend * 2.0
    bead_corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0),
                    (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    bead_vertices = []
    for node, index in flow._node_index.items():
        if node not in positions:
            continue
        cx, cy = positions[node]
        for corner_x, corner_y in bead_corners:
            bead_vertices.append(
                (corner_x, corner_y, cx, cy, quad_radius, float(index))
            )
    beads = np.asarray(bead_vertices, dtype=np.float32)
    bead_vao = upload(beads, ((0, 2, 0), (1, 2, 8), (2, 1, 16), (3, 1, 20)))

    palette_rgb, _reserved = bake_palette()
    palette = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, palette)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, palette_rgb.shape[1],
                 palette_rgb.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE,
                 np.ascontiguousarray(palette_rgb))
    for parameter in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
        glTexParameteri(GL_TEXTURE_2D, parameter, GL_LINEAR)
    for parameter in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T):
        glTexParameteri(GL_TEXTURE_2D, parameter, GL_CLAMP_TO_EDGE)

    baked_category = (
        flow.categories.index("baked") if "baked" in flow.categories else -1
    )
    mvp = _orthographic(float(args.width), float(args.height))
    background = (0.055, 0.055, 0.07)
    water = (0.60, 0.72, 0.80)

    def bind_shared(program, slots, peak):
        glUseProgram(program)
        glUniformMatrix4fv(slots["uMVP"], 1, GL_TRUE, mvp)
        glUniform1f(slots["uPeakWeight"], peak)
        glUniform1f(slots["uDyeStrength"], args.dye_strength)
        glUniform3f(slots["uWaterColour"], *water)
        glUniform1ui(slots["uCells"], gpu.cells)
        glUniform1i(slots["uCategories"], gpu.categories)
        glUniform1i(slots["uBakedCategory"], baked_category)
        glUniform1ui(slots["uMaxFragments"], lists.capacity)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, palette)
        glUniform1i(slots["uPalette"], 0)

    # Peak concentration normalises the density channel. Reading it back every
    # frame would stall the pipeline for a number that barely moves, so it is
    # refreshed occasionally and held between refreshes.
    peak = 1e-3
    clock = pygame.time.Clock()
    frame = 0
    running = True
    started = time.perf_counter()
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (
                event.type == KEYDOWN and event.key == K_ESCAPE
            ):
                running = False
        for _ in range(max(1, args.steps_per_frame)):
            gpu.step(dt)
        if frame % 30 == 0:
            # A high percentile, not the maximum. Diffusion-free transport
            # concentrates a whole emission pulse into very few cells, so the
            # max is a rare outlier; normalising against it drags every
            # ordinary cell toward the water colour and the dye disappears.
            live = gpu.read_state()[..., 0].sum(axis=2)
            carrying = live[live > 1e-9]
            peak = max(1e-3, float(
                np.percentile(carrying, 96.0) if carrying.size else 1e-3
            ))

        glViewport(0, 0, args.width, args.height)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, B_PIPE_IN, gpu.state_buffer)
        lists.begin()

        bind_shared(pipe_draw, pipe_uniforms, peak)
        glUniform1f(pipe_uniforms["uPipeWidth"], args.pipe_width)
        glUniform1f(pipe_uniforms["uPipeDepth"], 0.5)
        glBindVertexArray(pipe_vao)
        glDrawArrays(GL_TRIANGLES, 0, len(mesh))

        bind_shared(bulb_draw, bulb_uniforms, peak)
        glUniform1f(bulb_uniforms["uBeadRadius"], bead_radius)
        glUniform1f(bulb_uniforms["uBlend"], args.blend)
        glUniform1f(bulb_uniforms["uBulbDepth"], 0.3)
        glBindVertexArray(bead_vao)
        glDrawArrays(GL_TRIANGLES, 0, len(beads))
        glBindVertexArray(0)

        lists.resolve(background)

        pygame.display.flip()
        clock.tick(60)
        frame += 1
        if args.frames and frame >= args.frames:
            running = False

    used = lists.used_fragments()
    if args.screenshot:
        raw = glReadPixels(
            0, 0, args.width, args.height, GL_RGB, GL_UNSIGNED_BYTE
        )
        from PIL import Image

        image = Image.frombytes(
            "RGB", (args.width, args.height), raw
        ).transpose(Image.FLIP_TOP_BOTTOM)
        Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.screenshot)

    elapsed = time.perf_counter() - started
    print(
        f"tubes={len(mesh) // 6} beads={len(beads) // 6} cells={gpu.cells} "
        f"frames={frame} in {elapsed:.1f}s "
        f"({frame / max(elapsed, 1e-6):.0f} fps) solver t={gpu.time:.1f}s "
        f"peak={peak:.4f} list-fragments={used}/{lists.capacity}"
    )
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
