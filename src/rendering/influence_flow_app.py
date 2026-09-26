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


DEFAULT_PROGRAM_FILE = "examples/xor_project/train_xor.py"
DEFAULT_ENTRYPOINT = "train"
DEFAULT_FEEDS = {"steps": 200, "lr": 1e-2}
DEFAULT_MUTABLE = ("steps", "lr")


def build_scene(program_file: str, entrypoint: str, width: int, height: int):
    """Compile a real program and lay out the Dual IR it produced.

    ``precompile_only`` selects graph ingestion over the discovery tape. That
    is the documented route for a program capture: the tape linearises one
    observed run and cannot carry a graph portion's full operand set, so it
    drops boundary operands and needs every operation present in its captured
    kernel tables.

    The parameters stay symbolic. Concrete feeds are compile-time constants,
    so the compiler folds the body away and returns the answer -- correctly,
    but leaving no program to watch.
    """

    from pathlib import Path as _Path

    from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot
    from ..compiler.influence_field import (
        DYNAMIC, BAKED, RECURRENT, InfluenceContract, field_from_dual_ir,
    )
    from .influence_field_shader import reduce_crossings

    root = _Path(__file__).resolve().parents[2]
    source = _Path(program_file)
    if not source.is_absolute():
        source = root / program_file

    aot = compile_ast_aot(
        source.read_text(encoding="utf-8"),
        entrypoint,
        dict(DEFAULT_FEEDS),
        precompile_only=True,
        unroll_limit=8,
        mutable_parameters=DEFAULT_MUTABLE,
        boundary_namespace=str(root / "boundary_namespaces"),
        source_language="python",
        extraction_contract=str(
            root / "extraction_contracts" / "program_extraction.yaml"
        ),
    )
    regions = dict(aot.region_programs)
    field = field_from_dual_ir(
        aot.shell,
        InfluenceContract(enabled=True,
                          categories=(DYNAMIC, BAKED, RECURRENT)),
        regions=regions,
    )
    field.propagate()
    print(f"dual ir: {len(regions)} region(s), "
          f"{sum(len(getattr(getattr(c, 'program', c), 'steps', []) or []) for c in regions.values())} "
          f"numeric steps")

    edges = [(a, b) for a, outs in field._outgoing.items() for b, _r in outs]
    nodes = sorted(field._nodes, key=str)
    rank = {key: index for index, key in enumerate(field.activation_order)}

    successors: dict = {}
    for a, b in edges:
        successors.setdefault(a, []).append(b)
    depth = {key: 0 for key in nodes}
    for key in sorted(nodes, key=lambda k: rank.get(k, 0)):
        for nxt in successors.get(key, ()):
            if rank.get(nxt, 0) > rank.get(key, 0):
                depth[nxt] = max(depth.get(nxt, 0), depth[key] + 1)

    layers: dict[int, list] = collections.defaultdict(list)
    for key in nodes:
        layers[depth[key]].append(key)
    ordered = reduce_crossings(layers, edges)

    deepest = max(ordered) if ordered else 1
    widest = max((len(v) for v in ordered.values()), default=1)
    margin = 60.0
    positions = {}
    for level, members in ordered.items():
        for index, key in enumerate(members):
            positions[key] = (
                margin + (width - 2 * margin) * level / max(1, deepest),
                height / 2.0
                + (index - (len(members) - 1) / 2.0)
                * (height - 2 * margin) / max(6, widest),
            )
    lengths = {
        (a, b): math.hypot(positions[b][0] - positions[a][0],
                           positions[b][1] - positions[a][1])
        for a, b in edges if a in positions and b in positions
    }
    return field, positions, lengths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-file", default=DEFAULT_PROGRAM_FILE)
    parser.add_argument("--entrypoint", default=DEFAULT_ENTRYPOINT)
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
        B_BEADS, B_GEOMETRY, B_INCIDENT_ENTRY, B_INCIDENT_OFFSET, B_PIPE_IN,
        GLASS_FRAGMENT, GLASS_VERTEX, GPUInfluenceFlow,
        draw_program, storage_buffer,
    )

    field, positions, lengths = build_scene(
        args.program_file, args.entrypoint, args.width, args.height
    )
    print(f"program: {len(field._nodes)} nodes, "
          f"{sum(len(v) for v in field._outgoing.values())} edges, "
          f"{len(field.activation_order)} activations")
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

    glass = draw_program(GLASS_VERTEX, GLASS_FRAGMENT)
    glass_uniforms = locations(glass, (
        "uOrigin", "uExtent", "uPalette", "uWaterColour", "uDyeStrength",
        "uPeakWeight", "uBakedCategory", "uCells", "uCategories",
        "uCapsules", "uBeadCount", "uBlend", "uThickness", "uBackground",
        "uReferenceSpan",
    ))

    # The scene is described, not tessellated: one capsule per tube and one
    # sphere per junction, both read by a single fullscreen field evaluation.
    capsules = np.zeros(len(flow.edges), dtype=np.dtype([
        ("from", np.float32, 2), ("to", np.float32, 2),
        ("radius", np.float32), ("edge", np.uint32),
    ]))
    drawn = 0
    spans = [
        math.hypot(positions[b][0] - positions[a][0],
                   positions[b][1] - positions[a][1])
        for (a, b, _r) in flow.edges if a in positions and b in positions
    ]
    reference_span = float(np.median(spans)) if spans else 1.0
    for index, (source, target, _role) in enumerate(flow.edges):
        if (source, target) not in lengths:
            continue
        span = math.hypot(positions[target][0] - positions[source][0],
                          positions[target][1] - positions[source][1])
        # A stretched pipe is a thinner pipe: the same dye occupies more
        # length, so its cross-section narrows rather than its contents
        # thinning out of sight.
        radius = args.pipe_width * math.sqrt(
            min(1.0, reference_span / max(span, 1e-6)))
        capsules[index] = (
            positions[source], positions[target],
            max(radius, args.pipe_width * 0.35), index
        )
        drawn += 1

    bead_radius = args.pipe_width * args.bead_radius
    beads = np.zeros((max(1, len(positions)), 4), dtype=np.float32)
    for slot, node in enumerate(positions):
        beads[slot] = (positions[node][0], positions[node][1], bead_radius, 0.0)

    storage_buffer(B_GEOMETRY, capsules)
    storage_buffer(B_BEADS, beads)

    screen = np.asarray([-1, -1, 3, -1, -1, 3], dtype=np.float32)
    screen_vao = glGenVertexArrays(1)
    screen_vbo = glGenBuffers(1)
    glBindVertexArray(screen_vao)
    glBindBuffer(GL_ARRAY_BUFFER, screen_vbo)
    glBufferData(GL_ARRAY_BUFFER, screen.nbytes, screen, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glBindVertexArray(0)

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
            live = gpu.read_state()[..., 0].sum(axis=2)
            carrying = live[live > 1e-9]
            peak = max(1e-3, float(
                np.percentile(carrying, 96.0) if carrying.size else 1e-3
            ))

        glViewport(0, 0, args.width, args.height)
        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, B_PIPE_IN, gpu.state_buffer)

        glUseProgram(glass)
        glUniform2f(glass_uniforms["uOrigin"], 0.0, 0.0)
        glUniform2f(glass_uniforms["uExtent"], float(args.width), float(args.height))
        glUniform1f(glass_uniforms["uPeakWeight"], peak)
        glUniform1f(glass_uniforms["uDyeStrength"], args.dye_strength)
        glUniform3f(glass_uniforms["uWaterColour"], *water)
        glUniform3f(glass_uniforms["uBackground"], *background)
        glUniform1ui(glass_uniforms["uCells"], gpu.cells)
        glUniform1i(glass_uniforms["uCategories"], gpu.categories)
        glUniform1i(glass_uniforms["uBakedCategory"], baked_category)
        glUniform1ui(glass_uniforms["uCapsules"], len(capsules))
        glUniform1ui(glass_uniforms["uBeadCount"], len(beads))
        glUniform1f(glass_uniforms["uBlend"], args.blend)
        glUniform1f(glass_uniforms["uThickness"], bead_radius)
        glUniform1f(glass_uniforms["uReferenceSpan"], reference_span)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, palette)
        glUniform1i(glass_uniforms["uPalette"], 0)
        glBindVertexArray(screen_vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        pygame.display.flip()
        clock.tick(60)
        frame += 1
        if args.frames and frame >= args.frames:
            running = False

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
        f"tubes={drawn} beads={len(beads)} cells={gpu.cells} "
        f"frames={frame} in {elapsed:.1f}s "
        f"({frame / max(elapsed, 1e-6):.0f} fps) solver t={gpu.time:.1f}s "
        f"peak={peak:.4f}"
    )
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
