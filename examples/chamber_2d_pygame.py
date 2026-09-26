"""The chamber as a 2D room of tiles, drawn in pygame.

WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT

It is a VIEW plus the tiling that lets a kernel of one width cover a room of
any size.  It builds no state and it computes no physics.  The scenario
parameters come from ``chamber_raincloud_demo``, the chamber is assembled by
``chamber_raincloud_record.build_native``, and the stepping is the shared
``dt_controller.run_superstep``.  Everything else is imported, on purpose: a
viewer that retypes a law parameter is a viewer that can disagree with the
thing it claims to be showing.

WHY A 2D ROOM IS THE SAME CHAMBER

``chamber_raincloud_demo`` is ``shape=(1, 1, 8)`` -- one cell wide, one cell
deep, eight tall.  It is ALREADY flattened in y.  Widening x to make a room
is a change to one number, because ``chamber_sim._neighbour_tables`` resolves
a face with no neighbour to the cell itself: the front and back faces then
carry zero pressure difference and nothing flows through them.  They are
walls, and they were already walls.  ``z`` stays vertical, so settling,
autoconversion and rain work exactly as they do in the column.

No law was modified, no 2D variant of a law exists, and nothing was
recompiled.  These are the same durable LLVM pieces in
``artifacts/llvm_pieces`` that the recorder uses, run on a slab.

BATCH IS THE KERNEL WIDTH.  IT IS NOT THE MAP SIZE.

A compiled piece computes exactly the number of lanes it was authored at --
hand a b8 piece a longer column and it returns eight values, quietly.  That
is a property of the KERNEL, and the room is not obliged to be the shape of
it: ``tiled_law`` below slices the columns into kernel-width chunks, calls
the piece once per chunk, and concatenates.  The neighbour columns are
gathered by ``chamber_sim`` before any of this, over the whole grid, so a
chunk boundary is not a halo and carries no special case.

So the grid is whatever the picture wants.  ``tiled_law`` picks the widest
available piece that divides the cell count and falls back to b1, which
divides everything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

from chamber_dt_join import CompiledLaw  # noqa: E402
from chamber_raincloud_record import _col, _load_demo, build_native  # noqa: E402
from src.common.dt_system.dt_controller import STController, Targets, run_superstep  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402
from src.compiler.native_law_kernels import LLVMPiece  # noqa: E402


PIECES = HERE.parent / "artifacts" / "llvm_pieces"

#: The authored widths the artifacts were emitted at, widest first.
AVAILABLE_BATCHES = (64, 16, 8, 1)


def tiled_law(law: str, cells: int) -> CompiledLaw:
    """One law over ``cells`` cells, on whatever kernel width is available.

    Eight chunked b8 calls reproduce one b64 call exactly -- the kernel is a
    pure function of its lanes and the neighbour gather already happened --
    so which width gets used is a throughput choice, not a physics one."""
    batch = next(b for b in AVAILABLE_BATCHES if cells % b == 0)
    piece = LLVMPiece.load(PIECES / law / f"b{batch}" / f"{law}.piece")

    def stage(*columns):
        arrays = [
            np.asarray(column.tolist() if isinstance(column, AbstractTensor)
                       else column, dtype=np.float64)
            for column in columns
        ]
        # Per-cell columns are sliced; scalars -- every law parameter, and dt
        # -- are handed to each chunk unchanged.
        chunked: list[list[np.ndarray]] | None = None
        for start in range(0, cells, batch):
            slice_args = [
                a[start:start + batch] if (a.ndim and a.size == cells) else a
                for a in arrays
            ]
            produced = piece(*slice_args)
            if chunked is None:
                chunked = [[] for _ in produced]
            for index, value in enumerate(produced):
                chunked[index].append(np.asarray(value, dtype=np.float64))
        return tuple(
            AbstractTensor.tensor(np.concatenate(parts).tolist())
            for parts in (chunked or [])
        )

    return CompiledLaw(law, tuple(piece.argument_names),
                       tuple(piece.output_names), stage, piece)


def tiled_chamber_laws(cells: int) -> dict:
    """The four laws ``ChamberSim`` requires, each tiled for this grid.

    Surfaces and pools are one cell each, so they resolve to b1 and the
    tiling is a single call -- the same thing ``build_native`` does by hand
    for them today."""
    return {
        "voxel_air_step": tiled_law("voxel_air_step", cells),
        "voxel_species_step": tiled_law("voxel_species_step", cells),
        "surface_step": tiled_law("surface_step", 1),
        "pool_step": tiled_law("pool_step", 1),
    }


# ---------------------------------------------------------------------
# the picture
# ---------------------------------------------------------------------
# One entry per drawable channel: where it comes from and how it reads.
# `kind` is "air" (a column of the air engine) or "water" (a published
# output of the water species engine).
CHANNELS = (
    ("T",   "air",   "temperature",  (255, 176,  96)),
    ("S",   "water", "saturation",   (140, 220, 160)),
    ("LWC", "water", "cloud water",  (245, 245, 255)),
    ("IWC", "water", "ice",          (170, 225, 255)),
    ("RWC", "water", "rain",         ( 90, 150, 245)),
)

BACKGROUND = (14, 16, 22)
WALL = (46, 50, 62)
TEXT = (210, 216, 228)


def _channel(sim, name: str, kind: str) -> np.ndarray:
    """One channel as a flat per-cell array, or zeros if it is not published.

    A law publishes its outputs only after its first accepted step, so every
    read here tolerates absence rather than assuming a frame has run."""
    water = sim.species[sim.water]
    if kind == "air":
        if name in sim.air.state.columns:
            return _col(sim.air.state.columns[name])
        source = sim.air.state.outputs
    else:
        source = water.state.outputs
    if name in source:
        return _col(source[name])
    return np.zeros(sim.n, dtype=np.float64)


class Scale:
    """A per-channel autoscale that rises fast and falls slowly.

    Cloud water is ~1e-4 kg/m^3 and rain an order below it, so a fixed range
    shows either a black screen or a white one. The decay keeps one burst of
    rain from flattening everything that follows it for the rest of the run."""

    def __init__(self, floor: float = 1e-12, decay: float = 0.98):
        self.peak = floor
        self.floor = floor
        self.decay = decay

    def __call__(self, values: np.ndarray) -> np.ndarray:
        high = float(np.max(values)) if values.size else 0.0
        self.peak = max(high, self.peak * self.decay, self.floor)
        return np.clip(values / self.peak, 0.0, 1.0)


def _mix(base, colour, weight: float):
    w = max(0.0, min(1.0, weight))
    return tuple(int(b + (c - b) * w) for b, c in zip(base, colour))


def run(seconds: float, tile: int, plate_k: float | None, round_s: float,
        shape: tuple[int, int, int]) -> None:
    import pygame

    demo = _load_demo()
    nx, ny, nz = shape
    cells = nx * ny * nz
    # `laws=None` is deliberate: every law ChamberSim requires is supplied
    # here as a compiled piece, so the law module -- and its SymPy
    # construction -- is never touched. Passing it would only matter if a law
    # were missing, and a silent interpreter fallback is exactly what must
    # not happen.
    sim = build_native(demo, None, shape, plate_temperature=plate_k,
                       compiled_laws=tiled_chamber_laws(cells))

    dt_cfg = dict(demo.DT)
    targets = Targets(cfl=dt_cfg["cfl"], div_max=dt_cfg["div_max"],
                      mass_max=dt_cfg["mass_max"],
                      energy_exchange_fraction=dt_cfg["energy_exchange_fraction"])
    ctrl = STController()
    dt = float(dt_cfg["dt_initial"])

    pygame.init()
    pygame.display.set_caption("chamber 2D -- the raincloud laws, side on")
    margin, hud = 24, 96
    surface = pygame.display.set_mode(
        (nx * tile + margin * 2, nz * tile + margin * 2 + hud))
    font = pygame.font.SysFont("consolas", 15)
    small = pygame.font.SysFont("consolas", 13)
    clock = pygame.time.Clock()

    scales = {name: Scale() for name, _, _, _ in CHANNELS}
    single = 0                      # 0 = composite, 1..5 = one channel
    paused = False
    t = 0.0
    wall = time.time()
    substeps = rejected = 0

    def cell_rect(x: int, z: int):
        return pygame.Rect(margin + x * tile,
                           margin + (nz - 1 - z) * tile, tile, tile)

    running = True
    while running and t < seconds:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif pygame.K_0 <= event.key <= pygame.K_5:
                    single = event.key - pygame.K_0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Poke the room: warm or cool one cell's air. This writes the
                # air engine's own temperature column -- the same state the
                # law reads next step, not a display effect.
                mx, my = event.pos
                x = (mx - margin) // tile
                row = (my - margin) // tile
                if 0 <= x < nx and 0 <= row < nz:
                    z = nz - 1 - row
                    index = int(x + nx * (0 + ny * z))
                    column = _col(sim.air.state.columns["T"]).copy()
                    column[index] += 12.0 if event.button == 1 else -12.0
                    sim.air.state.columns["T"] = AbstractTensor.tensor(
                        column.tolist())

        if not paused:
            log: list = []
            total, dt, metrics = run_superstep(
                sim.state, round_s, dt, sim.dx, targets, ctrl, sim.advance,
                attempt_log=log)
            t += float(total)
            substeps = len(log)
            rejected = sum(1 for attempt in log if not attempt["accepted"])

        fields = {name: _channel(sim, name, kind) for name, kind, _, _ in CHANNELS}
        normed = {name: scales[name](values) for name, values in fields.items()}

        surface.fill(BACKGROUND)
        pygame.draw.rect(surface, WALL,
                         pygame.Rect(margin - 2, margin - 2,
                                     nx * tile + 4, nz * tile + 4), width=2)

        for z in range(nz):
            for x in range(nx):
                index = x + nx * (0 + ny * z)
                rect = cell_rect(x, z)
                if single:
                    name, _, _, colour = CHANNELS[single - 1]
                    surface.fill(_mix(BACKGROUND, colour,
                                      float(normed[name][index])), rect)
                    continue

                # Composite: air temperature is the ground tone, the
                # condensed phases paint over it in the order they occlude.
                shade = _mix((26, 34, 58), (196, 128, 70), float(normed["T"][index]))
                shade = _mix(shade, (70, 120, 90), float(normed["S"][index]) * 0.25)
                surface.fill(shade, rect)

                cloud = float(normed["LWC"][index])
                if cloud > 0.02:
                    fog = pygame.Surface((tile, tile), pygame.SRCALPHA)
                    fog.fill((245, 245, 255, int(210 * cloud)))
                    surface.blit(fog, rect.topleft)
                ice = float(normed["IWC"][index])
                if ice > 0.02:
                    frost = pygame.Surface((tile, tile), pygame.SRCALPHA)
                    frost.fill((170, 225, 255, int(200 * ice)))
                    surface.blit(frost, rect.topleft)

                rain = float(normed["RWC"][index])
                if rain > 0.02:
                    for d in range(1 + int(5 * rain)):
                        px = rect.x + 3 + (d * 7 + index * 13) % max(1, tile - 6)
                        py = rect.y + 2 + (d * 11 + int(t * 90)) % max(1, tile - 4)
                        pygame.draw.line(surface, (120, 170, 250),
                                         (px, py), (px, py + 3), 1)

        # The surfaces the laws actually carry: the ceiling plate and floor.
        for spec, engine in sim.surfaces:
            index = int(spec.voxel)
            x, z = index % nx, index // (nx * ny)
            rect = cell_rect(x, z)
            film = engine.state.outputs.get("h_film")
            frost = engine.state.outputs.get("h_frost")
            film_m = float(_col(film)[0]) if film is not None else 0.0
            frost_m = float(_col(frost)[0]) if frost is not None else 0.0
            colour, thickness = None, 0
            if frost_m > 1e-7:
                colour, thickness = (225, 240, 255), min(tile // 2, 1 + int(frost_m * 4e4))
            elif film_m > 1e-7:
                colour, thickness = (150, 190, 235), min(tile // 3, 1 + int(film_m * 6e4))
            if colour is not None:
                top = rect.y if z == nz - 1 else rect.bottom - thickness
                surface.fill(colour, pygame.Rect(rect.x, top, tile, thickness))

        # The pool on the floor, as a bar under the room.
        for _, engine in sim.pools:
            depth = engine.state.outputs.get("h_pool")
            depth_m = float(_col(depth)[0]) if depth is not None else 0.0
            width = int(nx * tile * min(1.0, depth_m / 0.05))
            pygame.draw.rect(surface, (60, 110, 180),
                             pygame.Rect(margin, margin + nz * tile + 4,
                                         max(0, width), 6))

        base = margin + nz * tile + 16
        label = "composite" if not single else CHANNELS[single - 1][2]
        lines = [
            f"t {t:7.2f} s   dt {dt:.5f}   substeps {substeps:3d}   "
            f"rejected {rejected:3d}   {clock.get_fps():4.1f} fps"
            f"{'   PAUSED' if paused else ''}",
            f"view: {label}    [0] composite  [1-5] channel  "
            f"[space] pause  [click] +12 K  [right-click] -12 K",
            f"{nx}x{nz} cells of {sim.dx:.2f} m, one cell deep -- "
            f"front and back are walls   |   wall clock {time.time() - wall:5.1f} s",
        ]
        for i, line in enumerate(lines):
            surface.blit((font if i == 0 else small).render(line, True, TEXT),
                         (margin, base + i * 18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=600.0,
                        help="simulated seconds before the window closes")
    parser.add_argument("--tile", type=int, default=40, help="pixels per cell")
    parser.add_argument("--plate", type=float, default=None,
                        help="ceiling temperature K (default: the demo's)")
    parser.add_argument("--round", type=float, default=1.0 / 30.0, dest="round_s",
                        help="simulated seconds advanced per drawn frame")
    parser.add_argument("--width", type=int, default=24, help="cells across")
    parser.add_argument("--height", type=int, default=16, help="cells tall")
    args = parser.parse_args()
    run(args.seconds, args.tile, args.plate, args.round_s,
        (args.width, 1, args.height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
