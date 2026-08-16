"""Live shell for the SymPy -> repository SSA -> LLVM fluid program.

The shell owns presentation only.  Fluid mathematics remains in
``symbolic_fluid_model.py`` as SymPy equations, and timestep rollback/retry
remains in the repository ``dt_system``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState
from src.compiler.symbolic_fluid_native_runtime import (
    compile_native_symbolic_fluid_step,
    load_symbolic_fluid_managed_functions,
)


def _rgb(state: SymbolicFluidGridState) -> np.ndarray:
    """Presentation-only mapping from published fields to an RGB image."""

    height = state.height
    anomaly = np.clip((height - 0.98) * 6.0, 0.0, 1.0)
    tracer = np.clip(state.tracer / np.maximum(height, 1.0e-30), 0.0, 1.0)
    speed = np.sqrt(state.momentum_x ** 2 + state.momentum_y ** 2)
    speed = np.clip(speed * 14.0, 0.0, 1.0)
    return np.stack((tracer, speed + 0.25 * anomaly, anomaly), axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--frames", type=int, default=0, help="zero runs until closed")
    parser.add_argument("--frame-duration", type=float, default=1.0 / 30.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scale", type=int, default=20)
    parser.add_argument(
        "--build-directory", type=Path,
        default=Path("build/symbolic-fluid-live/llvm"),
    )
    args = parser.parse_args()

    native = compile_native_symbolic_fluid_step(args.build_directory)
    advance = load_symbolic_fluid_managed_functions(native)[
        "symbolic_fluid_advance"
    ]
    state = SymbolicFluidGridState.initial(args.grid, args.grid)
    targets = Targets(
        cfl=0.45,
        div_max=1.0,
        mass_max=1.0e-8,
        error_limits={"height_positivity": 0.0, "tracer_bounds": 0.0},
    )
    controller = STController(dt_min=1.0e-8, dt_max=args.frame_duration)
    dt = float(args.dt)

    pygame = None
    screen = None
    clock = None
    if not args.headless:
        import pygame as pygame_module

        pygame = pygame_module
        pygame.init()
        screen = pygame.display.set_mode(
            (args.grid * args.scale, args.grid * args.scale)
        )
        pygame.display.set_caption("SymPy fluid · repository SSA · LLVM")
        clock = pygame.time.Clock()

    frame = 0
    running = True
    try:
        while running and (args.frames == 0 or frame < args.frames):
            if pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
            if not running:
                break
            attempts: list[dict] = []
            advanced, dt, metrics = run_superstep(
                state,
                args.frame_duration,
                dt,
                state.dx,
                targets,
                controller,
                advance,
                max_iters=256,
                attempt_log=attempts,
            )
            rejected = sum(not item["accepted"] for item in attempts)
            print(
                f"frame={frame:05d} advanced={float(advanced):.8g} "
                f"dt_next={float(dt):.8g} attempts={len(attempts)} "
                f"rejected={rejected} mass_error={metrics.mass_err:.3e} "
                f"wave_speed={state.last_wave_speed:.6g}"
            )
            if pygame is not None:
                pixels = np.ascontiguousarray((_rgb(state) * 255.0).astype(np.uint8))
                surface = pygame.surfarray.make_surface(np.swapaxes(pixels, 0, 1))
                screen.blit(
                    pygame.transform.scale(surface, screen.get_size()), (0, 0)
                )
                pygame.display.flip()
                clock.tick(60)
            frame += 1
    finally:
        if pygame is not None:
            pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
