"""Link the pure-SymPy fluid stencil into the repository managed-dt system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..common.dt_system.dt_controller import STController, Targets, run_superstep
from ..common.dt_system.dt_scaler import Metrics
from ..common.tensors.accelerator_backends.aot_compile import (
    AOTCompilation,
    compile_ast_aot,
)
from .symbolic_fluid_model import compile_symbolic_fluid_step


@dataclass
class SymbolicFluidGridState:
    """Transaction-safe grid storage consumed by managed ``dt_system``."""

    height: np.ndarray
    momentum_x: np.ndarray
    momentum_y: np.ndarray
    tracer: np.ndarray
    next_height: np.ndarray
    next_momentum_x: np.ndarray
    next_momentum_y: np.ndarray
    next_tracer: np.ndarray
    force_x: np.ndarray
    force_y: np.ndarray
    dx: float
    gravity: float
    viscosity: float
    tracer_diffusivity: float
    linear_drag: float
    coriolis: float
    minimum_height: float
    last_wave_speed: float = 0.0
    last_height_violation: float = 0.0
    last_tracer_violation: float = 0.0

    @classmethod
    def initial(cls, width: int = 24, height: int = 24) -> "SymbolicFluidGridState":
        if width < 4 or height < 4:
            raise ValueError("symbolic fluid grid dimensions must be at least four")
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        xx = (xx + 0.5) / float(width)
        yy = (yy + 0.5) / float(height)
        wave = 0.12 * np.exp(-((xx - 0.30) ** 2 + (yy - 0.50) ** 2) / 0.008)
        dye = 0.8 * np.exp(-((xx - 0.68) ** 2 + (yy - 0.50) ** 2) / 0.012)
        base = np.ones((height, width), dtype=np.float64)
        zeros = np.zeros_like(base)
        # A weak counter-clockwise seed current makes interaction between the
        # pressure wave and transported tracer immediately visible.
        vortex_x = -0.08 * (yy - 0.5) * np.exp(
            -((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.08
        )
        vortex_y = 0.08 * (xx - 0.5) * np.exp(
            -((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.08
        )
        h = base + wave
        return cls(
            height=h,
            momentum_x=h * vortex_x,
            momentum_y=h * vortex_y,
            tracer=h * dye,
            next_height=zeros.copy(),
            next_momentum_x=zeros.copy(),
            next_momentum_y=zeros.copy(),
            next_tracer=zeros.copy(),
            force_x=zeros.copy(),
            force_y=zeros.copy(),
            dx=1.0 / float(max(width, height)),
            gravity=1.0,
            viscosity=2.0e-4,
            tracer_diffusivity=1.0e-4,
            linear_drag=2.0e-3,
            coriolis=0.08,
            minimum_height=1.0e-4,
        )

    def copy_shallow(self):
        return (
            self.height.copy(), self.momentum_x.copy(), self.momentum_y.copy(),
            self.tracer.copy(), self.next_height.copy(),
            self.next_momentum_x.copy(), self.next_momentum_y.copy(),
            self.next_tracer.copy(), float(self.last_wave_speed),
            float(self.last_height_violation), float(self.last_tracer_violation),
        )

    def restore(self, snapshot) -> None:
        (
            self.height, self.momentum_x, self.momentum_y, self.tracer,
            self.next_height, self.next_momentum_x, self.next_momentum_y,
            self.next_tracer, self.last_wave_speed,
            self.last_height_violation, self.last_tracer_violation,
        ) = snapshot


SYMBOLIC_FLUID_DT_SOURCE = """
def symbolic_fluid_advance(state, dt):
    previous_mass = 0.0
    next_mass = 0.0
    max_wave_speed = 0.0
    max_height_violation = 0.0
    max_tracer_violation = 0.0
    height_count = state.height.shape[0]
    width_count = state.height.shape[1]
    for row in range(height_count):
        north = (row - 1) % height_count
        south = (row + 1) % height_count
        for column in range(width_count):
            west = (column - 1) % width_count
            east = (column + 1) % width_count
            previous_mass = previous_mass + state.height[row, column]
            (
                height_next,
                momentum_x_next,
                momentum_y_next,
                tracer_next,
                velocity_x,
                velocity_y,
                vorticity,
                speed,
                wave_speed,
                height_violation,
                tracer_violation,
            ) = symbolic_fluid_step(
                state.coriolis,
                dt,
                state.dx,
                state.force_x[row, column],
                state.force_y[row, column],
                state.gravity,
                state.height[row, column],
                state.height[row, east],
                state.height[north, column],
                state.height[south, column],
                state.height[row, west],
                state.linear_drag,
                state.minimum_height,
                state.momentum_x[row, column],
                state.momentum_x[row, east],
                state.momentum_x[north, column],
                state.momentum_x[south, column],
                state.momentum_x[row, west],
                state.momentum_y[row, column],
                state.momentum_y[row, east],
                state.momentum_y[north, column],
                state.momentum_y[south, column],
                state.momentum_y[row, west],
                state.tracer[row, column],
                state.tracer_diffusivity,
                state.tracer[row, east],
                state.tracer[north, column],
                state.tracer[south, column],
                state.tracer[row, west],
                state.viscosity,
            )
            state.next_height[row, column] = height_next
            state.next_momentum_x[row, column] = momentum_x_next
            state.next_momentum_y[row, column] = momentum_y_next
            state.next_tracer[row, column] = tracer_next
            next_mass = next_mass + height_next
            max_wave_speed = max(max_wave_speed, wave_speed)
            max_height_violation = max(max_height_violation, height_violation)
            max_tracer_violation = max(max_tracer_violation, tracer_violation)
    state.height = state.next_height + 0.0
    state.momentum_x = state.next_momentum_x + 0.0
    state.momentum_y = state.next_momentum_y + 0.0
    state.tracer = state.next_tracer + 0.0
    mass_error = abs(next_mass - previous_mass) / max(abs(previous_mass), 1.0e-30)
    state.last_wave_speed = max_wave_speed
    state.last_height_violation = max_height_violation
    state.last_tracer_violation = max_tracer_violation
    metrics = Metrics(
        max_vel=max_wave_speed,
        max_flux=max_wave_speed,
        div_inf=0.0,
        mass_err=mass_error,
        dt_limit=0.45 * state.dx / max(max_wave_speed, 1.0e-30),
        error_channels={
            "height_positivity": max_height_violation,
            "tracer_bounds": max_tracer_violation,
        },
    )
    return max_height_violation == 0.0 and max_tracer_violation == 0.0, metrics


def symbolic_fluid_frame(state, targets, controller, frame_duration, dt_initial):
    advanced, dt_next, metrics = run_superstep(
        state,
        frame_duration,
        dt_initial,
        state.dx,
        targets,
        controller,
        symbolic_fluid_advance,
        max_iters=256,
    )
    return (
        state.height,
        state.momentum_x,
        state.momentum_y,
        state.tracer,
        dt_next,
        state.last_wave_speed,
        state.last_height_violation,
        state.last_tracer_violation,
    )
"""


def capture_symbolic_fluid_dt_program(
    state: SymbolicFluidGridState,
    *,
    frame_duration: float = 1.0 / 30.0,
    dt_initial: float = 1.0e-3,
    checkpoint: bool | str = False,
    progress: Any = None,
) -> AOTCompilation:
    """Capture real managed-time control with the SymPy graph linked in."""

    symbolic = compile_symbolic_fluid_step()
    return compile_ast_aot(
        SYMBOLIC_FLUID_DT_SOURCE,
        "symbolic_fluid_frame",
        {
            "state": state,
            "targets": Targets(
                cfl=0.45,
                div_max=1.0,
                mass_max=1.0e-8,
                error_limits={
                    "height_positivity": 0.0,
                    "tracer_bounds": 0.0,
                },
            ),
            "controller": STController(dt_min=1.0e-8, dt_max=frame_duration),
            "frame_duration": float(frame_duration),
            "dt_initial": float(dt_initial),
        },
        precompile_only=True,
        python_bindings={
            "Metrics": Metrics,
            "run_superstep": run_superstep,
        },
        mutable_parameters=("dt_initial",),
        linked_process_graphs={
            "symbolic_fluid_step": symbolic.process_graph,
        },
        checkpoint=checkpoint,
        progress=progress,
    )


__all__ = [
    "SYMBOLIC_FLUID_DT_SOURCE",
    "SymbolicFluidGridState",
    "capture_symbolic_fluid_dt_program",
]
