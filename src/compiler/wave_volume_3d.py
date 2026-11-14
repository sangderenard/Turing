"""Specialized three-dimensional scalar wave stencil from a SymPy law.

This is a time-domain volume update, not the transverse angular-spectrum T4
propagator. Its seven samples are the center and six axial neighbors of one
voxel. The state owner supplies those samples and handles exterior boundaries;
the compiled law owns the interior arithmetic for each independent voxel.
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp

from .native_package import piece_from_law
from .symbolic_equation_compiler import compile_sympy_equations


def volume_wave_equation() -> tuple[sp.Equality, ...]:
    """Centered 3D wave equation with centered linear damping and source.

    The continuum law is u_tt + damping*u_t = speed**2 * Laplacian(u)
    + source. The denominator comes from centering u_t at the current time;
    no damping or source is hidden in a display transform.
    """
    names = (
        "current", "previous", "x_minus", "x_plus", "y_minus",
        "y_plus", "z_minus", "z_plus", "dx", "dy", "dz", "dt",
        "speed", "damping", "source",
    )
    symbols = {name: sp.Symbol(name, real=True) for name in names}
    v = symbols
    laplacian = (
        (v["x_minus"] - 2*v["current"] + v["x_plus"]) / v["dx"]**2
        + (v["y_minus"] - 2*v["current"] + v["y_plus"]) / v["dy"]**2
        + (v["z_minus"] - 2*v["current"] + v["z_plus"]) / v["dz"]**2
    )
    half_damping_step = v["damping"] * v["dt"] / 2
    next_value = (
        2*v["current"] - (1-half_damping_step)*v["previous"]
        + v["speed"]**2*v["dt"]**2*laplacian
        + v["dt"]**2*v["source"]
    ) / (1+half_damping_step)
    return (sp.Eq(sp.Symbol("next_value"), next_value, evaluate=False),)


def volume_wave_cfl(speed: float, dt: float, dx: float, dy: float,
                    dz: float) -> float:
    """Return the 3D Courant number; interior update requires at most one."""
    if speed < 0 or dt <= 0 or min(dx, dy, dz) <= 0:
        raise ValueError("speed must be nonnegative and spacings/time positive")
    return speed*dt*(dx**-2 + dy**-2 + dz**-2)**0.5


def compile_volume_wave_kernel(batch: int, *, directory: Path | str):
    """Compile the 3D stencil over a batch of independent interior voxels."""
    if batch < 1:
        raise ValueError("volume wave batch must be positive")
    name = "wave_volume_3d_step"
    compiled = compile_sympy_equations(volume_wave_equation(),
                                       name=name, dtype="float64")
    return piece_from_law(compiled, name, batch, directory=directory)


def compile_volume_wave_compute_shader(*, local_size: int = 64):
    """Emit a separate f32 GPU specialization of the same voxel law."""
    from .ssa_spirv_backend import emit_deployable_compute_shader

    equations = volume_wave_equation()
    name = "wave_volume_3d_step_f32"
    compiled = compile_sympy_equations(equations, name=name,
                                       dtype="float32")
    values = {
        instruction.res.id: instruction.res
        for block in compiled.function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    outputs = tuple(values[compiled.output_ids[str(eq.lhs)]]
                    for eq in equations)
    return emit_deployable_compute_shader(
        compiled.function, outputs=outputs, dtype="float32",
        local_size=local_size, name=name, module=compiled.module,
    )


def publish_volume_wave_product(directory: Path | str) -> Path:
    """Publish the validated 3D shader and its existing deployment record."""
    from .shader_deployment import publish_spirv_product

    product = publish_spirv_product(
        Path(directory), compile_volume_wave_compute_shader(),
    )
    if product.module_path is None:
        raise RuntimeError("3D wave volume has no validated SPIR-V product")
    return product.module_path


__all__ = ["volume_wave_equation", "volume_wave_cfl",
           "compile_volume_wave_kernel", "compile_volume_wave_compute_shader",
           "publish_volume_wave_product"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(publish_volume_wave_product(args.output))
