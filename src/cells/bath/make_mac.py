from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, Tuple

from .voxel_fluid import CFL, VoxelMACFluid, VoxelFluidParams


def make_mac(
    *,
    dim: int = 3,
    resolution: Iterable[int] | int = 8,
    viscosity: float = 1.0e-6,
    buoyancy: Tuple[float, float, float] = (0.0, -9.81, 0.0),
    cfl: float = CFL,
) -> SimpleNamespace:
    """Construct a VoxelMACFluid solver wrapped for demos."""

    if isinstance(resolution, int):
        res = (resolution,) * dim
    else:
        res = tuple(resolution)
        if len(res) != dim:
            raise ValueError("resolution length must match dim")
    res3 = list(res) + [1] * (3 - dim)
    nx, ny, nz = res3

    params = VoxelFluidParams(
        nx=nx,
        ny=ny,
        nz=nz,
        nu=viscosity,
        gravity=buoyancy,
        cfl=cfl,
    )
    engine = VoxelMACFluid(params)

    def export_positions_vectors_droplets():
        pts, vec = engine.export_vector_field()
        return pts, vec, None

    return SimpleNamespace(
        engine=engine,
        step=engine.step,
        step_with_controller=engine.step_with_controller,
        export_vector_field=engine.export_vector_field,
        export_positions_vectors=engine.export_vector_field,
        export_droplets=lambda: None,
        export_positions_vectors_droplets=export_positions_vectors_droplets,
        sample_at=getattr(engine, "sample_at", None),
    )


__all__ = ["make_mac"]
