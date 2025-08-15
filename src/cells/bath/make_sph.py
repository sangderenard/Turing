from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, Sequence, Tuple

import numpy as np

from .discrete_fluid import DiscreteFluid, FluidParams


def make_sph(
    *,
    dim: int = 3,
    resolution: Iterable[int] | int = 8,
    viscosity: float = 1.0e-6,
    buoyancy: Tuple[float, float, float] = (0.0, -9.81, 0.0),
) -> SimpleNamespace:
    """Create a weakly compressible SPH fluid configured for demos.

    Parameters
    ----------
    dim:
        Spatial dimension (1, 2 or 3).
    resolution:
        Number of particles per axis.  If an int is supplied it is used for
        all axes.  For ``dim < 3`` the remaining axes are filled with ``1``.
    viscosity:
        Kinematic viscosity (``nu``) in ``m^2/s``.
    buoyancy:
        Gravity vector.

    Returns
    -------
    ``SimpleNamespace``
        Wrapper exposing ``step`` and ``export_*`` helpers used by demos.
    """

    if isinstance(resolution, int):
        res = (resolution,) * dim
    else:
        res = tuple(resolution)
        if len(res) != dim:
            raise ValueError("resolution length must match dim")
    res3 = list(res) + [1] * (3 - dim)
    n_x, n_y, n_z = res3

    # Simple rectangular block of particles similar to demo_dam_break
    h = 0.05
    dx = h * 0.9
    xs = np.arange(n_x) * dx
    ys = np.arange(n_y) * dx
    zs = np.arange(n_z) * dx
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pos[:, 1] += 0.2

    params = FluidParams(
        smoothing_length=h,
        particle_mass=0.02,
        viscosity_nu=viscosity,
        gravity=buoyancy,
        bounce_damping=0.2,
    )
    engine = DiscreteFluid(
        pos,
        velocities=None,
        temperature=None,
        salinity=None,
        params=params,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(2.0, 2.0, 2.0),
    )

    def export_positions_vectors_droplets():
        pts, vec = engine.export_positions_vectors()
        drops = engine.droplet_p.copy()
        return pts, vec, drops

    return SimpleNamespace(
        engine=engine,
        step=engine.step,
        step_with_controller=engine.step_with_controller,
        export_positions_vectors=engine.export_positions_vectors,
        export_vertices=engine.export_vertices,
        export_droplets=lambda: engine.droplet_p.copy(),
        export_positions_vectors_droplets=export_positions_vectors_droplets,
        apply_sources=getattr(engine, "apply_sources", None),
        sample_at=getattr(engine, "sample_at", None),
    )


__all__ = ["make_sph"]
