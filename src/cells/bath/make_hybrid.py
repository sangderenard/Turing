from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, Tuple

import numpy as np

from .hybrid_fluid import HybridFluid, HybridParams


def make_hybrid(
    *,
    dim: int = 3,
    resolution: Iterable[int] | int = 8,
    n_particles: int = 512,
    viscosity: float = 1.0e-6,
    buoyancy: Tuple[float, float, float] = (0.0, -9.81, 0.0),
    phi_condense: float = 0.85,
    phi_shatter: float = 0.25,
    p_shatter_max: float = 0.0,
) -> SimpleNamespace:
    """Create a HybridFluid solver (particle+grid) for demos."""

    if isinstance(resolution, int):
        res = (resolution,) * dim
    else:
        res = tuple(resolution)
        if len(res) != dim:
            raise ValueError("resolution length must match dim")

    params = HybridParams(
        nu=viscosity,
        gravity=buoyancy,
        phi_condense=phi_condense,
        phi_shatter=phi_shatter,
        p_shatter_max=p_shatter_max,
    )
    engine = HybridFluid(shape=res, n_particles=n_particles, params=params)

    # Seed a block occupying roughly half the domain for quick visual tests
    try:  # pragma: no cover - seeding not critical for unit tests
        hi = tuple(r * params.dx * 0.5 for r in res)
        lo = tuple(0.0 for _ in res)
        engine.seed_block(lo, hi, mode="both")
    except Exception:
        pass

    def export_positions_vectors_droplets():
        parts = engine.export_particles()
        grid_pts, grid_vecs = engine.export_vector_field()

        # ``export_vector_field`` always returns 3D points/vectors even when the
        # simulation is 1D or 2D.  Slice them so they match the particle
        # dimensionality before concatenation.
        dim = parts["x"].shape[1]
        if grid_pts.shape[1] != dim:
            grid_pts = grid_pts[:, :dim]
            grid_vecs = grid_vecs[:, :dim]

        pts = np.concatenate([parts["x"], grid_pts], axis=0)
        vecs = np.concatenate([parts["v"], grid_vecs], axis=0)
        return pts, vecs, None

    return SimpleNamespace(
        engine=engine,
        step=engine.step,
        step_with_controller=engine.step_with_controller,
        export_particles=engine.export_particles,
        export_vector_field=engine.export_vector_field,
        export_positions_vectors=lambda: (
            engine.export_particles()["x"], engine.export_particles()["v"]
        ),
        export_droplets=lambda: None,
        export_positions_vectors_droplets=export_positions_vectors_droplets,
        sample_at=getattr(engine, "sample_at", None),
        apply_sources=getattr(engine, "apply_sources", None),
    )


__all__ = ["make_hybrid"]
