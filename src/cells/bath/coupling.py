"""Lightweight coupling between Bath (0D) and stateful fluid solvers.

This module provides a minimal adapter that exchanges mass/pressure signals
between the 0D Bath model and spatial fluid engines:

- DiscreteFluid (SPH): supports mass sources/sinks around cell COMs via
  ``apply_sources`` using dM = -rho * dV per cell.
- VoxelMACFluid (MAC): no mass injection (incompressible); we only sample
  pressure for diagnostics.

The adapter is intentionally conservative and side-effect free on the Bath
physics aside from optionally updating its ``pressure`` for observers.  It
does not alter ion content or osmotic computations; those remain inside the
cellsim API.  For now, solute sources are set to zero.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Sequence

import numpy as np


FluidKind = Literal["discrete", "voxel"]


@dataclass
class BathFluidCoupler:
    """Exchange mass/pressure between Bath and a spatial fluid engine.

    Parameters
    ----------
    bath:
        The 0D Bath object (holds density/pressure/viscosity diagnostics).
    engine:
        A fluid engine instance.  Supported kinds:
          - "discrete": must implement ``apply_sources`` and ``step``.
          - "voxel": must implement ``step`` and ``sample_at``.
    kind:
        Fluid kind selector: "discrete" or "voxel".
    radius:
        Source influence radius in world units when applying mass sources
        (used by discrete/SPH coupling).  Ignored for voxel.
    density_hint:
        Fallback density (kg/m^3) to convert volumes to mass; if None, uses
        ``bath.density``.
    """

    bath: object
    engine: object
    kind: FluidKind
    radius: float = 0.05
    density_hint: Optional[float] = None

    # Internal state: previous volumes snapshot to compute dV
    _prev_vols: Optional[np.ndarray] = None

    def prime_volumes(self, vols: np.ndarray) -> None:
        self._prev_vols = np.array(vols, dtype=float, copy=True)

    def exchange(self, *, dt: float, centers: Optional[np.ndarray], vols: Optional[np.ndarray]) -> None:
        """Apply sources based on cell dV and advance the fluid engine.

        This method is safe to call even if geometry or volume arrays are
        temporarily unavailable; it will simply advance the engine if present.
        """

        # Discrete: convert per-cell volume changes to mass sources around COMs
        if centers is not None and vols is not None and self._prev_vols is not None:
            try:
                dV = np.asarray(vols, dtype=float) - np.asarray(self._prev_vols, dtype=float)
                self._prev_vols = np.asarray(vols, dtype=float)
            except Exception:
                dV = None
            if dV is not None and self.kind == "discrete":
                rho = float(self.bath.density) if getattr(self.bath, "density", None) is not None else float(self.density_hint or 1000.0)
                dM = -rho * dV.astype(float)  # bath loses volume -> remove mass from fluid (negative dM)
                # solute mass changes unknown; set to zero for now
                dS_mass = np.zeros_like(dM, dtype=float)
                centers = np.asarray(centers, dtype=float)
                if centers.ndim == 2 and centers.shape[0] == dM.shape[0] and centers.shape[1] >= 3:
                    # apply mass sources within radius
                    if hasattr(self.engine, "apply_sources"):
                        try:
                            self.engine.apply_sources(centers[:, :3], dM, dS_mass, radius=float(self.radius))
                        except Exception:
                            # Non-fatal: continue stepping
                            pass

        # Advance fluid engine
        try:
            if hasattr(self.engine, "step"):
                self.engine.step(float(dt))
        except Exception:
            pass

        # Sample pressure for diagnostics and update bath.pressure as a gentle average
        try:
            if centers is not None and hasattr(self.engine, "sample_at"):
                centers = np.asarray(centers, dtype=float)
                if centers.ndim == 2 and centers.shape[1] >= 3 and centers.shape[0] > 0:
                    samp = self.engine.sample_at(centers[:, :3])
                    P = np.asarray(samp.get("P", None))
                    if P is not None and P.size:
                        pmean = float(np.nanmean(P))
                        # Low-pass blend to avoid abrupt jumps
                        if hasattr(self.bath, "pressure"):
                            self.bath.pressure = 0.9 * float(self.bath.pressure) + 0.1 * pmean
        except Exception:
            # diagnostics only; ignore failures
            pass


__all__ = ["BathFluidCoupler", "FluidKind"]
