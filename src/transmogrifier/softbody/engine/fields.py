# fields.py
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    # Only imported for type checking to avoid runtime cycles
    from .hierarchy import Cell, Hierarchy

# Vector field function: (X, t, cell, world) -> per-vertex vectors
VecFn = Callable[[np.ndarray, float, "Cell", "Hierarchy"], np.ndarray]

@dataclass
class VectorField:
    """A field that contributes either acceleration, velocity, or direct Δx."""
    fn:      VecFn
    units:   str = "velocity"   # 'accel' | 'velocity' | 'displacement'
    selector: Optional[Callable[["Cell"], bool]] = None
    dim:     int = 3
    com_neutral: bool = False   # only used by stochastic fields you define

    def active(self, cell) -> bool:
        return True if self.selector is None else bool(self.selector(cell))

class FieldStack:
    def __init__(self, fields: Optional[Sequence[VectorField]] = None):
        self.fields = list(fields) if fields else []

    def add(self, field: VectorField):
        self.fields.append(field)

    def apply(self, cells, dt: float, t: float, world, stage: str):
        """
        stage: 'prepredict' (update V via accel), 'postpredict' (advect X via velocity/displacement)
        """
        for f in self.fields:
            if stage == "prepredict" and f.units != "accel":
                continue
            if stage == "postpredict" and f.units == "accel":
                continue

            for c in cells:
                if not f.active(c):
                    continue
                X = c.X  # (N, D)
                D = X.shape[1]
                assert D == f.dim, f"Field dim {f.dim} != cell dim {D}"

                val = f.fn(X, t, c, world)  # shape (N,D) or (D,)
                if val.ndim == 1:
                    val = np.broadcast_to(val, X.shape)

                # Respect pins/infinite mass
                if getattr(c, "invm", None) is not None:
                    mask = (c.invm > 0).astype(X.dtype)[:, None]
                else:
                    mask = 1.0

                if f.units == "accel":
                    # Symplectic Euler: v += a*dt
                    c.V += (val * dt) * mask
                elif f.units == "velocity":
                    # Advect positions by u*dt; velocities will be rebuilt from X later
                    c.X += (val * dt) * mask
                elif f.units == "displacement":
                    # Direct Δx (e.g., Brownian kick already scaled)
                    c.X += (val) * mask
                else:
                    raise ValueError(f"Unknown units: {f.units}")
