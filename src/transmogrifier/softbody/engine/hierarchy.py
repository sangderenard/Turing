
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .mesh import make_icosphere, mesh_volume, volume_gradients, build_adjacency
from .constraints import VolumeConstraint
from .xpbd_core import XPBDSolver

@dataclass
class Organelle:
    pos: np.ndarray
    radius: float
    viscosity: float  # 0 free, 1 co-moving
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

@dataclass
class Cell:
    id: str
    X: np.ndarray
    V: np.ndarray
    invm: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    bends: np.ndarray
    constraints: dict
    organelles: List[Organelle]
    membrane_tension: float = 0.0
    osmotic_pressure: float = 0.0
    external_pressure: float = 0.0

    def enclosed_volume(self) -> float:
        return mesh_volume(self.X, self.faces)

    def surface_area(self) -> float:
        tris = self.X[self.faces]
        cross = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
        return 0.5 * np.linalg.norm(cross, axis=1).sum()

    def contact_pressure_estimate(self) -> float:
        V = abs(self.enclosed_volume())
        A = max(1e-12, self.surface_area())
        return float(self.membrane_tension) * (A / (V**(2/3))) if V>1e-12 else 0.0

@dataclass
class Hierarchy:
    box_min: np.ndarray
    box_max: np.ndarray
    cells: List[Cell]
    solver: XPBDSolver
    params: object

    def integrate(self, dt):
        for c in self.cells:
            self.solver.integrate(c.X, c.V, c.invm, dt)

    def project_constraints(self, dt):
        for c in self.cells:
            self.solver.project(
                c.constraints, c.X, c.invm, c.faces, mesh_volume, volume_gradients,
                dt, self.params.iterations, self.box_min, self.box_max
            )

    def update_organelle_modes(self, dt):
        for c in self.cells:
            if not c.organelles:
                continue

            v_adv = np.mean(c.V, axis=0)

            pos = np.array([o.pos for o in c.organelles], dtype=np.float64)
            vel = np.array([o.vel for o in c.organelles], dtype=np.float64)
            visc = np.array([o.viscosity for o in c.organelles], dtype=np.float64)[:, None]
            alpha = np.clip(visc, 0.0, 1.0)

            vel = (1.0 - alpha) * vel + alpha * v_adv
            pos = pos + dt * vel
            pos = np.clip(pos, self.box_min, self.box_max)

            for i, o in enumerate(c.organelles):
                o.vel = vel[i]
                o.pos[:] = pos[i]

def build_cell(id_str, center, radius, params, subdiv=1, mass_per_vertex=1.0, target_volume=None):
    X, F = make_icosphere(subdiv=subdiv, radius=radius, center=center)
    V = np.zeros_like(X)
    invm = np.full(X.shape[0], 1.0 / mass_per_vertex, dtype=np.float64)
    edges, bends = build_adjacency(F)

    edges = np.asarray(edges, dtype=np.int32)
    e_rest = np.linalg.norm(X[edges[:, 1]] - X[edges[:, 0]], axis=1)
    stretch = {
        "indices": edges,
        "rest": e_rest,
        "compliance": np.full(len(edges), params.stretch_compliance, dtype=np.float64),
        "lamb": np.zeros(len(edges), dtype=np.float64),
    }

    bends = np.asarray(bends, dtype=np.int32)
    if bends.size:
        a, b, c, d = X[bends[:, 0]], X[bends[:, 1]], X[bends[:, 2]], X[bends[:, 3]]
        n1 = np.cross(c - a, b - a)
        n2 = np.cross(b - d, a - d)
        n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-12
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12
        rest = np.arccos(np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0))
    else:
        rest = np.zeros(0, dtype=np.float64)
    bending = {
        "indices": bends,
        "rest": rest,
        "compliance": np.full(len(bends), params.bending_compliance, dtype=np.float64),
        "lamb": np.zeros(len(bends), dtype=np.float64),
    }

    V0 = abs(mesh_volume(X, F)) if target_volume is None else float(target_volume)
    volc = VolumeConstraint(target=V0, compliance=params.volume_compliance)

    constraints = {"stretch": stretch, "bending": bending, "volume": volc}
    return X, F, V, invm, edges, bends, constraints
