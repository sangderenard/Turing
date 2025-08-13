
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from .mesh import make_icosphere, mesh_volume, volume_gradients, build_adjacency
from .constraints import StretchConstraint, VolumeConstraint, DihedralBendingConstraint
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
    edges: List[Tuple[int,int]]
    bends: List[Tuple[int,int,int,int]]
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
            centroid = np.mean(c.X, axis=0)
            v_adv = np.mean(c.V, axis=0)
            for o in c.organelles:
                alpha = float(np.clip(o.viscosity, 0.0, 1.0))
                o.vel = (1-alpha)*o.vel + alpha*v_adv
                o.pos += dt * o.vel
                o.pos[:] = np.minimum(np.maximum(o.pos, self.box_min), self.box_max)

def build_cell(id_str, center, radius, params, subdiv=1, mass_per_vertex=1.0, target_volume=None):
    X, F = make_icosphere(subdiv=subdiv, radius=radius, center=center)
    V = np.zeros_like(X)
    invm = np.full(X.shape[0], 1.0/mass_per_vertex, dtype=np.float64)
    edges, bends = build_adjacency(F)

    stretch = []
    for (i,j) in edges:
        rest = np.linalg.norm(X[j]-X[i])
        stretch.append(StretchConstraint(i=i,j=j,rest=rest,compliance=params.stretch_compliance))

    bending = []
    for (i,j,k,l) in bends:
        a,b,c,d = X[i],X[j],X[k],X[l]
        n1 = np.cross(c-a, b-a); n2 = np.cross(b-d, a-d)
        n1/= (np.linalg.norm(n1)+1e-12); n2/= (np.linalg.norm(n2)+1e-12)
        rest = float(np.arccos(np.clip(np.dot(n1,n2), -1.0, 1.0)))
        bending.append(DihedralBendingConstraint(i=i,j=j,k=k,l=l,rest_angle=rest,compliance=params.bending_compliance))

    V0 = abs(mesh_volume(X, F)) if target_volume is None else float(target_volume)
    volc = VolumeConstraint(target=V0, compliance=params.volume_compliance)

    constraints = {"stretch": stretch, "bending": bending, "volume": volc}
    return X, F, V, invm, edges, bends, constraints
