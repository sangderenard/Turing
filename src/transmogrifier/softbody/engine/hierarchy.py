
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
        idx = self.faces
        a = self.X[idx[:, 0]]
        b = self.X[idx[:, 1]]
        c = self.X[idx[:, 2]]
        cross = np.cross(b - a, c - a)
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

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
        if not self.cells:
            return
        X = np.vstack([c.X for c in self.cells])
        V = np.vstack([c.V for c in self.cells])
        invm = np.concatenate([c.invm for c in self.cells])
        offsets = np.cumsum([0] + [len(c.X) for c in self.cells])
        self.solver.integrate(X, V, invm, dt)
        for c, start, end in zip(self.cells, offsets[:-1], offsets[1:]):
            c.X[:] = X[start:end]
            c.V[:] = V[start:end]

    def project_constraints(self, dt):
        if not self.cells:
            return

        sizes = [len(c.X) for c in self.cells]
        offsets = np.cumsum([0] + sizes)
        X = np.vstack([c.X for c in self.cells])
        invm = np.concatenate([c.invm for c in self.cells])

        # Aggregate stretch and bending constraints with index offsets
        stretch_idx = []
        stretch_rest = []
        stretch_comp = []
        stretch_lamb = []
        bend_idx = []
        bend_rest = []
        bend_comp = []
        bend_lamb = []
        for off, c in zip(offsets[:-1], self.cells):
            sc = c.constraints.get("stretch")
            if sc is not None:
                stretch_idx.append(sc["indices"] + off)
                stretch_rest.append(sc["rest"])
                stretch_comp.append(sc["compliance"])
                stretch_lamb.append(sc["lamb"])
            bc = c.constraints.get("bending")
            if bc is not None:
                bend_idx.append(bc["indices"] + off)
                bend_rest.append(bc["rest"])
                bend_comp.append(bc["compliance"])
                bend_lamb.append(bc["lamb"])

        cons = {}
        if stretch_idx:
            cons["stretch"] = {
                "indices": np.vstack(stretch_idx),
                "rest": np.concatenate(stretch_rest),
                "compliance": np.concatenate(stretch_comp),
                "lamb": np.concatenate(stretch_lamb),
            }
        if bend_idx:
            cons["bending"] = {
                "indices": np.vstack(bend_idx),
                "rest": np.concatenate(bend_rest),
                "compliance": np.concatenate(bend_comp),
                "lamb": np.concatenate(bend_lamb),
            }

        faces_dummy = np.empty((0, 3), dtype=np.int32)
        self.solver.project(
            cons, X, invm, faces_dummy, mesh_volume, volume_gradients,
            dt, self.params.iterations, self.box_min, self.box_max
        )

        # Volume constraints handled per cell on the aggregated arrays
        for off, c in zip(offsets[:-1], self.cells):
            vc = c.constraints.get("volume")
            if vc is not None:
                sl = slice(off, off + len(c.X))
                vc.project(X[sl], invm[sl], c.faces, mesh_volume, volume_gradients, dt)

        for c, start, end in zip(self.cells, offsets[:-1], offsets[1:]):
            c.X[:] = X[start:end]

    def update_organelle_modes(self, dt):
        counts = [len(c.organelles) for c in self.cells]
        if sum(counts) == 0:
            return

        centroids = np.array([np.mean(c.X, axis=0) for c in self.cells])
        v_adv = np.array([np.mean(c.V, axis=0) for c in self.cells])

        pos_list = []
        vel_list = []
        alpha_list = []
        cell_idx = []
        for idx, c in enumerate(self.cells):
            n = counts[idx]
            if n == 0:
                continue
            pos_list.append(np.array([o.pos for o in c.organelles], dtype=np.float64))
            vel_list.append(np.array([o.vel for o in c.organelles], dtype=np.float64))
            alpha_list.append(np.clip([o.viscosity for o in c.organelles], 0.0, 1.0))
            cell_idx.extend([idx] * n)

        pos = np.vstack(pos_list)
        vel = np.vstack(vel_list)
        alpha = np.concatenate(alpha_list)[:, None]
        v_adv_full = v_adv[cell_idx]

        vel = (1.0 - alpha) * vel + alpha * v_adv_full
        pos = pos + dt * vel
        pos = np.clip(pos, self.box_min, self.box_max)

        start = 0
        for c in self.cells:
            n = len(c.organelles)
            if n == 0:
                continue
            v_slice = vel[start:start+n]
            p_slice = pos[start:start+n]
            for j, o in enumerate(c.organelles):
                o.vel = v_slice[j]
                o.pos[:] = p_slice[j]
            start += n

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
