
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

    # softbody/engine/hierarchy.py
    def project_constraints(self, dt):
        if not self.cells:
            return

        sizes   = [len(c.X) for c in self.cells]
        offsets = np.cumsum([0] + sizes)
        X       = np.vstack([c.X   for c in self.cells])
        invm    = np.concatenate([c.invm for c in self.cells])

        # ---- Batch stretch with offsets
        st_idx, st_rest, st_comp, st_lamb = [], [], [], []
        # ---- Batch bending with offsets
        bn_idx, bn_rest, bn_comp, bn_lamb = [], [], [], []

        for off, c in zip(offsets[:-1], self.cells):
            s = c.constraints.get("stretch")
            if s is not None and len(s["indices"]):
                st_idx.append(s["indices"] + off)                # <— offset
                st_rest.append(s["rest"])
                st_comp.append(s["compliance"])
                st_lamb.append(s["lamb"])

            b = c.constraints.get("bending")
            if b is not None and len(b["indices"]):
                bn_idx.append(b["indices"] + off)                # <— offset
                bn_rest.append(b["rest"])
                bn_comp.append(b["compliance"])
                bn_lamb.append(b["lamb"])

        cons = {}
        if st_idx:
            cons["stretch"] = {
                "indices":     np.vstack(st_idx),
                "rest":        np.concatenate(st_rest),
                "compliance":  np.concatenate(st_comp),
                "lamb":        np.concatenate(st_lamb),
            }
        if bn_idx:
            cons["bending"] = {
                "indices":     np.vstack(bn_idx),
                "rest":        np.concatenate(bn_rest),
                "compliance":  np.concatenate(bn_comp),
                "lamb":        np.concatenate(bn_lamb),
            }

        # Faces are per-cell for volume; don’t try to batch volume unless you
        # also build a batched face array + per-constraint face ranges.
        faces_dummy = np.empty((0, 3), dtype=np.int32)
        self.solver.project(
            cons, X, invm, faces_dummy,  # faces unused for stretch/bending
            mesh_volume, volume_gradients,
            dt, self.params.iterations, self
        )

        # Scatter back
        for c, a, b in zip(self.cells, offsets[:-1], offsets[1:]):
            c.X[:] = X[a:b]

        # Do volume per cell explicitly (no coupling, correct faces)
        for c in self.cells:
            vc = c.constraints.get("volume")
            if vc is not None:
                vc.project(c.X, c.invm, c.faces, mesh_volume, volume_gradients, dt)


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
    def step(self, dt: float, t: float = 0.0, fields=None):
        """XPBD step: predict -> (optional fields) -> project -> rebuild V"""
        if not self.cells or dt <= 0.0:
            return

        # 0) save previous positions for velocity rebuild
        for c in self.cells:
            # allocate once if you want; keeping it simple:
            c.X_prev = c.X.copy()

        # 1) accelerations (fields) BEFORE prediction (optional)
        # convention: fields with units='accel' do v += a*dt here
        if fields is not None:
            try:
                fields.apply(self.cells, dt, t, self, stage="prepredict")
            except Exception:
                pass  # tolerate absence / different API

        # 2) predictor (damps V, advances X)
        self.integrate(dt)

        # 3) advection/noise AFTER prediction (optional)
        # convention: fields with units in {'velocity','displacement'} adjust X here
        if fields is not None:
            try:
                self.dt = dt  # so noise fields can read world.dt if they want
                fields.apply(self.cells, dt, t, self, stage="postpredict")
            except Exception:
                pass

        # 4) constraints (stretch/bend/contacts + per-cell volume)
        self.project_constraints(dt)

        # 5) rebuild velocities from corrected positions
        inv_dt = 1.0 / dt
        for c in self.cells:
            c.V[:] = (c.X - c.X_prev) * inv_dt

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

