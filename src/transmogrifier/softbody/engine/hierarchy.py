
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .mesh import (
    make_icosphere,
    mesh_volume,
    volume_gradients,
    build_adjacency,
    mesh_area2d,
    area_gradients2d,
    polyline_length,
    polyline_length_gradients,
)
from src.transmogrifier.softbody.geometry.primitives import planar_ngon, line_segment
from .constraints import VolumeConstraint
from .xpbd_core import XPBDSolver
from .fields import FieldStack
from .collisions import project_self_contacts_streamed
from src.transmogrifier.cells.cellsim.membranes.membrane import (
    Membrane, MembraneConfig, MembraneHooks,
)

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
    dim: int = 3

    def enclosed_volume(self) -> float:
        if self.dim == 2:
            return mesh_area2d(self.X, self.faces)
        if self.dim == 1:
            return polyline_length(self.X, self.faces, dim=1)
        return mesh_volume(self.X, self.faces)

    def surface_area(self) -> float:
        if self.dim == 2:
            e = self.edges
            if len(e) == 0:
                return 0.0
            d = self.X[e[:, 1], :2] - self.X[e[:, 0], :2]
            return float(np.linalg.norm(d, axis=1).sum())
        if self.dim == 1:
            return polyline_length(self.X, self.faces, dim=1)
        idx = self.faces
        a = self.X[idx[:, 0]]
        b = self.X[idx[:, 1]]
        c = self.X[idx[:, 2]]
        cross = np.cross(b - a, c - a)
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

    def contact_pressure_estimate(self) -> float:
        V = abs(self.enclosed_volume())
        A = max(1e-12, self.surface_area())
        if self.dim == 2:
            return float(self.membrane_tension) * (A / max(V, 1e-12))
        if self.dim == 1:
            return float(self.membrane_tension) / max(V, 1e-12)
        return float(self.membrane_tension) * (A / (V**(2 / 3))) if V > 1e-12 else 0.0

@dataclass
class Hierarchy:
    box_min: np.ndarray
    box_max: np.ndarray
    cells: List[Cell]
    solver: XPBDSolver
    params: object
    fields: FieldStack = field(default_factory=FieldStack)
    # Membrane surface physics toggle (replaces XPBD stretch+bending)
    use_membrane_surface_physics: bool = True
    membranes: List[Membrane] = field(default_factory=list)

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

    # -- membrane wiring --------------------------------------------------
    def _ensure_membranes(self):
        """Create/refresh one Membrane per cell when enabled."""
        if not self.use_membrane_surface_physics:
            self.membranes = []
            return
        if self.membranes and len(self.membranes) == len(self.cells):
            return
        self.membranes = []
        for c in self.cells:
            # Membrane owns area + bending. Keep XPBD volume for mild correction.
            cfg = MembraneConfig(
                bending_kappa=8e-20,
                area_k_local=2e-1,
                area_k_global=5e-2,
                preferred_shape_k=0.0,
                drag_coefficient=0.0,
                ib_mode="none",
                xpbd_compliance_area_local=0.0,
                xpbd_compliance_area_global=0.0,
                xpbd_compliance_volume=0.0,
            )
            def _deltaP(V, A, t, _c=c):
                pin = float(getattr(_c, "osmotic_pressure", 0.0) + getattr(_c, "internal_pressure", 0.0))
                pext = float(getattr(_c, "external_pressure", 0.0))
                return pin - pext
            hooks = MembraneHooks(deltaP=_deltaP)
            self.membranes.append(Membrane(F=c.faces, X0=c.X.copy(), cfg=cfg, hooks=hooks))

    # softbody/engine/hierarchy.py
    def project_constraints(self, dt):
        if not self.cells:
            return

        sizes = [len(c.X) for c in self.cells]
        offsets = np.cumsum([0] + sizes)
        X = np.vstack([c.X for c in self.cells])
        invm = np.concatenate([c.invm for c in self.cells])

        # ---- Batch stretch/bending only if membrane physics is OFF
        st_idx, st_rest, st_comp, st_lamb = [], [], [], []
        bn_idx, bn_rest, bn_comp, bn_lamb = [], [], [], []

        for off, c in zip(offsets[:-1], self.cells):
            s = None
            b = None
            if not self.use_membrane_surface_physics:
                s = c.constraints.get("stretch")
                b = c.constraints.get("bending")
            if s is not None and len(s["indices"]):
                st_idx.append(s["indices"] + off)                # <— offset
                st_rest.append(s["rest"])
                st_comp.append(s["compliance"])
                st_lamb.append(s["lamb"])
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

        dim = getattr(self.params, "dimension", 3)
        cell_ids = np.concatenate(
            [np.full(len(c.X), i, dtype=np.int32) for i, c in enumerate(self.cells)]
        )
        if dim == 1:
            all_faces = (
                np.vstack([c.faces + off for c, off in zip(self.cells, offsets[:-1])])
                if self.cells
                else np.empty((0, 2), dtype=np.int32)
            )
            vol_func = lambda X, f: polyline_length(X, f, dim=1)
            vol_grads = lambda X, f: polyline_length_gradients(X, f, dim=1)
        elif dim == 2:
            all_faces = (
                np.vstack([c.faces + off for c, off in zip(self.cells, offsets[:-1])])
                if self.cells
                else np.empty((0, 3), dtype=np.int32)
            )
            vol_func = mesh_area2d
            vol_grads = area_gradients2d
        else:
            all_faces = (
                np.vstack([c.faces + off for c, off in zip(self.cells, offsets[:-1])])
                if self.cells
                else np.empty((0, 3), dtype=np.int32)
            )
            vol_func = mesh_volume
            vol_grads = volume_gradients

        # XPBDSolver.project expects explicit bounding box limits.  The
        # previous call passed ``self`` which was interpreted as ``box_min``
        # and left ``box_max`` unset, effectively turning the projection step
        # into a no-op and preventing velocities from being rebuilt from the
        # advected positions.  Supplying both limits ensures constraint
        # projection (including contact with the world box) runs correctly and
        # produces non-zero velocities when fields like uniform_flow are
        # applied.
        self.solver.project(
            cons,
            X,
            invm,
            all_faces,
            vol_func,
            vol_grads,
            dt,
            self.params.iterations,
            self.box_min,
            self.box_max,
        )

        if getattr(self.params, "enable_self_contacts", True):
            project_self_contacts_streamed(
                X,
                all_faces,
                invm,
                cell_ids,
                self.params.contact_voxel_size,
                dt,
                min_separation=getattr(self.params, "min_separation", 0.0),
                compliance=self.params.contact_compliance,
                iters=2,
                max_vox_entries=self.params.contact_max_vox_entries,
                vbatch=self.params.contact_vbatch,
                ram_limit_bytes=self.params.contact_ram_limit,
                adjacency="self",
            )

        # Scatter back
        for c, a, b in zip(self.cells, offsets[:-1], offsets[1:]):
            c.X[:] = X[a:b]

        # Do volume per cell explicitly (no coupling, correct faces)
        for c in self.cells:
            vc = c.constraints.get("volume")
            if vc is not None:
                topo = c.faces
                vc.project(c.X, c.invm, topo, vol_func, vol_grads, dt)

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

        # 1.5) membrane surface forces (area + bending + pressure) -> accelerations
        if self.use_membrane_surface_physics:
            self._ensure_membranes()
            for c, m in zip(self.cells, self.membranes):
                F_mem, parts, geom = m.forces(c.X, c.V, t)
                # a = F/m = F * invm
                c.V[:] += (F_mem * c.invm[:, None]) * dt
                c._measured_volume = geom["V"]
                c._measured_area = geom["A_total"]

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
    dim = getattr(params, "dimension", 3)
    if dim == 2:
        n = int(8 * (2 ** max(0, int(subdiv))))
        X, F, edges = planar_ngon(n_segments=n, radius=radius, center=center, plane="xy", dtype=np.float64)
        # 2D membranes don't use dihedral bending; keep empty bends
        bends = np.zeros((0, 4), dtype=np.int32)
        V0 = mesh_area2d(X, F) if target_volume is None else float(target_volume)
    elif dim == 1:
        n = int(8 * (2 ** max(0, int(subdiv))))
        X, edges = line_segment(n_segments=n, radius=radius, center=center, axis="x", dtype=np.float64)
        F = edges.copy()
        bends = np.zeros((0, 4), dtype=np.int32)
        V0 = polyline_length(X, F, dim=1) if target_volume is None else float(target_volume)
    else:
        X, F = make_icosphere(subdiv=subdiv, radius=radius, center=center)
        edges, bends = build_adjacency(F)
        V0 = abs(mesh_volume(X, F)) if target_volume is None else float(target_volume)

    V = np.zeros_like(X)
    invm = np.full(X.shape[0], 1.0 / mass_per_vertex, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32)
    e_rest = np.linalg.norm(X[edges[:, 1]] - X[edges[:, 0]], axis=1)
    stretch = {
        "indices": edges,
        "rest": e_rest,
        "compliance": np.full(len(edges), params.stretch_compliance, dtype=np.float64),
        "lamb": np.zeros(len(edges), dtype=np.float64),
    }

    bends = np.asarray(bends, dtype=np.int32)
    if dim == 3 and bends.size:
        a, b, c, d = X[bends[:, 0]], X[bends[:, 1]], X[bends[:, 2]], X[bends[:, 3]]
        n1 = np.cross(c - a, b - a)
        n2 = np.cross(b - d, a - d)
        n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-12
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12
        rest = np.arccos(np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0))
    else:
        rest = np.zeros(len(bends), dtype=np.float64)
    bending = {
        "indices": bends,
        "rest": rest,
        "compliance": np.full(len(bends), params.bending_compliance, dtype=np.float64),
        "lamb": np.zeros(len(bends), dtype=np.float64),
    }

    volc = VolumeConstraint(target=V0, compliance=params.volume_compliance)
    constraints = {"stretch": stretch, "bending": bending, "volume": volc}
    return X, F, V, invm, edges, bends, constraints

