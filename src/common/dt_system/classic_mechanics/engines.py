# -*- coding: utf-8 -*-
"""Classic mechanics demo engines for dt graph orchestration.

Engines (all DtCompatibleEngine):
- GravityEngine: applies constant gravity to vertices
- ThrustersEngine: applies 4-direction thrust from COM
- SpringEngine: evaluates Hooke forces for springs between vertices
- PneumaticDamperEngine: idealized gas cylinder-like directional damping per spring
- GroundCollisionEngine: simple predictive/elastic+friction ground correction
- IntegratorEngine: integrates acceleration to velocity and position
- MetaCollisionEngine: semi-elastic friction collision system that consults
    spring + damper networks and integrates, supporting ground and inter-craft
    collisions. Designed to be used with the dt lookahead solver.

Each engine reads/writes from a shared state dict to keep the demo simple.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING, Callable, Optional
import math

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine
from ..debug import dbg, is_enabled
if TYPE_CHECKING:  # typing only to avoid hard dependency
    from ..solids.api import SolidRegistry, WorldConfinement, WorldPlane, SurfaceMaterial
from ..solids.api import GLOBAL_SOLIDS, GLOBAL_WORLD, MATERIAL_ELASTIC, MATERIAL_SOIL

Vec = Tuple[float, float]


def v_add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def v_scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s)


def v_len(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def v_norm(a: Vec) -> Vec:
    l = v_len(a) or 1.0
    return (a[0] / l, a[1] / l)


@dataclass
class DemoState:
    pos: List[Vec]
    vel: List[Vec]
    acc: List[Vec]
    mass: List[float]
    # edges as pairs of vertex indices
    springs: List[Tuple[int, int]]
    rest_len: Dict[Tuple[int, int], float]
    k_spring: Dict[Tuple[int, int], float]
    # pneumatics damping (dir-dependent): (along, against)
    pneu_damp: Dict[Tuple[int, int], Tuple[float, float]]
    # ground y=0
    ground_k: float = 1000.0
    ground_b: float = 10.0
    mu: float = 0.3
    g: float = 9.81
    # Entropic/efficiency factors (0..1 for efficiencies); drag in 1/s
    spring_eff: float = 1.0
    thruster_eff: float = 1.0
    pneumatic_eff: float = 1.0
    linear_drag: float = 0.0

    # Optional snapshot/restore for bisect solver compatibility
    def snapshot(self):  # pragma: no cover - lightweight
        return {
            "pos": list(self.pos),
            "vel": list(self.vel),
            "acc": list(self.acc),
        }

    def restore(self, snap) -> None:  # pragma: no cover - lightweight
        try:
            self.pos = list(snap["pos"])  # type: ignore[index]
            self.vel = list(snap["vel"])  # type: ignore[index]
            self.acc = list(snap["acc"])  # type: ignore[index]
        except Exception:
            pass


@dataclass
class Contact:
    """Lightweight contact report for softbody harmonization stubs."""
    p: Vec
    n: Vec
    pen: float
    material_kind: str
    state_idx: int
    vertex_idx: int
    source: str  # 'plane' | 'solid'


class GravityEngine(DtCompatibleEngine):

    def get_state(self, state_table=None):
        # Return a dict of UUIDs to identity objects (pos, mass, acc, etc.)
        if state_table is None:
            return {}
        return {uuid: dict(identity) for uuid, identity in state_table.identity_registry.items()}

    def __init__(self, state: DemoState, state_table=None, group_label=None, uuids=None, dedup: bool = True):
        self.s = state
        self.state_table = state_table
        self.group_label = group_label or "gravity_group"
        self.uuids = uuids or []
        if self.state_table is not None:
            self.uuids = []
            for i, (pos, mass) in enumerate(zip(self.s.pos, self.s.mass)):
                uuid_str = self.state_table.register_identity(pos, mass, dedup=dedup)
                self.uuids.append(uuid_str)
            self.state_table.register_group(self.group_label, set(self.uuids))

    # Snapshot proxies (optional)

    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float, state, state_table):
        # Use the state_table identity registry for all updates
        if is_enabled():
            dbg("eng.gravity").debug(f"dt={float(dt):.6g}")
        if state_table is None:
            state_table = self.state_table
        if state_table is None:
            raise ValueError("GravityEngine requires a StateTable for identity-based state management.")
        for uuid in self.uuids:
            identity = state_table.get_identity(uuid)
            if identity is None:
                continue
            mass = identity.get('mass', 0.0)
            if mass <= 0:
                continue
            acc = identity.get('acc', (0.0, 0.0))
            acc = v_add(acc, (0.0, -self.s.g))
            state_table.update_identity(uuid, pos=identity.get('pos'), mass=mass)
            # Also update acceleration in the identity object
            identity['acc'] = acc
        metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)
        return True, metrics, self.get_state(state_table)


class ThrustersEngine(DtCompatibleEngine):

    def get_state(self, state_table=None):
        if state_table is None:
            raise ValueError("SpringEngine requires a StateTable for identity-based state management.")
        return {uuid: dict(identity) for uuid, identity in state_table.identity_registry.items()}

    def __init__(self, state: DemoState, thrust: Vec = (0.0, 0.0), state_table=None, group_label=None, uuids=None, dedup: bool = True):
        self.s = state
        self.thrust = thrust
        self.state_table = state_table
        self.group_label = group_label or "thrusters_group"
        self.uuids = uuids or []
        if self.state_table is not None:
            self.uuids = []
            for i, (pos, mass) in enumerate(zip(self.s.pos, self.s.mass)):
                uuid_str = self.state_table.register_identity(pos, mass, dedup=dedup)
                self.uuids.append(uuid_str)
            self.state_table.register_group(self.group_label, set(self.uuids))


    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float, state, state_table):
        if is_enabled():
            dbg("eng.thrusters").debug(f"dt={float(dt):.6g} thrust={self.thrust}")
        if state_table is None:
            state_table = self.state_table
        if state_table is None:
            raise ValueError("ThrustersEngine requires a StateTable for identity-based state management.")
        # Compute total mass from identities
        total_mass = 0.0
        for uuid in self.uuids:
            identity = state_table.get_identity(uuid)
            if identity is not None:
                total_mass += max(identity.get('mass', 0.0), 1e-9)
        eff = max(0.0, min(1.0, getattr(self.s, "thruster_eff", 1.0)))
        if total_mass == 0.0:
            a = (0.0, 0.0)
        else:
            a = (eff * self.thrust[0] / total_mass, eff * self.thrust[1] / total_mass)
        for uuid in self.uuids:
            identity = state_table.get_identity(uuid)
            if identity is None:
                continue
            mass = identity.get('mass', 0.0)
            if mass <= 0.0:
                continue
            acc = identity.get('acc', (0.0, 0.0))
            acc = v_add(acc, a)
            state_table.update_identity(uuid, pos=identity.get('pos'), mass=mass)
            identity['acc'] = acc
        metrics = Metrics(0.0, 0.0, 0.0, 0.0)
        return True, metrics, self.get_state(state_table)


class SpringEngine(DtCompatibleEngine):

    def get_state(self, state_table=None):
        if state_table is None:
            raise ValueError("SpringEngine requires a StateTable for identity-based state management.")
        return {uuid: dict(identity) for uuid, identity in state_table.identity_registry.items()}

    def __init__(self, state: DemoState, state_table=None, group_label=None, uuids=None, dedup: bool = True):
        self.s = state
        self.state_table = state_table
        self.group_label = group_label or "spring_group"
        self.uuids = uuids or []
        self.edge_ids: list[tuple[str, str]] = []
        self.edge_uuids: list[str] = []
        self.dedup = dedup
        if self.state_table is not None:
            self._register_with_state_table(self.state_table)

    def _register_with_state_table(self, state_table) -> None:
        if state_table is None:
            raise ValueError("SpringEngine requires a StateTable for identity registration.")
        if self.uuids and self.state_table is state_table:
            return
        self.state_table = state_table
        self.uuids = []
        self.edge_ids = []
        self.edge_uuids = []
        for i, (pos, mass) in enumerate(zip(self.s.pos, self.s.mass)):
            uuid_str = state_table.register_identity(pos, mass, dedup=self.dedup)
            self.uuids.append(uuid_str)
        for (i, j) in self.s.springs:
            edge = (self.uuids[i], self.uuids[j])
            self.edge_ids.append(edge)
            # Register edge as an identity object with mass proportional to resting length
            L0 = self.s.rest_len[(i, j)]
            edge_uuid = state_table.register_identity(edge, mass=L0, dedup=True)
            self.edge_uuids.append(edge_uuid)
        state_table.register_group(self.group_label, set(self.uuids), edges={"spring": set(self.edge_uuids)})


    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float, state, state_table):
        if is_enabled():
            dbg("eng.spring").debug(f"dt={float(dt):.6g} springs={len(self.s.springs)}")
        if state_table is None:
            raise ValueError("SpringEngine requires a StateTable for identity-based state management.")
        self._register_with_state_table(state_table)
        eff = max(0.0, min(1.0, getattr(self.s, "spring_eff", 1.0)))
        for idx, (i, j) in enumerate(self.s.springs):
            uuid_i = self.uuids[i]
            uuid_j = self.uuids[j]
            identity_i = state_table.get_identity(uuid_i)
            identity_j = state_table.get_identity(uuid_j)
            if identity_i is None or identity_j is None:
                continue
            p_i, p_j = identity_i.get('pos'), identity_j.get('pos')
            k = self.s.k_spring[(i, j)]
            L0 = self.s.rest_len[(i, j)]
            d = v_sub(p_j, p_i)
            L = v_len(d)
            dir_ = v_norm(d)
            F = eff * k * (L - L0)
            f = v_scale(dir_, F)
            mass_i = identity_i.get('mass', 1e-9)
            mass_j = identity_j.get('mass', 1e-9)
            acc_i = identity_i.get('acc', (0.0, 0.0))
            acc_j = identity_j.get('acc', (0.0, 0.0))
            acc_i = v_add(acc_i, v_scale(f, +1.0 / max(mass_i, 1e-9)))
            acc_j = v_add(acc_j, v_scale(f, -1.0 / max(mass_j, 1e-9)))
            state_table.update_identity(uuid_i, pos=p_i, mass=mass_i)
            state_table.update_identity(uuid_j, pos=p_j, mass=mass_j)
            identity_i['acc'] = acc_i
            identity_j['acc'] = acc_j
            # Edge identity update: store force magnitude as a property
            edge_uuid = self.edge_uuids[idx]
            edge_identity = state_table.get_identity(edge_uuid)
            if edge_identity is not None:
                edge_identity['force'] = F
                state_table.update_identity(edge_uuid, pos=(p_i, p_j), mass=L0)
        metrics = Metrics(0.0, 0.0, 0.0, 0.0)
        return True, metrics, self.get_state(state_table)



class PneumaticDamperEngine(DtCompatibleEngine):

    def get_state(self, state_table=None):
        if state_table is None:
            raise ValueError("PneumaticDamperEngine requires a StateTable for identity-based state management.")
        return {uuid: dict(identity) for uuid, identity in state_table.identity_registry.items()}

    def __init__(self, state: DemoState, state_table=None, group_label=None, uuids=None, dedup: bool = True):
        self.s = state
        self.state_table = state_table
        self.group_label = group_label or "pneumatic_group"
        self.uuids = uuids or []
        self.edge_ids: list[tuple[str, str]] = []
        self.edge_uuids: list[str] = []
        self.dedup = dedup
        if self.state_table is not None:
            self._register_with_state_table(self.state_table)

    def _register_with_state_table(self, state_table) -> None:
        if state_table is None:
            raise ValueError("PneumaticDamperEngine requires a StateTable for identity registration.")
        if self.uuids and self.state_table is state_table:
            return
        self.state_table = state_table
        self.uuids = []
        self.edge_ids = []
        self.edge_uuids = []
        for i, (pos, mass) in enumerate(zip(self.s.pos, self.s.mass)):
            uuid_str = state_table.register_identity(pos, mass, dedup=self.dedup)
            self.uuids.append(uuid_str)
        for (i, j) in self.s.springs:
            edge = (self.uuids[i], self.uuids[j])
            self.edge_ids.append(edge)
            # Register edge as an identity object with mass proportional to resting length
            L0 = self.s.rest_len[(i, j)] if hasattr(self.s, 'rest_len') and (i, j) in self.s.rest_len else 1.0
            edge_uuid = state_table.register_identity(edge, mass=L0, dedup=True)
            self.edge_uuids.append(edge_uuid)
        state_table.register_group(self.group_label, set(self.uuids), edges={"spring": set(self.edge_uuids)})


    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float, state=None, state_table=None):
        if state is not None:
            self.s.restore(state)
        if is_enabled():
            dbg("eng.pneumatic").debug(f"dt={float(dt):.6g} springs={len(self.s.springs)}")
        if state_table is None:
            raise ValueError("PneumaticDamperEngine requires a StateTable for identity-based state management.")
        self._register_with_state_table(state_table)
        eff = max(0.0, min(1.0, getattr(self.s, "pneumatic_eff", 1.0)))
        for idx, (i, j) in enumerate(self.s.springs):
            uuid_i = self.uuids[i]
            uuid_j = self.uuids[j]
            identity_i = state_table.get_identity(uuid_i)
            identity_j = state_table.get_identity(uuid_j)
            if identity_i is None or identity_j is None:
                continue
            p_i, p_j = identity_i.get('pos'), identity_j.get('pos')
            v_i, v_j = identity_i.get('vel', (0.0, 0.0)), identity_j.get('vel', (0.0, 0.0))
            d = v_sub(p_j, p_i)
            dir_ = v_norm(d)
            rel_v = v_sub(v_j, v_i)
            along = rel_v[0] * dir_[0] + rel_v[1] * dir_[1]
            damp_a, damp_b = self.s.pneu_damp[(i, j)]
            coeff = damp_a if along > 0 else damp_b
            f = v_scale(dir_, eff * coeff * along)
            mass_i = identity_i.get('mass', 1e-9)
            mass_j = identity_j.get('mass', 1e-9)
            acc_i = identity_i.get('acc', (0.0, 0.0))
            acc_j = identity_j.get('acc', (0.0, 0.0))
            acc_i = v_add(acc_i, v_scale(f, +1.0 / max(mass_i, 1e-9)))
            acc_j = v_add(acc_j, v_scale(f, -1.0 / max(mass_j, 1e-9)))
            state_table.update_identity(uuid_i, pos=p_i, mass=mass_i)
            state_table.update_identity(uuid_j, pos=p_j, mass=mass_j)
            identity_i['acc'] = acc_i
            identity_j['acc'] = acc_j
            # Edge identity update: store force magnitude as a property
            edge_uuid = self.edge_uuids[idx]
            edge_identity = state_table.get_identity(edge_uuid)
            if edge_identity is not None:
                edge_identity['force'] = eff * coeff * along
                state_table.update_identity(edge_uuid, pos=(p_i, p_j), mass=state_table.get_identity(edge_uuid).get('mass', 1.0))
        metrics = Metrics(0.0, 0.0, 0.0, 0.0)
        return True, metrics, self.get_state(state_table)



class GroundCollisionEngine(DtCompatibleEngine):

    def get_state(self, state_table=None):
        if state_table is None:
            return {}
        return {uuid: dict(identity) for uuid, identity in state_table.identity_registry.items()}

    def __init__(self, state: DemoState, state_table=None, group_label=None, uuids=None, dedup: bool = True):
        self.s = state
        self.state_table = state_table
        self.group_label = group_label or "ground_group"
        self.uuids = uuids or []
        if self.state_table is not None:
            self.uuids = []
            for i, (pos, mass) in enumerate(zip(self.s.pos, self.s.mass)):
                uuid_str = self.state_table.register_identity(pos, mass, dedup=dedup)
                self.uuids.append(uuid_str)
            self.state_table.register_group(self.group_label, set(self.uuids))


    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float, state=None, state_table=None):
        if state is not None:
            self.s.restore(state)
        if is_enabled():
            dbg("eng.ground").debug(f"dt={float(dt):.6g}")
        if state_table is None:
            state_table = self.state_table
        if state_table is None:
            raise ValueError("GroundCollisionEngine requires a StateTable for identity-based state management.")
        k = self.s.ground_k
        b = self.s.ground_b
        mu = self.s.mu
        for idx, uuid in enumerate(self.uuids):
            identity = state_table.get_identity(uuid)
            if identity is None:
                continue
            mass = identity.get('mass', 0.0)
            if mass <= 0.0:
                continue
            pos = identity.get('pos', (0.0, 0.0))
            vel = identity.get('vel', (0.0, 0.0))
            if pos[1] < 0.0:
                pen = -pos[1]
                vy = vel[1]
                Fy = k * pen - b * vy
                Fx = -mu * k * pen * math.copysign(1.0, vel[0]) if abs(vel[0]) > 1e-6 else 0.0
                a = (Fx / max(mass, 1e-9), Fy / max(mass, 1e-9))
                acc = identity.get('acc', (0.0, 0.0))
                acc = v_add(acc, a)
                state_table.update_identity(uuid, pos=pos, mass=mass)
                identity['acc'] = acc
        metrics = Metrics(0.0, 0.0, 0.0, 0.0)
        return True, metrics, self.get_state(state_table)




class IntegratorEngine(DtCompatibleEngine):
    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        for k in ("pos", "vel", "acc", "mass"):
            if hasattr(self.s, k):
                out[k] = list(getattr(self.s, k))
        return out
    def __init__(self, state: DemoState):
        self.s = state

    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float, state=None, state_table=None):
        if state is not None:
            self.s.restore(state)
        if is_enabled():
            dbg("eng.integrate").debug(f"dt={float(dt):.6g} n={len(self.s.pos)}")
        drag = max(0.0, float(getattr(self.s, "linear_drag", 0.0)))
        # precompute exponential decay for unconditional stability
        damp = math.exp(-drag * max(0.0, float(dt))) if drag > 0.0 else 1.0
        for i in range(len(self.s.pos)):
            m = self.s.mass[i]
            if m <= 0.0:
                # Keep anchors fixed and clean any numeric junk
                self.s.acc[i] = (0.0, 0.0)
                self.s.vel[i] = (0.0, 0.0)
                # Do not modify position
                continue
            ax, ay = self.s.acc[i]
            vx, vy = self.s.vel[i]
            # semi-implicit Euler for stability
            vx += ax * dt
            vy += ay * dt
            # apply linear drag
            vx *= damp
            vy *= damp
            x, y = self.s.pos[i]
            x += vx * dt
            y += vy * dt
            # write back
            self.s.vel[i] = (vx, vy)
            self.s.pos[i] = (x, y)
            # clear acceleration for next accumulation cycle
            self.s.acc[i] = (0.0, 0.0)
        # Robust max velocity ignoring non-finite values
        max_v = 0.0
        for (vx, vy) in self.s.vel:
            try:
                if not (math.isfinite(vx) and math.isfinite(vy)):
                    continue
                max_v = max(max_v, v_len((vx, vy)))
            except Exception:
                pass
        metrics = Metrics(max_vel=max_v, max_flux=0.0, div_inf=0.0, mass_err=0.0)
        return True, metrics, self.get_state()



class MetaCollisionEngine(DtCompatibleEngine):
    def get_state(self, state=None):
        # state is expected to be a list of dicts, one per DemoState
        out = state if isinstance(state, list) and len(state) == len(self.states) else [{} for _ in self.states]
        for i, s in enumerate(self.states):
            for k in ("pos", "vel", "acc", "mass"):
                if hasattr(s, k):
                    out[i][k] = list(getattr(s, k))
        return out
    """Meta engine: resolves ground + inter-craft collisions with springs/dampers.

    Semielastic with Coulomb-like friction and position pushes to enforce
    non-penetration. This engine also evaluates spring and damper forces and
    integrates positions/velocities, so outer orchestration should not include
    a separate integrator for the same states.

    Intended to be paired with the bisect dt solver using an objective like
    max penetration depth (exposed via Metrics.div_inf) approaching zero.
    """

    def __init__(
        self,
        states: List[DemoState],
        *,
    restitution: float = 0.2,
    friction_mu: float = 0.5,
    body_radius: float = 0.12,
    solids: "SolidRegistry" | None = None,
    world: "WorldConfinement" | None = None,
    plastic_beta: float = 0.0,
    enable_plastic_relax: bool = False,
    softbody_contact_cb: Optional[Callable[[Contact], None]] = None,
    ) -> None:
        self.states = states
        self.e = float(max(0.0, min(1.0, restitution)))
        self.mu = float(max(0.0, friction_mu))
        self.r = float(max(1e-6, body_radius))
        self.solids = solids or GLOBAL_SOLIDS
        self.world = world or GLOBAL_WORLD
        # Optional plastic deformation factor and gate flag
        self.plastic_beta = float(max(0.0, min(1.0, plastic_beta)))
        self.enable_plastic_relax = bool(enable_plastic_relax)
        # Optional contact callback for softbody coupling (stub-friendly)
        self.softbody_contact_cb = softbody_contact_cb

    # Snapshot both states for solver lookahead compatibility
    def snapshot(self):  # pragma: no cover - lightweight
        return [s.snapshot() for s in self.states]

    def restore(self, snaps):  # pragma: no cover - lightweight
        try:
            for s, snap in zip(self.states, snaps):
                s.restore(snap)
        except Exception:
            pass

    # --- helpers ---
    def _apply_springs_dampers(self, dt: float, state_table) -> None:
        if state_table is None:
            raise ValueError("MetaCollisionEngine requires a StateTable for spring/damper evaluation.")
        for s in self.states:
            # accumulate spring and damper accelerations
            _ = SpringEngine(s).step(dt, s, state_table)
            _ = PneumaticDamperEngine(s).step(dt, s, state_table)

    def _integrate(self, dt: float) -> None:
        # Semi-implicit Euler (same as IntegratorEngine)
        for s in self.states:
            drag = max(0.0, float(getattr(s, "linear_drag", 0.0)))
            damp = math.exp(-drag * max(0.0, float(dt))) if drag > 0.0 else 1.0
            for i in range(len(s.pos)):
                if s.mass[i] <= 0.0:
                    # anchors remain fixed; also sanitize
                    s.acc[i] = (0.0, 0.0)
                    s.vel[i] = (0.0, 0.0)
                    continue
                ax, ay = s.acc[i]
                vx, vy = s.vel[i]
                vx += ax * dt
                vy += ay * dt
                vx *= damp
                vy *= damp
                x, y = s.pos[i]
                x += vx * dt
                y += vy * dt
                s.vel[i] = (vx, vy)
                s.pos[i] = (x, y)
                s.acc[i] = (0.0, 0.0)  # clear for next cycle

    def _resolve_world_planes(self) -> float:
        """Resolve contacts against world planes with material behavior.

        Plane half-space: n·x + d >= 0 is inside. Project bodies if outside.
        """
        max_pen = 0.0
        planes = getattr(self.world, "planes", [])
        if not planes:
            return 0.0
        for si, s in enumerate(self.states):
            for i, m in enumerate(s.mass):
                if m <= 0:
                    continue
                x, y = s.pos[i]
                p3 = (float(x), float(y), 0.0)
                for pl in planes:
                    n = getattr(pl, "normal", (0.0, 1.0, 0.0))
                    d = float(getattr(pl, "offset", 0.0))
                    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
                    # 2D point assumed z=0
                    dist = nx * p3[0] + ny * p3[1] + nz * 0.0 + d
                    if dist < 0.0:
                        pen = -dist
                        max_pen = max(max_pen, pen)
                        # Project to plane
                        x_corr = p3[0] + nx * pen
                        y_corr = p3[1] + ny * pen
                        s.pos[i] = (x_corr, y_corr)
                        # Velocity response based on material
                        vx, vy = s.vel[i]
                        v_n = vx * nx + vy * ny
                        v_t_x, v_t_y = vx - v_n * nx, vy - v_n * ny
                        mat = getattr(pl, "material", MATERIAL_ELASTIC)
                        kind = getattr(mat, "kind", "elastic")
                        rest = float(getattr(mat, "restitution", self.e))
                        mu = float(getattr(mat, "friction", self.mu))
                        # Report contact to softbody callback if relevant
                        if kind == "softbody_stub" and self.softbody_contact_cb:
                            try:
                                self.softbody_contact_cb(Contact(
                                    p=(x_corr, y_corr), n=(nx, ny), pen=float(pen),
                                    material_kind=kind, state_idx=si, vertex_idx=i, source="plane"
                                ))
                            except Exception:
                                pass
                        if kind == "soil":
                            # strong damping, no bounce
                            v_n_post = 0.0
                            damp = max(0.0, min(1.0, float(getattr(mat, "embed_damping", 0.7))))
                            v_t_x *= (1.0 - min(1.0, mu)) * (1.0 - damp)
                            v_t_y *= (1.0 - min(1.0, mu)) * (1.0 - damp)
                        elif kind == "softbody_stub":
                            # placeholder: treat like elastic for now
                            v_n_post = -rest * v_n
                            v_t_x *= max(0.0, 1.0 - mu)
                            v_t_y *= max(0.0, 1.0 - mu)
                        else:  # elastic
                            v_n_post = -rest * v_n
                            v_t_x *= max(0.0, 1.0 - mu)
                            v_t_y *= max(0.0, 1.0 - mu)
                        vx = v_t_x + v_n_post * nx
                        vy = v_t_y + v_n_post * ny
                        s.vel[i] = (vx, vy)
        return max_pen

    def _resolve_pairs(self) -> float:
        """Resolve pairwise craft collisions; return max penetration."""
        max_pen = 0.0
        # Collect all (state, index, mass>0) nodes as bodies
        bodies: List[Tuple[DemoState, int]] = []
        for s in self.states:
            for i, m in enumerate(s.mass):
                if m > 0.0:
                    bodies.append((s, i))
        R = self.r
        for a in range(len(bodies)):
            sA, iA = bodies[a]
            xA, yA = sA.pos[iA]
            for b in range(a + 1, len(bodies)):
                sB, iB = bodies[b]
                xB, yB = sB.pos[iB]
                dx = xB - xA
                dy = yB - yA
                dist = math.hypot(dx, dy)
                min_d = 2.0 * R
                if dist < min_d and dist > 1e-12:
                    pen = min_d - dist
                    max_pen = max(max_pen, pen)
                    nx, ny = dx / dist, dy / dist
                    # Split position correction by inverse mass
                    mA = max(sA.mass[iA], 1e-9)
                    mB = max(sB.mass[iB], 1e-9)
                    wA = 1.0 / mA
                    wB = 1.0 / mB
                    denom = wA + wB
                    pushA = pen * (wA / denom)
                    pushB = pen * (wB / denom)
                    sA.pos[iA] = (xA - nx * pushA, yA - ny * pushA)
                    sB.pos[iB] = (xB + nx * pushB, yB + ny * pushB)
                    # Velocity response along normal (restitution) and friction on tangent
                    vAx, vAy = sA.vel[iA]
                    vBx, vBy = sB.vel[iB]
                    # Relative velocity
                    rvx, rvy = vBx - vAx, vBy - vAy
                    v_rel_n = rvx * nx + rvy * ny
                    # Reflect component along normal with restitution
                    v_rel_n_post = -self.e * v_rel_n
                    dv_n = v_rel_n_post - v_rel_n
                    # Distribute impulse by inverse mass
                    jnA = -(dv_n) * (wA / denom)
                    jnB = +(dv_n) * (wB / denom)
                    vAx += jnA * nx
                    vAy += jnA * ny
                    vBx += jnB * nx
                    vBy += jnB * ny
                    # Tangential friction: damp tangent component proportionally
                    tx, ty = -ny, nx
                    v_rel_t = rvx * tx + rvy * ty
                    ft_scale = max(0.0, 1.0 - self.mu)
                    v_rel_t_post = v_rel_t * ft_scale
                    dv_t = v_rel_t_post - v_rel_t
                    jtA = -(dv_t) * (wA / denom)
                    jtB = +(dv_t) * (wB / denom)
                    vAx += jtA * tx
                    vAy += jtA * ty
                    vBx += jtB * tx
                    vBy += jtB * ty
                    sA.vel[iA] = (vAx, vAy)
                    sB.vel[iB] = (vBx, vBy)
        return max_pen

    def _resolve_solids(self) -> float:
        """Project bodies out of static solid AABBs (projected to XY); return max penetration."""
        if not getattr(self, "solids", None):
            return 0.0
        # Build AABBs lazily each call (registry may change)
        aabbs: List[Tuple[float, float, float, float, object]] = []
        try:
            for _name, mesh in self.solids.all():  # type: ignore[union-attr]
                v = mesh.as_vertex_array()
                xs = v[:, 0]
                ys = v[:, 1]
                mnx, mny = float(xs.min()), float(ys.min())
                mxx, mxy = float(xs.max()), float(ys.max())
                mat = getattr(mesh, "material", MATERIAL_ELASTIC)
                aabbs.append((mnx, mny, mxx, mxy, mat))
        except Exception:
            return 0.0
        max_pen = 0.0
        for si, s in enumerate(self.states):
            for i, m in enumerate(s.mass):
                if m <= 0.0:
                    continue
                x, y = s.pos[i]
                for (mnx, mny, mxx, mxy, mat) in aabbs:
                    if (mnx <= x <= mxx) and (mny <= y <= mxy):
                        # Inside: push out along smallest penetration to box face
                        pen_left = x - mnx
                        pen_right = mxx - x
                        pen_bottom = y - mny
                        pen_top = mxy - y
                        # Choose minimal push
                        pens = [
                            (pen_left, (-1.0, 0.0)),
                            (pen_right, (1.0, 0.0)),
                            (pen_bottom, (0.0, -1.0)),
                            (pen_top, (0.0, 1.0)),
                        ]
                        pen, n = min(pens, key=lambda t: t[0])
                        max_pen = max(max_pen, float(pen))
                        nx, ny = n
                        s.pos[i] = (x + nx * pen, y + ny * pen)
                        # velocity reflection along normal with restitution; simple tangent damp
                        vx, vy = s.vel[i]
                        v_n = vx * nx + vy * ny
                        kind = getattr(mat, "kind", "elastic")
                        rest = float(getattr(mat, "restitution", self.e))
                        mu = float(getattr(mat, "friction", self.mu))
                        if kind == "soil":
                            # no bounce, strong tangent damping
                            v_n_post = 0.0
                            damp = max(0.0, min(1.0, float(getattr(mat, "embed_damping", 0.7))))
                            v_t_x = vx - v_n * nx
                            v_t_y = vy - v_n * ny
                            v_t_x *= (1.0 - min(1.0, mu)) * (1.0 - damp)
                            v_t_y *= (1.0 - min(1.0, mu)) * (1.0 - damp)
                            vx = v_t_x + v_n_post * nx
                            vy = v_t_y + v_n_post * ny
                        elif kind == "softbody_stub":
                            # placeholder: treat like elastic for now
                            vx -= (1.0 + rest) * v_n * nx
                            vy -= (1.0 + rest) * v_n * ny
                            vx *= max(0.0, 1.0 - mu)
                            vy *= max(0.0, 1.0 - mu)
                            # Report to softbody callback
                            if self.softbody_contact_cb:
                                try:
                                    self.softbody_contact_cb(Contact(
                                        p=(x + nx * pen, y + ny * pen), n=(nx, ny), pen=float(pen),
                                        material_kind=kind, state_idx=si, vertex_idx=i, source="solid"
                                    ))
                                except Exception:
                                    pass
                        else:
                            vx -= (1.0 + rest) * v_n * nx
                            vy -= (1.0 + rest) * v_n * ny
                            vx *= max(0.0, 1.0 - mu)
                            vy *= max(0.0, 1.0 - mu)
                        s.vel[i] = (vx, vy)
        return max_pen

    def _plastic_relax_rest_lengths(self) -> None:
        """Gently relax spring rest lengths toward current lengths (softbody-like plasticity).

        Controlled by self.plastic_beta; when 0, disabled. This models
        quasi-plastic deformation for spring networks without a full softbody.
        """
        b = self.plastic_beta
        if b <= 0.0:
            return
        for s in self.states:
            for (i, j) in s.springs:
                pi = s.pos[i]; pj = s.pos[j]
                L = v_len(v_sub(pj, pi))
                key = (i, j)
                L0 = s.rest_len.get(key, L)
                s.rest_len[key] = (1.0 - b) * L0 + b * L

    def step(self, dt: float, state=None, state_table=None):
        if state is not None:
            # Replace internal states with provided state list
            if isinstance(state, list) and len(state) == len(self.states):
                for s, snap in zip(self.states, state):
                    s.restore(snap)
        if is_enabled():
            dbg("eng.collision").debug(f"dt={float(dt):.6g} n_states={len(self.states)}")
        # 1) consult spring+damper networks to accumulate acc
        self._apply_springs_dampers(dt, state_table)
        # 2) integrate to predict motion
        self._integrate(dt)
        # 3) resolve collisions (world planes + pairwise bodies + solids); report max penetration
        pen_g = self._resolve_world_planes()
        pen_p = self._resolve_pairs()
        pen_s = self._resolve_solids()
        max_pen = max(pen_g, pen_p, pen_s)
        # 4) optional plastic relaxation of spring rest lengths (flagged off by default)
        if self.enable_plastic_relax and self.plastic_beta > 0.0:
            self._plastic_relax_rest_lengths()
        # metrics: div_inf encodes penetration for solver objective compatibility
        max_vel = 0.0
        for s in self.states:
            for vx, vy in s.vel:
                if not (math.isfinite(vx) and math.isfinite(vy)):
                    continue
                max_vel = max(max_vel, math.hypot(vx, vy))
        m = Metrics(max_vel=max_vel, max_flux=0.0, div_inf=max_pen, mass_err=0.0)
        # Return new state list
        return True, m, [s.snapshot() for s in self.states]
