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
from typing import Dict, List, Tuple
import math

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine
from ..debug import dbg, is_enabled

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


class GravityEngine(DtCompatibleEngine):
    def __init__(self, state: DemoState):
        self.s = state

    # Snapshot proxies (optional)
    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float):
        if is_enabled():
            dbg("eng.gravity").debug(f"dt={float(dt):.6g}")
        for i, m in enumerate(self.s.mass):
            if m <= 0:  
                continue
            self.s.acc[i] = v_add(self.s.acc[i], (0.0, -self.s.g))
        return True, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)


class ThrustersEngine(DtCompatibleEngine):
    def __init__(self, state: DemoState, thrust: Vec = (0.0, 0.0)):
        self.s = state
        self.thrust = thrust

    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float):
        if is_enabled():
            dbg("eng.thrusters").debug(f"dt={float(dt):.6g} thrust={self.thrust}")
        total_mass = sum(max(m, 1e-9) for m in self.s.mass)
        a = (self.thrust[0] / total_mass, self.thrust[1] / total_mass)
        for i in range(len(self.s.pos)):
            self.s.acc[i] = v_add(self.s.acc[i], a)
        return True, Metrics(0.0, 0.0, 0.0, 0.0)


class SpringEngine(DtCompatibleEngine):
    def __init__(self, state: DemoState):
        self.s = state

    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float):
        if is_enabled():
            dbg("eng.spring").debug(f"dt={float(dt):.6g} springs={len(self.s.springs)}")
        for (i, j) in self.s.springs:
            p_i, p_j = self.s.pos[i], self.s.pos[j]
            k = self.s.k_spring[(i, j)]
            L0 = self.s.rest_len[(i, j)]
            d = v_sub(p_j, p_i)
            L = v_len(d)
            dir_ = v_norm(d)
            F = k * (L - L0)
            f = v_scale(dir_, F)
            self.s.acc[i] = v_add(self.s.acc[i], v_scale(f, +1.0 / max(self.s.mass[i], 1e-9)))
            self.s.acc[j] = v_add(self.s.acc[j], v_scale(f, -1.0 / max(self.s.mass[j], 1e-9)))
        return True, Metrics(0.0, 0.0, 0.0, 0.0)


class PneumaticDamperEngine(DtCompatibleEngine):
    def __init__(self, state: DemoState):
        self.s = state

    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float):
        if is_enabled():
            dbg("eng.pneumatic").debug(f"dt={float(dt):.6g} springs={len(self.s.springs)}")
        for (i, j) in self.s.springs:
            p_i, p_j = self.s.pos[i], self.s.pos[j]
            v_i, v_j = self.s.vel[i], self.s.vel[j]
            d = v_sub(p_j, p_i)
            dir_ = v_norm(d)
            rel_v = v_sub(v_j, v_i)
            along = rel_v[0] * dir_[0] + rel_v[1] * dir_[1]
            damp_a, damp_b = self.s.pneu_damp[(i, j)]
            coeff = damp_a if along > 0 else damp_b
            f = v_scale(dir_, -coeff * along)
            self.s.acc[i] = v_add(self.s.acc[i], v_scale(f, +1.0 / max(self.s.mass[i], 1e-9)))
            self.s.acc[j] = v_add(self.s.acc[j], v_scale(f, -1.0 / max(self.s.mass[j], 1e-9)))
        return True, Metrics(0.0, 0.0, 0.0, 0.0)


class GroundCollisionEngine(DtCompatibleEngine):
    def __init__(self, state: DemoState):
        self.s = state

    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float):
        if is_enabled():
            dbg("eng.ground").debug(f"dt={float(dt):.6g}")
        k = self.s.ground_k
        b = self.s.ground_b
        mu = self.s.mu
        for i, p in enumerate(self.s.pos):
            if p[1] < 0.0:
                # penalty force upward proportional to penetration and velocity
                pen = -p[1]
                vy = self.s.vel[i][1]
                Fy = k * pen - b * vy
                # friction along x against motion when in contact
                Fx = -mu * k * pen * math.copysign(1.0, self.s.vel[i][0]) if abs(self.s.vel[i][0]) > 1e-6 else 0.0
                a = (Fx / max(self.s.mass[i], 1e-9), Fy / max(self.s.mass[i], 1e-9))
                self.s.acc[i] = v_add(self.s.acc[i], a)
        return True, Metrics(0.0, 0.0, 0.0, 0.0)


class IntegratorEngine(DtCompatibleEngine):
    def __init__(self, state: DemoState):
        self.s = state

    def snapshot(self):  # pragma: no cover
        return self.s.snapshot()

    def restore(self, snap):  # pragma: no cover
        return self.s.restore(snap)

    def step(self, dt: float):
        if is_enabled():
            dbg("eng.integrate").debug(f"dt={float(dt):.6g} n={len(self.s.pos)}")
        for i in range(len(self.s.pos)):
            ax, ay = self.s.acc[i]
            vx, vy = self.s.vel[i]
            # semi-implicit Euler for stability
            vx += ax * dt
            vy += ay * dt
            x, y = self.s.pos[i]
            x += vx * dt
            y += vy * dt
            # write back
            self.s.vel[i] = (vx, vy)
            self.s.pos[i] = (x, y)
            # clear acceleration for next accumulation cycle
            self.s.acc[i] = (0.0, 0.0)
        return True, Metrics(max_vel=max(v_len(v) for v in self.s.vel), max_flux=0.0, div_inf=0.0, mass_err=0.0)


class MetaCollisionEngine(DtCompatibleEngine):
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
    ) -> None:
        self.states = states
        self.e = float(max(0.0, min(1.0, restitution)))
        self.mu = float(max(0.0, friction_mu))
        self.r = float(max(1e-6, body_radius))

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
    def _apply_springs_dampers(self, dt: float) -> None:
        for s in self.states:
            # accumulate spring and damper accelerations
            _ = SpringEngine(s).step(dt)
            _ = PneumaticDamperEngine(s).step(dt)

    def _integrate(self, dt: float) -> None:
        # Semi-implicit Euler (same as IntegratorEngine)
        for s in self.states:
            for i in range(len(s.pos)):
                ax, ay = s.acc[i]
                vx, vy = s.vel[i]
                vx += ax * dt
                vy += ay * dt
                x, y = s.pos[i]
                x += vx * dt
                y += vy * dt
                s.vel[i] = (vx, vy)
                s.pos[i] = (x, y)
                s.acc[i] = (0.0, 0.0)  # clear for next cycle

    def _resolve_ground(self) -> float:
        """Project nodes above ground and adjust velocities; return max penetration."""
        max_pen = 0.0
        for s in self.states:
            k = s.ground_k
            mu = s.mu
            for i, (x, y) in enumerate(s.pos):
                if y < 0.0:
                    pen = -y
                    max_pen = max(max_pen, pen)
                    # position projection
                    s.pos[i] = (x, 0.0)
                    # velocity response: reflect normal with restitution
                    vx, vy = s.vel[i]
                    vy = -self.e * vy
                    # friction: damp tangent when in contact; approximate Coulomb
                    # using a simple multiplier bounded by mu.
                    fx_scale = max(0.0, 1.0 - mu)
                    vx *= fx_scale
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

    def step(self, dt: float):
        if is_enabled():
            dbg("eng.collision").debug(f"dt={float(dt):.6g} n_states={len(self.states)}")
        # 1) consult spring+damper networks to accumulate acc
        self._apply_springs_dampers(dt)
        # 2) integrate to predict motion
        self._integrate(dt)
        # 3) resolve collisions (ground + pairwise bodies); report max penetration
        pen_g = self._resolve_ground()
        pen_p = self._resolve_pairs()
        max_pen = max(pen_g, pen_p)
        # metrics: div_inf encodes penetration for solver objective compatibility
        max_vel = 0.0
        for s in self.states:
            for vx, vy in s.vel:
                max_vel = max(max_vel, math.hypot(vx, vy))
        m = Metrics(max_vel=max_vel, max_flux=0.0, div_inf=max_pen, mass_err=0.0)
        return True, m
