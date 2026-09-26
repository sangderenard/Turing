"""One thermodynamic chamber as a lockstep package under one dt controller.

A ``CascadeMonitor`` watches every tick: an energy cascade, an endothermic
swing, a flash freeze, a breakdown field or a non-finite value refuses the
step, publishes the last committed state with the refused step's outputs
(``sim.monitor.bundle``), and holds the package until ``release()``.

Every law in symbolic_chamber_solvers runs on the same admitted ``dt``: the
package's ``advance`` steps the surfaces and pools, the air, each condensable
species and the aerosol in turn, wires their couplings by value, and merges
their published diagnostics into one ``Metrics`` whose ``dt_limit`` is the
tightest of all of them.  ``run_superstep`` then owns the clock:

    sim = ChamberSim(laws, shape=(1, 1, 8), dx=0.25, ...)
    total, dt_next, metrics = run_superstep(sim.state, 1.0, 0.01, sim.dx, targets, ctrl, sim.advance)

Couplings (all by value, one tick of lag where the arrow points backwards
in the step order):
    air F_*                 -> species and aerosol face flows
    species Q_lat (lagged)  -> air Q_ext;   surface / pool Q_gas -> air Q_ext
    species C_species, e    -> air C_cond, P_vap (recomputed from state, no lag)
    aerosol tau_cond_out    -> species tau_cond (lagged)
    species S, LWC, RWC,
      rain_fraction         -> aerosol
    voxel rain_out etc.     -> the cell below (settling), or the floor surface
    surface runoff          -> its pool's phi_in

Voxels are columns indexed ``x + nx * (y + ny * z)``; a face with no
neighbour gathers the cell itself, so the pressure difference across a wall
is zero and nothing flows through it.
"""

from __future__ import annotations

import copy

from src.common.dt_system.error_channels import (
    DT_CHANNEL_NAMES, empty_channels, channel_report,
)
from src.common.dt_system.time_contracts import BIND, HOLD

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.engine_api import DtCompatibleEngine
from src.common.tensors import AbstractTensor

from chamber_dt_join import CompiledLaw, LawEngine, LawState, METRIC_FIELDS, _as_tensor, _reduce

FACES = ("xm", "xp", "ym", "yp", "zm", "zp")


def column(values: Sequence[float]) -> AbstractTensor:
    return AbstractTensor.tensor([float(v) for v in values])


def gather(col: AbstractTensor, index: Sequence[int]) -> AbstractTensor:
    return col[list(index)]


@dataclass
class SurfaceSpec:
    voxel: int                      # the cell whose gas it sees
    state: Mapping[str, Any]        # m_film, n_s_film, m_frost, rho_frost, m_crust, m_sed, T_s
    params: Mapping[str, Any]
    pool: int | None = None         # pool that takes its runoff
    catches_rain: bool = False      # a floor: the cell's falling rain lands on it


@dataclass
class PoolSpec:
    voxel: int
    state: Mapping[str, Any]        # m_p, n_s_pool, T_l
    params: Mapping[str, Any]


class PackageState:
    """All law states, snapshotted and restored together for rollback."""

    def __init__(self, members: Mapping[str, LawState]):
        self.members = dict(members)
        self.dt_limit_hint: float | None = None

    def copy_shallow(self) -> "PackageState":
        snap = PackageState({name: member.copy_shallow() for name, member in self.members.items()})
        snap.dt_limit_hint = self.dt_limit_hint
        return snap

    def restore(self, snap: "PackageState") -> None:
        for name, member in self.members.items():
            member.restore(snap.members[name])
        self.dt_limit_hint = snap.dt_limit_hint


class ChamberSim(DtCompatibleEngine):
    def __init__(self, laws, *, shape: tuple[int, int, int], dx: float,
                 air_state: Mapping[str, Any], air_params: Mapping[str, Any],
                 species: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
                 aerosol: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
                 surfaces: Sequence[SurfaceSpec] = (), pools: Sequence[PoolSpec] = (),
                 water: str = "water", monitor: "CascadeMonitor | None" = None,
                 compiled_laws: Mapping[str, CompiledLaw] | None = None):
        self.laws = laws
        self.shape = shape
        self.dx = dx
        self.n = shape[0] * shape[1] * shape[2]
        self.water = water
        self.neighbours = self._neighbour_tables()
        compiled = dict(compiled_laws or {})
        required = (
            "voxel_air_step", "voxel_species_step",
            *(("aerosol_step",) if aerosol is not None else ()),
            *(("surface_step",) if surfaces else ()),
            *(("pool_step",) if pools else ()),
        )
        for name in required:
            if name not in compiled:
                compiled[name] = CompiledLaw.from_module(laws, name)

        self.air = LawEngine(compiled["voxel_air_step"], air_state, {**air_params, "dx": dx})
        self.species = {
            name: LawEngine(compiled["voxel_species_step"], state, {**params, "dx": dx})
            for name, (state, params) in species.items()}
        self.aerosol = (LawEngine(compiled["aerosol_step"], aerosol[0], {**aerosol[1], "dx": dx})
                        if aerosol is not None else None)
        self.surfaces = [(spec, LawEngine(compiled["surface_step"], spec.state, spec.params))
                         for spec in surfaces]
        self.pools = [(spec, LawEngine(compiled["pool_step"], spec.state, spec.params))
                      for spec in pools]

        members = {"air": self.air.state}
        members.update({f"species:{name}": engine.state for name, engine in self.species.items()})
        if self.aerosol is not None:
            members["aerosol"] = self.aerosol.state
        members.update({f"surface:{i}": engine.state for i, (_s, engine) in enumerate(self.surfaces)})
        members.update({f"pool:{i}": engine.state for i, (_s, engine) in enumerate(self.pools)})
        self.state = PackageState(members)
        # the order _advance_all visits the engines, so per-law metrics can
        # be keyed by name after the merge (the merge keeps only the max)
        self._law_names = ([f"surface:{i}" for i in range(len(self.surfaces))]
                           + [f"pool:{i}" for i in range(len(self.pools))]
                           + ["air"] + [f"species:{n}" for n in self.species]
                           + (["aerosol"] if self.aerosol is not None else []))
        self.metrics_by_law: dict[str, Metrics] = {}
        self.frame = 0
        self.world_time = 0.0
        self.observer_time = 0.0
        self.monitor = monitor if monitor is not None else CascadeMonitor()

    def causal_ceiling_dt(self) -> float:
        """The most recently measured chamber-law stability ceiling."""
        hint = self.state.dt_limit_hint
        return float("inf") if hint is None else float(hint)

    def step(self, dt: float, state=None, state_table=None):
        active = self.state if state is None else state
        ok, metrics = self.advance(active, float(dt))
        return ok, metrics, active

    def get_state(self, state=None):
        return self.state if state is None else state

    def snapshot(self):
        return {
            "state": self.state.copy_shallow(),
            "frame": int(self.frame),
            "law_frames": tuple(
                int(engine.frame) for engine in (
                    [self.air]
                    + list(self.species.values())
                    + ([self.aerosol] if self.aerosol is not None else [])
                    + [engine for _spec, engine in self.surfaces]
                    + [engine for _spec, engine in self.pools]
                )
            ),
            "metrics_by_law": copy.deepcopy(self.metrics_by_law),
            "monitor": copy.deepcopy(vars(self.monitor)),
            "world_time": float(self.world_time),
            "observer_time": float(self.observer_time),
            "boundary_state": (
                self.boundary_state.copy_shallow()
                if hasattr(self, "boundary_state") else None),
        }

    def restore(self, snapshot) -> None:
        self.state.restore(snapshot["state"])
        self.frame = int(snapshot["frame"])
        engines = (
            [self.air]
            + list(self.species.values())
            + ([self.aerosol] if self.aerosol is not None else [])
            + [engine for _spec, engine in self.surfaces]
            + [engine for _spec, engine in self.pools]
        )
        for engine, frame in zip(engines, snapshot["law_frames"]):
            engine.frame = int(frame)
        self.metrics_by_law = copy.deepcopy(snapshot["metrics_by_law"])
        vars(self.monitor).clear()
        vars(self.monitor).update(copy.deepcopy(snapshot["monitor"]))
        self.world_time = float(snapshot.get("world_time", 0.0))
        self.observer_time = float(snapshot.get("observer_time", 0.0))
        if snapshot["boundary_state"] is not None:
            self.boundary_state.restore(snapshot["boundary_state"])

    # ------------------------------------------------------------ topology
    def _neighbour_tables(self) -> dict[str, list[int]]:
        nx, ny, nz = self.shape
        tables = {face: [] for face in FACES}
        above, has_above = [], []
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    me = x + nx * (y + ny * z)
                    def idx(i, j, k):
                        inside = 0 <= i < nx and 0 <= j < ny and 0 <= k < nz
                        return (i + nx * (j + ny * k)) if inside else me
                    tables["xm"].append(idx(x - 1, y, z)); tables["xp"].append(idx(x + 1, y, z))
                    tables["ym"].append(idx(x, y - 1, z)); tables["yp"].append(idx(x, y + 1, z))
                    tables["zm"].append(idx(x, y, z - 1)); tables["zp"].append(idx(x, y, z + 1))
                    above.append(idx(x, y, z + 1)); has_above.append(1.0 if z + 1 < nz else 0.0)
        tables["above"] = above
        self.has_above = column(has_above)
        return tables

    def _gather_faces(self, prefix: str, col: AbstractTensor) -> dict[str, AbstractTensor]:
        return {f"{prefix}_{face}": gather(col, self.neighbours[face]) for face in FACES}

    # ------------------------------------------------------------ one lockstep tick
    def advance(self, state: PackageState, dt) -> tuple[bool, Metrics]:
        if self.monitor.paused:
            return False, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0, hard_failure=True,
                                  unresolved_report=[f"paused: {self.monitor.event['rule']}"])
        before = state.copy_shallow()
        ok, merged = self._advance_all(state, dt)
        tripped = self.monitor.inspect(self, before, merged, dt)
        if tripped is not None:
            state.restore(before)              # the refused step is not committed
            merged.hard_failure = True
            merged.unresolved_report = [f"cascade: {tripped}: {self.monitor.event['reason']}"]
            return False, merged
        return ok, merged

    def _advance_all(self, state: PackageState, dt) -> tuple[bool, Metrics]:
        p = self.air.params
        V = self.dx ** 3
        air = self.air.state.columns
        T, m_a = air["T"], air["m_a"]
        ok_all = True
        metrics_all: list[Metrics] = []

        # Recomputed from state: the species' contribution to the cell's
        # heat capacity and pressure.
        C_cond = 0.0
        P_vap = 0.0
        for engine in self.species.values():
            c, q = engine.state.columns, engine.params
            C_cond = C_cond + c["m_v"] * q["cv_v"] + (c["m_l"] + c["m_r"]) * q["c_w"] + c["m_i"] * q["c_i"]
            P_vap = P_vap + c["m_v"] * q["R_v"] * T / V
        water = self.species[self.water]
        e_water = water.state.columns["m_v"] * water.params["R_v"] * T / V

        # Surfaces and pools see the gas as it is at the start of the tick.
        Q_gas = 0.0
        pool_inflow = [0.0 for _ in self.pools]
        pool_solute = [0.0 for _ in self.pools]
        for spec, engine in self.surfaces:
            feeds = {"T": T[[spec.voxel]], "e_amb": e_water[[spec.voxel]]}
            if spec.catches_rain:
                w = water.state.outputs
                feeds["phi_drop"] = ((w["rain_out"] + w["drizzle_out"] + w["snow_out"])[[spec.voxel]]
                                     / _as_tensor(dt)) if w else _as_tensor(0.0)
            engine.feeds = feeds
            ok, m = engine.advance(engine.state, dt)
            ok_all &= ok; metrics_all.append(m)
            Q_gas = Q_gas + _scatter_one(engine.state.outputs["Q_gas"], spec.voxel, self.n)
            if spec.pool is not None:
                pool_inflow[spec.pool] = pool_inflow[spec.pool] + engine.state.outputs["runoff"] / _as_tensor(dt)
        for i, (spec, engine) in enumerate(self.pools):
            engine.feeds = {"T": T[[spec.voxel]], "e_amb": e_water[[spec.voxel]],
                            "phi_in": _as_tensor(pool_inflow[i]), "s_in": _as_tensor(pool_solute[i])}
            ok, m = engine.advance(engine.state, dt)
            ok_all &= ok; metrics_all.append(m)
            Q_gas = Q_gas + _scatter_one(engine.state.outputs["Q_gas"], spec.voxel, self.n)

        # Air: face flows from the pressure field, one shared temperature.
        Q_lat = 0.0
        for engine in self.species.values():
            if engine.state.outputs:
                Q_lat = Q_lat + engine.state.outputs["Q_lat"]
        P = m_a * p["R_a"] * T / V + P_vap
        self.air.feeds = {**self._gather_faces("P", P), **self._gather_faces("T", T),
                          "P_vap": P_vap, "C_cond": C_cond,
                          "Q_ext": Q_lat + Q_gas}
        ok, m = self.air.advance(self.air.state, dt)
        ok_all &= ok; metrics_all.append(m)
        F = {f"F_{face}": self.air.state.outputs[f"F_{face}"] for face in FACES}

        # Each condensable rides the same flows; settling arrives from above.
        tau_feed = {}
        if self.aerosol is not None and self.aerosol.state.outputs:
            tau_feed["tau_cond"] = self.aerosol.state.outputs["tau_cond_out"]
        for engine in self.species.values():
            c = engine.state.columns
            q = c["m_v"] / (m_a + 1e-40)
            engine.feeds = {**F, **self._gather_faces("q", q), "T": T, "m_a": m_a, **tau_feed,
                            "m_l_up": gather(c["m_l"], self.neighbours["above"]) * self.has_above,
                            "m_i_up": gather(c["m_i"], self.neighbours["above"]) * self.has_above,
                            "m_r_up": gather(c["m_r"], self.neighbours["above"]) * self.has_above}
            ok, m = engine.advance(engine.state, dt)
            ok_all &= ok; metrics_all.append(m)

        if self.aerosol is not None:
            c = self.aerosol.state.columns
            w = water.state.outputs
            self.aerosol.feeds = {**F, **self._gather_faces("N", c["N_a"]), **self._gather_faces("M", c["M_a"]),
                                  "T": T, "m_a": m_a, "rho_a": self.air.state.outputs["rho_a"],
                                  "S_amb": w["S"], "LWC": w["LWC"], "RWC": w["RWC"],
                                  "rain_fraction": w["rain_fraction"],
                                  "dep_in_N": gather(c["N_a"], self.neighbours["above"]) * 0.0,
                                  "dep_in_M": gather(c["M_a"], self.neighbours["above"]) * 0.0}
            ok, m = self.aerosol.advance(self.aerosol.state, dt)
            ok_all &= ok; metrics_all.append(m)

        self.frame += 1
        self.metrics_by_law = dict(zip(self._law_names, metrics_all))
        merged = merge_metrics(metrics_all, self.frame)
        state.dt_limit_hint = merged.dt_limit
        return ok_all, merged


def _scatter_one(value: AbstractTensor, index: int, n: int) -> AbstractTensor:
    mask = column([1.0 if i == index else 0.0 for i in range(n)])
    return mask * value


def merge_metrics(rows: Sequence[Metrics], frame: int) -> Metrics:
    limits = [m.dt_limit for m in rows if m.dt_limit is not None]
    channels = empty_channels()
    present = empty_channels()
    for m in rows:
        channels = channels + AbstractTensor.where(
            m.error_present, m.error_channels, AbstractTensor.zeros_like(m.error_channels))
        present = AbstractTensor.maximum(present, m.error_present)
    energy_slot = DT_CHANNEL_NAMES.index("energy_j")
    power_slot = DT_CHANNEL_NAMES.index("power_w")
    energy = float(channels[energy_slot].item())
    power = float(channels[power_slot].item())
    exchange_time_present = bool(present[energy_slot].item()) and bool(
        present[power_slot].item()) and power > 0.0
    dt_limit = min(limits) if limits else None
    # The chamber is one top-level simulation.  Its surfaces, pools, air,
    # species and aerosol keep their causal order and local metrics internally;
    # they do not become dt-system participants individually.
    return Metrics(
        **{name: max(float(getattr(m, name)) for m in rows) for name in METRIC_FIELDS},
        sim_frame=frame,
        dt_limit=dt_limit,
        error_channels=channels,
        error_present=present,
        pub_exchange_time=AbstractTensor.tensor([energy / power if exchange_time_present else 0.0]),
        pub_exchange_time_present=AbstractTensor.tensor([float(exchange_time_present)]),
        pub_contract=AbstractTensor.tensor([BIND if exchange_time_present else HOLD]),
        pub_dt_limit=AbstractTensor.tensor([0.0 if dt_limit is None else dt_limit]),
        pub_dt_limit_present=AbstractTensor.tensor([float(dt_limit is not None)]),
        pub_values=channels.copy(),
        pub_present=present.copy(),
        pub_limits=AbstractTensor.zeros_like(channels),
        pub_limits_present=AbstractTensor.zeros_like(present),
        hard_failure=any(m.hard_failure for m in rows),
    )



# =====================================================================
# Cascade detection: publish the state and stop, never step through it
# =====================================================================
@dataclass
class CascadeRule:
    """One condition on a tick that must halt the package.

    ``check(sim, metrics, dt)`` returns None to pass or a short reason.  The
    package refuses the step it fired on, so the published bundle holds the
    last committed state, the outputs of the refused step, and the reason.
    """

    name: str
    check: Any


def _member_max(sim, output: str) -> float:
    worst = None
    for member in sim.state.members.values():
        value = member.outputs.get(output)
        if value is None:
            continue
        v = _reduce(value, "max")
        worst = v if worst is None else max(worst, v)
    return worst


def energy_cascade(fraction: float = 0.5) -> CascadeRule:
    """More than ``fraction`` of the stored energy moved in one step."""
    def check(sim, metrics, dt):
        e = float(metrics.error_channels[0].item())
        pw = float(metrics.error_channels[1].item())
        if e > 0 and pw * float(dt) > fraction * e:
            return f"power {pw:.3g} W over dt {float(dt):.3g} s moved {pw * float(dt) / e:.2%} of {e:.3g} J"
        return None
    return CascadeRule("energy_cascade", check)


def endothermic_swing(power_w: float) -> CascadeRule:
    """A reaction turned endothermic harder than ``power_w``."""
    def check(sim, metrics, dt):
        v = _member_max(sim, "endothermic_power")
        return f"endothermic {v:.3g} W" if v is not None and v > power_w else None
    return CascadeRule("endothermic_swing", check)


def flash_freeze() -> CascadeRule:
    """A solution stepped below its own freezing point."""
    def check(sim, metrics, dt):
        worst = None
        for member in sim.state.members.values():
            value = member.outputs.get("freeze_margin")
            if value is not None:
                v = _reduce(value, "min")
                worst = v if worst is None else min(worst, v)
        return f"freeze margin {worst:.3g} K" if worst is not None and worst < 0 else None
    return CascadeRule("flash_freeze", check)


def breakdown() -> CascadeRule:
    """The field reached the breakdown field somewhere: a bolt is starting."""
    def check(sim, metrics, dt):
        v = _member_max(sim, "breakdown_margin")
        return f"|E| / E_bd = {v:.3g}" if v is not None and v >= 1.0 else None
    return CascadeRule("breakdown", check)


def non_finite() -> CascadeRule:
    def check(sim, metrics, dt):
        return "non-finite output" if metrics.hard_failure else None
    return CascadeRule("non_finite", check)


DEFAULT_RULES = (non_finite(), energy_cascade(), endothermic_swing(1e3), flash_freeze(), breakdown())


class CascadeMonitor:
    def __init__(self, rules: Sequence[CascadeRule] = DEFAULT_RULES):
        self.rules = list(rules)
        self.event: dict[str, Any] | None = None
        self.bundle: dict[str, Any] | None = None

    @property
    def paused(self) -> bool:
        return self.event is not None

    def inspect(self, sim, before: PackageState, metrics: Metrics, dt) -> str | None:
        for rule in self.rules:
            reason = rule.check(sim, metrics, dt)
            if reason is not None:
                self.event = {"rule": rule.name, "reason": reason, "frame": sim.frame, "dt": float(dt)}
                self.bundle = publish_state(sim, before=before, metrics=metrics)
                return rule.name
        return None

    def release(self) -> None:
        self.event = None
        self.bundle = None


def _listify(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return float(value)


def publish_state(sim, *, before: PackageState | None = None, metrics: Metrics | None = None,
                  path: str | None = None) -> dict[str, Any]:
    """Every member's committed columns and last outputs as plain lists.

    Written as JSON when ``path`` is given.  ``before`` is the state as it
    was before a refused step, so the bundle carries both the last good
    state and the outputs that tripped the monitor.
    """
    import json

    bundle: dict[str, Any] = {"frame": sim.frame, "members": {}}
    for name, member in sim.state.members.items():
        source = before.members[name] if before is not None else member
        bundle["members"][name] = {
            "columns": {k: _listify(v) for k, v in source.columns.items()},
            "outputs": {k: _listify(v) for k, v in member.outputs.items()},
        }
    if metrics is not None:
        bundle["metrics"] = {
            **{k: float(getattr(metrics, k)) for k in METRIC_FIELDS},
            "dt_limit": metrics.dt_limit, "error_channels": channel_report(metrics.error_channels, metrics.error_present),
            "hard_failure": bool(metrics.hard_failure),
        }
    if path is not None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=1)
    return bundle
