# -*- coding: utf-8 -*-
"""Graph-based composition for adaptive dt controllers.

This module provides a minimal graph abstraction and a standardized meta-loop
runner that composes existing timestep controllers (STController) and
superstep logic (run_superstep/run_superstep_plan) into nested rounds. It
enables:

- Describing complex nested time-stepping as a graph of controllers and
  advance-adapters.
- Arbitrary introspection: read latest Metrics for any node at any depth.
- A single MetaLoopRunner that "just reads the graph" to execute a frame.

The implementation is intentionally lightweight and engine-agnostic. It builds
on the proven dt controller behaviour in ``src/common/dt_system/dt_controller.py``
and the small dataclasses in ``src/common/dt.py``.

Future work can map these node types into the Transmogrifier ProcessGraph,
but this standalone module is sufficient to begin migrating managed-dt loops
to a declarative graph.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .dt import SuperstepPlan, SuperstepResult
from .dt_controller import STController, Targets, Metrics, run_superstep
from .dt_scaler import coerce_metrics
from .engine_api import EngineRegistration
from .debug import dbg, is_enabled, pretty_metrics
from .dt_solver import solve_window_bisect
from .state_table import sync_engine_from_table, publish_engine_to_table
from .integrator.integrator import Integrator

import time
from .realtime import RealtimeConfig, RealtimeState, compile_allocations

# 
# Node types
# 


# Accept both legacy two-tuple and new three-tuple signatures at runtime.
# Annotation uses the broader three-tuple, but call sites handle both.
AdvanceFn = Callable[[Any, float], Tuple[bool, Metrics, Any]]


@dataclass
class StateNode:
    """Holds a reference to a simulator subset/state."""

    state: Any
    label: str = "state"


@dataclass
class AdvanceNode:
    """Leaf that advances a StateNode and returns Metrics.

    The callable must mirror the engine-specific ``advance(state, dt)`` and
    return a strict three-tuple: ``(ok: bool, metrics: Metrics, state: Any)``.
    The returned state will be written back to the attached StateNode.
    """

    advance: AdvanceFn
    state: StateNode
    label: str = "advance"
    transaction_owner: Any = None


@dataclass
class ControllerNode:
    """Adaptive controller node (wraps STController + targets + dx)."""

    ctrl: STController
    targets: Targets
    dx: float
    label: str = "controller"


@dataclass
class EngineNode:
    """Shim that adapts a registered engine to an AdvanceNode.

    This ensures engines conform to the DtCompatibleEngine interface and
    isolates the advancement callable used by the dt controller.
    """

    registration: EngineRegistration
    label: str = "engine"

    def to_advance_node(self, state: StateNode) -> "AdvanceNode":
        def advance(state_obj: Any, dt: float, *, realtime: bool = False, state_table=None):
            if is_enabled():
                dbg("engine").debug(f"step: name={self.registration.name} dt={float(dt):.6g} realtime={realtime}")
            if state_table is None:
                raise ValueError("Explicit state_table argument is required")
            sync_engine_from_table(self.registration.engine, self.registration.name, state_table)
            eng = self.registration.engine
            ok, metrics, state_new = eng.step_with_state(state_obj, float(dt), realtime=realtime, state_table=state_table)  # type: ignore[misc]
            
            if realtime:
                penalty = 0.0
                targets = getattr(self.registration, "targets", None)
                if targets is not None:
                    if hasattr(metrics, "max_vel") and hasattr(targets, "cfl"):
                        if metrics.max_vel > targets.cfl:
                            penalty += float(metrics.max_vel - targets.cfl)
                    if hasattr(metrics, "div_inf") and hasattr(targets, "div_max"):
                        if metrics.div_inf > targets.div_max:
                            penalty += float(metrics.div_inf - targets.div_max)
                    if hasattr(metrics, "mass_err") and hasattr(targets, "mass_max"):
                        if abs(metrics.mass_err) > targets.mass_max:
                            penalty += float(abs(metrics.mass_err) - targets.mass_max)
                metrics.penalty = penalty
            publish_engine_to_table(self.registration.engine, self.registration.name, state_table)
            if is_enabled():
                dbg("engine").debug(
                    f"done: name={self.registration.name} ok={ok} metrics=({pretty_metrics(metrics)})"
                )
            return bool(ok), metrics, state_new

        return AdvanceNode(
            advance=advance,
            state=state,
            label=f"advance:{self.registration.name}",
            transaction_owner=self.registration.engine,
        )


@dataclass
class RoundNode:
    """A time-window (superstep) driven by a controller.

    Children may be "advance" leaves or nested RoundNodes. Nested rounds allow
    delegating an outer dt request to an inner controller that may subdivide.
    """

    plan: SuperstepPlan
    controller: ControllerNode
    children: List[Union[AdvanceNode, "RoundNode"]] = field(default_factory=list)
    allow_increase_mid_round: bool = False
    # Optional scheduling: 'sequential' (default), 'interleave', 'parallel'
    schedule: str = "sequential"
    # Optional nonlinear distribution hook: (metrics, targets, dx) -> dt_penalty scalar
    distribution: Optional[Callable[[Metrics, Targets, float], float]] = None
    label: str = "round"
    # Reference to the state_table, only set at the root
    state_table: Any = None


# 
# Introspection and runner
# 


@dataclass
class NodeStats:
    last_metrics: Optional[Metrics] = None
    attempted: List[float] = field(default_factory=list)
    advanced_total: float = 0.0


def _combine_metrics(values: List[Metrics]) -> Metrics:
    """Conservatively aggregate diagnostics from coupled child advances."""

    if not values:
        return Metrics(0.0, 0.0, 0.0, 0.0)
    limits = [
        float(value.dt_limit)
        for value in values
        if value.dt_limit is not None
    ]
    error_names = {
        name for value in values for name in value.error_channels
    }
    return Metrics(
        max_vel=max(float(value.max_vel) for value in values),
        max_flux=max(float(value.max_flux) for value in values),
        div_inf=max(float(value.div_inf) for value in values),
        mass_err=max(float(value.mass_err) for value in values),
        osc_flag=any(bool(value.osc_flag) for value in values),
        stiff_flag=any(bool(value.stiff_flag) for value in values),
        sim_frame=max(int(value.sim_frame) for value in values),
        proc_ms=sum(float(value.proc_ms) for value in values),
        dt_limit=min(limits) if limits else None,
        error_channels={
            name: max(
                float(value.error_channels.get(name, 0.0))
                for value in values
            )
            for name in error_names
        },
        hard_failure=any(bool(value.hard_failure) for value in values),
        advanced_dt=min(
            (
                float(value.advanced_dt)
                for value in values
                if value.advanced_dt is not None
            ),
            default=None,
        ),
    )


class _RoundTransaction:
    """Rollback adapter consumed by ``run_superstep``.

    It preserves StateNode identities, engine-owned snapshot state, controller
    memory, and the shared StateTable across a rejected parent micro-step.
    """

    def __init__(self, root: RoundNode, state_table: Any) -> None:
        self.root = root
        self.state_table = state_table

    def _collect(self):
        state_nodes: dict[int, StateNode] = {}
        owners: dict[int, Any] = {}
        controllers: dict[int, STController] = {}

        def visit(node):
            if isinstance(node, AdvanceNode):
                state_nodes[id(node.state)] = node.state
                owner = node.transaction_owner
                if owner is not None:
                    owners[id(owner)] = owner
            elif isinstance(node, RoundNode):
                controllers[id(node.controller.ctrl)] = node.controller.ctrl
                for child in node.children:
                    visit(child)

        visit(self.root)
        return state_nodes, owners, controllers

    def copy_shallow(self):
        state_nodes, owners, controllers = self._collect()
        state_snapshots = {}
        for key, node in state_nodes.items():
            value = node.state
            if hasattr(value, "copy_shallow") and hasattr(value, "restore"):
                state_snapshots[key] = ("restore", value.copy_shallow())
            else:
                state_snapshots[key] = ("replace", copy.deepcopy(value))
        owner_snapshots = {}
        for key, owner in owners.items():
            if hasattr(owner, "snapshot") and hasattr(owner, "restore"):
                owner_snapshots[key] = ("restore", owner.snapshot())
            else:
                owner_snapshots[key] = ("clocks", {
                    name: copy.deepcopy(getattr(owner, name))
                    for name in ("world_time", "observer_time")
                    if hasattr(owner, name)
                })
        controller_snapshots = {
            key: copy.deepcopy(vars(controller))
            for key, controller in controllers.items()
        }
        table_snapshot = (
            self.state_table.snapshot()
            if self.state_table is not None
            and hasattr(self.state_table, "snapshot")
            else copy.deepcopy(self.state_table)
        )
        return (
            state_snapshots,
            owner_snapshots,
            controller_snapshots,
            table_snapshot,
        )

    def restore(self, snapshot) -> None:
        state_snapshots, owner_snapshots, controller_snapshots, table_snapshot = snapshot
        state_nodes, owners, controllers = self._collect()
        for key, (strategy, saved) in state_snapshots.items():
            node = state_nodes[key]
            if strategy == "restore":
                node.state.restore(saved)
            else:
                node.state = copy.deepcopy(saved)
        for key, (strategy, saved) in owner_snapshots.items():
            owner = owners[key]
            if strategy == "restore":
                owner.restore(saved)
            else:
                for name, value in saved.items():
                    setattr(owner, name, copy.deepcopy(value))
        for key, saved in controller_snapshots.items():
            controller = controllers[key]
            vars(controller).clear()
            vars(controller).update(copy.deepcopy(saved))
        if self.state_table is not None:
            if hasattr(self.state_table, "restore"):
                self.state_table.restore(table_snapshot)
            else:
                self.state_table = copy.deepcopy(table_snapshot)

from .realtime import RealtimeConfig, RealtimeState
class MetaLoopRunner:
    """Standardized meta-loop runner that executes a RoundNode tree.

    Key properties:
    - Uses existing run_superstep/run_superstep_plan to guarantee identical
      dt semantics.
    - Captures per-node metrics and attempted dt values for introspection.
    - Supports arbitrary nesting depth.
    """

    def __init__(self, realtime_config: RealtimeConfig = None, realtime_state: RealtimeState = None, realtime: bool = False, state_table=None) -> None:
        self._stats: Dict[str, NodeStats] = {}
        self._process_graph = None
        self._adapter = None
        self._root_round = None
        self._ilp_scheduler = None
        self._schedule = None
        self._adv_map = None
        self._schedule_method = "asap"
        self._schedule_order = "dependency"
        self._realtime_config = realtime_config
        self._realtime_state = realtime_state
        self._realtime_allocations = None
        self._last_timings: list[float] = []  # timings for last frame, in schedule order
        self._realtime = realtime
        self._constructed_state_table = state_table

    def set_process_graph(self, round_node: RoundNode, *, schedule_method: str = "asap", schedule_order: str = "dependency") -> None:
        """Build and store a process graph, adapter, and schedule from the given RoundNode. Only called once unless graph changes."""
        from .dt_process_adapter import DtToProcessAdapter
        from ...transmogrifier.ilpscheduler import ILPScheduler
        if round_node.state_table is None:
            round_node.state_table = self._constructed_state_table
        self._adapter = DtToProcessAdapter()
        self._process_graph = self._adapter.build(round_node)
        self._root_round = round_node
        self._ilp_scheduler = ILPScheduler(self._process_graph)
        self._schedule_method = schedule_method
        self._schedule_order = schedule_order
        # Build id->AdvanceNode mapping by traversing the RoundNode tree
        adv_map = {}
        def collect_adv(node):
            if isinstance(node, AdvanceNode):
                adv_map[id(node)] = node
            elif isinstance(node, RoundNode):
                for c in node.children:
                    collect_adv(c)
        collect_adv(self._root_round)
        self._adv_map = adv_map
        # Compute and cache the schedule ONCE
        levels = self._ilp_scheduler.compute_levels(schedule_method, schedule_order)
        self._schedule = [self._adv_map[nid] for nid, _ in sorted(levels.items(), key=lambda x: (x[1], x[0])) if nid in self._adv_map]

    def _renew_allocations(self) -> None:
        if self._realtime_config is not None and self._realtime_state is not None:
            id_list = [id(adv) for adv in self._schedule]
            self._realtime_allocations = compile_allocations(self._realtime_config, self._realtime_state, id_list)

    # set_realtime_config is now removed; realtime config/state must be provided at construction


    def run_round(
        self,
        round_node: Optional[RoundNode] = None,
        dt: Optional[float] = None,
        *,
        realtime: Optional[bool] = None,
        state_table=None,
    ) -> SuperstepResult:
        """Execute one frame. Optionally accept a RoundNode on first call. Logs dt tape to state_table if provided."""
        if round_node is not None:
            self.set_process_graph(round_node)
        if self._schedule is None or self._root_round is None:
            raise RuntimeError("No process graph set. Call set_process_graph() first.")
        # Use the state_table from the root node if not explicitly provided
        table = state_table if state_table is not None else getattr(self._root_round, 'state_table', None)
        # Use the instance default if not explicitly provided
        if realtime is None:
            realtime = self._realtime
        self._last_timings = []
        if realtime:
            self._renew_allocations()
            if not self._schedule:
                return SuperstepResult(
                    advanced=0.0,
                    dt_next=0.0,
                    steps=0,
                    clamped=False,
                    metrics=None,
                )
            for adv, alloc in zip(self._schedule, self._realtime_allocations):
                step_dt_ms = alloc if alloc is not None else dt*1000.0 if dt is not None else 1.0
                step_dt = step_dt_ms / 1000.0  # convert ms to seconds
                t0 = time.perf_counter()
                ok, metrics, new_state = adv.advance(adv.state.state, step_dt, realtime=True, state_table=table)
                metrics = coerce_metrics(metrics)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                self._last_timings.append(elapsed)
                adv.state.state = new_state
                if table is not None:
                    table.set("dt_tape", adv.label, "metrics", metrics)
                    table.set("dt_tape", adv.label, "dt", step_dt)
            return SuperstepResult(
                advanced=step_dt,
                dt_next=step_dt,
                steps=len(self._schedule),
                clamped=False,
                metrics=metrics if self._schedule else None,
            )

        self._realtime_allocations = [None] * len(self._schedule)
        window = (
            float(dt)
            if dt is not None
            else float(self._root_round.plan.round_max)
        )
        return self._run_scientific_round(self._root_round, window, table)

    def _run_scientific_round(
        self,
        round_node: RoundNode,
        window: float,
        state_table: Any,
    ) -> SuperstepResult:
        transaction = _RoundTransaction(round_node, state_table)
        window_checkpoint = transaction.copy_shallow()
        round_stats = self._stats.setdefault(round_node.label, NodeStats())
        attempted_before = len(round_stats.attempted)
        advanced_before = round_stats.advanced_total
        metrics_before = round_stats.last_metrics
        timings_before = len(self._last_timings)
        attempt_log: list[dict] = []

        def advance(_transaction, dt_step):
            round_stats.attempted.append(float(dt_step))
            ok, metrics = self._advance_children(
                round_node, float(dt_step), state_table
            )
            return ok, metrics

        try:
            total, dt_next, metrics = run_superstep(
                transaction,
                round_max=float(window),
                dt_init=min(
                    max(float(round_node.plan.dt_init), 1e-30),
                    float(window),
                ),
                dx=round_node.controller.dx,
                targets=round_node.controller.targets,
                ctrl=round_node.controller.ctrl,
                advance=advance,
                allow_increase_mid_round=(
                    round_node.allow_increase_mid_round
                    or round_node.plan.allow_increase_mid_round
                ),
                eps=round_node.plan.eps,
                event_boundaries=round_node.plan.event_boundaries,
                attempt_log=attempt_log,
            )
        except Exception:
            transaction.restore(window_checkpoint)
            del round_stats.attempted[attempted_before:]
            round_stats.advanced_total = advanced_before
            round_stats.last_metrics = metrics_before
            del self._last_timings[timings_before:]
            raise
        attempts = len(round_stats.attempted) - attempted_before
        round_stats.advanced_total += float(total)
        round_stats.last_metrics = metrics
        if state_table is not None:
            state_table.set("dt_tape", round_node.label, "metrics", metrics)
            state_table.set("dt_tape", round_node.label, "attempted", list(
                round_stats.attempted
            ))
            state_table.set("dt_tape", round_node.label, "advanced", float(total))
        return SuperstepResult(
            advanced=float(total),
            dt_next=float(dt_next),
            steps=attempts,
            clamped=any(
                attempted < float(round_node.plan.dt_init)
                for attempted in round_stats.attempted[attempted_before:]
            ),
            metrics=metrics,
            attempted_dts=tuple(
                float(item["dt"]) for item in attempt_log
            ),
            accepted_dts=tuple(
                float(item["dt"])
                for item in attempt_log
                if item["accepted"]
            ),
            rejected_attempts=sum(
                1 for item in attempt_log if not item["accepted"]
            ),
            landed_boundaries=self._landed_boundaries(
                attempt_log,
                round_node.plan.event_boundaries,
                round_node.plan.eps,
            ),
        )

    @staticmethod
    def _landed_boundaries(
        attempt_log: list[dict],
        boundaries: tuple[float, ...],
        eps: float,
    ) -> tuple[float, ...]:
        cumulative = 0.0
        landed: list[float] = []
        for item in attempt_log:
            if not item["accepted"]:
                continue
            cumulative += float(item["dt"])
            for boundary in boundaries:
                if (
                    abs(cumulative - float(boundary)) <= eps
                    and not any(
                        abs(existing - float(boundary)) <= eps
                        for existing in landed
                    )
                ):
                    landed.append(float(boundary))
        return tuple(landed)

    def get_last_schedule_timings(self) -> list[float]:
        """Return the list of per-node timings (in seconds) for the last frame, in schedule order."""
        return list(self._last_timings)

    def get_latest_metrics(self, node: Union[RoundNode, AdvanceNode, ControllerNode]) -> Optional[Metrics]:
        """Return the latest Metrics observed at the given node, if any."""
        st = self._stats.get(node.label)
        return st.last_metrics if st else None

    def get_attempted_dts(self, node: Union[RoundNode, AdvanceNode, ControllerNode]) -> List[float]:
        st = self._stats.get(node.label)
        return list(st.attempted) if st else []

    # ------------------------------ helpers
    def _advance_children(
        self, parent_round: RoundNode, dt: float, state_table: Any
    ) -> tuple[bool, Metrics]:
        """Advance all children to cover dt; returns a composite Metrics.

        Policy: iterate children in order, delegating dt entirely to each child
        in sequence. This is a conservative default; schedulers can evolve.
        The returned Metrics is the last child's metrics (matches current
        step_with_dt_control contract that uses the engine's metrics to update
        the controller). If no children exist, returns a zeroed Metrics.
        """

        observed: List[Metrics] = []

        def run_one(child, slice_dt: float) -> Metrics:
            if isinstance(child, AdvanceNode):
                st = self._stats.setdefault(child.label, NodeStats())
                st.attempted.append(float(slice_dt))
                if is_enabled():
                    dbg("graph").debug(f"  leaf advance: label={child.label} dt={float(slice_dt):.6g}")
                t0 = time.perf_counter()
                res = child.advance(
                    child.state.state,
                    slice_dt,
                    realtime=False,
                    state_table=state_table,
                )
                self._last_timings.append(time.perf_counter() - t0)
                # Strict contract: must return (ok, Metrics, state)
                if not (isinstance(res, tuple) and len(res) == 3):
                    raise ValueError(
                        f"AdvanceNode advance() must return a three-tuple (ok, Metrics, state); got {type(res)} with len={len(res) if isinstance(res, tuple) else 'n/a'}"
                    )
                ok, m, new_state = res  # type: ignore[misc]
                m = coerce_metrics(m)
                if not ok:
                    return Metrics(
                        max_vel=m.max_vel,
                        max_flux=m.max_flux,
                        div_inf=m.div_inf,
                        mass_err=m.mass_err,
                        osc_flag=m.osc_flag,
                        stiff_flag=m.stiff_flag,
                        sim_frame=m.sim_frame,
                        proc_ms=m.proc_ms,
                        dt_limit=m.dt_limit,
                        error_channels=dict(m.error_channels),
                        hard_failure=True,
                        advanced_dt=m.advanced_dt,
                    )
                st.last_metrics = m
                st.advanced_total += float(slice_dt)
                # Update the StateNode with the returned state
                child.state.state = new_state
                if is_enabled():
                    dbg("graph").debug(f"  leaf metrics: label={child.label} {pretty_metrics(m)}")
                return m
            elif isinstance(child, RoundNode):
                inner = child
                if is_enabled():
                    dbg("graph").debug(
                        f"  inner round: label={inner.label} "
                        f"dt={float(slice_dt):.6g}"
                    )
                res = self._run_scientific_round(
                    inner, float(slice_dt), state_table
                )
                return res.metrics or Metrics(0.0, 0.0, 0.0, 0.0)
            else:
                raise TypeError(f"Unsupported child node type: {type(child)}")

        sched = parent_round.schedule or "sequential"
        children = list(parent_round.children)
        if not children:
            observed.append(Metrics(0.0, 0.0, 0.0, 0.0))
        elif sched == "sequential":
            for child in children:
                metrics = run_one(child, dt)
                observed.append(metrics)
                if metrics.hard_failure:
                    break
        elif sched == "interleave":
            # Break dt into equal slices across children, preserving order.
            slice_dt = dt / max(len(children), 1)
            for child in children:
                metrics = run_one(child, slice_dt)
                observed.append(metrics)
                if metrics.hard_failure:
                    break
        elif sched == "parallel":
            # Cooperative parallel stub: run each child for full dt and combine metrics conservatively.
            # True threading is engine-dependent; this aggregates maxima.
            for child in children:
                observed.append(run_one(child, dt))
        else:
            # Fallback to sequential
            for child in children:
                metrics = run_one(child, dt)
                observed.append(metrics)
                if metrics.hard_failure:
                    break

        combined = _combine_metrics(observed)
        # Also track on the parent round for introspection
        self._stats.setdefault(parent_round.label, NodeStats()).last_metrics = combined
        if is_enabled():
            dbg("graph").debug(
                f"_advance_children: sched={sched} children={len(children)} "
                f"-> {pretty_metrics(combined)}"
            )
        return not combined.hard_failure, combined


@dataclass
class GraphBuilder:
    """Helper to build round graphs from engine registrations.

    Example
    -------
    gb = GraphBuilder(ctrl=STController(dt_min=1e-6), targets=Targets(...), dx=0.1)
    round_node = gb.round(dt=0.016, engines=[fluid_reg, softbody_reg])
    runner = MetaLoopRunner()
    res = runner.run_round(round_node)
    """

    ctrl: STController
    targets: Targets
    dx: float

    def round(
        self,
        dt: float,
        engines: List[EngineRegistration],
        *,
        allow_increase_mid_round: bool = False,
        schedule: str = "sequential",
        realtime_config: RealtimeConfig = None,
        realtime_state: RealtimeState = None,
        parent_label: str = None,
        state_table: Any = None,
    ) -> RoundNode:
        plan = SuperstepPlan(round_max=float(dt), dt_init=float(dt))
        controller = ControllerNode(ctrl=self.ctrl, targets=self.targets, dx=self.dx)
        state_stub = StateNode(state=None)
        children: List[Union[AdvanceNode, RoundNode]] = []
        # Compose a unique label prefix for this round
        label_prefix = parent_label + "/" if parent_label else ""
        for idx, reg in enumerate(engines):
            adv = EngineNode(reg).to_advance_node(state_stub)
            unique_label = f"{label_prefix}{reg.name}_{idx}"
            localize = getattr(reg, "localize", True)
            if realtime_config is not None:
                localize = False
            if len(engines) == 1 and parent_label is not None:
                localize = False
            if reg.solver_config is not None and realtime_config is None:
                def make_adv_with_bisect(reg_local: EngineRegistration, label: str) -> AdvanceNode:
                    def advance(
                        _state_obj: Any,
                        dt: float,
                        *,
                        realtime: bool = False,
                        state_table=None,
                    ):
                        if is_enabled():
                            dbg("graph").debug(
                                f"bisect solve: name={reg_local.name} dt={float(dt):.6g} realtime={realtime}"
                            )
                        if state_table is None:
                            raise ValueError(
                                "bisect engine advance requires StateTable"
                            )
                        m = solve_window_bisect(
                            reg_local.engine,
                            float(dt),
                            reg_local.solver_config,
                            state_table=state_table,
                            registration_name=reg_local.name,
                        )
                        penalty = 0.0
                        targets = getattr(reg_local, "targets", None)
                        if targets is not None:
                            if hasattr(m, "div_inf") and hasattr(targets, "div_max"):
                                if m.div_inf > targets.div_max:
                                    penalty += float(m.div_inf - targets.div_max)
                        m.penalty = penalty
                        if realtime and realtime_state is not None:
                            realtime_state.update_proc_ms(label, getattr(m, 'proc_ms', 0.0), getattr(realtime_config, 'ema_alpha', 0.2))
                        return True, m, _state_obj
                    return AdvanceNode(
                        advance=advance,
                        state=state_stub,
                        label=f"bisect:{label}",
                        transaction_owner=reg_local.engine,
                    )

                e_round = RoundNode(
                    plan=SuperstepPlan(round_max=float(dt), dt_init=float(dt)),
                    controller=ControllerNode(ctrl=self.ctrl, targets=self.targets, dx=self.dx),
                    children=[make_adv_with_bisect(reg, unique_label)],
                    allow_increase_mid_round=False,
                    schedule="sequential",
                    label=f"bisect-round:{unique_label}",
                    state_table=state_table,
                )
                children.append(e_round)
            elif reg.solver_config is not None and realtime_config is not None:
                
                def adv_with_timing(state_obj, dt, *, realtime=False, adv=adv, label=unique_label, state_table=None):
                    # Forward state_table if the underlying advance supports it
                    
                    ok, m, state_new = adv.advance(state_obj, dt, realtime=realtime, state_table=state_table)

                    if realtime and realtime_state is not None:
                        realtime_state.update_proc_ms(label, getattr(m, 'proc_ms', 0.0), getattr(realtime_config, 'ema_alpha', 0.2))
                    return ok, m, state_new
                children.append(AdvanceNode(
                    advance=adv_with_timing,
                    state=state_stub,
                    label=f"advance:{unique_label}",
                    transaction_owner=reg.engine,
                ))
            elif localize:
                e_ctrl = reg.ctrl if reg.ctrl is not None else STController(
                    Kp=self.ctrl.Kp,
                    Ki=self.ctrl.Ki,
                    A=self.ctrl.A,
                    shrink=self.ctrl.shrink,
                    dt_min=self.ctrl.dt_min,
                    dt_max=self.ctrl.dt_max,
                )
                e_controller = ControllerNode(ctrl=e_ctrl, targets=reg.targets, dx=reg.dx)
                nested_round = GraphBuilder(ctrl=e_ctrl, targets=reg.targets, dx=reg.dx).round(
                    dt=dt,
                    engines=[reg],
                    allow_increase_mid_round=False,
                    schedule="sequential",
                    realtime_config=realtime_config,
                    realtime_state=realtime_state,
                    parent_label=unique_label,
                    state_table=state_table,
                ) if not (realtime_config is not None and len(engines) == 1) else None
                if nested_round is not None:
                    if reg.distribution is not None:
                        nested_round.distribution = reg.distribution
                    children.append(nested_round)
                else:
                    def adv_with_timing(state_obj, dt, *, realtime=False, adv=adv, label=unique_label, state_table=None):
                        ok, m, state_new = adv.advance(state_obj, dt, realtime=realtime, state_table=state_table)
                        if realtime and realtime_state is not None:
                            realtime_state.update_proc_ms(label, getattr(m, 'proc_ms', 0.0), getattr(realtime_config, 'ema_alpha', 0.2))
                        return ok, m, state_new
                    children.append(AdvanceNode(
                        advance=adv_with_timing,
                        state=state_stub,
                        label=f"advance:{unique_label}",
                        transaction_owner=reg.engine,
                    ))
            else:
                def adv_with_timing(state_obj, dt, *, realtime=False, adv=adv, label=unique_label, state_table=None):
                    ok, m, state_new = adv.advance(state_obj, dt, realtime=realtime, state_table=state_table)
                    if realtime and realtime_state is not None:
                        realtime_state.update_proc_ms(label, getattr(m, 'proc_ms', 0.0), getattr(realtime_config, 'ema_alpha', 0.2))
                    return ok, m, state_new
                children.append(AdvanceNode(
                    advance=adv_with_timing,
                    state=state_stub,
                    label=f"advance:{unique_label}",
                    transaction_owner=reg.engine,
                ))
        # Ensure an integrator finalizes the round by default
        if not any(isinstance(reg.engine, Integrator) for reg in engines):
            integ = Integrator()
            if state_table is not None:
                try:
                    integ.register(state_table, lambda _: {"pos": (0.0, 0.0), "mass": 0.0}, [0])
                except Exception:
                    pass
            int_reg = EngineRegistration(name="integrator", engine=integ, targets=self.targets, dx=self.dx, localize=False)
            adv_int = EngineNode(int_reg).to_advance_node(state_stub)
            int_label = f"{label_prefix}integrator_{len(children)}"
            children.append(AdvanceNode(
                advance=adv_int.advance,
                state=state_stub,
                label=f"advance:{int_label}",
                transaction_owner=adv_int.transaction_owner,
            ))

        return RoundNode(
            plan=plan,
            controller=controller,
            children=children,
            allow_increase_mid_round=allow_increase_mid_round,
            schedule=schedule,
            label=label_prefix.rstrip("/") or "root",
            state_table=state_table,
        )


__all__ = [
    "StateNode",
    "AdvanceNode",
    "ControllerNode",
    "EngineNode",
    "RoundNode",
    "MetaLoopRunner",
    "GraphBuilder",
]
