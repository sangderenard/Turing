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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .dt import SuperstepPlan, SuperstepResult
from .dt_controller import STController, Targets, Metrics, run_superstep
from .engine_api import EngineRegistration
from .debug import dbg, is_enabled, pretty_metrics
from .dt_solver import solve_window_bisect
from .state_table import GLOBAL_STATE_TABLE, sync_engine_from_table, publish_engine_to_table


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
        def advance(state_obj: Any, dt: float, *, realtime: bool = False):
            if is_enabled():
                dbg("engine").debug(f"step: name={self.registration.name} dt={float(dt):.6g} realtime={realtime}")
            sync_engine_from_table(self.registration.engine, self.registration.name, GLOBAL_STATE_TABLE)
            eng = self.registration.engine
            ok, metrics, state_new = eng.step_with_state(state_obj, float(dt), realtime=realtime)  # type: ignore[misc]
            if realtime:
                # Penalty: accumulate error if controller metric violated (e.g., max_vel, div_inf, etc.)
                # This is a placeholder: you may want to customize which metric and how penalty is computed
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
                # You can log or accumulate penalty as needed (e.g., attach to metrics, or log elsewhere)
                metrics.penalty = penalty
            publish_engine_to_table(self.registration.engine, self.registration.name, GLOBAL_STATE_TABLE)
            if is_enabled():
                dbg("engine").debug(
                    f"done: name={self.registration.name} ok={ok} metrics=({pretty_metrics(metrics)})"
                )
            return bool(ok), metrics, state_new

        return AdvanceNode(advance=advance, state=state, label=f"advance:{self.registration.name}")


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


# 
# Introspection and runner
# 


@dataclass
class NodeStats:
    last_metrics: Optional[Metrics] = None
    attempted: List[float] = field(default_factory=list)
    advanced_total: float = 0.0


class MetaLoopRunner:
    """Standardized meta-loop runner that executes a RoundNode tree.

    Key properties:
    - Uses existing run_superstep/run_superstep_plan to guarantee identical
      dt semantics.
    - Captures per-node metrics and attempted dt values for introspection.
    - Supports arbitrary nesting depth.
    """

    def __init__(self) -> None:
        self._stats: Dict[int, NodeStats] = {}
        self._process_graph = None
        self._adapter = None
        self._root_round = None
        self._ilp_scheduler = None
        self._schedule = None
        self._adv_map = None
        self._schedule_method = "asap"
        self._schedule_order = "dependency"
        self._last_timings: list[float] = []  # timings for last frame, in schedule order

    def set_process_graph(self, round_node: RoundNode, *, schedule_method: str = "asap", schedule_order: str = "dependency") -> None:
        """Build and store a process graph, adapter, and schedule from the given RoundNode. Only called once unless graph changes."""
        from .dt_process_adapter import DtToProcessAdapter
        from ...transmogrifier.ilpscheduler import ILPScheduler
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

    def run_round(
        self,
        round_node: Optional[RoundNode] = None,
        dt: Optional[float] = None,
        *,
        realtime: bool = False,
    ) -> SuperstepResult:
        """Execute one frame. Optionally accept a RoundNode on first call."""
        if round_node is not None:
            self.set_process_graph(round_node)
        if self._schedule is None or self._root_round is None:
            raise RuntimeError("No process graph set. Call set_process_graph() first.")
        # Use the round's requested window unless overridden
        step_dt = dt if dt is not None else self._root_round.plan.round_max
        import time
        last_metrics = None
        self._last_timings = []
        for adv in self._schedule:
            if realtime:
                t0 = time.perf_counter()
                ok, metrics, new_state = adv.advance(adv.state.state, step_dt, realtime=True)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                self._last_timings.append(elapsed)
            else:
                ok, metrics, new_state = adv.advance(adv.state.state, step_dt, realtime=False)
                self._last_timings.append(0.0)
            adv.state.state = new_state
            last_metrics = metrics
        from .dt import SuperstepResult
        return SuperstepResult(
            advanced=step_dt,
            dt_next=step_dt,
            steps=len(self._schedule),
            clamped=False,
            metrics=last_metrics if last_metrics is not None else None,
        )

    def get_last_schedule_timings(self) -> list[float]:
        """Return the list of per-node timings (in seconds) for the last frame, in schedule order."""
        return list(self._last_timings)

    def get_latest_metrics(self, node: Union[RoundNode, AdvanceNode, ControllerNode]) -> Optional[Metrics]:
        """Return the latest Metrics observed at the given node, if any."""

        st = self._stats.get(id(node))
        return st.last_metrics if st else None

    def get_attempted_dts(self, node: Union[RoundNode, AdvanceNode, ControllerNode]) -> List[float]:
        st = self._stats.get(id(node))
        return list(st.attempted) if st else []

    # ------------------------------ helpers
    def _advance_children(self, parent_round: RoundNode, dt: float) -> Metrics:
        """Advance all children to cover dt; returns a composite Metrics.

        Policy: iterate children in order, delegating dt entirely to each child
        in sequence. This is a conservative default; schedulers can evolve.
        The returned Metrics is the last child's metrics (matches current
        step_with_dt_control contract that uses the engine's metrics to update
        the controller). If no children exist, returns a zeroed Metrics.
        """

        last_metrics: Optional[Metrics] = None

        def run_one(child, slice_dt: float) -> Metrics:
            if isinstance(child, AdvanceNode):
                st = self._stats.setdefault(id(child), NodeStats())
                st.attempted.append(float(slice_dt))
                if is_enabled():
                    dbg("graph").debug(f"  leaf advance: label={child.label} dt={float(slice_dt):.6g}")
                res = child.advance(child.state.state, slice_dt)
                # Strict contract: must return (ok, Metrics, state)
                if not (isinstance(res, tuple) and len(res) == 3):
                    raise ValueError(
                        f"AdvanceNode advance() must return a three-tuple (ok, Metrics, state); got {type(res)} with len={len(res) if isinstance(res, tuple) else 'n/a'}"
                    )
                ok, m, new_state = res  # type: ignore[misc]
                if not ok:
                    return Metrics(max_vel=m.max_vel, max_flux=m.max_flux, div_inf=1e9, mass_err=1e9)
                st.last_metrics = m
                st.advanced_total += float(slice_dt)
                # Update the StateNode with the returned state
                child.state.state = new_state
                if is_enabled():
                    dbg("graph").debug(f"  leaf metrics: label={child.label} {pretty_metrics(m)}")
                return m
            elif isinstance(child, RoundNode):
                inner = child
                inner_plan = SuperstepPlan(round_max=float(slice_dt), dt_init=max(inner.plan.dt_init, 1e-30))
                saved = inner.plan
                inner.plan = inner_plan
                try:
                    if is_enabled():
                        dbg("graph").debug(f"  inner round: label={inner.label} dt={float(slice_dt):.6g}")
                    res = self.run_round(inner)
                    return res.metrics
                finally:
                    inner.plan = saved
            else:
                raise TypeError(f"Unsupported child node type: {type(child)}")

        sched = parent_round.schedule or "sequential"
        children = list(parent_round.children)
        if not children:
            last_metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)
        elif sched == "sequential":
            for child in children:
                last_metrics = run_one(child, dt)
        elif sched == "interleave":
            # Break dt into equal slices across children, preserving order.
            slice_dt = dt / max(len(children), 1)
            for child in children:
                last_metrics = run_one(child, slice_dt)
        elif sched == "parallel":
            # Cooperative parallel stub: run each child for full dt and combine metrics conservatively.
            # True threading is engine-dependent; this aggregates maxima.
            agg = None
            for child in children:
                m = run_one(child, dt)
                if agg is None:
                    agg = m
                else:
                    agg = Metrics(
                        max_vel=max(agg.max_vel, m.max_vel),
                        max_flux=max(agg.max_flux, m.max_flux),
                        div_inf=max(agg.div_inf, m.div_inf),
                        mass_err=max(agg.mass_err, m.mass_err),
                    )
            last_metrics = agg
        else:
            # Fallback to sequential
            for child in children:
                last_metrics = run_one(child, dt)

        if last_metrics is None:
            # empty sequence: return inert metrics
            last_metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)
        # Also track on the parent round for introspection
        self._stats.setdefault(id(parent_round), NodeStats()).last_metrics = last_metrics
        if is_enabled():
            dbg("graph").debug(
                f"_advance_children: sched={sched} children={len(children)} -> {pretty_metrics(last_metrics)}"
            )
        return last_metrics


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
    ) -> RoundNode:
        plan = SuperstepPlan(round_max=float(dt), dt_init=float(dt))
        controller = ControllerNode(ctrl=self.ctrl, targets=self.targets, dx=self.dx)
        state_stub = StateNode(state=None)
        children: List[Union[AdvanceNode, RoundNode]] = []
        for reg in engines:
            adv = EngineNode(reg).to_advance_node(state_stub)
            if reg.solver_config is not None:
                # Wrap engine with a RoundNode whose adapter uses the bisect solver
                def make_adv_with_bisect(reg_local: EngineRegistration) -> AdvanceNode:
                    def advance(_state_obj: Any, dt: float, *, realtime: bool = False):
                        if is_enabled():
                            dbg("graph").debug(
                                f"bisect solve: name={reg_local.name} dt={float(dt):.6g} realtime={realtime}"
                            )
                        if realtime:
                            # In realtime, just do one bisect solve and accumulate penalty
                            m = solve_window_bisect(reg_local.engine, float(dt), reg_local.solver_config)  # type: ignore[arg-type]
                            penalty = 0.0
                            targets = getattr(reg_local, "targets", None)
                            if targets is not None:
                                if hasattr(m, "div_inf") and hasattr(targets, "div_max"):
                                    if m.div_inf > targets.div_max:
                                        penalty += float(m.div_inf - targets.div_max)
                            m.penalty = penalty
                            return True, m, _state_obj
                        else:
                            m = solve_window_bisect(reg_local.engine, float(dt), reg_local.solver_config)  # type: ignore[arg-type]
                            return True, m, _state_obj

                    return AdvanceNode(advance=advance, state=state_stub, label=f"bisect:{reg_local.name}")

                # Build a single-child round that directly consumes the parent slice via bisect
                e_round = RoundNode(
                    plan=SuperstepPlan(round_max=float(dt), dt_init=float(dt)),
                    controller=ControllerNode(ctrl=self.ctrl, targets=self.targets, dx=self.dx),
                    children=[make_adv_with_bisect(reg)],
                    allow_increase_mid_round=False,
                    schedule="sequential",
                    label=f"bisect-round:{reg.name}",
                )
                children.append(e_round)
            elif getattr(reg, "localize", True):
                # Wrap in a nested round so the engine can run finer than the parent request
                e_ctrl = reg.ctrl if reg.ctrl is not None else STController(
                    Kp=self.ctrl.Kp,
                    Ki=self.ctrl.Ki,
                    A=self.ctrl.A,
                    shrink=self.ctrl.shrink,
                    dt_min=self.ctrl.dt_min,
                    dt_max=self.ctrl.dt_max,
                )
                e_controller = ControllerNode(ctrl=e_ctrl, targets=reg.targets, dx=reg.dx)
                e_round = RoundNode(
                    plan=SuperstepPlan(round_max=float(dt), dt_init=float(dt)),
                    controller=e_controller,
                    children=[adv],
                    allow_increase_mid_round=False,
                    schedule="sequential",
                    label=f"round:{reg.name}",
                )
                # Apply per-engine distribution hook if provided
                if reg.distribution is not None:
                    e_round.distribution = reg.distribution
                children.append(e_round)
            else:
                children.append(adv)
        return RoundNode(
            plan=plan,
            controller=controller,
            children=children,
            allow_increase_mid_round=allow_increase_mid_round,
            schedule=schedule,
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
