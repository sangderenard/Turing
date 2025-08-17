# DT Graph Design and Migration Plan

This document evaluates the current managed-dt system and proposes a graph-based organization that enables arbitrary access to details from any subset in nested loops, with a standardized meta-loop runner.

## Current State (Managed dt)

- Controllers: `src/cells/bath/dt_controller.py` provides `STController`, `Targets`, `Metrics`, `step_with_dt_control_*`, and `run_superstep`/`run_superstep_plan`.
- Contracts:
  - `advance(state, dt) -> (ok, Metrics)` dictates the micro-step API.
  - `SuperstepPlan`/`SuperstepResult` in `src/common/dt.py` define a frame window and result.
- Capabilities:
  - PI-smoothed dt proposal, CFL envelope (`update_dt_max`), exact landing, non-increasing dt policy (optional growth mid-round).
  - Nesting is supported ad-hoc (see tests like `test_nested_supersteps_compose`).
- Gaps:
  - No standard way to express an entire hierarchy of controllers/advancers.
  - Introspection across nested loops is manual.
  - Orchestrators re-implement glue code per-engine.

## Proposal: Graph-based Organization

Represent time-stepping as a directed acyclic graph composed of three node types:

- StateNode: Wraps a simulator or subset state.
- AdvanceNode: Binds an `advance(state, dt)` callable to a StateNode.
- ControllerNode: Holds `STController`, `Targets`, `dx`.
- RoundNode: Encapsulates a time window (`SuperstepPlan`) and references a ControllerNode plus children (AdvanceNode or nested RoundNode).

A standardized `MetaLoopRunner` walks a RoundNode tree to run a frame. It delegates dt windows downward using the existing `run_superstep` semantics, collecting per-node `Metrics` and attempted dt values for complete introspection.

### Benefits

- Declarative: The loop topology is data, not code.
- Portable: Works across engines (cellsim, spatial fluids) because it reuses the common dt contracts.
- Introspectable: The runner captures per-node `Metrics`, attempted dts, and totals; any subset can be queried.
- Composable: Nested rounds and multiple children per round.

## Minimal Implementation

`src/common/dt_graph.py` adds:

- Node dataclasses: `StateNode`, `AdvanceNode`, `ControllerNode`, `RoundNode`.
- `MetaLoopRunner` that:
  - Builds an `advance_adapter` to route the outer dt to children.
  - Calls `run_superstep` with the parent controller and returns `SuperstepResult`.
  - Iterates children: `AdvanceNode` calls the engine advance; `RoundNode` recursively constructs an inner plan of size `dt`.
  - Captures `NodeStats` for metrics/attempted/advanced totals.

This is intentionally small but functional, preserving all existing dt semantics while standardizing the meta-loop.

## Migration Path

1. Wrap existing subsystems:
   - Build `StateNode` for each engine/subset (e.g., bath, membrane, organelles).
   - Create `AdvanceNode` adapters from existing `advance(state, dt)` lambdas.
   - Attach `ControllerNode` using current `STController`, `Targets`, `dx`.
   - Compose into per-frame `RoundNode` with `SuperstepPlan(round_max=frame_dt, dt_init=ctrl_dt)`.
2. Replace hand-rolled orchestrators with `MetaLoopRunner.run_round(round_root)`.
3. Expose an introspection panel to query `runner.get_latest_metrics(node)` and `runner.get_attempted_dts(node)`.
4. Incrementally push the representation into `transmogrifier` if desired, by mapping these nodes into `ProcessGraph` types.

## Integration with Transmogrifier (Future)

- Create a thin shim that converts `RoundNode` trees into `ProcessGraph` nodes with roles:
  - `Controller` op with inputs: `dx`, `targets`, `dt_init`; outputs: `dt_next`, `metrics`.
  - `Advance` op consumes `dt` and state refs.
  - `Round` op materializes the run_superstep loop.
- Use existing ProcessGraph scheduling to interleave multiple subsystem rounds or to pipeline them across memory regions.

## Edge Cases and Guarantees

- Empty round (no children): returns inert metrics (all zeros) and proposes `dt_next` based on controller state.
- Child failure: surfaced as large penalty metrics so `run_superstep` halves dt and retries (same as current behaviour).
- Nesting: inner `RoundNode` receives `dt` via a temporary plan `SuperstepPlan(round_max=dt, dt_init=inner.plan.dt_init)`.
- Non-increasing policy: obeyed by `run_superstep` unless `allow_increase_mid_round=True`.

## Next Steps

- Add unit tests mirroring `test_nested_supersteps_compose` but using `MetaLoopRunner`.
- Add simple visualization of the round tree with per-node `dt` attempts.
- Provide convenience builders to reduce boilerplate when wiring complex sims.
