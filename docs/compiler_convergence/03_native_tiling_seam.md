# Task 3 — carry deployment choices into native products

## Observed code state

- `plan_region_deployments()` already produces per-region backend choices with
  `strategy`, `workers`, `chunk`, nesting depth, and reasons.
- `site_bundle.py` serializes those choices into
  `compiler.region_deployment_strategies`, but this is bundle metadata rather
  than native execution.
- `render_pooled_control_c()` is compiled and run by
  `tests/test_deployment_native_emission.py`.  It accepts one global `workers`
  value and emits `turing_pool_deploy(..., lane_count, 1)`, so the planned chunk
  never reaches the C runtime.
- The emitter handles only waves of nullary scheduled-region calls.  Intra-
  region chunking needs a context-bearing ABI and is explicitly absent.
- `WorkContract` currently accepts only `deployment="serial"`; native pooling
  cannot be advertised through the contract until the emitted product consumes
  it.

## Work sequence

1. Define the exact adapter from `RegionDeploymentPlan` to control waves.  A
   wave may pool only if all participating regions choose the C/LLVM pool and
   their choices are compatible; otherwise retain the current serial render.
2. Extend the renderer to consume a plan or a minimal immutable native-plan
   record.  Carry workers and chunks explicitly; do not infer them again.
3. Keep the current lane-level nullary ABI as the first integrated seam.  Add
   context-bearing intra-region chunking only with a separate correctness test
   proving disjoint spans.
4. Wire the renderer into the actual native C-shell emission point and link
   `turing_pool.c` only when pooled waves exist.
5. Persist the consumed plan and fallback reasons in product metadata.
6. Once the path is real, extend `WorkContract` deployment values and tests so
   an accepted nonserial contract is never a lie.

## Acceptance

- A planned worker and chunk choice appears literally in emitted C.
- The built artifact executes every region exactly once and matches the serial
  fallback.
- Nested plans respect tempered worker budgets and never recursively
  oversubscribe the pool.
- A missing/failed pool still executes the serial SSA fallback.
- The finished executable has no Python runtime or `HostDeploymentPool`
  dependency.
