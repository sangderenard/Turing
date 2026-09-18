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

## Result and GEMM product adoption — 2026-08-20

Implemented and verified:

- `turing_pool_deploy_span(context, item_count, chunk_size)` partitions a
  finite span with persistent native workers and executes every item once.
- Deployment planning emits frame-level C choices; the native renderer
  consumes literal workers/chunk values and retains serial fallback reasons.
- Site manifests carry `native_deployment_frames`.
- GEMM artifacts prebake runtime ABI order, concrete shape/stride/offset
  records, deterministic bindings, every tile parameter permutation, and
  every launch span. Arbitrary edges use zero-filled margins on one verified
  square module.
- `compile_native_gemm_product()` is now the canonical GEMM product seam. From
  the ordinary bank source and a problem shape it builds/adopts candidate
  cores, performs the composed-path tile decision, chooses workers/chunk,
  prebakes the complete launch matrix, and links the selected LLVM core and
  `turing_pool.c` into one shared library.
- The product deduplicates and parallel-packs each A/B window once per call,
  executes context-bearing lane spans, publishes only valid edge windows, and
  embeds no Python dependency. It exports a serial control using the identical
  packing and lane spans; a failed pool takes that same fallback.
- A 65-cubed integration test proves padded edge tiles for both pooled and
  serial entries against NumPy.

The 256-cubed product selects tile 64 and seven background workers plus the
caller. It measures 4.71 ms / 7.12 GF/s, 6.34x over its native serial control,
5.92x over the original parametric kernel, and 0.27x NumPy, with 1.56e-13
error. At 1024 cubed it selects tile 128 and measures 221.13 ms / 9.71 GF/s,
7.16x over native serial, 8.89x over the original kernel, and 0.31x NumPy,
with 4.69e-13 error.

The generic `fortran_c_shell`/`profiled_c_shell` control renderer remains a
separate adoption seam: it still handles nullary scheduled-region waves, not
GEMM's context-bearing calls. The GEMM product is real and canonical for this
BLAS path; this result does not falsely advertise every native product as
pooled.
