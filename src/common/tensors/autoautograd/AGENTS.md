# Autoautograd Whiteboard Pipeline Notes

This directory hosts the whiteboard cache, runtime, and related scheduling glue.
It implements the "spot" integrator design: jobs operate on node-local state with
per-node versions and stress-based updates.

## Design Spec Recap
- **Timeless integration**: updates depend on the multiset of impulses, not wall-clock.
- **Node-local causality**: each node has a version counter; cache keys combine op identity with these tokens.
- **Cache packages**: `(forward, grads)` are reusable when node versions and op identity match.
- **Single-backward discipline**: each whiteboard session runs exactly one forward/backward pair.
- **Queue → triage → cache-or-compute → bin → whiteboard → separate → return** pipeline batches misses by shape.

## Current Implementation & Drift
- Modules are stubs only; no cache backend, whiteboard execution, or batching logic exists yet.
- Root-level scripts still call the legacy bridge and ignore these modules.
- Node version tracking and stress integrator hooks are not wired, so cache safety and
  error-relative updates are theoretical.
- Tests fail outside of a narrow matmul case; expect breakage until the bridge and
  scheduling layers are implemented.

## Potential Issues / Compromises
- Stubs risk diverging from the pure math spec if filled piecemeal.
- Cache keys depend on version or value digests; without a concrete hash policy we
  may see unintended collisions.
- Batch whiteboard execution will require allocator pooling; failure to honour
  the NumPy-first policy could violate `src/common/tensors/AGENTS.md`.
- Integrator remains time-free but the rest of the repo assumes epoch counters;
  migrating will need careful coordination.

## Migration Guidance
- Root-level scripts (`backend.py`, demos in repository root) should gradually
  replace per-call packing with `integration/bridge_v2.push_impulses_from_op_v2`.
- When implementing, forward results should be emitted via `ResultSink` instead of
  printing to stdout.
- The legacy autograd bridge can be deleted once `bridge_v2` is wired through
  `Ops.call` and tests cover caching behaviour.

Run targeted `pytest` suites after edits. Follow the NumPy-first policy and avoid
introducing heavy dependencies.
