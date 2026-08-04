# WASM mixed-extent output handoff

Status date: 2026-08-03

## Scope

This handoff covers one open design decision in the Turing WASM fused-program
backend: what to do when a single `FusedProgram` both (a) reduces an axis away
(producing a value of extent `N`) and (b) wants to materialize a value that
still spans the reduced axis (extent `N*K`, "grid"-shaped) as a public output.

It was triggered by a real `build_site_page.py` build that failed emission with:

```
WebAssembly emission shortfalls:
- step -1 (output): output 'next_position_x' is grid/kaxis-shaped;
  the flat model cannot materialise an N*K output
```

It does not cover the reshape/keepdim NumPy-oracle divergence (tracked
separately) or the direct grid-*feed* ABI-sizing limitation.

## Why the backend is flat (the correct framing)

The WASM backend was built to lower *fused elementwise programs*. Its emitted
function is `run(count, ...offsets)`:

- `count` is a runtime argument; the emitter never derives an extent from
  shapes.
- One `(loop $body)` runs `$i` from `0` to `count`.
- Every output store is `$outN + i*element_bytes` — exactly one element per
  lane.

For a pure elementwise map this is not a compromise. Rank carries no
information the kernel needs: an `(N,K)` tensor and a flat `(N*K,)` tensor are
the same bytes, and one counter over all elements is correct regardless of
rank. The `(count, ...offsets)` ABI with no shape metadata is the *minimum*
correct contract for that workload.

Axis reduction (`_plan_axis_reductions` in
`src/compiler/fused_program_wasm_backend.py`) is the single grafted-on
exception, because a reduction is the one case where rank matters — you must
know `K` to fold it. The planner reconstructs `N` and `K` from `program.meta`
and emits a nested loop (outer `$i` over `N`, inner `$k` over `K`), but it
deliberately keeps the *outer* contract flat: `count = N`, one output per lane.

## The general truth (one rule, not two answers)

A fused program is a DAG of values over a set of iteration axes. Every value
has a **domain** — the subset of axes it spans. There is exactly one placement
rule:

> Evaluate each op, and store each output, at the innermost loop point where
> its full index is available.

Everything the backend does today is a projection of that rule:

| Case | Axis set | Domains | Store site |
|---|---|---|---|
| Pure elementwise | `{flat}` | all equal | outer loop, at `i` |
| Axis reduction | `{N, K}`, K folded | reduced → `{N}` | outer loop, at `i` |
| **Mixed (this case)** | `{N, K}` | grid → `{N,K}`, reduced → `{N}` | grid at inner `i*K+k`, reduced at outer `i` |

The mixed case is **not** new topology. It is the same `{N,K}` nest the
reduction path already builds. `next_position_x` simply has domain `{N,K}` and
must be stored inside the inner fold loop; the reduced outputs have domain
`{N}` and store in the outer loop.

The tell that this is one rule, not a special case: the backend **already
computes every value's domain** (`classify()` → `value_class`, labeling
`grid`/`row`/`kaxis`/`scalar`, which *is* the axis domain
`{N,K}`/`{N}`/`{K}`/`{}`), and the inner loop **already computes the `i*K+k`
address** for grid *loads* (`index_lines`). The only non-general piece in the
entire backend is that the output-store pass (`emit_wasm_module`, the block that
walks `output_ids` and stores at `$outN + i*element_bytes`) is hardcoded to the
outer loop, one element per lane. It can therefore only place `{N}`/`{}`-domain
outputs, and grid/kaxis outputs are rejected at
`_plan_axis_reductions` (the `output` shortfall).

So the rejection is a **placement gap**, not a topological limit.

## Option A — Domain-driven store pass within the existing 2-axis nest

Make the store pass place each output by its domain, using loop machinery that
already exists.

**What changes**
- `_plan_axis_reductions`: stop rejecting `grid`/`kaxis` outputs. Instead,
  record which outputs are inner-domain and which grid/kaxis values must stay
  live long enough to be stored (they are currently transient inside a single
  reduction's fold loop).
- Emission: for grid/kaxis outputs, emit the store inside the inner `$k` loop
  at the address `index_lines` already computes (`i*K+k` for grid, `k` for
  kaxis). Row/scalar outputs keep storing in the outer loop.
- A grid output value that is not naturally a dependency of any reduction needs
  its own inner `$k` loop (a "materialization loop") that recomputes the grid
  value and stores it. When the grid value *is* already recomputed inside an
  existing reduction's fold loop, the store can piggyback on that loop.
- Both emitters must move in lockstep: WAT (`_emit_reduction_body_wat` /
  `_step_instructions`) and binary (`_assemble` reduction path via
  `CodeBuilder`). They share `_plan_axis_reductions`, so the plan-level changes
  propagate; only the store-emission code is duplicated and must be edited in
  both.
- `wasm_fidelity._invocation_extent` and the run-offset/plan logic must learn
  that a grid output occupies `N*K` elements in memory (the reduced outputs
  still occupy `N`). The oracle already produces the correct grid array, so the
  comparison is just a matter of sizing the output buffer.

**Pros**
- Bounded, uses the existing `{N,K}` nest and the existing domain
  classification. No new IR, no polyhedral machinery.
- Directly unblocks `next_position_x` and any "reduce-and-also-emit-the-grid"
  program, which is a common physics/neighbor pattern.
- Keeps the flat outer ABI (`count = N`); only output buffer sizing changes.

**Cons**
- Still hardcoded to two axes. A program with two independent reduction axes,
  or a grid output spanning three axes, remains unsupported.
- The "materialization loop" for a grid output with no host reduction adds a
  second inner loop; care needed so it recomputes exactly the inner
  dependencies (mirror `inner_dependencies` construction).
- Two-emitter lockstep is error-prone; needs fidelity coverage for every new
  placement (grid output alone, grid + reduced output, kaxis output).

**Risk**: medium. Contained to the reduction path, but touches both emitters
and the fidelity sizing.

## Option B — Axis-set / loop-nest generalization

Replace the hardcoded 1-axis flat model and 2-axis reduction bolt-on with a
general loop nest built from the program's axis set.

**What changes**
- Introduce an explicit axis model: enumerate the distinct axis extents present
  across `program.meta`, assign each value a domain (ordered subset of axes),
  and build a loop nest over the axes.
- Placement becomes the single general rule: each op is evaluated at the
  innermost point where its inputs' indices exist; each reduction is a fold over
  its axis's loop; each output is stored at its domain's point with a strided
  address computed from the axis order.
- The ABI generalizes from `count` to a tuple of extents (one per axis), or a
  shape vector passed to `run`. Pure elementwise collapses to a single axis
  (today's flat model as the 1-axis instance); axis reduction is the 2-axis
  instance.
- Both emitters are rewritten to consume the nest description rather than the
  ad-hoc `_AxisReductionPlan`. `_plan_axis_reductions` is subsumed by the
  general planner.
- Grid *feeds* (currently rejected because the count ABI cannot size them)
  become expressible, since each feed is sized by its domain's axis extents.

**Pros**
- One rule, uniformly. Removes the "two answers" framing entirely and matches
  the user's stated preference for a general answer.
- Extends to arbitrary axis counts, multiple reduction axes, and grid feeds —
  not just the specific mixed case that failed.
- The classification and address arithmetic that already exist become the
  general primitives rather than special cases.

**Cons**
- Substantially larger. It is a loop-nest / mini-polyhedral construction:
  axis discovery, domain assignment, legal nesting order (reductions constrain
  ordering), broadcast semantics per point, and strided address generation in
  both emitters.
- The ABI change (count → extent tuple) ripples into every caller: the HTML
  shell, `wasm_class_modules`/coordinator wiring, `site_bundle`, the fidelity
  run script, and any existing published pages that assume `run(count, ...)`.
- Higher regression risk against the current green elementwise and reduction
  paths; needs the full fidelity suite plus new multi-axis cases as a safety
  net before cutover.

**Risk**: high. Correct long-term shape, but a backend rewrite with ABI
surface-area changes.

## Recommendation for the decision-maker

The two options are not rivals so much as the same rule realized at two scopes.
Option A **is** the general rule ("store each value at the innermost loop where
its index exists") applied within the loop nest that already exists; Option B is
that rule applied to a loop nest that does not yet exist.

- If `next_position_x` is a genuine, intended `N*K` output and the near-term
  goal is to unblock this and similar reduce-and-emit-grid programs, **Option A**
  delivers the general placement rule at bounded cost and keeps the flat outer
  ABI.
- If the roadmap needs arbitrary-rank kernels, multiple reduction axes, or grid
  feeds — i.e. the flat/2-axis projections are becoming a recurring wall —
  **Option B** is the correct investment, and Option A's store-by-domain logic
  is a strict subset of it (so A is not throwaway work if B follows).

Open question that gates both: is `next_position_x` legitimately grid-shaped, or
did an upstream keepdim/reshape fail to collapse it back to `{N}`? If the
latter, neither option is needed for this build — the fix is upstream shape
propagation, and the grid-output work should be scheduled on its own merits
rather than as a bug fix for this page.

## Key code references

- `src/compiler/fused_program_wasm_backend.py`
  - `_plan_axis_reductions` — domain classification (`classify` / `value_class`)
    and every reduction-shape rejection, including the grid/kaxis *output*
    shortfall.
  - `_emit_reduction_body_wat` — WAT inner fold loop; `index_lines` already
    computes `i*K+k` for grid loads.
  - `emit_wasm_module` — the outer `(loop $body)` over `count` and the
    hardcoded outer-loop output-store pass.
  - `_assemble` — the binary emitter that must move in lockstep with the WAT
    emitter.
- `src/common/tensors/accelerator_backends/c_primitive_program.py`
  - `_CAPTURED_NATIVE_KERNELS` — capture-time op → kernel-kind map (the `prod`
    fix this session; relevant here because grid feeds/outputs are classified
    upstream of the WASM planner).
- `src/compiler/wasm_fidelity.py`
  - `_invocation_extent`, `verify_wasm_module`, `verify_wasm_source` — the
    NumPy-oracle fidelity harness that any new output placement must be
    validated against (it already generates correct grid references).
