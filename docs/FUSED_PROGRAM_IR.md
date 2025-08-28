FusedProgramIR
===============

Status: Draft (Unified Program IR; training/eval integrated)

Purpose
-------
Define a tape‑free, scheduler‑driven, self‑contained program for any
AbstractTensor model. The IR is the single source of truth for composition and
codegen: a single modular function that runs deterministically and accepts a
`training: bool` mode. We do not split into forward/backward IRs; operators
that behave differently in training vs evaluation honor the flag internally.

Goals
-----
- Tape‑free: No autograd tape is referenced or required by the IR.
- Scheduler‑driven: Step order is exactly the ILPScheduler order.
- Minimal boundary: Only bind feeds (inputs, parameters, buffers). All other
  values are produced by steps.
- Unified mode: One modular function with `training: bool`; ops that differ by
  mode must honor it, others ignore it.
- Precise diagnostics: Report exact “cut‑wire” locations with shapes, ids, op
  names and neighbor context.
- Codegen‑ready: Translates directly into one function or fused kernel chain.

Scope
-----
- Phase 1 (this doc): Unified IR schema and builder from the scheduled
  process graph; program runner semantics with a `training` flag. No demo work.
- Phase 2: State update semantics and codegen mapping for training updates.

Core Concepts
-------------
- Feeds: External values bound at call time. Exactly the tensor nodes with no
  producing op in the scheduled graph (inputs, parameters, buffers). Constants
  are modeled as creation steps, not feeds.

- Steps: Linearized op nodes in strict ILPScheduler order. Each step names an
  AbstractTensor method and its immutable attrs. Steps may be declared
  `mode_sensitive: true` if their semantics differ when `training=True`.

- Outputs: Named ids to return to the caller (e.g., `pred`, `loss`, optionally
  projections of intermediate tensors). The IR does not require a `loss`.

- State (optional): Subset of feeds considered stateful (parameters, moving
  stats, optimizer slots). Phase 2 may add explicit `state_out` mapping to
  return updated state when `training=True`.

- Meta (optional): Per‑id shape/dtype/device snapshots for diagnostics.

IR Schema (unified)
-------------------
```
FusedProgram:
  version: int
  feeds: set[int]                 # ids to bind at call time
  steps: list[OpStep]
  outputs: dict[str, int]         # e.g., {pred: id, loss: id}
  state_in: set[int] | null       # optional, subset of feeds
  meta: dict[int, Meta] | null    # id -> {shape, dtype, device}

OpStep:
  step_id: int
  op_name: str                    # AbstractTensor method name
  input_ids: list[int]
  attrs: dict[str, Any]
  result_id: int
  mode_sensitive: bool            # if op semantics differ when training
```

Scheduler Integration
---------------------
1. Build the composed process graph (no backward/tape/control nodes). The graph
   must expose:
   - tensor nodes (data) and op nodes (operations)
   - edges: tensor → op (inputs), op → tensor (results)
2. Feed the process graph to ILPScheduler via GraphTranslator; capture per‑node
   levels.
3. Linearize steps in increasing level order. Within the same level, use a
   deterministic secondary key (e.g., node id) for stability.
4. Feeds = all tensor nodes with no producing op in the graph.
5. Steps = all op nodes; for each op node:
   - input_ids = predecessor tensor ids
   - result_id = successor tensor id (the primary result)
   - attrs = immutable parameters attached to the op node
   - mode_sensitive = true if the op has training/eval differences
6. Outputs = caller‑declared named ids (e.g., prediction, loss). No special
   casing of “forward” or “backward”.

Execution Semantics (Program Runner)
-----------------------------------
Inputs:
- `program: FusedProgram`
- `feeds: dict[int, AT]` (must bind all `program.feeds`)
- `training: bool`

Replay:
1. Allocate a value store. Insert all bound feeds.
2. For each `step` in order:
   - Gather `input_ids` from the store.
   - Dispatch `op_name` via the AbstractTensor method with `attrs`.
   - If `step.mode_sensitive` is true, pass the `training` flag to the op
     (ops that ignore it are fine; those that differ must honor it).
   - Store the result at `result_id`.
3. Produce named outputs from `program.outputs` ids.
4. (Phase 2) If `training=True` and the IR encodes state updates, also project
   updated state; otherwise, state is identity passthrough.

Diagnostics (cut‑wire reporting)
--------------------------------
On any failure (e.g., missing feed, unknown op, shape mismatch), the runner
emits a structured error with:
- step_id, op_name
- input_ids, missing/bound status, and meta if available
- result_id and expected vs actual shape (when known)
- neighbor context: previous producers of the inputs; next consumers of the
  result (limited depth) to show the break both sides.

Binding Contract
----------------
- Ids are stable integers tied to graph construction. The caller must bind the
  exact leaves the IR declares.
- Tensors must be AbstractTensor values (no raw backend scalars). The execution
  engine uses the AbstractTensor op surface exclusively.
- Constants are encoded as creation steps (e.g., zeros/full) with attrs, not as
  feeds.

Determinism and Versioning
--------------------------
- The IR carries a `version`. Any change to op dispatch or attrs semantics must
  bump the version.
- Steps are ordered deterministically (level + id). Execution and codegen must
  not reorder.

Gradient/Update Semantics (Phase 2)
-----------------------------------
- Model parameter/buffer updates as explicit state in/out on the same unified
  program. No separate backward tape or duplicate IR.
- Two acceptable implementations:
  1) Encode optimizer math as additional steps producing new state tensors.
  2) Treat updates as side‑effectful ops that also project updated state ids.
- With `training=False`, updates are disabled while ops like dropout/batchnorm
  honor evaluation behavior.

Codegen Path (Outline)
----------------------
1. Convert `steps` into a single function in the C backend (or any target).
   Emit only the operators used; represent `training` as a function parameter.
2. Bind feeds as function arguments (inputs, params, buffers). Inline creation
   steps or preallocate buffers.
3. Use scheduler levels for lifetime planning and buffer allocation.

Integration Hooks (No Demo Yet)
-------------------------------
1. Capture with the audit session (strict checks on capture).
2. Build `FusedProgram` from the scheduled process graph.
3. Provide a `ProgramRunner(program)` callable that accepts `feeds` and
   `training: bool` and returns named outputs (and, in Phase 2, updated state).

Appendix: Dynamic Operator Resolution
------------------------------------
There is no “minimal operator set.” The IR uses the AbstractTensor method names
directly. Operator resolution follows these rules:

- Instance‑based resolution: The program runner accepts an execution instance
  (e.g., an AbstractTensor or a backend tensor). It derives the class and
  resolves each `op_name` by checking that the class defines that method. This
  ensures we call exactly the same methods the model used, across both the
  autograd tape and the graph‑based total execution, assuming consistent method
  usage by the user.

- Parameter aliasing: Where backends accept aliased parameter names, a thin
  normalization layer adapts `attrs` to the resolved method signature without
  changing semantics.

- Method union (future): We will compute the union of methods across all
  registered backends and the AbstractTensor surface and assign unique
  enumerations for codegen. That mapping will allow us to know not only which
  operation is desired, but also which backend provides it and where it lives.
  This is deferred; do not block current IR on it.

- Fallback policy: If a method is missing on the execution class, emit a
  cut‑wire diagnostic naming the op, expected aliases (if any), and the class
  inspected.
