# Handoff: whole-object DLL emission → modular tensor-op SSA definitions

Branch: `nogodsnomasters`. Session date: 2026-08-11.

This document summarizes a long working session and sets the forward architecture.
Read the "Corrected direction" section first — several mid-session approaches were
wrong and were reverted; do not resurrect them.

---

## 1. The goal

Bootstrap the compiler one module at a time. Concretely: compile a whole Python
class (its public method surface) to a native **DLL** where each method is its own
`bind(C)` export and object state flows through field slots — no fused whole-program
reduction, no numeric projection. Then use `AbstractTensor` as the driving test case.

The compiler's job (stated explicitly by the user, late in the session, and it
reframes everything): **pursue the interior of code dependencies until they resolve
to primitives.** Backend-swapping is *Nodus's* job, not this compiler's. Tensor ops
stay tensor ops; their *implementation* is produced by **ingesting the pure-Python
(`pure_backend`) definition into SSA**. No per-backend tensor-op code, ever.

---

## 2. What was built and is CORRECT (committed, verified)

The whole-object emission path (all backend-neutral at the SSA level unless noted):

- **Non-numeric whole-object emission** — `precompile_to_ssa.lower_control_sections_to_ssa`
  + `fortran_c_shell._emit_class_surface_module`. Emits every planned method as its own
  `bind(C)` export from control + planner regions, with NO `FusedProgram`, NO
  `project_public_numerical_program`, NO `valid_precompile` gate. A void `__init__` and
  a `mul`-bearing `scale` lower the same way.
- **Field-slot ABI** — `self` is a field arena; `GetAttr`→`Load(GEP(self,slot))`,
  `setattr`→`Store(...)`. Field ops recovered from the process graph
  (`fortran_c_shell._field_slot_ops`) and injected (`precompile_to_ssa._inject_field_slot_access`)
  in the graph's SOURCE order (memory order — NOT the data-dependency topo sort, which
  reorders a write past a later read). Constant field writes (`self.x = None`/`5`/`"s"`)
  materialize their source. Verified via ctypes: Tiny=21, Vec2=16, Counter increments &
  returns new value, Box getter, Setter.
- **Multi-instance is free** — each object is its own arena (a `bind(C)` pointer); two
  Counter instances keep independent state. No per-class heap; that framing was wrong.
- **Universal SSA indexing** — `ir_indexing.lower_indexing_to_ssa_addressing`:
  `Indexed`/`IndexedStore` → `GetElementPtr` + `Load`/`Store`, the address vocabulary
  every backend shares. Dynamic arrays compile as assumed-size `d(*)` (bind(C) forbids
  assumed-shape, allows assumed-size); array-ness/intent propagate per-method by shared
  value id. Verified: `get(d,1)=20`, `put(99,d,2)` mutates `[10,20,99]`.
- **String/None as universal tokens** — `ir_string_interning.tokenize_ssa_string_constants`:
  a str/bytes `Const` → `string_token` (fnv1a-64), None → reserved `string_table.NONE_TOKEN`
  sentinel, `equal`/`not_equal` over a token tagged `string_compare`. Fortran realizes a
  token as `transfer(<i64>, 0._c_double)` (64-bit identity in the f64 working type) and a
  token compare as an i64 reinterpret. `_literal` is the universal backstop so no path
  leaks a None/str/bytes. `fnv1a_64` accepts bytes. Verified content-addressed identity.
- **Ingestion structural folds** (`transmogrifier/graph/node_special_cases.py`, applied in
  `graph_express2.build_graph`):
  - `fold_constant_getattr` — `getattr(o,"name",default)` with a declared name → `o.name`
    (a static attribute access, so it resolves structurally; the default is dead in AOT).
  - `expand_ellipsis_subscripts` — `o[...]` → `o[tuple([slice(None)]*(o.ndim-k))]`, so the
    subscript grows an explicit `o.ndim` dependency the graph schedules before it fires.
- **Process-graph shape-carrying (NEW, uncommitted-then-committed this session's tail)** —
  `hierarchical_plan.PlanClosure.value_shapes` + `plan_region_to_ssa_instrs` now give each
  region SSA value the shape/dtype from its process-graph node's `DomainNode`
  (`glsl_deployment_strategy` builds `value_shapes`, squeezing size-1 dims → () for scalars).
  This is the fix for the shapeless path (see §3). KEEP THIS.
- **`log(const)` operand layout fix** — `process_graph_fusion.py`: a unary op's lone
  constant operand stays a tensor value instead of being forced into `right_scalar`.

The user's dependency principle threads through all of the above: **the process graph
must grow the right dependencies (data, memory, structural like ndim/shape) so the
schedule is deterministic and everything an operator needs is known before it fires.**

---

## 3. What was WRONG and reverted — do not resurrect

Mid-session, chasing `AbstractTensor` op-by-op, I hand-rolled **per-op decompositions**
in a new `ir_tensor_ops.py` (`cbrt = sign·|x|^(1/3)`, casts → copy/trunc, view ops →
identity) and even added tensor ops to the Fortran `_UNARY` table. These were **reject
versions**: they reinvented — lossily — what `pure_backend` already defines, discarded
shape/dtype, and put tensor semantics in the wrong place. They only "passed" because
every test repro used **scalar** params, so shape was trivial.

Root cause of the whole detour: my path lowered regions via `plan_region_to_ssa_instrs`,
which built `SSAValue(id)` with **no shape** → the entire path was scalar-only → tensor
ops looked handleable when they weren't. Shape lives in the process graph's `DomainNode`,
never in a FusedProgram (reaching for FusedProgram was also wrong — this path must not use
it). `ir_tensor_ops.py` was deleted; the WASM `_VIEW_OPS` consolidation was reverted; the
shape-carrying fix (§2) replaced the shapeless path.

---

## 4. The reframe that is now the plan (pure_backend ingestion)

`src/common/tensors/pure_backend.py` (`class PurePythonTensorOperations(AbstractTensor)`)
implements **every** tensor op as universal simple list-math (`transpose_` is
`[list(row) for row in zip(*data)]`, etc.). These are the primitive-resolvable
definitions. Compiling AbstractTensor "right" = dissolving each op into its pure-backend
body, which is loops/indexing/arithmetic the compiler ingests. This also collapses the
"tensor ops" and "loop control-holes" into ONE problem.

**Experiment (this session):** compile the pure-backend class itself
(`compile_section_to_dll pure_backend.py PurePythonTensorOperations`). It got remarkably
far — dozens of ops lowered to regions (`transpose_`, `permute_`, `squeeze_`, `pad_`,
`unfold2d_` [15 regions], `topk_`, `log_softmax_tensor_`, ...) before a specific bug
(`log(const)` layout, now fixed). The list-math bodies DO resolve to primitives. The
approach is validated.

---

## 5. Forward architecture: modular tensor-op SSA definitions (TO BUILD)

The user's directive: make dual-IR→SSA use a **modular definition of tensor operations** —
objects that house their own SSA and dispense it via **compiled external links OR SSA
source**. The **pure-backend automatic ingestion is the most important initial module.**

Design sketch (to implement):
- A `TensorOpModule` protocol/registry: given a canonical op name (`transpose`, `where`,
  `scatter`, ...), dispense that op's SSA implementation — either as an SSA `Function`
  (source, to inline) or as an external symbol in a compiled DLL (to link, "recognize
  without lowering").
- The **pure-backend ingestion module**: auto-compiles each `PurePythonTensorOperations`
  op method to SSA once (its list-math body), caches it, and dispenses it by op name.
  Built on the existing `compile_ast_aot` / whole-object path.
- Dual-IR→SSA lowering (`lower_control_sections_to_ssa` / region lowering) consults the
  registry to resolve a tensor-op node to its SSA definition instead of emitting an opaque
  primitive — the concrete realization of "pursue dependencies to primitives" and of the
  opportunistic harness ("greedily use the largest compiled span; external links where
  available; SSA source otherwise"). See memory `project-opportunistic-pipeline-harness`.

This keeps tensor ops as tensor ops (identity preserved) while their implementation is
modular, swappable (compiled link vs source), and defined ONCE from pure Python.

---

## 6. Current phase boundary: the LOOP ENGINE (the unified remaining core)

After the `log(const)` fix, `pure_backend` compiles through ~all elementwise/shape/reduction
op bodies and now stops at the **loop engine**. Two concrete first symptoms:
- `PurePythonTensorOperations__pad_cat_`: `loop_carried (carried update value 57/10 has no
  producer inside the loop body)` — raised in `precompile_to_ssa.py` ~line 1323, in the
  control lowering's loop handling. A loop-carried accumulator's per-iteration update value
  is not produced inside the lowered loop body (the body lowering doesn't wire the update,
  likely because it comes from a region/nested construct the loop body lowering doesn't emit).
- `stack_`: `control-hole stack_ loop 35 [Raise]` — the loop is left entirely unlowered.

THIS IS THE KEY INSIGHT: the pure-backend op bodies are list-building loops with carried
accumulators, so **compiling pure_backend == clearing the AbstractTensor loop control-holes
== implementing the tensor ops**. All one problem: lower list-building loops with carried
accumulators (comprehensions and accumulator `for` loops that append/build a result). This
is the next major engine piece; it is NOT a quick fix.

## 7. Immediate next steps (in order)
1. **Loop engine** — lower list-building loops with carried accumulators. Start from the two
   symptoms above: (a) wire a carried accumulator's update producer inside the loop body;
   (b) lower a `[Raise]` control-hole loop (`stack_`). This unblocks most of pure_backend.
2. Get `pure_backend` to compile end-to-end (more bugs will surface; fix as they come).
3. Stand up the `TensorOpModule` registry + the pure-backend ingestion module (the plumbing
   can be scaffolded now; it becomes useful as ops compile, which the loop engine gates).
4. Wire region lowering to resolve tensor-op nodes via the registry.
5. Re-run `AbstractTensor` — the shape-dependent ops (`transpose`/`scatter`/`extent`/`where`/
   `max`-axis) resolve through their ingested pure-backend SSA.

Key files: `precompile_to_ssa.py`, `fortran_c_shell.py`, `hierarchical_plan.py`,
`glsl_deployment_strategy.py` (region construction ~1499), `ir_indexing.py`,
`ir_string_interning.py`, `string_table.py`, `node_special_cases.py`,
`process_graph_fusion.py`, `common/tensors/pure_backend.py`.
