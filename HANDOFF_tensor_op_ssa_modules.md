# Handoff: whole-object DLL emission → modular tensor-op SSA definitions

Branch: `codex/recursive-reduction-bridge`. Session date: 2026-08-11.

## 2026-08-12 static source-call linker continuation

Binary/PE work is paused in `BINARY_HEAD_IR_CONTINUATION.md`; the active focus
is again complete source pursuit and one statically linked repository-SSA
ecosystem.  No long recursive compile was run in this continuation.

Two call-linking defects are fixed and have direct source-ingestion regressions:

1. Source calls are now materialized to a dependency fixed point.  A chain such
   as `root -> middle -> leaf` no longer depends on function-table visitation
   order; resolving the leaf makes its caller eligible in the next round until
   the entire authored chain is ordinary repository-SSA `Call` instructions.
2. Zero-result authored calls are executable calls rather than omitted records.
   Outside retained loops, the process graph's exact source position identifies
   the next produced SSA value and the call is inserted before that producer.
   Calls inside retained loops remain unresolved until their control block owns
   an explicit call marker; hoisting a side effect out of a loop is forbidden.

Supporting invariants:

- source-declared parameters remain in `Function.args` even when their only
  consumer is a `PlanCall` materialized after control lowering;
- call-only functions retain their source output identity, so `return callee()`
  replaces the temporary empty `Ret` with a call and returned value;
- hierarchy callable-shell duplicates are discarded only when a complete frame
  for the same caller/callee exists and the duplicate has no argument/result
  bindings;
- unresolved-call emission diagnostics now report every missing callee storage
  value and every established frame binding.

Focused evidence: `tests/test_static_source_call_linking.py`,
`tests/test_precompile_to_ssa.py`, and `tests/test_function_table.py` pass
40 tests together.  The qualified-symbol whole-object emission test also
passes independently.

Constructor-owned record storage is now linked.  A class-construction node
allocates a distinct caller-owned raw record/sequence arena, invokes the real
source-linked constructor first, and binds later method frames to the same
field storage by the record descriptor's exact storage identity.  The
`Store().has(key)` regression now emits complete Fortran: the five dict-table
ABI values are supplied rather than reported as `missing_frame`.  Repeated
field reads/writes which have distinct local sequence descriptor ids are
correlated to the same physical field arena without dropping either
occurrence.  Two constructor occurrences are tested to receive disjoint
arenas, and authored constructor parameters are passed from their exact caller
operands.  No Python object, opaque handle, runtime dispatcher, scalar
substitute, or numerical projection was introduced.

Bounded evidence after this repair: the record/dict/sequence selections pass
7 tests; the static-call, precompile, and FunctionTable selection passes 40
tests.  The long recursive compile was not run.  Remaining boundaries are
explicit: construction inside retained loops needs a caller-owned instance
pool indexed by the loop, nested record-valued fields need recursive record
descriptor remapping, and constructor defaults require the same authored
default-literal binding already used by ordinary calls.

Retained-loop construction now distinguishes lifetime correctly.  A
non-escaping instance reuses one caller-owned arena, because its authored
constructor resets that arena on every iteration; the constructor is an
ordinary source-linked Call in the loop-body CFG and is never hoisted.  A
one-sequence-field instance which escapes the iteration (for example
`buckets.append(Bucket())`) uses the existing child-table pool ABI: the outer
list stores the induction-row handle, ordinary Mul/GetElementPtr instructions
derive the field row and per-instance length/status cells, and the constructor
initializes that row before the append publishes its handle.  The resulting
module emits complete Fortran.  Constructor calls are now first-class
hierarchy PlanCalls, so call-only loops survive control projection before the
ABI linker runs.  Multi-sequence-field escaping records now use a record-level
instance-pool descriptor grouping one child-pool layout per exact field
identity under the same outer handle. `Pair.left` and `Pair.right` therefore
have independent row strides/data/length/status storage while sharing only the
instance handle, and the complete module emits Fortran.

The multi-field boundary was audited with `Pair.left` and `Pair.right`, two
identically shaped list fields.  Repeated field views are now correlated only
when their exact record storage identity matches; shape equality alone no
longer collapses the fields.  Mixed scalar-plus-sequence records now join the
same record pool: packed scalar record storage gets an instance stride and
field offset, and ordinary Mul/Add/GetElementPtr derives the scalar field row.
The conceptual record identity is removed from the ABI after every consumer is
rewritten to physical field storage; a regression protects against local-id
collision with unrelated capacity values. Nested record-valued fields remain
the next explicit pool boundary.

The bounded loop/control/precompile/static-call selection passes 76 tests.

## 2026-08-11 bootstrap control-hole ledger

The real `ProcessGraph.build_from_ast` compile now reaches whole-object Fortran
emission in about 137--183 seconds instead of the earlier 26-minute / 73-GB
thrash.  Do not collapse the remaining obligations: the compiler reports every
specialized occurrence independently until it is actually lowered.

Current numbered occurrences from the last real compile:

1. `build_from_ast:159`, source line 2210: `existing_classes.add(identity)`.
2. `build_graph:333`, line 2058: `args.extend(value)` and `args.append(value)`.
3. `deduplicate_node:24`, line 1797: `G.remove_node(nid)`.
4. `_expand_unresolved_ast_parents:463`, line 1101:
   `unavailable_identities.add(...)` and `definitions.extend(...)`.
5. `_expand_unresolved_ast_parents:602`, line 1302: generator iteration over
   filtered `ast.walk(module)`; its append effects are already explicit.
6. `_attach_external_methods:75`, line 335: `present.add(target.attr)`.
7. Second specialized occurrence of item 2.
8. Second specialized occurrence of item 3.
9. Third specialized occurrence of item 3.
10. Fourth specialized occurrence of item 3.
11. Fifth specialized occurrence of item 3.
12. Third specialized occurrence of item 2.
13. Sixth specialized occurrence of item 3.
14. Seventh specialized occurrence of item 3.
15. Eighth specialized occurrence of item 3.
16. Ninth specialized occurrence of item 3.
17. Second specialized occurrence of item 4.
18. Second specialized occurrence of item 5.
19. Second specialized occurrence of item 6.

The next non-loop blocker is tracked separately as item 20: an empty list
aggregate reaches Fortran emission as scalar literal `[]` rather than through
its SSA sequence descriptor/arena.

Already fixed in this continuation: generator-backed `extend` now inserts each
yield directly through destination sequence policy (including generator
filters); `while pending` now reloads the sequence length and compares it with
zero at both initial test and latch; parameter memory flags are compiler table
metadata rather than runtime objects; callsite source identity propagates into
formal receivers; and the planner's `object()` missing-value sentinel was
replaced by separate integer status/value tables. No numerical projection or
Python/runtime collection handler was added.

This document summarizes a long working session and sets the forward architecture.
Read the "Corrected direction" section first — several mid-session approaches were
wrong and were reverted; do not resurrect them.

## 2026-08-11 continuation — authoritative tensor-backend status

The source-producing backend requested after the original handoff now exists.
This supersedes the document's proposal to make `pure_backend` the only initial
tensor definition module:

- `SSATensorOperations(AbstractTensor)` records the fundamental tensor surface
  into an `SSATensorProgram`; inherited compound methods therefore expand during
  ingestion and bottom out in the same fundamental calls.
- The C backend is represented by one complete LLVM/repository-SSA reference.
  Tensor lowering copies the required function dependency closure into the
  caller. It never calls an opaque tensor handler and never leaves a runtime
  tensor dispatcher behind.
- `SSATensorTable`/`SSATensorDescriptor` are first-class per-function SSA data:
  logical tensor identity, data value, dtype, static/dynamic shape and strides,
  storage, arena/allocation owner, byte span, view alias, and writability.
  These adopt Nodus's useful record/lease separation without adopting Nodus's
  opaque runtime handle/backend dispatch architecture.
- Incoming `AbstractTensor` leaves can be replaced by SSA-owned input records or
  detached SSA constants. The replacement retains no reference to the original
  object, payload, or backend. Recursive replacement covers nested feed trees.
- Complete ProcessGraph lowering now propagates a replacement feed's shape and
  dtype through planned regions. Hierarchy ID assignment preserves
  `PlanClosure.value_shapes`; it no longer erases them while canonicalizing IDs.
- Proven compound catalog currently includes clamp/clip, derived trigonometric
  reciprocals and ratios, degree/radian conversions, `nan_to_num`, stable
  mean, one-dimensional stable softmax, and one-dimensional stable log-softmax.
  They emit ordinary SSA with no remaining
  `tensor`/`tensor_operation` placeholder. Stable softmax also compiles to a
  Fortran DLL and executes correctly.
- Fortran calls now distinguish a one-element metadata vector from a scalar
  tensor arena. Shaped constants remain array designators for pointer arguments;
  one-element tensor arenas are explicitly indexed for scalar arguments.

Current honest boundary: structural singleton-axis broadcasting is complete
through the source-level `broadcast_double` definition. Stack/cat definitions
exist in the C reference but their pointer-table ABI remains source-only in the
current Fortran emitter. This is an explicit substrate gap, not a runtime
fallback.

Focused verification: 73 tests pass across the SSA tensor reference, complete
ProcessGraph lowering, hierarchy shape preservation, Fortran emission, and two
real DLL execution routes. Numerical projection remains forbidden for this path.

### Abstract-NN and broadcast correctness extension

The smallest representative `abstract_nn` network now compiles correctly:
`Linear(2→8) → tanh → Linear(8→1)`, including ordinary `(1, D)` biases and a
four-row XOR-shaped input. Parameter creation through `from_list_like` preserves
the owning `SSATensorProgram`, so weights and biases become detached SSA
constants rather than NumPy payloads hidden inside an SSA wrapper.

Both routes have native correctness proofs:

- direct `abstract_nn.Model` construction through `SSATensorOperations`;
- Python function ingestion through ProcessGraph, complete-region SSA lowering,
  source-reference linking, generated control ABI, and Fortran DLL execution.

The four native outputs agree with NumPy to a maximum absolute error of
`5.6e-17`. ProcessGraph singleton axes are authoritative and survive hierarchy
ID assignment. Shaped `MatMul` is routed to the row-major `matmul_double` source
instead of Fortran's column-major intrinsic.

The earlier general-broadcast boundary is closed. `broadcast_double` is now a
handwritten repository-SSA reproduction of the C backend function; shaped SSA
arithmetic materializes singleton-axis broadcasts through it before invoking
the ordinary binary kernel. The direct SSA backend uses the same definition.

Performance measurements (262,144 float64 elements, 7 warmups, medians) are
diagnostic only; optimization remains the reducer/dispatcher's later job:

| Program | Fortran | NumPy | Native compile |
|---|---:|---:|---:|
| `tanh(x)` | 6.871 ms | 6.726 ms | 1.33 s |
| `x * 1.25 + 0.5` | 205.756 ms | 0.736 ms | 1.54 s |
| 18-stage GELU/softplus/trig/clamp chain | 1564.530 ms | 40.136 ms | 2.54 s |
| batch-one 2→8→1 MLP | 21.2 µs | 11.1 µs | 1.87 s |

All comparisons matched NumPy (`0` error for the microbenchmarks/MLP and
`6.7e-16` for the long chain). The large cost begins in opcode-dispatched
binary kernels, not the ABI or unary libm loop; leave fusion/specialization to
topological reduction and deployment scheduling.

### First compiler-native bootstrap module

The second half of the outer goal has now begun with a deliberately
backend-neutral compiler module: lexicographical topological ordering. Its ABI
is only two integer edge arrays plus integer order/status outputs. Python owns
arbitrary graph node identities and lexical keys, normalizes them to dense
indices, and reconstructs the returned identities. No AST object, tensor
object, tensor operation, or backend handle crosses the DLL boundary.

`native_compiler_accelerators.py` now provides:

- a semantic-name registry which refuses native registration unless an exact
  Python correctness fallback already exists;
- explicit compile/load/registration (importing the compiler never invokes a
  toolchain);
- a complete `CompiledProgramAPI` record for the Fortran C ABI;
- registration as both the active callable provider and an opportunistic
  pipeline `Foundation`;
- canonical ProcessGraph relabeling through that provider when installed,
  otherwise the unchanged NetworkX implementation.

The native implementation agrees with NetworkX for empty and nontrivial DAGs,
custom lexical keys, equal-key insertion ordering, and cyclic-graph failure.
The focused joint gate is 7 tests: native compile/registration, live canonical
relabeler use, broadcast parity, shaped whole-object ABI, direct abstract-NN
program ownership, and four-row ProcessGraph network DLL execution.
The broader goal-focused gate is 77 tests and includes the complete repository
SSA reference, precompile/control lowering, and hierarchy-shape suites.

The next smallest clean modules, in order, are:

1. dependency-wave/level assignment over the same dense edge-table ABI;
2. dense parent/child offset-table construction used after canonical relabeling;
3. row-major extent/stride table construction (metadata only, never tensor-op
   semantics);
4. batched FNV-1a token calculation over byte-offset tables.

Do not put AST traversal, tensor dispatch, or backend operation policy in these
DLLs. Those remain Python/SSA orchestration; native modules consume and return
complete information tables.

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
  token as a typed signed `integer(c_int64_t)` and compares it directly. The earlier
  float-bit reinterpretation was prohibited numerical projection. `_literal` is the
  universal backstop so no path
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
