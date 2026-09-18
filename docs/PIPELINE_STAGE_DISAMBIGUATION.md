# Pipeline stage disambiguation

> **Application-entry warning:** The numerical objects and emitters described
> below are internal submechanisms reached only after Python/AST ingestion and
> compiler planning.  Their deliberately narrow shapes do not define the
> source language accepted by the Python recompiler.  Application code starts
> from Python source through the AST/ProcessGraph frontend; it must not create a
> `FusedProgram`, call `lower_fused_program_to_ssa`, or call a fused-program
> backend emitter as a substitute compiler path.

This repo has several representations that all sit "between the math and the
machine code" and are easy to conflate. This doc names each one precisely,
says what it actually is, what produces it, and what consumes it — grounded
in the current code, not the intended architecture.

There are **two parallel, currently-unconnected pipelines**. Keep that in
mind while reading:

- **Pipeline A (precompiler):** ProcessGraph → FusedProgram + ControlProgram
  → one unified SSA `IRModule`. This is stages 1–4 below.
- **Pipeline B (tape JIT backends):** a captured autograd tape → directly to
  a backend-specific target (C text, LLVM IR text, GLSL text, or — as of
  this session — a real `transmogrifier.ssa.Function`) → compiled and run.
  This is stage 5.

Nothing in the repo today converts a tape into a `FusedProgram`, or a
`FusedProgram` into a tape. The two pipelines produce structurally similar
artifacts (both eventually reach op-level instructions) but do not call into
each other. `fortran_jit_backend.py` is the one partial exception: it
reuses the SSA *dataclasses* from stage 4, populated directly from the tape,
without going through stages 1–3 or `precompile_to_ssa.py` at all.

---

## 1. Process graph / graph translation

**What it is:** `ProcessGraph` — `src/transmogrifier/graph/graph_express2.py:640`.
A `networkx.DiGraph` (optionally backed by a `BitTensorMemoryGraph`) where
each node is one tensor operation and edges carry the operand *role*
(`lhs`/`rhs`/`operand`/`result`/...). This is the semantic dataflow DAG:
what depends on what, and how — with no notion yet of fusion, scheduling, or
which backend will run it.

**What produces one:** several sources feed into `ProcessGraph` form —
`ProcessGraph.build_from_expression(...)` (`graph_express2.py:1256`, from a
SymPy expression), `symbolic_process_graph.py:275` (canonical SymPy
projection), `bitops_process_graph.py:62` (recorded-bitops provenance), and
`fused_program_to_process_graph()` (`process_graph_fusion.py:812`, the
*reverse* direction — projecting a `FusedProgram` back into `ProcessGraph`
form for further graph-level scheduling).

**Translation performed here:** `process_graph_fusion.py` is the
backend-neutral fusion planner over `ProcessGraph`. It does two jobs:

- `plan_process_graph_dispatches(graph, profile) -> ProcessGraphDispatchPlan`
  (`:915`) selects maximal connected, cost-positive `DispatchRegion`s —
  groups of ops worth fusing together.
- `dispatch_region_to_fused_program(graph, region) -> FusedProgram`
  (`:1006`) is the actual **ProcessGraph → FusedProgram** translation: walk
  the region's nodes in order, canonicalize each op name, fold
  scalar-constant parents into `attrs`, and emit a flat `OpStep` list. This
  is stage 1 → stage 2.

Separately, `process_graph_to_nodus_graph_ir(graph) -> str`
(`src/compiler/nodus_graph_ir.py:31`) is a **text emitter**, not an object
IR: it topologically sorts the graph and prints Nodus GraphIR source lines.
This is a different downstream target from `FusedProgram`/SSA entirely —
worth not confusing with stage 2 just because both start from a
`ProcessGraph`.

---

## 2. Pre-JIT fused program output — "the task itinerary"

**What it is:** `FusedProgram` — `src/common/tensors/fused_ir.py:37`.

```python
@dataclass
class Meta:
    shape: Iterable[int] | None = None
    dtype: str | None = None
    device: str | None = None

@dataclass
class OpStep:
    step_id: int
    op_name: str
    input_ids: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)
    result_id: int = -1
    mode_sensitive: bool = False
    level: Optional[int] = None

@dataclass
class FusedProgram:
    version: int
    feeds: Set[int]
    steps: List[OpStep]
    outputs: Dict[str, int]
    state_in: Set[int] | None = None
    meta: Dict[int, Meta] | None = None
    extras: Dict[str, int] | None = None
```

**Why "task itinerary" is the right name:** `OpStep` has no branch, no loop,
no basic-block field of any kind — `level` is a scheduling hint, not
control flow. `FusedProgram.steps` is a flat `List[OpStep]`, consumed in
list order by every downstream reader (e.g.
`lower_fused_program_to_ssa` just iterates `program.steps`). It is a
version-tagged record of "feed these values in, do these steps in this
order, these value ids are the outputs" — closer to a linear work order
than to a compiler IR. All control flow lives one stage later, in the
separate `ControlProgram` (stage 3).

**What produces one:** this is a common bridge format with several
producers, not a single frontend — `dispatch_region_to_fused_program`
(stage 1, above), `abstract_nn/fused_program.py:150` (the canonical
builder this module re-exports), `c_primitive_program.py` (capture-tape →
`FusedProgram`, several sites), `glsl_backend.py` /
`glsl_fused_network.py` (GLSL-side builders), and — the one that matters
for stage 3 — `glsl_deployment_strategy.py`, which builds the **per-region**
`FusedProgram`s that get threaded into the control/SSA lowering as
`region_programs`.

---

## 3. Pre-SSA graph IR — control and numeric isolated

**What it is:** `ControlProgram` — `src/compiler/control_source.py:118`.

```python
@dataclass(frozen=True)
class ControlProgram:
    root: ControlBlock
    region_indices: tuple[int, ...] = ()
    uniforms: tuple[ControlUniform, ...] = ()
    value_aliases: tuple[tuple[int, int], ...] = ()
    iterable_bindings: tuple[tuple[int, int, str], ...] = ()
    static_iterable_bindings: tuple[...] = ()
    collection_bindings: tuple[...] = ()
    closure_iterable_bindings: tuple[...] = ()
```

`ControlBlock` is a tagged union: `StatementBlock`, `SequenceBlock`,
`LoopBlock`, `StateMachineTick`, `ParallelDeployment`, `CallBlock`,
`ValidationBlock`, `StreamPublishBlock`. This is the "control shell" —
loops, branches, state machines, validation, stream publishing — and it
contains **zero numeric op steps**. The module docstring is explicit about
the intent: "the planner owns control flow. Backends render this
structure; they must not rediscover loops."

**How control and numeric stay isolated:** a `ControlProgram` never embeds
a `FusedProgram` inline. Instead, wherever numeric work belongs, the
control tree has a `StatementBlock` whose single line is a placeholder —
`__scheduled_region_N__` — matched by the same regex in three places
(`precompile_to_ssa.py:46`, `control_source.py:313`,
`hierarchical_control.py:29`). Each region index `N` maps, elsewhere, to
its own separate `FusedProgram` (stage 2). Control never knows what's
inside a region; it only knows region `N` exists and where it sits in the
control flow.

**What produces the split:** `LoopComposer`
(`src/compiler/loop_composer.py:1180`) is where a combined program actually
gets torn into "control shell + numeric region placeholders" — around
`loop_composer.py:2215` it emits the `__scheduled_region_N__` markers at
the correct lexical position inside each `LoopBlock.body`.
`glsl_deployment_strategy.py` is the higher-level driver that instantiates
`LoopComposer`, builds the resulting `ControlProgram`s, and builds the
matching per-region `FusedProgram`s (stage 2) side by side.
`hierarchical_plan.py` / `hierarchical_control.py` sit one layer above
that, composing several closures' worth of `ControlProgram`s into one using
the same region-marker convention.

---

## 4. SSA

**What it is:** the transmogrifier IR — `src/transmogrifier/ssa.py`:
`SSAValue`, `Instr` (`op`/`args`/`res`/`attributes`), `BasicBlock`,
`Function`, `IRModule`. Standard SSA: real basic blocks, real `Phi` nodes,
real control edges (`Br`/`CondBr`).

**The unification point:** `lower_precompile_and_control_to_ssa(artifact,
control, *, region_programs=None, ...)` — `src/compiler/precompile_to_ssa.py:983`.
This is the one function that actually merges stages 2 and 3 into stage 4:

1. Lowers the main numeric `artifact` (a `FusedProgram`) into one `Function`
   via `lower_fused_program_to_ssa` — every `OpStep` becomes an `Instr`,
   almost always wrapped as `Handler.Call` (because a matching C/LLVM
   kernel symbol exists for nearly every op), with the **original**
   canonical op name preserved under `attributes["tensor_operation"]` and
   the kernel symbol under `attributes["callee"]`.
2. Lowers each entry in `region_programs` (per-region `FusedProgram`s) into
   its own named `Function` (`numerical_region_{index}`), recording
   `region_callees` (index → function name) and `region_signatures`
   (feed ids, output ids).
3. Lowers the `ControlProgram` via `lower_control_program_to_ssa`, where
   `_ControlSSABuilder` turns every `__scheduled_region_N__` marker into a
   real `Handler.Call` instruction targeting `region_callees[N]`.
4. Assembles one `IRModule` containing the numeric function(s), the control
   function, and imported LLVM-algorithm functions.

Result: one `IRModule`, self-contained, where a control function's `Call`
sites point at real numeric functions by name — the seam between "control"
and "numeric" is now an ordinary function call, not a string marker.

---

## 5. AOT backend compiling (Fortran, C, GLSL — Pipeline B)

**What these are:** `c_jit_backend.py`, `llvm_jit_backend.py`,
`glsl_jit_backend.py`, `fortran_jit_backend.py`, all under
`src/common/tensors/accelerator_backends/`. Each compiles **one already-
captured, fixed program** (not a general `AbstractTensor` backend — see the
earlier distinction: these have no `tensor()`, no eager per-op API).

**Where their input actually comes from:** none of them touch stages 1–4.
Confirmed by import search — none import `fused_ir.FusedProgram`,
`control_source.ControlProgram`, or `precompile_to_ssa`. Instead, each one
independently walks an **autograd tape** — the recording produced by
`tensor_torture.capture_torture_case()` (`with autograd.forward_capture()
as tape: ...`) — and lowers that tape straight to its own target:

- `c_jit_backend.py` → C source text, via its own `_required_nodes`/
  `_emit_c_tape` tree-walk of `captured.tape._nodes`.
- `llvm_jit_backend.py` → delegates to `c_backend_llvm_ssa.py`'s
  `lower_abstract_tensor_tape_to_llvm_ssa`, which does its own independent
  tree-walk and emits LLVM IR **text** directly (not `transmogrifier.ssa`
  objects, despite the name).
- `glsl_jit_backend.py` → GLSL source text, same pattern.
- `fortran_jit_backend.py` → the one partial exception: it walks the tape
  itself (same `_required_nodes`-style pattern as `c_jit_backend.py`) but
  builds real `transmogrifier.ssa.Function`/`BasicBlock`/`Instr` objects
  instead of text, then hands them to the pre-existing
  `ssa_fortran_backend.emit_module`/`compile_module` machinery (previously
  only exercised by hand-built test fixtures, never a real lowering). It
  still never touches `FusedProgram`, `ControlProgram`, or
  `precompile_to_ssa.py` — it is a second, independent tape → SSA-`Function`
  lowering, not a reuse of stage 4's lowering.

All four then compile with a real toolchain (gcc/clang, llvmlite, a GLSL
compiler, gfortran) and execute through the shared `profiled_c_shell`
launch-boundary ABI, so their timings are directly comparable to each
other.

---

## Quick reference

| Stage | Type | File | Has control flow? | Flat or graph? |
|---|---|---|---|---|
| 1. Process graph | `ProcessGraph` | `transmogrifier/graph/graph_express2.py` | no (pure dataflow) | graph (DAG) |
| 2. Fused program | `FusedProgram` / `OpStep` | `common/tensors/fused_ir.py` | no | flat list |
| 3. Control shell | `ControlProgram` / `ControlBlock` | `compiler/control_source.py` | yes (loops/branches/state machines) | tree of blocks, region markers as leaves |
| 4. SSA | `Function` / `Instr` / `IRModule` | `transmogrifier/ssa.py` | yes (real CFG, `Phi`) | basic blocks + edges |
| 5. AOT backend | target-specific (C/LLVM-IR/GLSL text, or SSA for Fortran) | `accelerator_backends/*_jit_backend.py` | n/a (one fixed program) | flat, from tape not graph |

**The seam that doesn't exist yet:** nothing converts an autograd tape
(what feeds Pipeline B) into a `FusedProgram`/`ControlProgram`
(what Pipeline A consumes), or vice versa. Unifying that is a real,
separate piece of work — not something either pipeline does today.
