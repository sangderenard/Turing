# Whole-program compile frontiers — where they all are

**Snapshot:** 2026-08-05, branch `codex/recursive-reduction-bridge`
**Purpose:** four separate efforts have all converged on the same missing
capability. This document exists so that convergence isn't rediscovered from
scratch if momentum gets sidetracked again.

## The convergence

Four unrelated pieces of work — a binary-executor web page, Dream document
parsing, a sympy-to-physics compiler, and a "run whole classes as one
program" effort — have all independently run into the same wall:

> **`build_program_bundle()` / `discover_source_contract()` (`src/compiler/site_bundle.py:877`)
> only ever ingests a single top-level function as the entrypoint.** It filters
> `module.body` to `ast.FunctionDef` and raises if there is no public
> top-level function. There is no branch for `ast.ClassDef` at all.

Meanwhile, **whole-context/class ingestion already exists one layer down**,
in `src/common/tensors/topological_reducer.py::reduce_abstract_tensor_topology`
(around line 1968): it discovers every `ast.ClassDef`, builds
`graph.G.graph["class_table"]` (methods, fields, field defaults), and lets
methods call each other as real compiled calls sharing class fields as
structural state. This is real and used — but only reachable by feeding a
whole class's source through `compile_ast_aot` directly, never through the
page builder.

There are, in fact, **two independent, non-identical attempts** at building a
"whole object becomes a page" layer on top of that capability:

1. `src/compiler/parametric_card_program.py` — a retained, non-flattened
   "system of cards" (`ProgramCard`, `CardAliasEdge`, `ParametricCardProgram`)
   under an outer coordinator. Built for the `dt_system`/fluid work.
2. `src/compiler/wasm_class_coordinator.py` + `wasm_class_modules.py` —
   `build_class_inventory` / `emit_wasm_control_coordinator`, a *different*
   whole-class-to-Wasm mechanism, used by the bound-spring / `ComputationalWorld`
   work (`tests/test_bound_spring_aot.py`).

Neither is wired to `build_program_bundle`. Neither has been proven to
produce a working rendered page end to end. **Before building a third one for
the binary machine program, read both of these and figure out whether one of
them is already the right shape** — this doc's biggest open question is
whether `parametric_card_program.py` and `wasm_class_coordinator.py` should
converge into one mechanism, and which one (if either) is closer to feeding
`discover_source_contract`.

---

## Frontier 1: Binary machine program → page

**Owner doc:** `CMD_BINARY_EXECUTOR_COMPILATION_HANDOFF.md` (repo root)
**Status:** the executor itself is real and working natively
(`BinaryMachineProgram.load_pe` / `MachineExecutionOrchestrator.step`,
logging via `system_tape`, branch traversal via `create_path_forest`). It has
**never been compiled into a page** — every published version of the
`reversible-binary-machine` bundle still carries the forbidden
`"compiler.backend": "prebuilt-program-interior"` marker (confirmed by
reading all 6 published `bundle.json` files this session), meaning display
was hand-built while the real work stayed native Python.

Most recent commit against this specific goal (`7e081d2`, "Give javascript
Dream sections a real AST dependency graph") lists its own continued
shortfalls verbatim:

- `machine_web_publication.py` only recompiles a single entry Wasm block and
  falls back to snapshot-replay for everything else — the resident
  coordinator (`BinaryMachineProgram`, `MachineExecutionOrchestrator`, the PE
  loader, AMD64 semantics) is not compiled into the page at all.
- `CapabilityGatedExternalPort` is not wired to forward previously-recorded
  external-call completions for browser-resident execution.
- Python Dream sections (chip-setup, head-step) report no executable
  artifact, only a ProcessGraph description.
- The manifest still says `prebuilt-program-interior`.
- A real `cmd.exe` still requires `reversible_machine_web_host.py` running
  natively; nothing runs resident in the browser yet.

**This session's independent finding, same conclusion from a different
angle:** `X86TensorReadHead.transition()` (a *different*, tensor-native
decoder that nothing in the real pipeline actually calls — confirmed via
grep, it's a dead import in `binary_ingestion.py`) now compiles through
`compile_ast_aot` to a real, non-empty dispatch artifact (1670 ProcessGraph
nodes, real `dispatch_plan`) after fixing a genuine identity-aliasing bug in
`X86ReadHeadState.initial()` (multiple dataclass fields sharing one Python
object by reference instead of independent objects holding the same value —
fixed via `.clone()` per field). That fix is real and worth keeping, but
**`X86TensorReadHead` is not the thing `BinaryMachineProgram` runs** —
`MachineExecutionOrchestrator.step()` is, and `step()` is far too complex
(Windows loader/TLS/DLL-detach simulation) to hand-port into a simplified
free function, and doing so would violate the project's own rule against a
second interpreter.

**Also fixed this session, real and load-bearing regardless of the above:**
`GradTape`/`AbstractTensor` identity tracking switched from raw `id()` (a
reusable memory address) to a monotonic per-object token
(`abstraction.tensor_identity`) — `id()` reuse across a discovery trace was
producing spurious "same primitive, two endpoints" compiler errors. This
propagated through `autograd.py`, `c_primitive_program.py`, `fs_harness.py`,
and the relevant call sites in `glsl_deployment_strategy.py`'s
`_observe_process_graph_node`. Full regression sweep, 26/26 on the affected
suites (fixed a handful of tests that asserted raw `id()` values directly,
which is expected fallout, not a regression).

**Design constraint surfaced this session, not yet resolved:** the binary
file cannot be a compile-time constant (would mean redistributing someone
else's binary, and reverse-engineering/legal exposure). It has to be
loaded once at runtime, from a user-supplied file, and then carried forward
through the same per-tick `state_feedback` mechanism as the registers —
which in turn means the "decode once" phase (PE parsing → instruction table)
and the "step every tick" phase are two different-shaped, differently-timed
pieces of the *same* real program, not two programs. `shell_io.py`'s
`SystemPort(kind=FILE, direction=INPUT, entry_point=...)` is a real,
existing mechanism for "call this compiled function once, when a file
arrives" — confirmed by reading its validation logic — and is the natural
fit for the decode phase, separate from the per-tick `entrypoint`.

**Next step for this frontier specifically:** none of the above is
buildable through `build_program_bundle` until the class-ingestion gap
(above) is closed, because `MachineExecutionOrchestrator` is a class with
real inter-method structure, not a lone function.

---

## Frontier 2: Dream document parsing

**Status:** healthier than the other three. All `dream_document` tests pass
(16 passed, 15 skipped — skips are headless-Chrome/optional-dependency
gated, not failures). The most recent real work (`7e081d2`) replaced a
mislabeled fake artifact — the JavaScript route used to fall through to a
flat line-by-line node chain while still claiming `executable=True` with no
recorded shortfall — with a real AST-derived dependency graph (vendored
`acorn` via a Node helper script), feeding the same `ProcessGraph`
representation the Python (`ast`) and `sympy` routes already use.

**What's still open, per that commit's own accounting:** Python Dream
sections (chip-setup, head-step) still produce no executable artifact, only
a ProcessGraph description — this is the same "whole context, not a
document-order function" problem the class-ingestion gap describes.

This frontier doesn't need new investigation right now — it's in a
consistent, honestly-labeled state (no false "done" claims), and its
remaining gap is downstream of the same class-ingestion fix everything else
needs.

---

## Frontier 3: Sympy expression input → spring graph

**Owner doc:** `docs/SYMBOLIC_SPRING_WHOLE_PROGRAM_COMPILER_HANDOFF.md`
**Primary entrypoint:** `src/rendering/symbolic_spring_image.py::run_symbolic_spring_image`
**Status:** active, not complete, per its own header — and there is
currently **no regression test at all** covering it (`tests/test_symbolic_spring_program.py`
does not exist as source, only a stale compiled `.pyc` from a since-deleted
file). `tests/test_shader_extractor.py` (a supporting piece — recognizing
`compileShader`/`glCreateShader` call sites) passes clean, 6/6, slightly
ahead of the 5/5 the handoff doc recorded.

The required architecture, verbatim from the doc, is one real function
retaining everything from the runtime expression string through to a
rendered frame:

```
expression_text (runtime string)
  -> sympy.sympify(..., evaluate=False)
  -> ProcessGraph.build_from_expression(...)
  -> symbolically_reduce_process_graph(...)
  -> load_fluxspring_graph_shaders()
  -> run_precompiled_graph(..., duration=math.inf, shader_sources=...)
```

**The documented false-success diagnosis is the important part to not
repeat:** a generic run reported "hierarchy recomposition skipped" with
`ValueError: planned calls reference enclosing loops absent from closure
control: (161, 182)`, then silently fell back to a seven-region numerical
shell with an **empty public input/output map** — meaning `expression_text`
never actually made it into the compiled artifact as a real runtime input,
despite `control_shortfalls == ()` reporting success. This is the exact same
"no exception raised ≠ meaningful compile" trap this session repeatedly hit
with `X86TensorReadHead`/`MachineExecutionOrchestrator.step()`. The doc's own
prescribed fix: make whole-program hierarchy recomposition fail closed when
no valid prior hierarchical artifact exists, instead of silently selecting a
numerical child shell.

The doc explicitly notes it was blocked waiting on the *same* loop-carried
SSA identity fix that later landed in `5b8875c` (see Frontier 4) — but its
own blocking error (missing loop IDs 161/182 in `hierarchical_control.compose_hierarchical_control`)
is a **different, still-open** bug: a `PlanCall`'s `enclosing_loop_ids`
couldn't find a matching `LoopBlock` in its own closure's control. Whether
`5b8875c` incidentally fixed this too is unverified — nobody has rerun
`run_symbolic_spring_image` since that commit landed.

**Next step for this frontier:** rerun the symbolic-spring compile now that
the loop-carried identity fix is in, before assuming anything else is wrong.
If the 161/182 missing-loop error persists, it needs its own fix in
`hierarchical_control.py`/the loop planner, not another workaround.

---

## Frontier 4: Parametric card program (multicard / "whole file with many classes")

**Owner doc:** `AGENTS/experience_reports/1785934795_DOC_Turing_Parametric_Multicard_DT_Fluid_Handoff.md`
(in the sibling `speaktome` repo, not this one)
**Status:** further along than its own handoff doc says, but still not
wired to produce a page.

That handoff doc reports being blocked on six loop-carried SSA identity
shortfalls in the real CG viscosity/pressure solvers, with no commit made
that session. **This was fixed in a later commit, `5b8875c`** ("Checkpoint
concurrent compiler work and clear loop-carried SSA shortfalls" — already on
this branch, verified via `git merge-base --is-ancestor`), which states the
4x4 full-physics fluid compile now reaches Fortran emission. Two scope
corrections: `glsl_deployment_strategy.canonical_global` now treats a loop
backedge as a cut instead of walking through it (it was fusing a cycle's two
endpoints into one, degenerating the header Phi to `Phi(x, x)`), and
`loop_composer` now restricts carried-alias candidates to the loop body's
own induced subgraph.

**Verified this session:** ran the exact regression suite the handoff doc
names as its resume command — `test_canonical_forward_and_backward.py`,
the loop-composer/process-graph-shell focused regressions, `test_parametric_card_program.py`,
`test_voxel_fluid_engine.py`. 26/26 pass (one collateral failure from this
session's own `tensor_identity` change, fixed in the same pass).

**What's still actually open, confirmed by reading `test_parametric_card_program.py`
directly:** it tests card retention, control-region structure, exact alias
edges, rejecting dispatch to a missing card, and feedback naming validation.
**It does not test rendering or producing a page at all.** This matches the
user's own observation — the multicard mechanism "always fails when you tell
it to render" — and matches the handoff doc's own "Known limitations": the
card program is retained metadata/validation today; the native shell still
invokes the composed compiled entry point directly rather than the card
program driving a separately-schedulable runtime. Nothing currently asks it
to render anything, so nothing currently proves it can.

**Next step for this frontier:** find or write the first test that actually
asks `parametric_card_program` to drive a rendered frame end to end, and see
exactly where it breaks. That failure is the real next fact to gather —
right now the failure mode is only known anecdotally ("always fails"), not
diagnosed.

---

## Recommended order of attack

1. Read `parametric_card_program.py` and `wasm_class_coordinator.py`
   side by side. Determine whether they're solving the same problem twice or
   are legitimately different layers. This blocks a real decision on where
   class-ingestion support belongs in `discover_source_contract`.
2. Actually attempt a `parametric_card_program` render (Frontier 4's real
   gap) — this is the cheapest experiment (infrastructure already exists,
   `dt_system`/fluid demo already compiles past SSA) and will produce a
   concrete, diagnosable failure instead of an anecdote.
3. Rerun `run_symbolic_spring_image` (Frontier 3) now that `5b8875c` is in,
   to find out whether its missing-loop-placement bug survived the loop
   identity fix or was incidentally resolved.
4. Only after one whole-context path is proven to render something real,
   decide whether to extend `discover_source_contract` to accept a class
   entrypoint, and wire the binary machine program (Frontier 1) through it.

Do not start a fifth parallel "whole object ingestion" mechanism for the
binary machine program before finishing step 1. That is exactly how this
repo ended up with two.
