# Handoff — 2026-09-03, session 2: Route C fixed and verified, migration ready

Branch `codex/recursive-reduction-bridge`, commits `4348a2d8` (inventory) and
`9365c6ac` (fixes) on top of `b0b86e4e` (the approved plan). This document
supersedes the runtime-status claims in
[`docs/CONTINUATION_2026-09-03.md`](CONTINUATION_2026-09-03.md) (session 1,
same day): that report called the AbstractTensor stage "currently BROKEN at
runtime" and named the fix as unstarted work. It is no longer unstarted —
read this document first, then the plan, then session 1's report only for
historical rules/measurements that still hold (reproduced below where they
matter).

## What this session was asked to do, and how it relates to the plan

The user asked for the SymPy → AbstractTensor → backend pipeline to be
**demystified** before any build/lowering work resumed, explicitly not
starting with "securing any build." That is Phase 0/1 of the already-approved
[`docs/PLAN_SYMPY_TO_ABSTRACT_TENSOR_PATHS.md`](PLAN_SYMPY_TO_ABSTRACT_TENSOR_PATHS.md).
This session executed Phase 0 (inventory) and Phase 1 (dependencies) in full,
then went further than "read-only" into the first slice of Phase 4 (fixing
Route C) because the inventory surfaced concrete, reproducible bugs blocking
the plan's own stated preferred path — fixing them was in scope as "bugs
directly tied to the thing being investigated," not scope creep, and every
fix is a vocabulary-table addition, not an architecture change, in a module
neither runtime binding currently depends on for its live behavior.

**No build or native lowering was run this session.** Everything below was
verified with `python -m pytest` and short `python -c` repros only, per the
user's standing rule ("no broad test sweeps; verify with seconds-long
repros," "never run a build/lowering without approval").

## Where things stand right now

- Working tree is clean; nothing uncommitted.
- The full route inventory with file:line citations lives in
  [`docs/PHASE0_ROUTE_INVENTORY_2026-09-03.md`](PHASE0_ROUTE_INVENTORY_2026-09-03.md).
  This handoff summarizes and points into it rather than repeating it in
  full — read that document for the complete route table (A-F), Route G
  (the orthogonal extended-precision `signal_math` surface), and the full
  maintainability/consolidation assessment.
- **The one process left running by session 1 is STILL RUNNING**: PID
  `16920`, `python tools/build_vehicle_validator_native.py --output
  build/vehicle_validator_dually_o0 --assembly-profile dually-axle --contract
  deploy --optimization O0`, started 12:50:47, now past 15:11 (>2h20m) and
  its log (`.../scratchpad/native_build.log`) still reads only `[1/4]
  emitting compiler-owned C sections` — the same stage session 1 last saw it
  in at the ~45-CPU-minute mark. This is now roughly 3x the "historical
  stage 1 was 8-13 min" baseline session 1 recorded, and well past the
  "deploy contract's outlining may explain it" hypothesis session 1 offered.
  **The user has explicitly said to leave it alone and look at the outcome
  once it finishes or dies on its own** — do not kill it, do not start a
  second one, just check its exit state before touching anything that
  depends on its output (`build/vehicle_validator_dually_o0`).

## What was actually done this session

### 1. Full route inventory (Phase 0/1)

Six SymPy-to-AbstractTensor routes confirmed (A: hand-printer+AOT feeding the
WGSL contact kernel; B: the compiler's own SymPy→ProcessGraph→SSA pipeline,
disk-cached; C: SSA→Python materializer, the plan's preferred target; D: a
narrow one-consumer SymPy-matrix-to-WebGPU helper; E: `lambdify`, proof-only;
F: the fluid model, mechanically routed through B). Plus a seventh,
orthogonal mechanism (Route G) discovered while investigating: `AbstractTensor`'s
trig/hyperbolic methods have a runtime-global quality switch
(`abstraction_methods/trigonometry.py`) between the plain libm `"operator"`
route (default, what the SSA `Tanh` opcode already captures) and
`"signal_math"` baked high-precision cores measured against a 40-digit SymPy
reference — relevant to Phase 3's pinned-precision tests, not a required
change. Full table and file:line citations: inventory doc, "Route table" and
"Route G" sections.

### 2. Four real gaps found and fixed in Route C (`src/compiler/ssa_python_materializer.py`)

Route C — `materialize_function_body`/`materialize_ir_module` with
`tensor_vocabulary=True` — is the plan's Phase 2 preferred canonical path. It
turned out to already be catalog-driven and largely correct, but had never
been exercised against a real compiled vehicle law before this session. Four
gaps were found (by direct reproduction, not just reading) and fixed:

1. **`Tanh`** was entirely absent from `_UNARY_SPELLING` (fell into the
   generic `UNIMPLEMENTED` bucket with a misleading "needs a bit width"
   message). Added, mirroring how `Sqrt`/`Log`/etc. already redirect through
   `tensor_vocabulary`.
2. **`Max`/`Min`** were hardwired to Python's `max`/`min` builtins
   unconditionally — confirmed to raise `ValueError: The truth value of a
   tensor with more than one element is ambiguous` the moment a real
   multi-element `AbstractTensor` reaches one. Fixed with a
   `_BINARY_TENSOR_NAME` redirect to the catalogued `.maximum()`/`.minimum()`
   methods.
3. **The receiver-selection defect this whole investigation started from**
   (Route A's bug: `(0).maximum(x)` — a plain `int` literal picked as the
   method receiver, `AttributeError`) had to be actively avoided in the
   fix above, not just papered over: `_BodyMaterializer` now tracks which
   SSA values are provably constant (seeded from `Const`, propagated through
   pure-constant arithmetic) and the `Max`/`Min` redirect picks whichever
   operand is *not* constant as the receiver. This is the one piece of new
   logic in the fix, not a table entry, and is the part most worth review.
4. **`Pi`** had no case at all (`no Python form for 'Pi'`). `Pi` is
   deliberately a semantic operation until backend lowering
   (`symbolic_process_graph.py:569-576`); every native backend reads its
   value from `bounded_constants.materialize_pi` rather than restating a
   literal. Route C now does the same read.
5. **`Const`'s payload was read only under the key `"value"`**, but the
   compiler's own SSA builder (and `ssa_reference_evaluator`, `ssa_c_backend`)
   spell it `"constant"`, with `"values"`/`"llvm_literal"` as further
   fallbacks. **This was the actual blocking gap** — without it, Route C
   could not materialize *any* constant the compiler itself produces, which
   is presumably why it had never been exercised against a real vehicle law
   before. Fixed to read the same precedence `ssa_reference_evaluator` does.

All four fixes are additive (vocabulary-table entries plus one constant-
tracking helper), touch only `ssa_python_materializer.py`, and do not change
any existing passing test's behavior.

### 3. Verification (not build-dependent)

- Both laws the session-1 continuation report named as broken under Route A
  now materialize through Route C and **match the `sympy.lambdify` reference
  exactly** on random batch columns (`rtol=1e-9..1e-10`):
  `vehicle_member_material_step` (17 outputs) and `abstract_ui_vehicle_step`
  (144 outputs, the full vehicle body law).
- `tests/test_ssa_python_materializer.py`: 7 new regression tests added (one
  per fix, plus a same-constant-operand-stays-scalar counterpart for the
  Max/Min fix), all passing. Full-file before/after: baseline 12
  failures/29 passes → now 10 failures/45 passes. The 10 remaining failures
  are a pre-existing, unrelated counted-loop/storage-formal-metadata
  subsystem (confirmed present before this session's changes via `git
  stash`). The `Const` fix additionally resolved 2 of the original 12
  failures as a side effect (the file's own 291-instruction symbolic-fluid-
  step calibration case), which is a bonus, not a regression risk, since
  those tests assert agreement with `ssa_reference_evaluator` — an
  independent oracle.
- `tests/test_auto_port_numpy_to_abstract_tensor.py`: unchanged, all 8 pass.

### 4. A live discrepancy found, not yet acted on

While checking whether Route A's numeric-receiver bug also reaches the
contact law (it does — 5 of 33 subexpressions), a sharper problem surfaced:
`compile_wheel_contact_abstract_tensor()`'s real, cached, currently-live
generated source contains the same defect, and executing that exact string
with plain `exec()`+call raises `AttributeError` — confirmed directly — yet
the function itself, which feeds that same string through
`compile_ast_aot(backend="webgpu", precompile_only=True)`, completes without
error and has been building the live WGSL contact kernel successfully. Best
current explanation (not fully step-debugged): `compile_ast_aot`'s own
docstring points new callers at `fortran_c_shell.lower_ast_source_to_ssa`, a
structural AST-to-SSA frontend that recognizes `receiver.method(args)` call
*shapes* against its tensor-operation catalog rather than executing the
source through real CPython attribute lookup, so it never asks whether a
literal `0` really has a `.maximum` method. Whatever the mechanism, the
finding stands on its own: **the same byte-identical AbstractTensor-Python
text is silently valid to one of its current readers and fatally invalid to
another.** This is the strongest concrete evidence in this investigation for
consolidating onto one route, and it means the contact kernel is currently
running on unverified malformed subexpressions that happen not to matter to
its particular build path. Full detail: inventory doc, finding 4.

### 5. Maintainability/consolidation assessment (asked for directly by the user)

Answered in full in the inventory doc's dedicated section. Short version:
some vocabulary is genuinely single-authority and structurally enforced
(the SSA opcode table audit, the tensor-call-form derivation from
`AbstractTensor`'s own class); some is single-authority in intent but not
enforced anywhere, and had already drifted (`Pi`, `Const`'s key precedence —
three independent inline copies across `ssa_reference_evaluator`,
`ssa_c_backend`, and, until this session, `ssa_python_materializer`); and
Route A's printer is not shared at all — it is a second, disagreeing
definition of a fact Route C's catalog already owns. One person cannot
reliably maintain the current four-reader state; the plan's own Phase 2
decision criteria already call for consolidating onto one, and this
session's findings are concrete supporting evidence for that, not a new
argument.

## The frontier — what the next agent should do, in order

This is Phase 4 of the approved plan, steps 1 through 5, updated with what
is now known:

1. **Done this session**: Phase 4 step 1 (verify/extend Route C). Complete.
2. **Next, and now unblocked**: point
   `src/compiler/vehicle_python_compilation.py`'s `symbolic_abstract_tensor_source`
   at `materialize_function_body`/`materialize_ir_module` instead of
   `abstract_ui_vehicles._abstract_tensor_python`. Concretely:
   `compilation.function` (already produced by `compile_sympy_equations`,
   Route B) is the input Route C wants; `symbolic_abstract_tensor_source`
   currently ignores it and re-derives its own `sympy.cse` from
   `compilation.equations` instead. The replacement should call
   `materialize_function_body(compilation.function, tensor_vocabulary=True,
   parameter_names=argument_names)` and assemble the same
   `def name(args): ... return (...)` shape the current printer-based
   version produces, so `vehicle_python_runtime_bindings` and
   `tests/test_symbolic_abstract_tensor_stage.py` need no further change.
   This session already proved the numerical output is identical for both
   laws that function currently serves — this step is mechanical.
3. **Also check before deleting anything**: `_abstract_tensor_python` /
   `_AbstractTensorPythonPrinter` (`abstract_ui_vehicles.py:3938-3969`) is
   also called from `compile_wheel_contact_abstract_tensor`
   (`abstract_ui_vehicles.py:3972`, the live WGSL contact kernel). That
   consumer has its own numeric-receiver defects (finding 4 above) and its
   own re-derived `sympy.cse` — the same migration pattern applies, but this
   one **does require a build** (WGSL emission, `compile_ast_aot` with
   `backend="webgpu"`) to verify, so get explicit approval before touching
   it, per the standing rule.
4. **Only after both consumers move to Route C**: delete
   `_AbstractTensorPythonPrinter`/`_abstract_tensor_python` for real. Do not
   delete while either consumer still calls it.
5. **Phase 5 acceptance** (build-dependent, needs approval): re-run
   `tools/frame_parity.py` for all four laws (fixture, material, contact,
   vehicle body) against C/LLVM/Fortran at ULP level; confirm the native
   build lowers from the same SSA the eager stage now uses; N-frame
   Python-vs-DLL parity through the feedback loop. This is exactly what
   session 1 was mid-build toward (PID 16920) — check that build's outcome
   first, since a fresh build is expensive and the running one may already
   answer part of this.
6. **Lower priority, no urgency**: retire the dead `#if 0`-guarded
   `vehicle_native_graph_tick_batch` per-lane scalar wrapper in
   `vehicle_native_deployment.py:1121-1129` once the above is stable — it is
   already disabled, not compiled in, just not yet deleted.

## Open risks and questions for the next agent

- The AOT-vs-real-Python discrepancy (finding 4) was not root-caused to the
  bytecode level. If the next agent touches `compile_ast_aot` or
  `lower_ast_source_to_ssa` for any reason, revisit this — it may indicate
  the AOT frontend is more permissive than intended, not just permissive by
  fortunate accident.
- Route D (`compile_sympy_matrix_to_abstract_tensor_backend`) was read and
  is narrow/self-contained (one consumer, one degenerate matmul shape) — no
  action needed, but it was not empirically exercised this session.
  Confidence is from reading only.
- Route G (`signal_math`/extended precision) is orthogonal and untouched.
  Its relevance is to Phase 3's pinned-precision test tolerances, not to the
  route consolidation itself — flagged so the next agent doesn't rediscover
  it from scratch, not because it needs immediate action.
- The native build (PID 16920) status is the one live external dependency;
  nothing in Phase 4 step 2 (the pure-Python migration) needs it, but Phase
  4 step 3 and Phase 5 do.

## How to verify this session's claims quickly

```
python -m pytest tests/test_ssa_python_materializer.py tests/test_auto_port_numpy_to_abstract_tensor.py -q
```
Expect: 10 pre-existing unrelated failures (counted-loop/storage-formal
metadata), everything else green, including the 7 new tests for the fixes
documented here.

## Session 3 addendum (same day): Phase 4 step 2 DONE

`vehicle_python_compilation.symbolic_abstract_tensor_source` now calls
`ssa_python_materializer.materialize_function_body(compilation.function,
parameter_names=argument_names, tensor_vocabulary=True)` and assembles the
`def` with `ast`; no sympy printer, no re-derived CSE. The eager bindings
are therefore the compiler's own AbstractTensor materialization of the same
SSA the native product is lowered from. `tests/test_symbolic_abstract_tensor_stage.py`
(4 tests) passes: both laws match the sympy reference per lane on batch
columns; the source spells Max/Min as tensor methods and sqrt as the SSA Pow
with a constant exponent, with no `math.`/`abs(`/lambdify. Remaining
consumer of the old printer: `compile_wheel_contact_abstract_tensor`
(needs a WGSL build to verify; approval required). Native build PID 16920
still alive in stage 1 (left alone, per the user).
