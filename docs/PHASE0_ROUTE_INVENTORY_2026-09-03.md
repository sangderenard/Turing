# Phase 0/1 inventory — SymPy to AbstractTensor Python routes (2026-09-03)

Executes `docs/PLAN_SYMPY_TO_ABSTRACT_TENSOR_PATHS.md` Phase 0 (inventory,
read-only) and the load-bearing part of Phase 1 (dependencies). No build or
lowering was run; every claim below is either read from source (file:line) or
from a seconds-long `python -c` repro against the existing compiler, never a
new interpreter.

## Route table (filled)

| # | Route | Entry points (file:line) | Simplifies? | Tensor methods? | Feeds native? |
|---|---|---|---|---|---|
| A | SymPy printer -> AbstractTensor source -> AOT capture -> SSA | `_AbstractTensorPythonPrinter` `abstract_ui_vehicles.py:3938`; `_abstract_tensor_python` `:3965`; `compile_wheel_contact_abstract_tensor` `:3972` | `sympy.cse` only (`:4008`) | Hand table: `Pow(.5)->.sqrt()`, `tanh->.tanh()`, `Min/Max->.minimum()/.maximum()` (`:3941-3959`). **Numeric receivers break**: confirmed below. | Yes: wheel-contact WGSL (`emit_webgpu_module` `:4097`) |
| B | SymPy -> `ingest_sympy_expressions` -> ProcessGraph -> `process_graph_to_ssa_instrs` -> scalar SSA | `symbolic_equation_compiler.compile_sympy_equations` `:315`; `_compile_sympy_equations_uncached` `:64`; `ingest_sympy_expressions` `symbolic_process_graph.py:710` | SSA scheduling + `reduce_constant_exponent_pow` (`ir_identities.py`) + predicate/dtype identity pass (`:126-140`) | Scalar SSA opcode vocabulary only (Capitalized: `Add`/`Max`/`Tanh`/...); no tensor methods at this stage | Yes: `compile_*_c`/LLVM/Fortran for vehicle body, material, fixture, tire (`vehicle_native_assembly.py`, `vehicle_balloon_tire.py`, `vehicle_mechanical_material.py`) |
| C | SSA -> Python emission | `ssa_python_materializer.materialize_function_body`/`materialize_ir_module` (`ssa_python_materializer.py:780,1245`), `tensor_vocabulary=True` flag | Inherits B | **Catalog-driven, not a hand table** — reads `CANONICAL_ABSTRACT_TENSOR_OPERATORS` and `inspect.getattr_static(AbstractTensor, name)` (`:168-202`) to decide `x.op()` vs `AbstractTensor.op()`. Already used by `demo_kuramoto_field*.py`, `build_kuramoto_webgpu.py`, `probe_folded_sine.py`, `absorb_source.py` and covered by `tests/test_ssa_python_materializer.py` + `tests/test_auto_port_numpy_to_abstract_tensor.py`. **Two real gaps found, see below.** | Not today (no vehicle consumer binds this yet) |
| D | SymPy matrix graph -> AbstractTensor backend regions | `compile_sympy_matrix_to_abstract_tensor_backend` `abstract_ui_vehicles.py:3439`; `SympyTensorBackendPrecompile` `:3401` | N/A — the helper only accepts one degenerate `sympy.MatMul` of two `MatrixSymbol`s (`:3467`), so there is nothing to simplify | Yes, trivially: literal source is `f"{a}.matmul({b})"` (`:3477-3479`) | Yes: WebGPU only, single consumer `compile_vehicle_wrench_reduction_webgpu` (`:3553`), itself only used at `:4970` |
| E | `sympy.lambdify` | `tools/frame_parity.py:216` (`python_backend`, the *authority* oracle), `tools/differential_translation.py:85`, `tools/compare_balloon_tire_oracles.py:45`, `src/compiler/vehicle_balloon_tire.py:859` (one runtime binding, see below), `src/cells/cellsim/api/saline.py:65` (unrelated subsystem) | No | No — float per lane | Never (proof-only), except the one vehicle_balloon_tire binding noted below |
| F | Other symbolic programs | `symbolic_fluid_model.compile_symbolic_fluid_step` (`symbolic_fluid_model.py:218`) calls `compile_sympy_equations` directly — mechanically **route B** with a different physical model, not a separate mechanism. `symbolic_fluid_dt.py`, `symbolic_fluid_source.py`, `symbolic_fluid_native_runtime.py`, `symbolic_fluid_direct_control.py` are siblings in the same fluid subsystem. | via B | via B | via B |

Out of scope (found, not vehicle-law related, not investigated further):
`transmogrifier/graph/graph_express2.py:4686` and `graph_express_chalkboard_problem.py:90` (legacy transmogrifier lambdify),
`src/compiler/spectral_colorimetry.py:140` (CIE color-matching lambdify, unrelated model).

## Route G — the extended-precision transcendental surface (orthogonal to A-F)

Not a SymPy ingestion route at all, but load-bearing context for Phase 3's
pinned-precision tests, so it belongs in this inventory. `AbstractTensor`'s own
trig/hyperbolic methods (`.tanh()`, `.sin()`, `.sqrt()` is separate, see below)
are not a single fixed implementation:
`abstraction_methods/trigonometry.py:36-119` holds a module-global switch,
`_IMPLEMENTATION`, with three states, selected by
`use_backend_operator()` (default), `use_signal_math(quality="double2")`, and
`use_signal_kernels(quality)`:

- **`"operator"`** (default): `value._apply_operator("tanh", ...)`, lowering to
  the repository's plain `unary_double`/libm opcode — exactly the `Tanh`
  SSA opcode Route B/C already carry (`ssa_llvm_backend.py:86`,
  `ssa_python_materializer.py`'s now-fixed `_UNARY_SPELLING["Tanh"]`).
- **`"signal_math"`**: `common.tensors.signal_math.SignalMath`
  (`signal_math.py:1505`) — baked polynomial/Horner cores measured against a
  40-digit SymPy reference (`signal_math.py:191`, `_reference`), selectable
  quality/limb width (`"double2"` = two-limb double-double by default). The
  module docstring is explicit that this "captures into repository SSA like
  any other authored source" — i.e. when this switch is live, an AOT capture
  of `x.tanh()` does NOT emit a `Tanh` opcode at all; it captures the whole
  Horner expansion as ordinary `Add`/`Mul`/`Div`/`Const` operations, all of
  which route C already spells with no gap.
- **`"signal_kernels"`**: routes through compiled kernels the
  `kernel_bank` has admitted and verified against their own Python reference;
  falls back to the eager `signal_math` surface (never to `"operator"`) for
  anything not yet admitted; forward-only (no VJP wiring yet).

Empirically confirmed same-precision agreement at double resolution
(`operator` vs `signal_math` tanh on `[-2, -1, 0.5, 3]` agree to 5 displayed
digits; the disagreement, if any, only shows up at the ULP level the
`"double2"`+ tiers exist to improve). The first `use_signal_math` call in a
fresh environment measures and bakes every core against its reference and
took roughly 8 minutes in this session; `signal_math.prebake` persists the
result to disk keyed by setting so subsequent runs are instant
(`signal_math.py:1318-1389`), the same cold-then-cached shape as the SymPy
program cache Phase 1 already relies on.

**Relevance to Phase 3 (square-one tests):** the plan's pinned-exact law set
(`sqrt(x*x + 1e-30)`, `tanh(x)`, etc., against `sympy.N(..., 40)`) can be
checked at whichever precision tier this switch selects without touching the
SymPy-to-SSA pipeline at all — flip `use_signal_math` before the AOT capture
that produces the pinned test's reference program, and the captured SSA for
the transcendental becomes an ordinary arithmetic expansion instead of an
opaque libm call. This is a tool for tightening Phase 3's tolerance story, not
a required change to the route table above.

## Concrete findings (empirically verified, not just read)

### 1. Route A's "numeric receiver" bug, reproduced directly

```
_abstract_tensor_python(sympy.Max(0, x))            -> "(0).maximum(x)"
_abstract_tensor_python(sympy.Min(1, Max(0, x)))    -> "(1).minimum((0).maximum(x))"
_abstract_tensor_python(sympy.sqrt(3) * x)          -> "(3).sqrt()*x"
_abstract_tensor_python(sympy.tanh(x))              -> "(x).tanh()"      # fine, receiver is the arg
_abstract_tensor_python(sympy.pi * x)               -> "math.pi*x"       # fine, plain float
```

`(0)` and `(3)` are Python `int` literals from the printer's argument order; a
plain `int` has no `.maximum`/`.sqrt` method, so this raises `AttributeError`
the moment a numeric literal lands in the position the hand table always
prints as the receiver. This is the exact defect the 2026-09-03 continuation
report already named for the member material and vehicle body laws.

### 2. Route C already exists and mostly satisfies the plan's Phase 2 criteria — but had three real gaps of its own (now fixed, see below)

`ssa_python_materializer.py` is not a stub: it is catalog-driven (reads
`AbstractTensor`'s own class via `inspect.getattr_static`, never a hand list),
tested, and already used by four tools. It is the right adoption target per
plan Phase 4 step 1. Reproduced directly against `compile_sympy_equations`
output (`tensor_vocabulary=True`):

```
tanh(x)            -> MaterializationError: "'Tanh' has no exact Python
                       spelling here" (the generic UNIMPLEMENTED message,
                       factually misleading — Tanh needs no bit width, it is
                       just missing from _UNARY_SPELLING)
Max(0, x)          -> "t0 = 0\nt2 = max(t0, x)\nreturn t2"   (Python builtin
                       max(), NOT .maximum() — always, tensor_vocabulary
                       makes no difference for binary ops)
Min(1, Max(0, x))  -> same pattern, nested max()/min()
sqrt(3) * x        -> "t3 = 3 ** 0.5\nt4 = x * t3"           (fine: both
                       operands of ** are plain numbers, only the final Mul
                       touches the tensor, and Mul is a normal operator)
```

Executing the emitted `Max(0, x)` function against a real multi-element
`AbstractTensor` confirms this is a runtime defect, not a style nit:

```
ValueError: The truth value of a tensor with more than one element is ambiguous.
```

Root cause: `_BINARY_SPELLING["Max"]/["Min"]` (`ssa_python_materializer.py:89-90`)
are hardwired to Python's `max`/`min` builtins and consulted unconditionally
at `:431-435`, before the tensor-vocabulary branch that only exists for
`_UNARY_SPELLING` (`:437-455`). `Tanh` is simply absent from
`_UNARY_SPELLING` even though `ssa_llvm_backend._UNARY` (the table this file
audits itself against) declares it (`ssa_llvm_backend.py:86`) — so it lands
in `UNIMPLEMENTED` by default rather than being wired to the catalogued
`tanh` tensor method the same way `Sqrt`/`Log`/etc already are.

Both gaps are one-table fixes (add `Tanh` to `_UNARY_SPELLING`/`_NEEDS_MATH`
the way `Sqrt` is handled; give `_BINARY_SPELLING`'s `Max`/`Min` the same
`tensor_vocabulary` redirect through `TENSOR_CALL_FORMS["maximum"/"minimum"]`
that unary already has) — not an architecture problem. This is the opposite
kind of defect from Route A's: A guesses a hand table per SymPy node type and
breaks on numeric receivers; C is catalog-correct but was never extended past
its first four demo programs' operator set.

A third gap surfaced in the same pass: `Pi` was not in any table at all
(`sympy.pi * x` raised `no Python form for 'Pi'`), because `Pi` stays a
*semantic* operation until backend lowering by design
(`symbolic_process_graph.py:569-576`) — every real backend
(`ssa_c_backend.py:383`, `ssa_llvm_backend.py:3957`, `ssa_fortran_backend.py:3376`,
`ssa_wasm_backend.py:182`) reads its value from the one shared
`bounded_constants.materialize_pi` home rather than restating a literal. The
materializer had no case for it at all.

**All three (`Tanh`, `Max`/`Min`, `Pi`) are now fixed** in
`src/compiler/ssa_python_materializer.py`, with a fourth, more consequential
gap found and fixed alongside them: `Const`'s payload is read only under the
key `"value"`, but the compiler's own SSA builder spells it `"constant"` (as
do `ssa_reference_evaluator.py:524` and `ssa_c_backend.py:376`, both of which
check `"constant"` first with `"value"`/`"values"`/`"llvm_literal"` as
documented fallbacks). This meant `materialize_function_body` could not run
on *any* constant the compiler itself produces — confirmed by materializing
`compile_vehicle_member_material_ssa()`'s real output, which failed
immediately with `Const %t195 carries no 'value' attribute`. Fixing the
`Const` case to read the same precedence the reference evaluator does was
the change that actually made Route C usable on real laws; the `Tanh`/`Max`/
`Min`/`Pi` gaps only became visible in the first place once literals stopped
being the earlier blocker.

**Verified end-to-end after all four fixes**, against the two laws the
2026-09-03 continuation report named as broken under Route A
(`vehicle_member_material_step`, 17 outputs, and `abstract_ui_vehicle_step`,
144 outputs): both materialize through `materialize_function_body(...,
tensor_vocabulary=True)` with no shortfalls, execute on real batch
`AbstractTensor` columns, and match the `sympy.lambdify` reference
(`tools/frame_parity.python_backend`) to `rtol=1e-9..1e-10` on the same
random batch the existing `tests/test_symbolic_abstract_tensor_stage.py`
uses. This closes exactly the runtime break the continuation report left
open. Regression tests for all four fixes were added to
`tests/test_ssa_python_materializer.py`; running that file's full suite
before and after shows two additional PRE-EXISTING failures now pass as a
side effect of the `Const` fix (`test_materialized_python_reproduces_the_authored_mathematics`,
`test_materialized_python_agrees_with_the_ssa_reference_evaluator`, both
against the 291-instruction symbolic fluid step calibration case) — no new
failures were introduced; the remaining ten pre-existing failures are an
unrelated counted-loop/storage-formal-metadata subsystem.

**`vehicle_python_compilation.symbolic_abstract_tensor_source` (the actual
runtime binding) still uses Route A's printer, unchanged in this session.**
It re-derives its own `sympy.cse` from `compilation.equations` rather than
reusing `compilation.function`/`.module` (Route B's SSA, already computed).
Switching it to call `materialize_function_body`/`materialize_ir_module`
instead is the literal Phase 4 step 2 the plan describes, is now unblocked,
and was deliberately left undone this session pending confirmation, since it
changes the vehicle simulation's actual runtime behavior and its Phase 5
acceptance gates call for a native/DLL parity re-run.

### 3. Route D is narrow and self-contained

Single consumer, single degenerate shape (rank-2 `MatMul` of two
`MatrixSymbol`s for wheel-wrench reduction). Nothing else calls
`compile_sympy_matrix_to_abstract_tensor_backend`. No drift risk, no deletion
needed, out of scope for the migration.

### 4. The same generated Route A source is valid to one execution path and invalid to another

While checking whether Route A's numeric-receiver bug also affects the
contact law (feeding the live WGSL kernel), a sharper problem surfaced.
`compile_wheel_contact_abstract_tensor()`'s cached, real (not hypothetical)
generated source contains the identical defect pattern in 5 of its 33
subexpressions, e.g.:

```
tensor_tmp_0 = (1.65*tire_section_radius).minimum((0).maximum(tire_radial_compression))
tensor_tmp_19 = tensor_tmp_15*(...)*(1).minimum((0.58).maximum(-load_sensitivity*(0).maximum(...)-1)+1))*...
```

Executing this exact source string with plain `exec()` and a real call
(what the eager `--python-material` validator's
`_abstract_tensor_stage_callable` does) raises the same
`AttributeError: 'int' object has no attribute 'maximum'` — confirmed
directly. Yet `compile_wheel_contact_abstract_tensor()` itself — which feeds
this same string through `compile_ast_aot(..., backend="webgpu",
precompile_only=True, ...)` — completes without error and has been building
the live contact WGSL kernel successfully. `compile_ast_aot`'s own docstring
(`aot_compile.py:817-826`) says new callers should prefer
`fortran_c_shell.lower_ast_source_to_ssa`, a structural AST-to-SSA frontend
(the same one `test_auto_port_numpy_to_abstract_tensor.py` exercises) that
recognizes `receiver.method(args)` call *shapes* against a tensor-operation
catalog rather than executing the source through ordinary CPython attribute
lookup — a plausible, though not fully step-debugged, explanation for why it
tolerates a receiver whose real Python type has no such method: it never
asks the real type.

Whatever the exact mechanism, the observation itself is solid and is the
strongest single piece of evidence in this inventory: **the same
byte-identical "AbstractTensor Python" text is silently valid to one of its
current readers and fatally invalid to another.** The defect was caught only
because a human happened to exercise the reader that does enforce real
Python semantics (the eager validator) in the same session as the reader
that doesn't (the WGSL build). Nothing before this inventory pass flagged
that the WGSL contact kernel is running on paths that also contain unverified
malformed subexpressions.

## Maintainability and consolidation assessment

The user asked directly whether these routes can be maintained by one
person, whether they share common authorities, whether they can be kept
reliably available and updated, and whether this presents a clear need to
consolidate. Answering each in turn, from what was directly read or
reproduced above:

**Do the routes share common authorities?** Partially, and inconsistently —
this is the crux of the problem, not a side note.

- *Good precedent exists and works.* `ssa_python_materializer`'s scalar
  vocabulary is audited against `ssa_llvm_backend`'s own tables at import
  time (`INVENTED` must be the empty set or the module refuses to import,
  `ssa_python_materializer.py:141-161`); its tensor-call spellings
  (`TENSOR_CALL_FORMS`) are read live from `AbstractTensor` itself via
  `inspect.getattr_static`, never a hand-kept list
  (`ssa_python_materializer.py:168-205`). These are real single authorities
  with a structural enforcement mechanism, not a convention someone has to
  remember.
- *Some conventions are single-authority in intent but not enforced.* Every
  native backend (`ssa_c_backend`, `ssa_llvm_backend`, `ssa_fortran_backend`,
  `ssa_wasm_backend`) reads `Pi` from the one
  `bounded_constants.materialize_pi` home rather than restating a literal —
  but nothing checks that a *new* backend does the same, and
  `ssa_python_materializer` simply had no `Pi` case at all until this
  session's fix. Likewise, a `Const` instruction's payload key
  (`"constant"` vs `"value"` vs `"values"` vs `"llvm_literal"`) is
  re-implemented inline with the same precedence in
  `ssa_reference_evaluator.py:524`, `ssa_c_backend.py:376`, and (after this
  session's fix) `ssa_python_materializer.py` — three copies of the same
  four-line precedence, not one function every reader calls. This is exactly
  the silent-drift shape the repository's own docstrings warn about
  elsewhere ("a hand-kept list... free to drift") but had not itself been
  protected against here — confirmed by the fact that it *had* drifted:
  `ssa_python_materializer` was missing the primary key.
- *One route is not an authority at all — it is a second, disagreeing
  definition.* Route A's `_AbstractTensorPythonPrinter`
  (`abstract_ui_vehicles.py:3938`) hand-implements the same "how does SymPy
  `Max`/`Min`/`Pow`/`tanh` become an AbstractTensor method call" question
  Route C's catalog already answers correctly (post-fix). It is not reading
  from the catalog; it is a wholly separate table that gets the receiver
  selection wrong. Two definitions of the same fact, one of them broken, is
  the opposite of a shared authority.
- Route G (`signal_math`'s baked cores vs. the SSA `Tanh`/`unary_double`
  libm opcode) is a second example of two independently-arrived-at "correct"
  answers for the same mathematical operation (both ultimately checked
  against SymPy, but through unrelated code paths with no test asserting the
  two agree with each other to a stated tolerance).

**Can one person maintain this today?** Only with substantial tribal
knowledge and real risk, because a law can currently pass through up to four
different pieces of machinery that do not fully agree with each other on
what is valid AbstractTensor Python (Route A's printer, Route B/C's
catalog-driven pipeline, and the AOT frontend's own structural reading of
that same generated text), and at least one of those disagreements is
currently silent in production (the contact kernel). Discovering this kind
of defect currently depends on a human happening to exercise two different
readers of the same text in the same sitting, which is what happened by
accident this session and in the 2026-09-03 continuation report before it.

**Can they be kept reliably universally available and updated?** Not as
currently structured, no — the concrete finding above is the proof: identical
text, silently valid to one reader, fatal to another. Any new law is exposed
to the same risk the moment it contains a SymPy `Max`/`Min`/`sqrt` with a
numeric first argument, and whether it "works" depends on which of the two
readers happens to run it first.

**Is there a clear need to consolidate?** Yes — and the plan already reaches
this conclusion independently through its Phase 2 decision criteria ("exactly
one reduction feeds both eager and native"; "tensor-method spelling produced
by the compiler's vocabulary, not a hand table"). This session's findings are
concrete supporting evidence for that conclusion, not a new argument: Route
A's printer is a duplicate, disagreeing table that should be deleted once its
consumers (the eager vehicle-law bindings, already migrated in spirit if not
yet in code, and the contact-kernel precompile) move to Route C; the
`Const`-payload-key convention should become one function every reader calls
instead of three independent copies; and the Route A/AOT-frontend
discrepancy for the contact law specifically should be re-verified against
Route C once that migration happens, since Route C is real, ordinary,
CPython-executable Python and cannot silently disagree with itself the way a
printer-vs-AST-frontend pairing can.

### 6. Phase 1's named risk — per-lane scalar invocation in the native shell — is already retired in source

`vehicle_native_deployment.py:1121-1129`: `vehicle_native_graph_tick_batch`,
the per-lane loop that calls the scalar `vehicle_graph_tick_vector` core once
per batch lane, is wrapped in `#if 0` with the comment *"Iterative scalar
batch wrapper disabled. The canonical vectorized Python graph is the only
exported tick implementation."* The dead code is still in the source (not yet
deleted) but is not compiled in. The tire appendage microstep loop (48
substeps, `:984`) is a different, intentional temporal integration detail
(sub-dt integration within one tick), not a per-lane scalar dispatch, and is
out of scope for this migration.

## What this means for Phase 2/4

Route C (`ssa_python_materializer`, `tensor_vocabulary=True`) is the existing,
compiler-owned, catalog-audited path the plan's Phase 2 candidate 1 calls
for. Four fixes (`Tanh`, `Max`/`Min`, `Pi`, and the `Const` payload key) have
now been applied and verified against the two real laws that were broken —
this was materially less work than building a new emitter, and it keeps one
authority (the SSA Route B already produces) for both the eager and native
stages, per the plan's Phase 2 decision criteria.

**Remaining work, not done this session (Phase 4 steps 2-5):**
1. Point `vehicle_python_compilation.symbolic_abstract_tensor_source` at
   `materialize_function_body`/`materialize_ir_module` instead of
   `_abstract_tensor_python`, and delete the latter once nothing calls it
   (Route A's printer is also used by `compile_wheel_contact_abstract_tensor`
   for the contact law — check that consumer before deleting).
2. Re-run `tools/frame_parity.py` for all four laws and the native
   build/DLL comparison the plan's Phase 5 acceptance gates call for — both
   need a build/lowering step, which this session deliberately did not run
   without explicit approval.
3. Retire the dead `#if 0`-guarded per-lane scalar wrapper in
   `vehicle_native_deployment.py` once the above is confirmed stable.
