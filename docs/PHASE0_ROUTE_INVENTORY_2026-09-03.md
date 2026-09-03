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

### 2. Route C already exists and mostly satisfies the plan's Phase 2 criteria — but has two real gaps of its own

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

### 3. Route D is narrow and self-contained

Single consumer, single degenerate shape (rank-2 `MatMul` of two
`MatrixSymbol`s for wheel-wrench reduction). Nothing else calls
`compile_sympy_matrix_to_abstract_tensor_backend`. No drift risk, no deletion
needed, out of scope for the migration.

### 4. Phase 1's named risk — per-lane scalar invocation in the native shell — is already retired in source

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
for — it is not vaporware, it needs exactly two vocabulary entries fixed
(`Tanh` unary, `Max`/`Min` binary tensor redirect) before it can replace
Route A for the vehicle body/material/contact/fixture laws. This is
materially less work than building a new emitter, and keeps one authority
(the SSA route B already produces) for both the eager and native stages, per
the plan's Phase 2 decision criteria.

Recommended next step (not yet done, awaiting go-ahead per "never run a
build/lowering without approval" and to keep this session's scope to
demystification): extend `_UNARY_SPELLING`/`_BINARY_SPELLING` in
`ssa_python_materializer.py` for these two gaps, then re-run the Phase 3
square-one law set (`tests/test_symbolic_abstract_tensor_stage.py`) against
route C instead of route A's printer.
