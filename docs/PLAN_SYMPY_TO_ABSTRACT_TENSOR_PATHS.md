# Action plan: one clean path from SymPy to AbstractTensor Python to native

Goal (user, 2026-09-03): know exactly how many paths exist from a SymPy-authored
law to AbstractTensor Python, what the system depends on today, and what it
takes to have ONE path that (a) attempts simplification, (b) turns the law into
tensor methods, and (c) is the only thing native products are lowered from.
Going back to square one with basic tests is acceptable if the existing copies
have drifted.

## Phase 0 — Inventory (read-only, one session)

Enumerate every route and its consumers with these searches, then fill the
table below. Known candidates going in:

| # | Route | Entry points (known so far) | Simplifies? | Tensor methods? | Feeds native? |
|---|---|---|---|---|---|
| A | SymPy printer -> AbstractTensor source -> AOT capture -> SSA | `abstract_ui_vehicles._AbstractTensorPythonPrinter`, `_abstract_tensor_python`, `compile_wheel_contact_abstract_tensor` | sympy `cse` only | yes (hand table: Pow/tanh/Min/Max; numeric receivers break) | yes: contact kernel (WGSL/C/WASM) |
| B | SymPy -> `ingest_sympy_expressions` -> ProcessGraph -> `process_graph_to_ssa_instrs` -> scalar SSA | `symbolic_equation_compiler.compile_sympy_equations` (+ persistent cache), `symbolic_process_graph` | SSA scheduling + `reduce_constant_exponent_pow` + identity pipeline | SSA vocabulary (tensor ops by name), typed scalar | yes: `compile_*_c`, LLVM, Fortran, WASM for vehicle body, material, fixture (the bundle's `abstract_ui_vehicle_step.c`) |
| C | SSA -> Python emission | `ssa_python_materializer.py` (unread), `fused_program_python_backend.compile_single_region_python` (dialect tables; used as the Fortran fidelity reference) | inherits B | unknown: check which dialects spell tensor methods | not today |
| D | SymPy matrix graph -> AbstractTensor backend regions | `compile_sympy_matrix_to_abstract_tensor_backend`, `SympyTensorBackendPrecompile` (abstract_ui_vehicles ~3400) | unknown | yes | unknown |
| E | `sympy.lambdify` | `tools/frame_parity.python_backend`, tests | no | no (float per lane) | never (proof only) |
| F | Other symbolic programs | `symbolic_fluid_model.compile_symbolic_fluid_step`, any `symbolic_*` module | via B | via B | check |

Searches:
```
grep -rn "lambdify\|PythonCodePrinter\|_abstract_tensor_python\|ingest_sympy_expressions\|compile_sympy_equations\|compile_ast_aot(" src/ tools/ --include=*.py
grep -rn "sympy" src/compiler/*.py -l
grep -rn "def compile_.*_abstract_tensor\|SympyTensorBackendPrecompile\|Precompile\b" src/compiler/*.py
```
Deliverable: the table completed with file:line, plus for each route the list of
runtime consumers (validator bindings, contact precompile, native shell
sections, WASM/WebGPU plugins, tests).

## Phase 1 — What depends on what today

Facts already established this session:
- The eager validator (`--python-material`) bound the vehicle body and member
  material through E (lambdify) until today; now through A's printer (broken:
  numeric receivers). The contact law goes through A. So the eager run mixes
  two routes, and the native vehicle body comes from B. Two reductions of one
  law -> parity by measurement only (the material gate discrepancy was this).
- B is cached on disk per source revision (`compile_symbolic_program`), so its
  cost objection (the reason A was written) is gone.
- The native contact kernel is A -> AOT -> SSA -> WGSL/C; the native vehicle
  body/material/fixture are B -> scalar emitters. The user's rule: NO scalar
  cores at runtime; batched only. Verify how the native shell batches the B
  products (48 tire microsteps loop, `vehicle_native_batch_lane`) and whether
  the vehicle-body C section is invoked per lane (if so it is exactly the
  pattern to retire).

Deliverable: a dependency diagram (route -> consumer -> artifact) and the list
of every place a second reduction of the same law exists.

## Phase 2 — Choose the canonical path

Candidate 1 (preferred): SymPy -> optional `sympy.simplify`/`cse` at ingest ->
ProcessGraph -> SSA (CSE, identities, precision sections, work contract) ->
AbstractTensor Python emission with tensor methods and a batch axis (route C,
compiler-owned) -> AOT capture of that Python -> native. One authority, one
reduction; the eager stage and the native product share the SSA.

Candidate 2 (allowed by the user): ProcessGraph -> AbstractTensor Python
directly (skip SSA for the eager stage), native still from SSA. Fewer moving
parts for the eager stage but two reductions again unless the Python emitter
and the SSA builder share the same scheduling. Only acceptable if the emitter
is a thin, verified spelling of the process graph.

Decision criteria: (1) exactly one reduction feeds both eager and native;
(2) tensor-method spelling produced by the compiler's vocabulary, not a hand
table; (3) batch axis is explicit; (4) identities/precision contracts apply.

## Phase 3 — Square-one tests (write before migrating anything)

A tiny law set, each pinned exact (`sympy.N(..., 40)`) vs the tensor stage vs
each native backend, per element AND on batch columns:
`y = x + 1`, `Max(0, x)`, `Min(1, Max(0, x))`, `sqrt(3) * x`, `sqrt(x*x + 1e-30)`,
`tanh(x)`, `x ** 2`, `x / (x*x + 1)`, `Abs(x - 1)`, a two-output law with a
shared subexpression (checks CSE), and `pi * x`. Then the four real laws
(fixture, material, contact float32, vehicle body) through `tools/frame_parity.py`
with the tensor stage as the authority and lambdify demoted to a reference.

## Phase 4 — Migration

1. Implement/verify route C: SSA -> AbstractTensor Python (tensor methods,
   batch axis). If `ssa_python_materializer.py` already does this, adopt it;
   if it emits scalar/numpy dialect only, extend the dialect table there (the
   compiler's own vocabulary), never a new printer.
2. `vehicle_python_runtime_bindings` binds route C output. Delete
   `symbolic_abstract_tensor_source`'s printer dependence.
3. Contact precompile: replace A's printed source with route C's emission,
   keep the AOT capture and the WGSL/C lowering unchanged; confirm the
   captured program is identical or ULP-equal (frame parity).
4. Native shell: vehicle body/material/fixture products from the SAME SSA the
   eager stage uses; retire per-lane scalar invocation if found in Phase 1.
5. Delete `_AbstractTensorPythonPrinter` once no consumer remains; keep
   lambdify only in `frame_parity.python_backend` and tests.

## Phase 5 — Acceptance

- `tests/test_symbolic_abstract_tensor_stage.py` passes for all four laws.
- `tools/frame_parity.py` for fixture, material, contact, vehicle body: tensor
  stage vs C/LLVM/Fortran at ULP level over 64 fed-back frames.
- `--python-material` validator runs the tensor stage; native build lowers the
  same SSA; N-frame Python-vs-DLL parity holds through the feedback loop.

## Risks / unknowns to resolve early

- Does the AOT capture accept compiler-emitted Python unchanged (names, batch
  shapes, `mutable_parameters`)? Try on the fixture first.
- Contact law is float32 in SSA; the tensor stage must carry that dtype.
- Relational selects: the material comment says C has no scalar relational
  select; `Max`/`Min` are the sanctioned spellings (they lowered fine).
- Route D's consumers are unknown; do not delete anything before Phase 1.
