# AUDIT: sympy → process-graph op spelling coverage (2026-10-02)

Read-only audit. No source edited, nothing compiled. The matrix was computed
by importing the live tables (`BACKWARD_RULES`, `TENSOR_OPERATION_SCALAR_SPELLING`,
the materializer/evaluator/C/LLVM tables) and applying each reader's own
lookup rule to each emitted spelling.

Working-tree caveat: `src/compiler/symbolic_process_graph.py` and
`src/compiler/ssa_python_materializer.py` were uncommitted and being edited
while this ran (the approved Min/Max fix: ingestion now emits
`minimum`/`maximum`, and the materializer reads through
`TENSOR_OPERATION_SCALAR_SPELLING`). Line numbers in `symbolic_process_graph.py`
are as of the audit. Both the committed spelling (`Min`/`Max`) and the
working-tree spelling (`minimum`/`maximum`) are rows below.

## Where the SSA goes (the readers)

`compile_sympy_equations` → `ingest_sympy_expressions` →
`process_graph_to_ssa_instrs` (`ssa_builder.py:11`, copies `node["op"]`
unchanged into `Instr.op`) → `const`→`Const` rename, input strip
(`symbolic_equation_compiler.py:~345-385`) → `reduce_constant_exponent_pow`
(`ir_identities.py:78`, which adds `Mul`/`Div`/`Sqrt`/`Const`). From there:

- **Direct scalar lanes** receive these spellings unchanged:
  - C: `emit_ssa_function_to_c` (`ssa_c_backend.py:345`, dispatch 410-553),
    used by `vehicle_native_assembly.py:540,587,745,821`.
  - LLVM: `emit_ssa_function_to_llvm` single-function body
    (`ssa_llvm_backend.py:5361`; Pi 5727, Select 6172, libm Call 6041-6066,
    `scalar_likeness` 662), used by `sympy_native.py:176`.
- **Probe lane** (`tools/compiler_probes/probe_orbital_transfer.py` →
  `native_package.piece_from_law`): `symbolic_abstract_tensor_source`
  (`vehicle_python_compilation.py:52`) → **`ssa_python_materializer`** → Python
  AbstractTensor source → `lower_ast_source_to_ssa` → LLVM / `emit_ssa_module_to_c`.
  In this lane the native emitters see AST-lane spellings, not the sympy ones;
  the materializer is the only reader of the sympy spellings.
- **Reference evaluator** `ssa_reference_evaluator._step` (lookup 497-516 via
  `TENSOR_OPERATION_SCALAR_SPELLING`; `_BINARY` 123, `_UNARY` 155; Call intrinsics 713-735).
- **Backward**: `process_graph_autograd` casefolds the op (`_operation`,
  1177-1178), stops at `_PREDICATE_OPERATIONS` (1380-1384), then
  `graph_adjoint_rule_name` (47-51) → `BACKWARD_RULES` (lookup 2056-2058,
  raise 2071-2074). Only reached for a `Derivative` that SymPy cannot reduce
  (`symbolic_process_graph.py` Derivative branch → `obtain_graph_reverse`).

## Matrix

Legend: Y = handled; N = not handled (raises / shortfall); stop = predicate,
gradient stops there by design; key→X = key of the spelling table mapping to X;
value = already the table's scalar spelling; — = no entry.

| Emitted op (source) | (1) backward rule | (2) SCALAR_SPELLING | (3) materializer | (4) ref evaluator | (5) C (`emit_ssa_function_to_c`) | (6) LLVM (`emit_ssa_function_to_llvm`) |
|---|---|---|---|---|---|---|
| Add (Add) | Y add | key→Add | Y | Y | Y | Y |
| Mul (Mul) | Y mul | key→Mul | Y | Y | Y | Y |
| Pow (Pow) | Y pow | key→Pow | Y | Y | Y | Y |
| Sqrt (Pow ½) | Y sqrt | key→Sqrt | Y | Y | Y | Y |
| Div (Pow reduction) | Y div | key→Div | Y | Y | Y | Y |
| Mod (Mod) | Y mod | key→Mod | Y | Y | Y | Y |
| Abs (Abs) | Y abs | key→Abs | Y | Y | Y | Y |
| Sin (sin; cot/csc respell) | Y sin | — | Y | Y | Y | Y |
| Cos (cos; cot/sec respell) | Y cos | — | Y | Y | Y | Y |
| **Tan** (tan) | Y tan | — | **N** | **N** | Y | **N** |
| **Tanh** (tanh) | Y tanh | — | Y | **N** | Y | Y |
| Exp (exp) | Y exp | key→Exp | Y | Y | Y | Y |
| Log (log) | Y log | key→Log | Y | Y | Y | Y |
| Floor (floor) | Y floor | — | Y | Y | Y | Y |
| **Ceil** (ceiling) | **N** | — | Y | Y | Y | Y |
| Min / Max (committed spelling) | Y min/max | value | Y | Y | Y | Y |
| **minimum / maximum** (working-tree spelling) | Y | key→Min/Max | Y | Y | **N** | **N** |
| Eq Ne Lt Le Gt Ge (relations) | stop | value | Y | Y | **N** | Y |
| **LAnd LOr LNot** (And/Or/Not) | **N** (not in predicate stop-set) | value | Y | Y | **N** | Y |
| **LXor** (Xor) | **N** | — | **N** | **N** | **N** | **N** |
| **Select** (Piecewise, sign, Heaviside) | Y where | — | Y | Y | **N** | Y |
| **Pi** (pi) | **N** | — | Y | **N** | Y | Y |
| Const (literals) | leaf | — | Y | Y | Y | Y |
| **Indexed** (Indexed, getitem) | N | — | **N** | **N** | **N** | **N** |
| **Tuple** (Tuple) | N | — | **N** | **N** | **N** | **N** |
| Call sinh/cosh/asin/acos/atan/asinh/acosh/atanh (scalar) | **N** (`call`) | — | Y (tensor method) | Y (intrinsic) | Y | Y |
| **Call atan2 / erf / any other sympy Function** | **N** | — | **N** (bare call) | **N** | **N** | **N** |
| **Call r1, F1… (AppliedUndef)** | **N** | — | **N** (bare call) | **N** | **N** | **N** |
| unsqueeze (Sum/Integral lowering) | N | — | Y (catalog) | **N** | **N** | Y (shape alias) |
| sum / prod (reductions) | Y | — | Y (catalog) | **N** | **N** | **N** |
| max (unary extent reduction) | Y | — | Y (catalog) | **N** | **N** | **misread**: casefolds to binary `Max` template |
| arange / get_tensor | N | — | Y (catalog) | **N** | **N** | **N** |
| where (mask) | Y | — | Y (catalog) | Y | **N** | Y |

Note: `input`/`const` are structural (stripped/renamed before any reader) and omitted.

## Gaps, by reader

1. **`minimum`/`maximum` (the working-tree Max/Min fix) breaks both direct native scalar lanes.**
   - C: `ssa_c_backend.py:484` tests the exact strings `{"Max","Min"}`; `maximum` falls to `553` "no direct scalar C spelling". The module lane is also exact on the casefold (`ssa_c_backend.py:4987,5023`, `{"max","min"}`).
   - LLVM: `scalar_likeness` (`ssa_llvm_backend.py:662`) looks up `_BINARY` / `_BINARY_FOLDED` only. `"maximum"` is in neither, so it returns None and raises a shortfall (`sympy_native.py:176` lane).
   - Route: both should read through `hierarchical_plan.TENSOR_OPERATION_SCALAR_SPELLING`, as the evaluator (`ssa_reference_evaluator.py:513`) and now the materializer (`ssa_python_materializer.py:547`) do.
2. **`Tan`**
   - The materializer (`ssa_python_materializer.py:613` falls through to `648`) has no scalar form. Its catalog fallback is case-sensitive: `tan` is catalogued but `Tan` is not.
   - The evaluator (`ssa_reference_evaluator.py:155` table, raise at `702`) has no entry.
   - LLVM `_UNARY` (`ssa_llvm_backend.py:76`) has no Tan, and the casefold doesn't help.
   - `TENSOR_OPERATION_SCALAR_SPELLING` has no `tan` key. Route: the spelling table plus the LLVM `_UNARY` likeness table, which owns the vocabulary that both the materializer and the evaluator audit against.
3. **`Tanh`**: the evaluator lacks it (`ssa_reference_evaluator.py:155`; it shows up in `_UNIMPLEMENTED`). The LLVM `_UNARY` table has it. Route: LLVM `_UNARY` → evaluator `_UNARY`.
4. **`Ceil`**: there is no backward rule. Casefolded it is `ceil`, which is absent from `BACKWARD_RULES` (`floor` is at `backward_registry.py:1059`). Fails at `process_graph_autograd.py:2057`. Route: `BACKWARD_RULES`.
5. **Relations, `LAnd`/`LOr`/`LNot`, `Select` in C `emit_ssa_function_to_c`**: the dispatch (`ssa_c_backend.py:459-551`) never consults `_C_COMPARISONS` (`5727`), `_LOGICAL_BINARY` (`210`) or Select. Every Piecewise, sign, Heaviside or relation law shortfalls at `553`. Yet `supported_scalar_operations()` (`682-699`) advertises all of these, because that function describes the module lane. Route: the existing `_C_COMPARISONS` / `_LOGICAL_BINARY` tables.
6. **`LAnd`/`LOr`/`LNot`/`LXor` in backward**: the stop-set (`process_graph_autograd.py:1380-1384`) spells `logical_and`/`logical_or`/`logical_not`. The casefolded ingestion spellings `land`/`lor`/`lnot`/`lxor` therefore miss it and raise at `2071`. Route: `TENSOR_OPERATION_SCALAR_SPELLING` (`logical_and`→`LAnd`) or `hierarchical_plan.is_predicate_operation` (`PREDICATE_OPERATIONS` lists all four).
7. **`LXor`: no reader implements it.**
   - `hierarchical_plan.PREDICATE_OPERATIONS:115` declares it.
   - LLVM `_BINARY` has only `Xor` (`ssa_llvm_backend.py:56`).
   - The materializer, evaluator and C lanes have only `Xor` too.
   - `TENSOR_OPERATION_SCALAR_SPELLING` has no `logical_xor` key.
   - Route: the LLVM likeness `_BINARY` (`Xor`) through the spelling table.
8. **`Pi`**: the evaluator has no branch, so it raises at `ssa_reference_evaluator.py:702`; the materializer, C and LLVM all read `bounded_constants.materialize_pi`. Backward: `pi` is not skipped like `const` (`process_graph_autograd.py:2039`), so a gradient reaching it raises. Route: `bounded_constants.materialize_pi`; on the backward side, the `input`/`const` leaf rule.
9. **`Call` to a sympy Function outside the libm set** (atan2, erf, …) **and applied undefined functions** (r1…r3, F1…F3):
   - Every reader fails: the materializer emits a bare unbound name (`ssa_python_materializer.py:448-491`), the evaluator raises at `743`, C at `553`, and LLVM finds no libm callee.
   - Backward has no `call` rule (`process_graph_autograd.py:2057`). These are orbital work items 3 and 4.
   - The libm callee set is restated three times: `symbolic_equation_compiler.py:365`, `ssa_c_backend.py:492`, `ssa_llvm_backend.py:6046`. The evaluator keeps a fourth copy at `713-735`.
   - Route: the callee set has no single table today. The nearest existing owner is `ssa_llvm_backend._TENSOR` (`323`, the `unary_double` rows).
10. **Reduction/quadrature tensor ops** (`unsqueeze`, `sum`, `prod`, `max`, `arange`, `get_tensor`, `where` from `lower_sum_declaration` / `lower_integral_declaration`, `symbolic_process_graph.py:585-673`):
    - Only the materializer's catalog handles them.
    - The evaluator raises on all of them except `where`. Its `_operation_name` lookup does not map them, and `TENSOR_OPERATION_SCALAR_SPELLING` has no `max` key, only `maximum`.
    - Neither direct scalar native lane handles them.
    - LLVM `scalar_likeness("max")` casefolds onto the binary `Max` template for what is a one-operand reduction. That is a misread, not a miss.
    - Route: `ssa_llvm_backend._TENSOR` (`sum`/`prod`/`max`/`arange`/`where`) and `_SHAPE_ONLY` (`unsqueeze`).
11. **`Indexed`, `Tuple`**: no reader handles either.
    - `Tuple` reaches SSA only if it is not the expression-set envelope or a consumed Integral/Derivative argument.
    - `Indexed` reaches SSA only from `IndexedBase` laws.
    - Neither occurs in the orbital set.

## Orbital benchmark set: sympy constructs with no mapping

From `probe_orbital_transfer.build_program()`, classified with `_sympy_process_graph_rule`:

- **`Derivative` (6)**: no table entry. It is special-cased in `add_node`. Every occurrence is a Derivative of an AppliedUndef, so `sympy.diff` stays unevaluated and the path goes to `obtain_graph_reverse` → `call` → no adjoint rule (work item 4).
- **`Integral` (1)**: no table entry, special-cased. `doit` fails on `sqrt(F·F)` of undefined F, so it takes the quadrature path (`get_tensor`/`unsqueeze`/`sum`; gap 10).
- **AppliedUndef `r1 r2 r3 F1 F2 F3`**: there is no specific mapping. These fall through the MRO to `sympy.Function` → generic `Call(callee=name)`, which is unbound (work item 3; gap 9).
- **`ImmutableDenseMatrix`, `MutableDenseMatrix`, `MatrixSymbol`, `Str`**: no rule. They are reached only inside componentization (`declared_symbolic_outputs`) and `MatrixElement` inputs, never ingested as nodes.
- **`MatAdd` / `MatMul`**: no own entry. They resolve through the MRO to `Add` / `Mul`.

All other constructs in the set map directly: Add, Mul, Pow (sqrt via Pow ½), Integer/Rational/Half/One/Zero/NegativeOne, Symbol, MatrixElement, Equality, Tuple. The set contains no trig, Min/Max, Piecewise or relation, so gaps 1-8 are not exercised by it.
