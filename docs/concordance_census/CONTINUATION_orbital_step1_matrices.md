# Continuation: orbital benchmark, work item 1 (matrices and complicated lhs)

Base: main 09c579e7. Uncommitted. Lane M1, 2026-10-02.

## Status

Item 1 resolved. 2 of 9 laws now compile through the sanctioned lane and
match the sympy reference natively (C and LLVM, max relative error 0):
`initial_condition_rhs`, `terminal_condition_rhs`. The raw set
(`orbital_transfer_raw`) passes `declared_symbolic_outputs` and the
MatrixElement ingest, then stops at item 4 (Derivative adjoint). Every
other law stops at item 3 (unbound applied-undefined Call), item 4, or both.

    python -m pytest tests/test_orbital_transfer_compile.py -q
    4 passed, 7 xfailed

## What changed

- `src/compiler/symbolic_equation_compiler.py`
  - `declared_symbolic_outputs(equations, name)` replaces the "equation
    output must be a Symbol" refusal. A scalar equation declares one output.
    A matrix-valued equation (lhs Matrix, MatrixSymbol or matrix expression,
    rhs of the same shape) declares one output per component, row-major.
    Mismatched shapes, or a matrix on one side only, raise `TypeError`.
  - Naming: a Symbol lhs names its output. An element of a MatrixSymbol
    `M[i, j]` names it `matrix_component_name("M", (i, j))` = `M_i_j`. Any
    other lhs is unnamed: the output is `<name>_<k>` for equation `k`, its
    components `<name>_<k>_<i>_<j>`.
  - LHS convention (DEFAULT, the user can change it): a lhs that is not a
    declared name (e.g. `Derivative(r1(s), (s, 2))`, `r1(0)`) is a residual
    equation. The output is `lhs - rhs`, built unevaluated
    (`Add(lhs, Mul(-1, rhs, evaluate=False), evaluate=False)`), zero where
    the equation holds. This follows the lane's existing linear-system
    convention (`abstract_ui_vehicles.engine_playable_linear_equations`
    uses `equation.lhs - equation.rhs`). Solving for the lhs's unknown was
    not chosen: it is not always possible and would rewrite the law.
  - After ingest, input columns are checked: one column name declared by two
    different values, or an input column named like an output, raises
    `ValueError` (the old Symbol-only recursion check is kept as well).
  - Function metadata gains `symbolic_outputs` (output, equation index,
    component, form) and `matrix_element_inputs` (column, MatrixSymbol,
    shape, element index).
  - Book: `compile_sympy_equations` posts on every call (cache hit or not),
    into the active book: each Equality NOVEL(`INGEST_SOURCE`) on
    `symbolic_equation` (row = (program, k)); each output DERIVED from its
    equation's cell on `symbolic_equation_output` (row = (program, output),
    fact = equation index, component, `SymbolicOutputForm`). Program scope =
    `symbolic_program_scope` = (law name, sha256 of the authored sreprs), so
    two sets under one name never share a row. Mode CONCORD.
  - The new helpers are in `_pipeline_implementation`'s digest.
- `src/compiler/symbolic_process_graph.py`: `MatrixElement` rule in
  `SYMPY_PROCESS_GRAPH_TRANSLATIONS` and an `add_node` branch. An element of
  a MatrixSymbol with integer indices and shape is an `Input` node,
  `binding_name = matrix_component_name(M, index)`, attributes
  `matrix_symbol`, `matrix_shape`, `matrix_index`. Any other parent, or a
  symbolic index/shape, raises `TypeError`. `matrix_component_name` is the
  one spelling for both input columns and output components.
- `src/compiler/concordance_declarations.py`: pages `symbolic_equation`,
  `symbolic_equation_output`; facts `SymbolicEquationFact`,
  `SymbolicOutputFact`; enum `SymbolicOutputForm`.
- `tools/compiler_probes/probe_orbital_transfer.py`: `WORK_ITEMS`,
  `LAW_BLOCKERS`, `blocker_reason`, `benchmark_laws` (raw set + per
  entry/side). The reference evaluates the compiler's own declared outputs
  and reads each declared MatrixSymbol element from its column (the fixed
  center/boundary numbers are gone). `main` prints the work list and a
  per-law verdict; "work-list drift" counts unexpected fails and XPASSes.
  stdout uses `errors="backslashreplace"` (the Greek mu crashed cp1252).
- `tests/test_orbital_transfer_compile.py`: one case per law, strict xfail
  from `LAW_BLOCKERS`; a sync test (union of reasons == `WORK_ITEMS`); an
  item-1 identity test on the raw `initial_condition` Equality (3 residual
  outputs, 3 `r_start_i_0` columns, book rows DERIVED from the equation).

## Remaining work list

2. Greek-name sanitation. Laws: raw, equation_of_motion_rhs,
   total_energy_expression. Assigned by construct presence only. `mu_1`
   (U+03BC) now passes `compile_sympy_equations` as an argument name; its
   behaviour in lowering is NOT OBSERVED (those laws fail on item 3 first).
3. External functions as runtime-provided symbols. Laws: all seven
   remaining. Observed: equation_of_motion_rhs now reaches `piece_from_law`
   and fails `undefined_operands=6` (r1..r3, F1..F3 call results feeding
   planned region 0); force_cost `undefined_operands=3`; boundary lhs
   `call-result-unavailable`.
4. Live differentiation/integration. Laws: raw, equation_of_motion_lhs,
   total_energy_expression: `no graph-native adjoint rule for <n>:call`.

## Gate (this lane)

- pytest: 4 passed, 7 xfailed.
- probe_orbital_transfer: 2/9 laws reach the book (chains reach
  source_span); work-list drift 0; exits 1 while items 2-4 are open.
- probe_emission_chain: failures 0. probe_scalar_native_correctness:
  failures 0.
- audit: findings 0,1,0,1,5,0,0 (unchanged).

## Next

Item 3: a declared binding for an AppliedUndef accepted by
`compile_sympy_equations` and carried into the materialized source. When it
lands, remove 3 from `LAW_BLOCKERS`; the strict xfails will say which laws
flip.
