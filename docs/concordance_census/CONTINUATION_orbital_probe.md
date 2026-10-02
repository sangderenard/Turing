# Continuation: the orbital transfer set as the compiler's show-off example

Base: main dbc59f0a. Uncommitted. Lane OT, 2026-10-02.

## Status

The set does not compile yet. 0 of 8 laws reach the book; 9 failures. The
probe reports them and exits 1. No substitution layer: the user ruled that
every construct that fails is compiler work, not something to rewrite away.
The set made it into a ProcessGraph before (graph_express2 suite,
`ProcessGraph.build_from_expression`); it has never been compiled.

## What was added

- `tools/compiler_probes/probe_orbital_transfer.py`. Builds
  `Orbit.stable_orbit_transfer_solution(Orbit.symbolic_orbit('1'),
  Orbit.symbolic_orbit('2'))`, the graph_express2 call. Offers the raw
  Equalities to `compile_sympy_equations`, then one law per dict entry and
  side (outputs named `<entry>_<side>_<k>`; the expressions are untouched).
  Route per law: `compile_sympy_equations` -> `piece_from_law` -> C emission
  + compile -> native (C and LLVM) vs a sympy reference (concrete r(s), F(s)
  and center values on the reference side only, `doit`, `lambdify`,
  1e-12 relative). A law that reaches the book prints process-graph nodes,
  book cells, emission units sourced/unsourced, and one STATEMENT unit's
  chain through `edges_into`/`mint_of` to `source_span`.
  `lower_for_viewer(sink)` is the viewer's builder.
- `src/compiler/native_package.py` `piece_from_law`: new keyword
  `resolved_process_graph_sink=None`, passed unchanged to
  `lower_ast_source_to_ssa`. The viewer draws the graph of the same
  lowering instead of lowering twice.
- `tools/view_identity_concordance.py` `lower_probe`: `--probe orbital`
  calls `probe_orbital_transfer.lower_for_viewer`. Today it exits with the
  failure list (no law lowers, so there is no book and no ring). No PNG was
  produced.

## Failures, verbatim (ranked, smallest fix first)

1. Raw set, `compile_sympy_equations`:
   `TypeError: equation output must be a Symbol: Eq(Matrix([[Derivative(r1(s), (s, 2))], ...]), ...)`
   at `src/compiler/symbolic_equation_compiler.py:91`
   `_compile_sympy_equations_uncached`. Fix there: a Matrix lhs names one
   output per element.
2. `equation_of_motion_rhs`, `initial_condition_rhs`, `terminal_condition_rhs`:
   `TypeError: no SymPy to ProcessGraph translation rule for MatrixElement: c_1[0, 0]`
   (and `r_start[0, 0]`, `r_end[0, 0]`) at
   `src/compiler/symbolic_process_graph.py:1064` `add_node` (strict).
   Fix in `SYMPY_PROCESS_GRAPH_TRANSLATIONS` / `ingest_sympy_expression`:
   MatrixSymbol = parameter of declared shape; MatrixElement = index into it.
3. `force_cost_integral`, `initial_condition_lhs`, `terminal_condition_lhs`:
   `compile_sympy_equations` succeeds (the Integral lowers as 5-point
   Gauss-Legendre); `piece_from_law` -> `lower_ast_source_to_ssa` raises
   `FortranEmissionError: full-native execution contract rejected the linked repository SSA`
   at `src/compiler/fortran_c_shell.py:45626` `_lower_ast_source_to_ssa_impl`.
   Cost: `undefined_operands=3` (the `F1/F2/F3(t13)` call results, value
   name `t15` etc., feeding `force_cost_integral__force_cost_integral__planned_region_5`).
   Boundary lhs: `structural_outputs=(... (4, 'call', 'call-result-unavailable') ...)`.
   Cause, observed in the materialized source: `t15 = F1(t13)` with `F1`
   unbound. Ingest emits `Call callee='F1'` for an applied undefined
   Function and nothing binds it. Fix: a declared binding for an
   AppliedUndef (column, sampled Table or bound callee; `bitops.declare`
   already asks "Table or Function?") accepted by `compile_sympy_equations`
   and carried into `symbolic_abstract_tensor_source`.
4. `equation_of_motion_lhs`, `total_energy_expression`:
   `ProcessGraphAutogradError: ProcessGraph has no graph-native adjoint rule for 1:call`
   (`2:call` for the energy) at `src/compiler/process_graph_autograd.py:2072`
   `differentiate_process_graph`, reached from the Derivative branch of
   `symbolic_process_graph.add_node` (`obtain_graph_reverse`). Depends on 3:
   an unbound call has no derivative. Then an adjoint rule for `call`.
5. Greek symbol names (`mu_1` spelled with U+03BC): unknown. Every law that
   carries them fails at 2 or 4 first. The name is a valid Python
   identifier.

## Gate

- New probe: runs, reports the 9 failures, exits 1 (expected until 1-4 land).
- `probe_emission_chain.py`: failures 0.
- `audit_identity_concordance.py`: findings 0,1,0,1,5,0,0 (unchanged).
- Viewer: `python tools/view_identity_concordance.py --probe orbital --emit both`
  exits with the failure list. Snapshot command for when it lowers:
  `python tools/view_identity_concordance.py --probe orbital --emit both --snapshot shots/orbital_ring.png --exit-after 6`

## Next

Fix 1, then 2; rerun the probe (seconds per law). After 3, the cost law
should reach the book and exercise the native-vs-reference check, the counts
and the chain. 4 last.
