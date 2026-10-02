# Differentiation feasibility for the collocation planner (2026-10-02)

Lane FD. Probes only, no compiler edits. Context:
`docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md` step 4 (collocation planner
needs a residual Jacobian). Observed facts and inferences are kept apart.

## 2. Implicit differentiation of the Kepler stage

Probe: `tools/compiler_probes/probe_collocation_jacobian.py kepler` (87 s,
of which ~64 s is the orbital_plan import).

Observed:
- Implicit function theorem on R = E - e sin E - M (sympy):
  dE/dM = -1/(e cos E - 1) = 1/(1 - e cos E).
- LEO 7000 km -> GEO 42164 km plan, e_t = 0.715239, 64 transfer-leg samples
  (t_burn1 + T*[0.02..0.98]) through the compiled `_solve_kepler` (6 stages,
  max residual 1.79e-16). Central FD of the converged E over the compiled M
  vs 1/(1 - e cos E) (range 0.583..3.317), max relative error:
  step 1e-4 T: 2.79e-7; 1e-5 T: 2.79e-9; 1e-6 T: 9.4e-11.
  The error falls as step^2 down to the roundoff floor: it is the FD's own
  truncation error, not a mismatch.

Verdict: the implicit derivative matches. Nothing new is needed: it is one
closed-form law (1/(1 - e cos E), already the factor in eq_KE2's dE/dt)
evaluated at the converged E. Inferred: through the solve, a collocation
residual should use this law as the stage's adjoint instead of
differentiating the six unrolled Newton stages (which agree only at
convergence).

## 4. Sparsity of an N-slice collocation chain

Probe: `probe_collocation_jacobian.py sparsity` (7 s). The slice is built
from the jumper's own builders (`orbital_jumper.gravity_force_rhs` over two
centers, `orbital_actuation.actuation_force_rhs`/`thrust_magnitude`,
`eq_N1_2`, `eq_N1_1` with the same xreplace as `orbital_jumper_dt_pieces`),
six-axis jumper (1000 kg, 100 kN), design constants substituted as numbers.
Residual = x_{k+1} - step(x_k, u_k, dt_k), symplectic Euler (position uses
the updated momentum, as the sequential round does). Columns per slice:
r_k(3) p_k(3) u_k(6) dt_k(1); plus x_N.

Observed (slice 0; free-symbol pattern == sympy `jacobian` nonzeros):

    row 0 (p_x)  xxxx..xx.......x..x     cols: r0 xyz | p0 xyz | u0 0..5 |
    row 1 (p_y)  xxx.x...xx......x.x           r1 xyz | p1 xyz | dt0
    row 2 (p_z)  xxx..x....xx.....xx
    row 3 (r_x)  xxxx..xx....x.....x
    row 4 (r_y)  xxx.x...xx...x....x
    row 5 (r_z)  xxx..x....xx..x...x

48 nnz of 6x19; every row has 8. r_k is a dense 3x3 (gravity), p_k
diagonal, two throttles per axis, dt_k dense, x_{k+1} identity. The cost
row depends only on u_k and dt_k.

Chain (block bidiagonal: row block k touches column blocks k and k+1 only):

| N | J | nnz | density | CPR colors (natural / largest-first) |
|---|---|---|---|---|
| 1 | 6x19 | 48 | 42.1% | 8 / 8 |
| 2 | 12x32 | 96 | 25.0% | 11 / 9 |
| 5 | 30x71 | 240 | 11.3% | 11 / 9 |
| 20 | 120x266 | 960 | 3.0% | 11 / 9 |
| 100 | 600x1306 | 4800 | 0.61% | 11 / 9 |

Verdict: 9 colored evaluations (largest-first greedy) recover the full
residual Jacobian for any N; lower bound 8 (max row nnz). Independent of N.
Inferred: 9 forward-mode/complex-step sweeps or 6 reverse sweeps per slice
(rows) both suffice; with a per-slice graph adjoint (item 1) the block is
6 VJPs on a 19-column slice, reused for every k.

## 1. Collocation slice Jacobian, graph-native

Probe: `probe_collocation_jacobian.py jacobian [--relabel-minmax] [--compile]`.
Route: the item-4 slice (6 residuals + cost
`1e-4*fuel_rate + dt - 1e3*log(5e6 - dt*fuel_rate)`) ->
`symbolic_process_graph.ingest_sympy_expressions(strict=True)` ->
`process_graph_autograd.differentiate_process_graph(outputs=[root_i],
wrt=<variables in row i>)`, one adjoint per output row.

Observed, ingestion: 154 nodes, ops input 19, const 18, Mul 55, Add 43,
Max 6, Min 6, Pow 6, Log 1. Strict ingestion: no fallbacks. All 19 variable
leaves are `input` nodes.

Observed, differentiation as-is: 0/7 rows differentiate. Verbatim:
`backward graph produced no gradient for 10, 15` (rows 0, 3; 60, 64 rows
1, 4; 83, 87 rows 2, 5; all six for the cost). Those ids are u0_0..u0_5.
Cause: the throttle clamp `Min(Max(u, 0), 1)` ingests as binary nodes
`Max(parents=[const 0.0, u])`; `_operation` casefolds `Max` -> `max`, which
in `BACKWARD_RULES` is the UNARY reduction rule (gradient to its single
operand `x`), so the gradient lands on parent 0 -- the constant -- and the
throttle gets none. No rule was missing; the binary one (`maximum`/`minimum`)
exists and was not selected. Not refused loudly as "no rule": it surfaced
only as a missing gradient.

Observed with `--relabel-minmax` (probe-local relabel of Max/Min nodes to
`maximum`/`minimum`, simulating the edit): 7/7 rows differentiate, 0
refusals; backward graphs 225..290 nodes; rules used add, mul, pow,
maximum, minimum, log. No other op lacked a rule.

Smallest edit: `src/compiler/symbolic_process_graph.py:76-77`, translate
`sympy.Min`/`sympy.Max` to `"minimum"`/`"maximum"` (the reverse table at
:180-185 already accepts both spellings), or arity-route `max`/`min` with
two parents in `graph_adjoint_rule_name`. Unverified: which spelling the
forward SSA lowering of sympy laws expects.

Observed, compile (`--relabel-minmax --compile`): one adjoint over all 7
outputs, `fuse_forward_loss_backward(unit_loss_seed=False)` -> motion of
828 nodes (seeds 154..160) in 1 s; then
`lower_training_motion_to_repository_ssa(motion)` stops, verbatim:
`ValueError: node 157 has no ingestion scope: the graph was neither built
by build_from_ast nor entered lexical normalization`
at `src/common/tensors/topological_reducer.py:567 node_identity_cell`, via
`glsl_deployment_strategy.py:16298 _argument_identity_cells` <-
`:17114 _propagate_callsite_planner_specializations` <-
`strategize_shell_deployment` <- `process_graph_autograd.py:2571`.
Node 157 is a seed input. Inferred: the identity-book precondition
(`ingestion_value_scope` is minted only in `graph_express2.py:5247`, the
AST build; `operand_position_scope` only in lexical normalization) is not
met by ANY graph `lower_training_motion_to_repository_ssa` receives from
differentiation -- the motion graph is built fresh in
`fuse_forward_loss_backward`, and no caller (`llvm_training_runtime`,
`perforated_network_llvm`) mints a scope. So this is not sympy-specific;
the graph-native backward compile route is blocked upstream of my slice.
(Not run: `tests/test_process_graph_autograd.py`, whose hand-built-graph
native tests would show the same if the inference holds; baseline in
TEST_BASELINE_AND_HAZARDS.md is from 2026-08-19, before the book check.)

Smallest next edit: `fuse_forward_loss_backward` (or
`lower_training_motion_to_repository_ssa` entry) mints the motion graph's
`ingestion_value_scope` from the book (`book.mint_scope("ingestion:
training_motion:<name>")`) -- the same post `graph_express2.py:5247` makes,
so `node_identity_cell` posts rows instead of raising.

Observed, numeric check without compiling (`--render`): each row's fused
motion rendered by `process_graph_to_sympy_expressions` is not evaluable --
the backward rules appear as `getitem(Call(...))` (calls into the
`BACKWARD_RULES` closure), which sympy cannot evaluate. So no numeric
Jacobian-vs-sympy number exists yet: the only executor for those calls is
the native lowering that is blocked above. (Did not interpret the rule
bodies in Python: that would be a stand-in executor.)
Reference side is ready: sympy `jacobian` of the slice at the sample point
has 55 nonzeros in 7x19 (48 residual + 7 cost).

Verdict, item 1: graph-native differentiation of the slice is feasible
with one translation edit (Max/Min); the Jacobian could not be checked
numerically because backward-motion lowering raises on the identity-book
scope for every differentiated graph. Two edits, in order: Max/Min rule
spelling, then the motion graph's ingestion scope; then rerun
`jacobian --compile` (drop `--relabel-minmax`) for the relative error.

## 3. Complex step through the compiler

Probe: `tools/compiler_probes/probe_complex_step.py eager|native`,
x = 0.7, h = 1e-30.

(a) Eager AbstractTensor, observed: `AbstractTensor.tensor([0.7+1e-30j])`
selects `NumPyTensorOperations`, data complex128; every result stays
complex128. Im f/h vs analytic, relative error:
sin*exp 0; x**3+log 0; sqrt/(1+x^2) 2.2e-16; Squire-Trapp
exp/sqrt(cos^3+sin^3) 1.7e-16. Complex survives eagerly on the NumPy
backend (sin/exp/log/sqrt/pow/div). Not measured: the polyspline signal
cores (not on this path) or other backends.

(b) `lower_ast_source_to_ssa` + C emission + native, observed. Each form is
a batch AbstractTensor function lowered exactly as
`native_law_kernels._lower_law` lowers a law (x a span of 1 under a
`batch_contract`-shaped ExtractionContract, `c_backend_repository_ssa_reference`).
None reaches C emission; complex does not survive lowering in any form.

- literal, `z = x + 1e-30j; w = z.sin()*z.exp(); return w.imag()/1e-30`
  (x float64): stops in lowering, verbatim
  `TypeError: float() argument must be a string or a real number, not 'complex'`
  at `src/compiler/tensor_ssa_lowering.py:5660`
  (`constant(float(scalar_payload), "float64")` feeding
  `binary_scalar_double`), via `fortran_c_shell.py:41373
  _class_surface_ssa_program -> lower_tensor_calls_to_repository_ssa`.
- complex_op, `z = AbstractTensor.complex(x, x*0.0 + 1e-30)`: stops in
  lowering, verbatim `FortranEmissionError: full-native execution contract
  rejected the linked repository SSA: ... unaccounted_formals=({'function':
  'cs_complex_op__tick', 'detail': "2 formals but only 1 named or
  ABI-accounted; unnamed value ids [5] with accounting {5: {}} ..."},)`
  (`fortran_c_shell.py:45626`). Cause not traced (unknown which pass made
  the extra formal).
- fed, `return x.sin()*x.exp()` with x DECLARED complex128 in the
  contract: the dtype reaches the physical call, then, verbatim,
  `ValueError: 2 incompatible final physical call inputs; storage types are
  immutable: ('cs_fed__tick__planned_region_0', 0, 'complex128',
  'unary_double', 212, 'float64'); (...same...)` at
  `fortran_c_shell.py:41974`.

Observed common cause: the C tensor basis the lowering targets is
float64-only (`unary_double`, `binary_double`, `binary_scalar_double`);
`complex64/complex128` appear nowhere in `ssa_c_backend.py`,
`ssa_llvm_backend.py` or `tensor_ssa_lowering.py`. No Im f/h number exists
for the native lane.

Verdict, item 3: complex step works eagerly (1e-16), not through the
compiler. Smallest next edit is not one line: a complex128 storage type in
the tensor basis (complex variants of unary/binary/binary_scalar in
`c_backend_llvm_ssa`'s opcode table and their C/LLVM kernels), after which
the literal form also needs `tensor_ssa_lowering.py:5660` to carry a complex
scalar instead of `float()`. Inferred: for the planner, the graph adjoint
(item 1) or the closed-form implicit law (item 2) needs fewer new parts than
a complex lane.
