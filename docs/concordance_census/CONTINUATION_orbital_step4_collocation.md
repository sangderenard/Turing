# Continuation: orbital craft step 4, the collocation planner

Task: `engine_toy/orbital_collocation.py` + `engine_toy/tests/test_orbital_collocation.py`.
Design: `docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md` (decisions 4, 7; build
step 4). Jacobian: compiled graph-native reverse, unit-seed VJP, one motion per
residual row (explicit-seed fused motion is another agent's lane). Owned files:
those two only. Not owned: orbital_tracker/jumper/actuation (report hooks).
Do not commit.

## 2026-10-03 session start

State at start: turing has uncommitted `src/compiler/precompile_to_ssa.py`,
`docs/concordance_census/CONTINUATION_name_arm_alias.md`,
`tools/compiler_probes/probe_nested_inplace_arm.py` (other lanes; not mine).
No orbital_collocation.py existed.

Read: design doc, DIFFERENTIATION_FEASIBILITY, probe_collocation_jacobian.py,
CONTINUATION_jacobian_compile_fixes.md (its last entry: on the probe slice the
12 `Indexed(Call bw_*)` grads are DROPPED by lowering, and it says this is NOT
seed-specific -- one output, unit seed, same shortfalls).

Observed by reading: `orbital_tracker.tracking_command`/`fly` call
`orbital_plan.reference(plan, t)` (a module function over HohmannPlan legs).
A non-Hohmann plan cannot be flown by the tracker without a hook there.

## 2026-10-03 BLOCKER: the unit-seed compiled VJP is broken too (not seed-specific)

Observed (scratch `unit_row.py`: probe `slice_laws(0)` row 0 alone ->
`ingest_sympy_expressions(strict)` -> `compile_process_graph_backward(
packaging="combined", unit_loss_seed=True)` -> `lower_training_motion_to_
repository_ssa` -> `emit_ssa_function_to_llvm` -> `compile_artifact`, ~12 s):
- lowering shortfalls (), LLVM shortfalls (), native builds.
- loss_0 native -199679.20679840667 vs sympy -199679.20679840664 (forward OK).
- 7 of 8 grad outputs (grad_29/33/2/10/15/0/5) are NOT in `buffer_order`;
  the one present (grad_23, r0_x) reads 0 (ref -0.0232). buffer_order carries
  backward intermediates (182, 180, 185, ...) as formals.
Same shape as CONTINUATION_jacobian_compile_fixes.md "KeyError 806 lane".

Observed: `tests/test_llvm_training_runtime.py::test_graph_reverse_is_a_compiled_parametric_vjp`
(AbstractTensor left*right VJP, ~8 s) FAILS at clean HEAD de609156
(worktree C:\Users\alber\AppData\Local\Temp\wtc): `KeyError: 7` (grad 7 not
in the execution buffers). So the whole graph-reverse compile route is red,
AbstractTensor and sympy alike, unit seed and explicit seed alike.

Bisect (git bisect run on that test, bc5b347a good .. de609156 bad, wtc):
first bad commit **534a4941 "Fix concordant numeric solve compilation"**
(2026-09-22; fortran_c_shell, glsl_deployment_strategy +617, identity_concordance,
precompile_to_ssa, ssa_call_input_adapters, tensor_ssa_lowering, ...).
e39ff7cf (its parent side) passes. 87b37d5d skipped (RecursionError).
Not contained (2169-line commit; the same symptom is another agent's active
lane), so not fixed here. The grads fall to the structural-recovery fallback
`fortran_c_shell.py:21121-21132` (`basic-index-contract`) because the planned
region does not publish them; which hunk of 534a4941 drops them: UNKNOWN.

## 2026-10-03 File/hunk narrowing of 534a4941 (wtc worktree, overlays on e39ff7cf)

One file at a time from 534a4941 onto e39ff7cf, same test: only
`src/compiler/glsl_deployment_strategy.py` turns it red (`KeyError: 7`); the
other 8 files each pass alone. Hunks (git diff -U0, 27 hunks): hunks 0-9
applied = pass; 0-14 = KeyError 7; 0-12 = NameError publish_return_members
(hunk 3 adds a 162-line block that hunks 10-14 move out of
`_propagate_callsite_tensor_specializations`); 0-13 HUNG (>600 s, killed).
So the drop lives in the `_propagate_callsite_tensor_specializations` /
`publish_return_members` rework (534a4941 lines ~15990-16450). Stopped
narrowing there; wtc restored to de609156 clean.

## 2026-10-03 Observed at HEAD on the tiny VJP (scratch trace_members.py, hook only)

`compile_native_graph_reverse(left*right)`: motion grads {0: 7, 1: 8}, seed
{2: 3}; final `buffer_order (0, 1, 2)` -- grads 7, 8 AND seed 3 all absent.
Hook on `glsl_deployment_strategy._publish_callsite_return_members`
(HEAD :17427): backward rule `Call 6` (kind tuple, 2 descriptors) is
published as an aggregate on the first fixed-point round, incumbent () ->
leaves (7, 8): the existing `Indexed(6, const)` grad nodes are REUSED
(`resident_direct_projection`), gaining `authored_call_result_projection`;
no new nodes. Rounds 2-3: unchanged. Before 534a4941 this publication was
gated by `_tensor_descriptor(caller, node) is None` (hunk @@ -16229), so the
backward Call kept its own descriptor and was never turned into an aggregate
producer. Inferred, NOT traced: the aggregate-producer view of a backward
rule Call is what keeps its Indexed members (the grads) out of the planned
region's outputs, so lowering's structural fallback refuses them
(`basic-index-contract`: the members carry no `basic_index_axes`). The link
between the publication and the region's output list is the missing link.
Stopped here: not contained; same symptom is the KeyError-806 lane's.

## 2026-10-03 FIX (contained): reference-copy collision in the training-motion lowering

Observed: every graph-reverse motion whose backward uses `pow` (sympy
`sqrt(x**2+y**2)`, `x/y`) died in `lower_training_motion_to_repository_ssa`
at `tensor_ssa_lowering.py:5745` "repository SSA function collision for
'binary_value'". Cause (read + confirmed by the fix): `_class_surface_ssa_program`
deep-copies its `tensor_ssa_reference` (fortran_c_shell.py, c9607d25) and links
the COPY's Function objects; `process_graph_autograd.py` then lowered the
remaining tensor calls from the cached ORIGINAL (`c_backend_repository_ssa_reference`
is lru_cached) -> two `binary_value` objects, one name.
Fix: fortran_c_shell records the copy as `module_metadata["tensor_ssa_reference"]`;
`lower_training_motion_to_repository_ssa` links from `module.metadata
["tensor_ssa_reference"]` (raises if absent). x*y - c still compiles with
correct values (13, d/dx 5, d/dy 3, d/dc -1).
Regression check: tests/test_process_graph_autograd.py 4 failed / 20 passed
both at clean HEAD (wtc) and with the fix, same 4 tests.
tests/test_symbolic_fluid_native_runtime.py fails identically at clean HEAD
(ConcordanceRefusal planner_specialization 'advance'), not this fix.

## 2026-10-03 Next walls (not contained; recorded, not fixed)

W2. With the collision fixed, a pow backward reaches LLVM emission and stops:
`training_motion__bw_pow__specialized_*: Call: operation has no repository
LLVM emission` and `training_motion__unbroadcast__specialized_*: Call: ...`.
The unemitted Calls (scratch powcall.py, x/y): in unbroadcast, a Call with
`tensor_operation='extent'` and no callee; in bw_pow, `Call __plan_callsite_3__`
with no args. Repro (fresh process, ~15 s):
`compile_reverse_rows("p", [x / y], [x, y])` in engine_toy/orbital_collocation.py.
W3. Two compiles in one process share state: the second compile is refused
(`call_link_order_concordance disagreement`, `call_result_projection_concordance
disagreement`, `ConcordanceRefusal ... REVISE without a changed source`) when
both post into the one detached book. `orbital_collocation.compile_reverse_rows`
now opens one book per row (`begin_identity_book`/`end_identity_book`), which
fixes x*y-c followed by x*x*y+c ... except the second (pow) compile then stops
at `precompile_to_ssa.py:4671` "region 6 reads carried initial 56 through an
operand no lexical_read_binding row attributes" (precompile_to_ssa.py is dirty
from another lane; NOT baselined clean). Inferred: process-global state beyond
the book (e.g. the lru_cached backward-rule closure graph) leaks across compiles.
W1 (above) still stands for graphs that do lower: grads absent from buffer_order.

## 2026-10-03 State of the deliverable

engine_toy/orbital_collocation.py written: transcription, laws (jumper's own
builders + eq_KE1_3 vis-viva arrival + decision-7 cost), Hohmann warm start,
SLSQP over the compiled rows, CollocationPlan + reference(). Jacobian behind
`compile_reverse_rows` (one unit-seed motion per row today).
engine_toy/tests/test_orbital_collocation.py: 2 compile-free tests PASS
(arrival rows vanish on the circle to 1e-14; warm start = Hohmann burns through
the actuation matrix to 1e-6). Convergence test: ERROR at slice row 0, W2
(230 s). Flight test additionally needs a tracker hook: `orbital_tracker`
calls `orbital_plan.reference(plan, t)` (HohmannPlan legs only).

## 2026-10-03 Coordination note

At wrap-up HEAD is b3ad7e17 and other lanes have uncommitted edits in the same
files: fortran_c_shell.py (a plan_callsite marker retirement fix for the
bw_log graph-reverse motion -- the `__plan_callsite_N__` half of W2) and
glsl_deployment_strategy.py (likely W1). My compiler edits are ONLY:
fortran_c_shell.py `module_metadata["tensor_ssa_reference"] = ...` (+comment,
after `module_metadata["identity_book"] = ...`) and the `reference = ...`
block in process_graph_autograd.lower_training_motion_to_repository_ssa.
Worktree wtc (mine) removed. wtb was moved to e39ff7cf by someone else.
Not committed.

## 2026-10-03 (later) Walls down; rewrite for the new craft

Coordinator: W1 (42a689a2 plan_callsites), W2 (eps marker, per-callsite
folds), W3 (`reverse_compile_book`) fixed in turing 095a3c0c..51b4cebe; my
binary_value fix committed. engine_toy changed under me: variable mass
(N7.2), attitude in the actuation law, propellant supply, leapfrog kicks;
tracker reads `plan.reference(t)` / `plan.impulses()` (348cb5d).

orbital_collocation.py rewritten:
- state per node: momentum(3), position(3), propellant mass(1); slice law =
  the jumper's laws (N4.1, TS1.2/1.4 flow, N7.2, N1.1) stepped kick-drift-kick
  with supply 1 (INVENTED: the slice-level form of the jumper's leapfrog).
- planning design at a declared attitude (default identity).
- cost prices thrusters as `orbital_tracker.fuel_price`'s rule (kg/(N s)
  when the design burns propellant, else impulse).
- `compile_reverse_rows` = ONE explicit-seed fused motion per row set inside
  `reverse_compile_book`; one native run per seed.
- CollocationPlan: reference(t), impulses() (each thrusting slice at its
  midpoint), mu, ideal_delta_v, r2; collocation_replanner(problem, craft=).
First slice compile (7 rows x 23 wrt, KDK): still compiling after 600 s
CPU, 1.1 GB (tool moved it to the background; waiting for it).

## 2026-10-03 Measurements so far (explicit-seed fused motion)

- arrival rows (5 rows x 7 wrt): compile 96 s; values+5 seeds 0.19 ms per
  call; vs sympy jacobian max rel 1.8e-16; repeated runs bit-identical.
- slice rows, 2 thrusters: compile 476 s. Six-axis (6 thrusters, 7 rows x
  27 wrt): compile 874 s; values+7 seeds 0.76 ms per call (scalar arena
  reuse: prepare_artifact_execution once, indexed write/read; the first
  version re-prepared per run: 241 ms per call).
- sympy reference check of the 6-thruster slice: sympy evalf of the
  unexpanded KDK tree did not finish in 15 min (faulthandler dump: all in
  sympy evalf, not the compiler); killed. Verification moves to the
  converged plan's own defects + the flight.
- cost split: per-slice fuel motion (1 row, n_u+1 wrt) + trip-total motion
  (J on I, T); dJ/dI, dJ/dT broadcast to slices on the host (the adjoint of
  a sum). A single N-slice cost graph would compile in O(N) time.
- compiled rows now persist across processes in %TEMP%/orbital_collocation_rows,
  keyed by the laws' srepr and perforated_network_llvm._compiler_fingerprint
  (the precedent for caching a compiled reverse).
- first full test run: slice (6 thrusters) compiled 871.6 s and was cached;
  at exactly 15:00 the faulthandler watchdog dump fired while the arrival
  compile ran and the process died with "Windows fatal exception: access
  violation" inside the dump (frames: identity_concordance.py:3867
  _post_or_unsourced <- :3854 record_shape_transformation <-
  glsl_deployment_strategy). Inferred, not proven: the crash is the
  watchdog's frame walk racing the main thread (faulthandler's dump of a
  running thread is documented as unsafe), not the compiler. Rerun with
  the slice cached.

## 2026-10-03 First convergence, then accuracy fix (substeps)

N=40, one KDK step per slice: SLSQP 139 it, 992 Jacobians (7 ms each),
solve 34 s, defect 2.3e-9, fuel 0.9983 x Hohmann, time 0.9981 x; ended
"Positive directional derivative for linesearch". Flown by the tracker
(six-axis): arrived (|r|-r2 27.7 m, |v|-vc 0.068 m/s, |r-r_ref| 6.4 m) but
fuel 1.549 x plan. Cause (measured): the plan's per-slice discretization
error is 994 m (one 85 s step vs 64 sub-steps of the same compiled law);
the tracker trims onto that error. The "0.998 x Hohmann" was the coarse
law's own error, not a better trip.
Fix: a slice is `substeps` (16) steps of the compiled law; the slice
Jacobian is the chain of each step's compiled Jacobian (forward accumulation
of compiled derivatives; no new compile). Now: per-slice error 3.6 m vs 256
sub-steps; SLSQP "Optimization terminated successfully" in 3 iterations, 7
Jacobians, 0.9 s; defect 2.1e-12; fuel 1.0000 x Hohmann, time 1.0016 x;
2 burns (247.45 and 239.37 m/s).
