# Continuation: guarded tuple return published per lane (lane F, 2026-09-30)

Repro: `python -u tools/compiler_probes/probe_record_in_tuple_return.py`
(read its docstring for the six-link diagnosis; it is unchanged and correct).

## What changed

One file, `src/compiler/fortran_c_shell.py`, two places, nothing else.

1. `_normalize_top_level_guard_returns` (the single-exit rewrite for
   compile targets whose top-level guard bodies end in `return`).  Before,
   `result_assignment` bound the WHOLE returned expression to one private
   name (`__turing_single_exit_result`), so a guarded `return a, b` bound an
   aggregate in each arm with no SSA producer; the conditional lowering
   promoted the merge arms to unnamed formals and the return-surface
   expansion found no lanes.  Now the rewrite runs `nest` twice with an
   `emit` callback: a first pass only collects every return it will publish
   (guarded returns plus the terminal), the exit shape is chosen from all of
   them at once, and the second pass emits the assignments.  When every
   collected return is an `ast.Tuple` literal of one common arity N (no
   starred element), N private names `__turing_single_exit_result_{lane}`
   are assigned per lane at every site and the final statement is
   `return (name_0, ..., name_{N-1})`, exactly the shape
   `_normalize_direct_tail_recursion`'s `ExitReturnRewriter` already
   produces (`tuple_result_arity`, lane names, tuple-of-names return).  Name
   collisions step by N as the tail-recursion rewrite does.  Any other shape
   (mixed arities, a non-tuple return) keeps the previous single-name
   behaviour byte for byte, including the `_{suffix}` collision scheme.

2. New helper `_single_exit_tuple_arity(returns)` directly above it: returns
   the common tuple arity or 0.

Receipt: the existing mechanism is kept (`_normalize_top_level_guard_returns`
returns receipt dicts; `lower_ast_source_to_ssa` stores them at
`module.metadata["single_exit_guard_normalization"]`).  Each receipt still
carries `function`, `result_name` (now the first lane's name), `guard_count`,
`source_lines`, and gains `result_names` (tuple, one per lane) and
`tuple_result_arity` (0 when the whole value went through one name).  The
`report(...)` line "ssa-source: normalized return-only guards in N selected
function(s)" is unchanged.

Consumers checked: `glsl_deployment_strategy.py` matches the binding name by
`startswith("__turing_single_exit_result")`, which the per-lane names still
satisfy; the repro probe's own value-name check uses the same prefix.

## Verified (all on this tree, after the edit)

- `python -m py_compile src/compiler/fortran_c_shell.py`: ok; file stays
  uniform CRLF, ASCII.
- `probe_record_in_tuple_return.py`: exit 0.  One-site control and the
  guarded form both lower; in both the record parameter is a return slot
  with a field layout and the Ret publishes the record field(s) then the
  scalar lane.
- `probe_branch_written_field.py`: exit 0 (seam present: True).
- `probe_annotated_scalar_parameter.py`: exit 0 (plain / annotated /
  annotated with return).
- `probe_scalar_native_correctness.py`: 14/14, failures: 0.
- `python -u tools/audit_identity_concordance.py` first lines:
  view 493 rows / 24 functions / 0 findings; toplevel 322 / 31 / 1;
  energy 490 / 22 / 0; controller 528 / 27 / 1; controller_untyped
  531 / 27 / 5; mapping 24 / 2 / 0; oscillator 331 / 22 / 0.  No document
  in the repo records these exact counts, and a stash or checkout baseline
  is forbidden, so "unchanged" is established by construction: the rewrite
  only touches selected compile targets (`entrypoint` plus
  `dependency_seeds`, which the audit never passes), and none of the seven
  roots (`root` / `tick`) has a return-only top-level guard.
  `_energy_time_limit` has a guarded `return 0.0, 0.0` but is a callee, not
  a target, so it is not rewritten in `energy` or `toplevel`.

## What remains

- Nothing for this defect.  Not measured: the Woodshop / native dt-system
  lowerings (not allowed in this lane), and whether any selected target in
  those programs has guarded tuple returns that now take the per-lane path.
  The receipt's `tuple_result_arity` field tells you which path fired.
- `probe_record_in_tuple_return.py` still documents the pre-fix chain in its
  docstring as "Diagnosis"; it passes, so the text is now historical.

## Working-tree state the next session must know

- Uncommitted (do not commit from this lane): `src/compiler/fortran_c_shell.py`
  (this fix).  `src/compiler/glsl_deployment_strategy.py` is also modified
  in the working tree by a concurrent lane; it is not part of this fix.
  `shots/` is an untracked directory, also not mine.
- The repo is `C:\dev\Powershell\turing`, branch main, a nested git repo
  inside the root-only coordination repo.
