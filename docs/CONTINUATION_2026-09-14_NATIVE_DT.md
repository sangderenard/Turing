# Native managed tire and dt controller continuation

## Provisional use accepted; validator integration resumed

The user accepted the measured full-window precision for the moment and
requested use in the original Python validator. This permits integration to
proceed without first eliminating the 169/167 adaptive-schedule difference.
It does not turn the existing strict 38/48 comparison into a passing result
or establish a numerical error bound for the coupled validator. See
`CONTINUATION_2026-09-14_VALIDATOR_NATIVE.md` for that separate integration.

## Confirmed repair

The unfinished v167 investigation in `PATCH_SEQUENCE_2026-09-07_ALL_19.md`
is resolved at its recorded compiler boundary. The saved v167 module's
`conditional_phi_continuation_receipts` explicitly identifies the invalid
`run_superstep` header-Phi rewrite from raw proposal 274 to conditional Phi
565. This happens in `reconcile_conditional_phi_continuations`, after local
lowering and frame linking already preserved the two logical loop bindings.

That pass now retains an explicitly selected loop update when its unique,
exact SSA producer dominates the backedge. A dominating conditional merge
does not establish assignment to a different loop binding. The decision has
an `exact_loop_carried_update` receipt, with incumbent tie policy. Ordinary
stale captures still advance to their dominating conditional result.

Structural recovery also uses object identity for membership in its insertion
list. Read-only stack inspection repeatedly found the old replay spending
its time in recursive instruction equality at this exact-membership check.
The superseded pre-fix replay was stopped after the bad rewrite was proved
from v167 provenance and the new failing regression; it was not time-limited.

## Verification

- The new late-pass regression failed before the repair. It now covers both
  an owned raw snapshot and an ordinary stale capture, plus idempotence.
- The return-state/control batch passed 42 tests, with 72 deselected.
- Adjacent shell checks: 3 passed, 2 failed. Both failures were reproduced
  using the two compiler modules from HEAD in an isolated Python process,
  without changing the checkout: `test_whole_object_if_assignment_lowers_to_conditional_phi_merge`
  assumes the old function name, and `test_nested_if_threads_inner_phi_into_outer_phi`
  omits the now-required extraction contract. Their assertions were retained.
- The trusted source-side checkpoint was replayed through the current
  compiler: six frame rounds, 4,538 formals, three result-type rounds,
  zero incompatible result contracts, zero structural findings.
- Final production SSA has recurrence update 286 -> backedge 565 and
  snapshot update 274 -> backedge 274; value 328 has no definitions or uses.
- The explicit `-O0` standalone one-step executable completes with all
  48 public buffers matching eager execution at the existing tolerances.
  Window and initial dt are both `0.000244140625` seconds.
- Two consecutive short outer windows also match all 48 buffers using the
  same native executable and persistent state within each execution.

## Artifacts and reproduction

Run from the Turing repository with the existing working Python 3.11 runtime
(the old `.venv` still points at an absent Python 3.10 installation).

```powershell
python tools/replay_ssa_checkpoint.py build/patch_sequence_source_v147/pre-frame-link.pkl --output build/dt_native_owned_update_20260914/repository-ssa.pkl
python tools/build_managed_checkpoint_native.py build/dt_native_owned_update_20260914/repository-ssa.pkl --output build/dt_native_owned_update_20260914-one-step-o0 --window-duration 0.000244140625 --dt-initial 0.000244140625 --optimization O0
python tools/managed_dt_parity.py build/dt_native_owned_update_20260914-one-step-o0 --frames 1
```

The one-step result is preserved as
`build/dt_native_owned_update_20260914-one-step-o0/managed-dt-parity-one-frame.json`;
`managed-dt-parity.json` in that directory records the two-frame run.
Full-window inputs were prepared separately under
`build/dt_native_owned_update_20260914-full-window-o0`: requested window
`1/120`, initial proposal `1/360`. This uses the identical executable, with
SHA-256 `a8aec22c54a7ffcf81ee02206139c7f1cf6b5c945804943b5e5da1b761e7646c`.
All physical input arrays were regenerated and shape-checked against the
48-buffer manifest. The parity driver independently checks initial bytes
against its own reconstructed fixture before either execution.

## Full-window result and remaining frontier

Both executions complete the requested `1/120` second outer window exactly.
Native takes 169 successful substeps and eager takes 167. Both report zero
critical and zero nonfinite attempts. Maximum per-substep displacement is
`0.0001507525859315176` m native and `0.00015954991310288886` m eager, against
the existing `0.003` m limit. The running native process loaded only its
executable and Windows/C runtime DLLs; Python was not its runtime.

Strict pointwise comparison remains **38/48**, with the existing `rtol=1e-8`
and `atol=1e-10`. Mismatches cover material state/output, telemetry, controller
state, and the next proposal. Final `dt_next` is `1.9393494246175835e-05`
native versus `1.7811625127278842e-05` eager. The strict result was retained
as a failure; successful outer-window completion does not change that verdict.

Evidence is in the full-window directory's `managed-dt-parity.json`,
`managed-dt-parity-mismatches.npz`, and `managed-dt-window-observations.json`.
The latter separates measured outer-window completion from strict buffer
agreement. The maximum per-step displacement is not a bound on cumulative
native/reference state error.

To rerun the full-window comparison:

```powershell
python tools/managed_dt_parity.py build/dt_native_owned_update_20260914-full-window-o0 --frames 1
```

The next numerical investigation must locate the earliest causal divergence
using this current checkpoint. The older v145 substep-32 trace is historical
evidence, not proof of the cause of this new 169/167 schedule difference.
The old patch sequence's full domain/rollback coverage matrix also remains
open. No optimization, acceptance-tolerance change, dt-law change, or commit
was made. This work concerns the managed tire/controller target, not the
complete vehicle viewer. All processes launched for this continuation have
terminated; the two pre-existing Python jobs were left untouched.
