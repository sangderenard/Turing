# Scalar record return-state increment

The next record repair is deliberately partial. The reducer now records each
authored return's observed field state, keyed by source span and exact receiver
identity, and remaps those identities into the canonical graph. Return branches
retain their original source slot identities, which select the corresponding
source receipt. The physical record-return Phi expansion can select a scalar
when those matching sites agree on the receiver/field and its unique non-formal
SSA definition dominates the incoming return edge. It does not rebind earlier
reads or change the record's global field descriptor. Missing receipts at any
matching site also prevent selection. The helper is included in the compiler
toolchain fingerprint so changing it invalidates stale compilation products.

Projection alias removal now remaps the new return receipts with the existing
return slots. The linker retains the source receipts in function metadata
(`record_return_state_receipts`) for inspection. A bounded real-step lowering
completed in 106.37s and confirmed the receipt and return edge share the same
remapped record identity (`build/record_return_step_repro.pkl`). This bounded
artifact is not the full validator's linked native result.

The full saved SSA exposed a second issue: 592 and 593 have float64 result
types, despite their Boolean leaves and the Boolean output field ABI. Selecting
593 therefore needs an explicit conversion. The lookup permits that conversion
only for an acyclic conditional-carried Phi tree with Boolean leaves; the
linker emits Cast in the return predecessor before its branch. The existing
intermediate dtype remains unchanged, preserving its consumers' storage ABI.
Arbitrary numeric leaves do not qualify.

Selection is limited to conditional-carried definitions. It checks effects
between the selected definition and the return, cutting re-entry to that
definition so a future loop iteration does not invalidate its newly computed
version. Unknown calls through the field's storage refuse selection. Known
linked callees are checked transitively for formal redefinitions, declared
field publication, and stores through explicit aliases; copying the field as
a Store value into a different destination is allowed. Recursive or unknown
call effects remain conservative refusals. This is a bounded field check,
not the general source-aware stale-read audit still requested by the review.

Physical return-field Phis retain their incoming record identities and are
revisited in subsequent link rounds. Readiness can change when a callee's
physical fields or effect information become available. Rechecking starts
from each source record's physical field, never from a previously selected
override; casts are reused by edge/source/type. The public result slot remains
stable throughout this process.

The source return-slot index is retained separately from physical record
correlation. In the controller, the physical record is 433 while the source
slot and receipt are 434. Selection crosses that boundary only when the exact
source slot's record and physical record have identical identities and field
descriptors. Both receiver identities participate in the effect check. The
reachability pass trims incoming receiver metadata alongside Phi arguments
and updates the surviving scalar seed. Its source file is now also included
in the toolchain fingerprint.

Effect checking distinguishes the record object from its scalar field storage.
A call through the whole record remains subject to the conservative check;
projecting a different field does not make that field an alias of hard_failure.
The saved whole-SSA selector succeeds with both the source (434) and physical
(433) record identities included. The corresponding field-scope regressions
and native variants passed 12 tests in 15.87s.

## Separate lexical-reference cache repair

The early static-reference failure recurred while trying to verify the final
linker increment. A deterministic test evicts an already resolved callee
projection and later reuses its static symbol; it reproduced the same KeyError
at `reference_attributes`. The reference cache now verifies that its node is
still present, still a StaticReference, and still names the same compiler
symbol. Otherwise it reconstructs the reference from the known Python binding
using the existing factory. It does not execute that binding or create an ABI
input. The new repro plus existing static-reference and receipt regressions
passed 4 tests in 2.26s after first reproducing the failure in 3.17s.

The lookup preserves the previous field on ambiguous receipts, missing
definitions, incompatible scalar types, and non-dominating definitions. This
is incremental support, not a claim that those fallbacks implement mutable
records correctly. It currently applies to physical `return_merge` Phis;
direct returns without that merge still need support. It does not infer
receiver aliases through calls, synthesize loop state, or keep otherwise
discarded field writes alive.

Keyed fields are excluded. `error_channels` requires an ownership-preserving
assignment into the record's physical storage, including the correct behavior
of existing aliases. Choosing a dictionary-handle Phi is not that repair.
Duplicate effect-token Phis 496/497 have not been removed. Optional numeric
field presence remains a hard blocker before native validator parity; the
managed controller supplies dt_min=None and dt_max=None. Presence support will
restore runtime control paths currently folded under the numeric-only ABI.

## Validation

- Reducer return-site receipts, existing scalar ledger shape, and effect-order
  regressions: 7 passed in 3.16s.
- Scalar edge lookup and bounded native SSA execution: 2 passed; the authored
  child-record-return regression is 1 expected failure, total 13.59s.
  The native check reuses execution buffers across initial-field and branch
  combinations, verifies the returned field and preserves the earlier value.
  It tests the scalar edge mechanism, not authored child-call lowering.
- The child-return regression initially failed twice at the full-native
  provenance gate. Even with an explicit observation of the child's field,
  the child retains an unnamed formal and the caller loses field provenance.
  That exact refusal is retained as an expected failure; other errors fail.
  This is not the requested end-to-end native proof.
- Before adding exact return-edge matching, the scalar helper, reachability,
  and existing record-read tests gave 9 passed / 1 expected failure in 20.92s.
- Final edge-matching/native helper and effect-order tests: 9 passed in 10.13s.
  A preceding run exposed that this repository's NetworkX adapter omits the
  dominator root self-entry; explicitly restoring it fixed entry-defined
  candidates. The distinct-return-site check reproduced that omission.
- Toolchain fingerprint, reducer receipt, and authored child-return checks:
  2 passed / 1 expected failure in 6.25s.
- Return-edge source identities after physical alias resolution: 1 passed in
  2.37s, covering both predicated and unconditional returns.
- Final scalar-return tests, including projection remapping, Boolean-leaf
  proof, and both native dtype variants: 8 passed in 17.91s. Native executions
  remain bounded at 20s and reuse buffers across both branch outcomes.
- Expanded field-effect tests and existing ordering tests: 16 passed in
  16.03s. They include transitive calls, value-copy versus target-store
  distinction, unknown writers, and loop re-execution. Applying the checked
  selector to the saved whole SSA chooses 593 (float64) for the Boolean
  return field. The fresh source run must also show its emitted conversion.
- Return/alias/native component tests and the existing two conditional tuple
  native cases: 13 passed / 1 expected failure in 36.24s. The expected failure
  remains the authored child-record ABI probe, not a passing end-to-end proof.
- After slot correlation and provenance pruning, scalar-return, reachability,
  and conditional-tuple native tests: 19 passed in 53.13s.
- Existing guard-loss probes, managed-input admission, and toolchain checks:
  7 passed / 2 expected failures in 6.34s. None admission still refuses numeric
  fields without presence storage. The two old guard-loss failures remain
  visible and are not native proofs.
- The child-return probe also retained its expected refusal with wildcard ABI
  bindings (1 expected failure in 5.15s); exact helper-name binding was not the
  cause of that failure.

An intermediate whole-source run completed with 19 formals and zero undefined
operands/unresolved calls (`artifacts/compiler_evidence/record_return_full_formals_20260906.log`).
It preceded the completed edge-matching change and is not its verification.
The following edge-only diagnostic also completed at 19/zero undefined/zero
unresolved (`artifacts/compiler_evidence/record_return_edge_full_formals_20260906.log`), but still
returned 1378. This caught the missing receipt remap and provisional float64
Phi types. The alias-only diagnostic was stopped during planning before
changing the conversion code; its partial log is not a completed result.
The first typed diagnostic stopped earlier in lexical reduction with a
missing cached static-reference node at topological_reducer.py's
`reference_attributes` lookup (`artifacts/compiler_evidence/record_return_typed_full_formals_20260906.log`).
This was before return linking. A separate clean retry is recorded below;
the early failure is not a completed formal-count measurement.
That retry completed with 19 formals, zero undefined operands/unresolved calls
(`artifacts/compiler_evidence/record_return_typed_retry_full_formals_20260906.log`), but the then
over-conservative effect check still retained 1378 at the return. The exact
receipts were now correctly remapped to receiver 434. Inspection showed the
calls carry the full record ABI, while `_propose_dt_pen` never consumes its
hard-failure formal and `_apply_energy_sidechain` forwards it to a helper that
does not consume it. This motivated the transitive SSA effect check above,
not an exemption based on those function names.

## Final publication boundary and verification

Per-round linking remained too early: replaying the selector on the completed
module succeeded while the return still used its initial field. The final
publication pass now runs after constant reachability and paired signature
cleanup, before undefined-operand checks and the strict native gate. It uses
the retained source receipts, exact source/physical record layout agreement,
the same dominance/effect checks, and original field storage from the record
table. It preserves output Phi identities and reuses return-edge casts.

Replay on `build/full_formal_diagnostic/repository-ssa.pkl` from
`artifacts/compiler_evidence/record_return_field_scope_full_formals_20260906.log` changed exactly four
step return fields: div_inf 1368 -> 553, mass_err 1369 -> 573,
dt_limit 1374 -> 555, and hard_failure 1378 -> Cast(593) at 9564.
The returned hard_failure Phi remains 1482. The second publication changes
zero fields and undefined operands remain empty. Replay artifact:
`build/record_return_final_pass_replay.pkl`. This replay is separate from the
fresh source diagnostic below.

The final targeted command covered `test_ssa_record_return_state.py`,
`test_ssa_reachability.py`, and `test_native_record_return_state.py`:
**18 passed, 1 xfailed in 26.38s**, external timeout90s. The two native dtype
variants now invoke the actual publication pass against typed record tables,
check idempotence and stable output identity, and execute all branch outcomes
with repeatedly reused buffers. The authored child-record case remains the
explicit expected failure described above; it is not an end-to-end proof.

No full native validator build, frame-parity run, authored DT edit, commit,
or push. Keyed error_channels 577, duplicate effect Phis 496/497, optional
presence storage, and authored child-record ABI recovery remain open.

### Fresh source result

`artifacts/compiler_evidence/record_return_publication_full_formals_20260906.log` completed with
terminal exit1 at the strict provenance gate: **19 formals, zero undefined
operands, zero unresolved calls, zero unmaterialized boundaries**, no optional
merge/non-native findings. The saved module records four final publications:
div_inf 553, mass_err 573, dt_limit 555, and hard_failure Cast(593) at9564.
The physical returned hard_failure remains Phi1482, now consuming9564.
The early planned_region_28 call at loop_exit still consumes1378. Reapplying
publication to this fresh module changes zero fields. The authoritative
`build/full_formal_diagnostic` artifacts have been regenerated from source.

The formal groups are unchanged: step10, run_superstep6, _no_exchange_observed1,
_propose_dt_pen1, coerce_metrics1. In particular496/497 remain deliberately
retained. This is verified scalar return wiring, not completion of the record
repair or a full native parity claim. All launched processes are terminal.
