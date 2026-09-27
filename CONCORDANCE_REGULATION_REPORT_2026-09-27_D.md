# Regulation through the concordance — report D (2026-09-27)

Continues [report C](CONCORDANCE_REGULATION_REPORT_2026-09-26_C.md) from
commit `b8d47465`. The target was the Woodshop Newton dt-system compile through
the sanctioned whole-source compiler and LLVM backend, with no findings from
the identity concordance.

## Outcome

The compile completes and the final audit is green:

```text
identity concordance: 6629 rows across 445 functions, 0 finding(s)
  id groups: legacy=2668, minted=3961
```

The 13 findings recorded at `b8d47465` were reduced in measured stages:

| Stage | Findings | Remaining classes |
|---|---:|---|
| `b8d47465` | 13 | alias-target-missing ×9; descriptor-member-shared ×2; descriptor-member-unknown ×2 |
| Output identity classified correctly | 9 | alias-target-missing ×5; descriptor findings ×4 |
| Exact descriptor and record-identity retirement | 4 | alias-target-missing ×4 |
| Final dead planning occurrences retired | 0 | none |

## Defects and repairs

1. `output_identity_aliases` was audited as substitutable physical storage.
   It is semantic output history: its target may be an edge occurrence retired
   after the merged output is published. The audit now records it as
   `output-identity-of`; the existing output-concordance agreement check still
   verifies the durable snapshot against its authoritative page.

2. Exact positional aggregate-result legalization updated record descriptors
   but did not advance the provisional terminal slots already reached by
   source projections. The same positional proof now publishes both the
   authored field occurrence and its existing terminal resident to the emitted
   output slot.

3. Linked record arguments left duplicate sequence descriptors after proving
   an exact handle/column identity. The finalizer now consumes that planning
   proof, binds extent/status cells by their declared descriptor roles, removes
   only dead private bookkeeping, updates record fields and formal accounting,
   and deletes the duplicate descriptor row. This removed both shared-member
   and unknown-handle findings without changing sequence contracts.

4. Planning-only record aliases and terminal aggregate projection edges
   survived after neither endpoint existed in final SSA. They are now
   tombstoned at finalization only when output concordance proves the record
   identity, or when neither endpoint survives in SSA or any descriptor and no
   alias depends on the source. The identity pages retain the full history and
   the function metadata records a retirement receipt.

## Verification

Full Woodshop Newton compile (Windows, Python 3.11, warm law-piece cache):

```text
cd C:\dev\Powershell\engine_toy
py -3.11 -u -c "from woodshop import WoodshopSimulation; from src.compiler.identity_concordance import concordance_report; sim=WoodshopSimulation(); system=sim.world_rules.lower_newton_dt_system('C:/dev/Powershell/turing/build/woodshop_newton_llvm', backend='llvm', optimization='O0', piece_mode='link'); print(concordance_report(system.module, limit=40))"

identity concordance: 6629 rows across 445 functions, 0 finding(s)
  id groups: legacy=2668, minted=3961
```

Focused identity regressions:

```text
py -3.11 -m pytest tests/test_sequence_contract_concordance.py -k "concordant_sequence_descriptor_alias or output_concordance_closes or finalization_closes or function_alias_publication" -q
4 passed, 15 deselected in 2.49s

py -3.11 -m pytest tests/test_process_graph_function_linking.py -k "output_alias_publication_uses_distinct_concordance_from_storage or planning_alias_refinement_records_concorded_transition or late_record_result_rebinds_aliased_fields_in_following_call_frame" -q
2 passed, 1 failed, 90 deselected in 3.61s

py -3.11 -m pytest tests/test_sequence_contract_concordance.py -q
17 passed, 2 failed in 5.85s
```

The one focused-file failure is the pre-existing assertion that all SSA ids
remain below `1_000_000_000`; the current compiler intentionally emits minted
ids in the `230584301021369xxxx` range. The new concordance assertion in that
same test executes before the baseline assertion and passes.

The two full-file sequence failures expect keyed-tensor planned-region calls
that are absent. Both fail identically at `b8d47465` in a clean short-path
worktree (`C:\tb8`), confirming they are baseline failures rather than changes
from this repair.

## Status

**Green:** Woodshop's Newton block compiles through LLVM and the final
concordance reports zero findings.

**Next:** the unrelated baseline minted-id ceiling and the other defects listed
in report C remain separate work; none is required for this concordance goal.
