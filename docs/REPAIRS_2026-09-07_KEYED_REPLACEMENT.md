# Resident replacement prerequisite for keyed record assignment

The existing conditional sequence replacement cleared the destination length
in its caller, then called `lower_sequence_extend`. That erased a source
sharing the same length cell and could partially overwrite the destination
before reporting exhausted capacity. It is not a suitable implementation of
owned keyed-field copy assignment.

`lower_sequence_replace` now generates ordinary memory SSA that loads the
source length first, checks it against zero, source capacity, and destination
capacity, copies compatible rows to the same indices, and publishes the
destination length after the copy. Failure returns status2 without modifying
destination contents or length; success returns status1. Empty replacement
publishes length0. Self-aliasing keeps the source intact. The existing caller
records the status through its normal sequence status cell.

This operation requires matching column dtypes, key policy, and live-flag
layout. It preserves keys/values directly instead of running insertion policy
again. Cross-column storage overlap and nested child ownership are rejected.
Read-only and dynamic-growth destinations use the existing shortfall path.
No allocation size or capacity is invented. The conditional replacement
lowering now uses this helper, with no destructive pre-clear. Other extend
operations retain their existing implementation. The changed helper file was
already in the compiler toolchain fingerprint.

## Verification

`tests/test_native_sequence_replace.py` and
`test_conditional_sequence_assignment_replaces_one_resident_arena`:
**4 passed in15.68s**, external timeout90s, each native subprocess timeout20s.
The native cases cover ordinary keyed copies, empty copies, self-aliasing,
repeated buffer reuse, destination overflow, negative source length, and
source length exceeding its capacity. Failure checks compare every destination
column and the length cell; successful copies also preserve source contents.
The structural case checks replacement stays on its conditional branch and
does not retain an opaque scalar carry or clear the caller's length first.

Broader sequence-table and conditional-tuple group: **42 passed,1 failed
in41.39s**. The failure is
`test_compiled_retained_loop_mutates_caller_sequence_record` at line811:
it expects `artifact.c_source_path` to be empty, but the compiler emits its
C ABI shim. The fixture uses append and does not call the replacement path.
The assertion was not weakened and no baseline checkout/stash was used.
This is a measured remaining failure, not a claimed clean broad gate.

Fresh source diagnostic `artifacts/compiler_evidence/keyed_replace_full_formals_20260907.log`
completed with exit1 at the strict gate: **19 formals, zero undefined operands,
zero unresolved calls, zero unmaterialized boundaries**, no optional merge or
non-native findings. The authoritative full_formal_diagnostic artifacts have
been regenerated. All four scalar publications remain present; hard_failure
still returns through1482 -> Cast9564 -> ledger593, and the earlier bool call
at loop_exit still consumes1378. This increment does not reduce the formal
count. All launched processes are terminal.

## Why error_channels is still open

The previous saved full module has record433/434's error_channels as physical
length1375, keys1376, values1377. Source GetAttr461 already carries the exact
receiver434, field name, dict kind, int64 key and float64 value annotations.
It instead owns a separate anonymous sequence461 with length671 and unknown
column dtypes. The branch's local dictionary materializer493 owns another
anonymous sequence. Phis576/577 still carry those mapping identities.

More than return publication is missing. In the floor branch `if_true.9`,
neither the `dict(metrics.error_channels or {})` copy nor the two indexed
stores494/495 into materializer493 are present. `_field_slot_ops.table_sequence`
recognizes lexically named dictionaries but refuses this producer when its
`binding_name` is absent, despite declaring its exact arena identity earlier.
The second store also needs the proven IndexedStore-to-base alias chain.
Simply accepting more identities is insufficient: unscheduled table stores
can currently be emitted by the builder at function entry. Recognition and
lexical effect scheduling must therefore be repaired together.

Next bounded source regression: a dictionary materializer without lexical
binding metadata, two chained keyed writes inside a conditional, and assignment
to an owned record field. Verify the constructor copy precedes both writes,
the assignment copy follows them on that branch, local aliases after assignment
refer to the intended field storage, and later helper calls read that storage.
Recover the physical field descriptor's capacity/extent through its actual ABI
binding; never substitute the independent local arena's capacity for it.

The scalar publication from the previous increment is retained. Keyed577 and
duplicate effect Phis496/497 are not deleted or claimed repaired. Optional
presence storage and the authored child-record ABI proof remain open. No
authored DT edit, full native validator build/parity, commit, or push.
