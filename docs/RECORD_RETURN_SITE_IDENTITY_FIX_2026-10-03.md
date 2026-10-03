# Proposed record-return site identity repair

> Port note (2026-10-03): this is the proposal's own account against its
> baseline 3559a8f. The port onto 854e145c kept this head's existing
> return-site mechanism (lanes RS and WF) and re-expressed only the parts it
> lacked; the private AST key, the entry projection of every declared scalar
> field, and the span-free site receipts described below were NOT ported.
> What landed and why: `docs/concordance_census/CONTINUATION_return_site_port.md`.

Baseline: `3559a8fdcae91ea773dbb433a23f9bad9b9c117a`.
This is a local proposed patch, not a published or merged change.

## Reproduction and observed fault

`tools/compiler_probes/probe_record_return_merge.py` contains the real small
source/ProgramABI contract. A record is returned at three sites inside
`while True`: one sets `hard_failure=True`, one multiplies `value` by 0.25,
and one returns without changing either field.

Before any compiler edit, the new control and input-preservation regressions
both failed (2 failed in 2.37 s): only two return edges remained, and the incoming
`hard_failure` field was absent from the native signature. Fresh lowering and
C emission nevertheless reported complete with zero C shortfalls.

A fresh native run on the untouched baseline with Zig 0.16.0 confirmed:

- Input: `Metrics(False, 2.0)`, `rejected=False`
- CPython: `hard_failure=False`, `value=0.5`
- Baseline emitted C at O2: `hard_failure=True`, `value=0.5`

The original probe documents the historical spinning third path. The fresh
baseline run above deliberately used the terminating wrong-result case.
The proposed patch executes the formerly missing third path in a bounded
native subprocess, at both O0 and O2.

The probe-introducing commit `bb72c487c4d4222b76f85065a05b29eb4b9d09ca`
is not the defect's introducing commit. Its diagnosis predates this repair.

## Identity and source chain

Read-only hooks established the following chain:

1. The reducer already issued three distinct ingestion-value Ref cells for
   the authored return occurrences. Resolving each returned `Name` redirects
   it to the same record value 0 and removes the occurrence node.
2. `LoopComposer.describe` instead anchored all three returns at value 0.
   All three anchors were absent from `lexical_position`. Body assembly
   dropped them. Conditional-arm construction recreated two returns, but the
   unconditional third site became the loop latch edge.
3. The reducer's field cursor originally contained only fields previously
   read or written. The first returning arm introduced `hard_failure`, but
   its terminal-branch continuation did not initialize that field for the
   other paths. Those paths' receipts omitted the input flag.
4. `scalar_return_field_versions.lookup` matched predecessor sites by equal
   returned-slot tuples. Every predecessor therefore matched all three
   sites. It retained the descriptor fallback, which held the first arm's
   `True` value.
5. Preserving initial field projections exposed another exact consumer
   omission: `_fold_callsite_structural_values` removed GetAttr 15 and 16 as
   unused, although the return-site field-state pages consumed them. A
   read-only removal hook identified that writer directly.
6. Written-state graph attributes retained ingestion-scope Refs while
   return-site rows named canonical Refs. Writer and reader published/looked
   up different keys for the same field state.
7. Materializing record storage coalesced the initial `value` projection 16
   onto physical resident 6. The SSA field-version page must follow that
   exact alias, just as instruction operands do.
8. A field version starts at its authored assignment effect. Treating its
   own Store as an intervening write rejected a valid written version whose
   constant RHS had been evaluated earlier in entry.

The fix carries the captured authored-site Ref through copied ASTs, loop and
conditional control, emitted return edges, and serialized return receipts.
Returned object/value identities remain separate. Source coordinates only
order execution, including every tuple-return element producer.

Declared scalar parameter slots now use the existing GetAttr/OBSERVED field
state before branch snapshots. Span/table fields keep their existing storage
protocol. Canonical relabel refreshes the existing field-state attribute
views. Pruning recognizes return-field consumers. Storage coalescing advances
SSA_FIELD_VERSION using its exact alias receipt. Selection reads the one
site's field-state/version cells and retains dominance/effect checks.

Finite loops containing source return controls retain their control owner;
evaporating such a loop previously erased its exits. The finite-for variant
also failed C compilation on the untouched baseline and now executes natively.

Late publication uses the module's attached identity book, even after pickle
reload, and restores the caller's ambient book afterward.

## Final observations

For the core source, fresh SSA has:

- Three distinct authored return-site cells carrying the same record 0
- A public incoming Boolean `hard_failure` field 15
- `hard_failure` Phi: `[written True, input15, input15]`
- `value` Phi: `[input6, written13, input6]`
- The third return's predecessor branches to `function_exit`
- All six live field-selection rows resolve to sourced SSA_FIELD_VERSION cells

Concordance before: 21 rows, 3 functions, 0 gated findings; 189 unsourced facts,
0 unsourced identities, 23 unsourced groups.
Concordance after: 23 rows, 3 functions, 0 gated findings; 203 unsourced facts,
0 unsourced identities, 23 unsourced groups. The existing migration worklist
remains open; zero gated findings does not mean every existing fact is sourced.
The new return-site audit detects a missing site key or a wrong slot receipt.

The existing repository audit cases also completed with exit 0 on the final
source: mapping has 24 rows / 2 functions / 0 gated findings, with 197
unsourced facts and 0 unsourced identities; oscillator has 331 rows / 22
functions / 0 gated findings, with 2,951 unsourced facts and 0 unsourced
identities. Both retain an OPEN latch and match the baseline observations.

```sh
python -u tools/audit_identity_concordance.py mapping oscillator
```

## Verification

Focused final batch: 60 passed in 14.23 s:

```sh
python -m pytest tests/test_record_return_site_identity.py tests/test_ssa_record_return_state.py tests/test_control_source.py tests/test_pruned_loop_return.py tests/test_native_record_return_state.py -q --tb=short
```

The new file contributes 15 tests. Its native matrix is 72 exact CPython/C
comparisons: 8 inputs at O0 and O2 for the core source, plus 8 inputs at O0 for each
of nested-while, finite-for, zero-trip/post-loop return, nested else, aliased
receiver, constant-pruned branch, and field-reassignment variants. Native
execution uses child processes with 20-second timeouts. Compilation was bounded
by the external 90-second test command. No tolerance or interpreter-lane
substitute is used for the expected results.

Independent review also executed tuple-return sequencing for
`return m, m.value * 2, m.value * 3` across 8 input combinations. A separate
fresh-process replay test corrupted a written flag Phi input, loaded the
pickled module under an unrelated ambient book, verified that publication
repaired the exact original input, and verified the next publication made
zero changes and restored the ambient book.

Additional adjacent measurements:

- Three existing reducer field-state tests pass; the aggregate seed test
  passes separately.
- The focused loop-composer batch reports 6 passed, 1 failed on both baseline
  and proposal. `test_synthesized_record_field_seed_keeps_loop_ownership`
  encounters the pre-existing shared `function_address('kernel')` collision
  in that combined batch, but passes alone in 1.20 s.
- `test_native_loop_return_dominates_and_does_not_repeat_mutation` stops at
  the same pre-existing `run_all` gate on baseline and proposal: 10 findings,
  beginning with id-scale findings that flag issued tagged IDs as
  memory-address-scale. Its assertion was
  not relaxed; that test's native section was not reached.
- The four-test reducer batch similarly has an existing shared
  `function_address('update')` collision (3 passed, 1 failed on both trees).

Cloud environment: Python 3.12.14, NumPy 2.3.5, networkx 3.7, pytest 9.1.1,
Zig 0.16.0. The nodus native tensor arena is unavailable; the compiler and C
artifact path above were still exercised. Cloud-only environment settings:

```sh
CC=gcc CXX=g++ LDSHARED='gcc -shared' CFFI_TMPDIR=/tmp/turing-audit-cffi ZIG_GLOBAL_CACHE_DIR=/tmp/turing-zig-global-cache ZIG_LOCAL_CACHE_DIR=/tmp/turing-zig-local-cache PYTHONDONTWRITEBYTECODE=1 timeout 90 /tmp/turing-fix-venv/bin/python -m pytest tests/test_record_return_site_identity.py tests/test_ssa_record_return_state.py tests/test_control_source.py tests/test_pruned_loop_return.py tests/test_native_record_return_state.py -q --tb=short
```

## Limits

No full suite, full controller lowering, Windows run, LLVM native execution,
or Fortran native execution was performed. The measurements prove the listed
source/record-return cases, not arbitrary record aliasing or whole-controller
parity. Rebuild fresh source graphs; no compatibility claim is made for old
serialized planning artifacts that lack captured authored return-site keys.
