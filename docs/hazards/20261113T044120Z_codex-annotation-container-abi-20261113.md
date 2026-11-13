2026-11-13 list(set) snapshot must own storage (MEASURED, isolated compiler worktree)

Real compile_nodes contains `out_keys = list(node_keys)` for a set comprehension. A minimal dataclass key/set/list/filter reproduction reached `native heap context requires a pointer-cell input`. Traced every dtype writer: the canonical sequence pass correctly set data108 to ptr, then helper correlation changed it to int64 because list constructor99 had been aliased to set108 while carrying its own fixed/no-owner descriptor and counters. This is not a pointer-cast problem.

Producer fault: _source_sequence_snapshots accepted list/tuple/array but not set sources. _sequence_concat_ops therefore treated the unrecorded list conversion as an immutable view. Admit list(set) as a snapshot, and preserve the source's actual resolved unique schema instead of querying it with destination duplicate policy. The set snapshot resets its own list length on every execution and invokes the existing extend/source-iteration/destination-insert implementation. Bulk replace's matching-policy guard and heap pointer check are unchanged. No new allocator or copier, no guessed dtype, no shared-source changes.

Verification so far:
- New test_set_sequence_snapshot.py: 3 passed, including no shared keyed/list columns and zero concordance findings, real LLVM differential across four input pairs with duplicates/empty input, destination mutation, and repeated constructor inside a loop. Same 3 tests fail clean base554ac55e with the pointer-cell error.
- Existing constructor-snapshot, extend-scoped-helper and native-replace suites: 11 passed.
- Mapping identity audit: 28 rows,2 functions,0 findings,0 unsourced,latch CLOSED.
- Broader test_ir_sequence_tables.py: 41 passed,15 failed on both repaired and clean baseline, same failing tests. Existing failures include strict native physical-layout proofs, changed constructor schema expectations and old ad-hoc graph fixtures without canonical identities. No new failures.

Limits: the actual class-equipped compile_nodes, using the exact lazy-import body in place of its module-return helper, now reaches final concordance with 50 findings. It has not produced LLVM. Findings include record columns lacking bound/formal definitions, span-input operation mismatch, undefined condition87, and exact default-expression provenance. The minimal string-record reproducer similarly reaches a separate span-input finding. None is bypassed. No deployment or merge.
