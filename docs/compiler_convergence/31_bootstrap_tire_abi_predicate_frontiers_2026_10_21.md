# Bootstrap/tire ABI and predicate convergence — 2026-10-21

This continues report 30 in the canonical repository
`C:/dev/Powershell/turing`. The compiler was exercised as a two-oracle system:
every repair was checked against both the whole-compiler bootstrap and the
managed balloon-tire whole-program build. Neither program compiles completely
yet, but both now reach later, named refusals.

## Current outcome

The fresh bootstrap admits 23 of 330 units: 22 compiled-unverified, one
partial, zero failed, and 307 pending. Unit 035
`IdentityPage._virtual_fact` newly compiles; unit 024 remains complete. Unit 034 no
longer fails on a conditional arm changing from primitive integer storage to
`CompilerSSAExtent` record storage; it now reaches the older structural gap in
`IdentityBook._virtual_edges_into`:

```text
unresolved-required-source-values:
  value 43 Tuple, source value 42 via elts
unexplained-public-arguments:
  value 19 Ref row constructed by (origin, *ref.row[1:])
  value 13 scalar projection from _fork_of_scope
```

The managed tire now lowers `balloon_tire_vector_step`, all 396 of its planned
regions, and its 369-function repository module with `shortfalls=0`. It no
longer fails on the `isfinite` bool dtype, the collapsed `(4,41) -> (164,)`
storage view, or the six `face_scatter[i]` matmul operands. It then exposed two
strict-provenance gaps in returned-record and call-table linking. Both are
repaired. A clean rerun passes those stages and now stops at the next semantic
return-surface refusal:

```text
linked callee lost an authored output from its return surface:
  callee=balloon_tire_managed_advance
  lost(callee_id, caller_id)=((2305843010213709235, 263),)
  callee_ret_ids=(258,)
  callee_semantic_output_ids=(258, 2305843010213709235)
```

The missing semantic output is the authored `Metrics(...)` record returned
beside `finite` by `balloon_tire_managed_advance`. The physical `Ret` contains
only `finite`; late linking correctly refuses to classify the missing record
as an unobservable non-return handle. This is the next record-return repair,
not permission to weaken the refusal.

## General compiler repairs

### Primitive identities do not alias ProgramABI records

Two independent ABI writers canonicalized a semantic type through the
ProgramABI record table. Because `CompilerSSAExtent` declares semantic
identity `builtins.int`, ordinary integer values could become record-backed
extent values. `_program_abi_record_for_type_identity` is now the shared rule:
explicit record names and non-primitive defining class identities resolve to a
record; primitive semantic identities never do. Both conditional-union
canonicalization and physical-arm construction use that rule.

### Unary tensor metadata preserves semantic dtype

Elementwise unary results inherit the source shape, but predicates such as
`isfinite`, `isnan`, `isinf`, and `logical_not` retain their authored Boolean
dtype. The old unconditional dtype copy changed a valid bool result into
float64 and then attempted an unsupported revision of its shape-transformation
row.

### Storage aliases do not publish false identity self-transforms

A lowered storage alias can carry a distinct `ssa_storage_view`. When the
semantic reshape is already represented by its result identity, the storage
owner must not also publish a shape-changing self-edge. The settlement pass
now recognizes only shape-preserving same-identity transport.

### Terminal raises retain their compiler-authored constant predicate

An unconditional terminal `raise` is represented by a `ValidationBlock` with
an exact constant-false predicate. Graph-expression attachment previously
replaced that predicate with a value leaf named by the `Raise` AST node,
inventing a public Boolean input. Constant validation predicates are now
preserved; structured nonconstant validation predicates are still attached
from the graph so their resident query dependencies can be scheduled.

### Region consumers reread unanimous tensor descriptors

The six tire `matmul` regions captured `face_scatter[0:3]` before call-edge
settlement, so their local SSA occurrences remained shape `()` even after the
repository descriptor proved `(144,288)`. Operand settlement now rereads a
descriptor only when every tensor table for the same source integral agrees.
It fills only an absent shape and does not overwrite an explicit
`ssa_storage_view`. This moved the full tire through the complete
`balloon_tire_vector_step` repository link with zero shortfalls.

### Returned-record descriptors name their source cells

Both caller-side paths that create a record view of a callee result now call
`SSARecordTable.register` with the exact resident/callee descriptor cell and
`FRAME_BINDING` stage. They no longer introduce derived record rows through a
raw table assignment. The strict writer gate remains intact.

### Dead entry-record pruning names its structural evidence

`_prune_dead_entry_field_aliases` now posts its `fields_only` decision from
the aggregate formal and matching declared field formal cells before pruning
the handle. The focused native returned-record compile passes end to end.

### Incoming call-table pruning revises the incumbent row

Pruning unused callee storage formerly replaced one `SSACallRecord` through
list syntax, discarding the row provenance it had just read. It now uses the
table's sourced `replace`, derived from the exact incumbent record cell. A
clean tire run passes all 26 incoming-storage reconciliations and publishes 39
source-call records before reaching the independent missing `Metrics` return.

### Windows progress publication is serialized

Worker threads can publish the same `compile-progress.json` concurrently.
Their atomic replacements are now serialized in-process and both worker and
parent publishers tolerate a five-second transient Windows sharing interval.
This repaired two observed bootstrap orchestration failures without changing
compiler semantics or compilation receipts.

## Full-run receipts

Bootstrap command:

```powershell
python -u -m tools.bootstrap_compiler_concordance --lower-only `
  --output build/bootstrap_dual_oracle_after_call_record_sources_20261021
```

Result:

```text
admitted=23, peak-workers=2, cumulative=23/330, pending=307
verified=0, compiled-unverified=22, partial=1, failed=0
unit 034 IdentityBook._virtual_edges_into: partial
```

Receipt:
`build/bootstrap_dual_oracle_after_call_record_sources_20261021/compartment.json`

Managed-tire command:

```powershell
python -u tools/build_balloon_tire_native.py --managed-dt `
  --output build/balloon_tire_after_call_record_sources_20261021
```

Failed identity log:
`artifacts/identity_logs/balloon_tire_managed_python.20261021T013912.failed.log.xz`

This clean run failed only after all 41 local functions were lowered, the
396-region/369-function repository linked with zero shortfalls, and the
whole-program call table was built.

## Cross-reference

The useful commonality was not that bootstrap unit 034 and the tire had the
same opcode. Both exposed a consumer occurrence whose physical aggregate or
tensor contract existed elsewhere but was not read at the consumer boundary.
For the tire, the exact descriptor already existed and a conservative reread
was sufficient. For bootstrap unit 034, the exact structure does not yet
exist: `Ref.row` remains opaque and the variadic tuple reconstruction has no
lowering, while `_fork_of_scope`'s mapping value still publishes one column
for a two-member tuple. Report 22's tuple-mapping design remains the required
work; treating either value as a guessed scalar or opaque handle would only
hide the missing structure.

## Focused verification

The combined current gate passes 26 tests:

```text
15 passed  tests/test_value_union_abi.py
 2 passed  shaped isfinite + collapsed storage-view regressions
 3 passed  threaded/transient Windows progress publication
 2 passed  structured validation + terminal raise predicate retention
 1 passed  later unanimous tensor descriptor reread
 1 passed  native returned-record linking and dead-handle pruning
 2 passed  incoming-storage pruning regressions
```

`py_compile` passes for the changed compiler modules.

## Next work

1. Preserve the authored `Metrics` aggregate on
   `balloon_tire_managed_advance`'s physical return surface using the existing
   record-return tables and their declaration cells; do not classify it as a
   non-return binding or add a substitute snapshot convention.
2. Implement the existing general tuple-mapping value-position design for
   bootstrap unit 034: publish the value element schema, carry all value
   columns through keyed storage and lookup, and lower the variadic `Ref.row`
   reconstruction from structural leaves.
3. Keep both full commands as the acceptance gate. A repair is provisional
   until neither oracle regresses and a fresh receipt records the new frontier.
