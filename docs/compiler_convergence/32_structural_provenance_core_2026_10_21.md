# Structural provenance core — work in progress

User authorization: implement recursive structural descriptors, boundary
transport receipts, and dependency-aware truth maintenance in the existing
provenance/concordance machinery FIRST, then get farther in BOTH the compiler
bootstrap and the full managed validator+tire compilation.

## Implemented surface (not a completion receipt)

### Deterministic ownership/publication repair after the 12:27 bootstrap receipt

Source changes only. No bootstrap, validator/tire compilation, tests, or Python
execution was performed in this repair turn. Static source inspection and
`git diff --check` are not behavioral validation. Existing edits are preserved.

The saved new bootstrap and older artifacts establish two new regression chains:
unit032's predicate9 entered result settlement as None instead of bool; unit043's
region formal7 entered as int instead of int64, was adapted to int64, and then
was revised back to int by scalar_dtype_settlement. Units002/045 both failed at
allocate_result_storage passing an accepted callee's foreign structural Ref to
the caller's mint. Unit028's absent recursive inherited descriptor14 predates
the latest source-only continuation. The paired tire evidence remains the
producer extent15 versus scalar leased SSA fact, plus its Boolean result feed.

Architectural changes:

- IdentityBook now owns digest-qualified cross-book proof edges. An import
  retains the exact source Ref, its fact and its owning accepted book; it does
  not invent a root, merge independently scoped coordinates, or relabel a
  foreign Ref as local. Source validation and the structural audit check the
  retained evidence. Projection carries the import receipt and owner. Concurrent
  publication uses the existing book lock. Scalar and optional result readers
  transport proofs through this same operation; accepted bodies stay immutable.
- The source-backed dict-truth producer publishes bool before changing its
  formal. Planned scalar captures resolve their provisional type from the exact
  source VALUE_TYPE_ABI before consumers run. Scalar inference seeds from the
  book, reads declared literal/load widths there, and publishes before changing
  occurrences. An unresolved type publication cannot erase a known resident
  type. These rules apply to both bootstrap and validator/tire lowering.
- Result-type decisions require concordance for equal-priority competing
  contracts instead of silently keeping whichever arrived first. A changed
  revision of the same recorded producer can refresh its retained decision.
  This does not claim a proof of global compiler convergence; existing bounded
  iteration guards remain, and real incompatible producers remain errors.
- Structural record fields retain explicit record links, including deferred
  and recursive targets. Missing target descriptors are tracked dependencies,
  not imaginary withdrawn by-value records. Physical flattening still refuses
  an unmaterialized link; no scalar substitute or empty physical ABI is supplied.
- Optional payload views use the same shape/storage scopes as ordinary values.
  Graph extents complete provisional empty shapes without overwriting an
  already published nonempty physical view. The existing result-storage lease
  writers consequently share this rule for the bootstrap and Metrics spans.
  New core pages now have explicit lifecycle classifications.

All these source changes remain unexecuted. Neither full compilation is proven
fixed or advanced, and the active objective must not be marked complete.

### Source-only continuation after the user's execution prohibition

No builds, tests, Python probes, or process cancellation were performed during
this source-edit continuation. Only source inspection, edits, and a whitespace
diff check were performed. The diff check passed; it is not syntax or behavioral
validation. The earlier 157-test result below predates these changes.

The repair family is correlated by the saved full bootstrap and full managed
tire artifacts: consumers used mutable SSA/metadata publications inconsistent
with the producer's concordance facts. Bootstrap unit027 lost return positions;
unit045 carried resident `int` while SSA_VALUE recorded `int64`. The tire leased
Metrics.error_present with a scalar SSA fact despite its producer's extent 15.

Canonical source changes in this continuation:

- Added AGGREGATE_RETURN_LAYOUT beside the existing structural declarations.
  Control return-arm joins publish the complete ordered member list before
  downstream optional lowering. Optional glyph and native union expansions
  publish their physical layouts, and optional forwarding publishes the missing
  OPTIONAL_VALUE_STRUCTURE relationship.
- Migrated the located aggregate-layout readers in the shell, optional lowering,
  GLSL planning and product acceptance off metadata. Aggregate source lookup
  follows the layout's actual provenance chain. Accepted readers use the retained
  owner book. Record-layout readers use the existing RECORD_RETURN_LAYOUT page;
  missing record-expansion writers now publish there as well.
- Output recovery preserves the semantic aggregate root when a named-output
  hint is its first physical Phi. emit_outputs no longer rewrites explicit Ret
  instructions. Acceptance no longer replaces multiple authored paths with a
  singleton merely because only one physical value survived, and compares arity
  as well as the set of output IDs.
- The structural SSA view reads SSA_VALUE plus the exact graph-copy shape state,
  tracking absent shape rows as dependencies. Declared scalar normalization is
  read from its existing concordance page rather than inferred from empty shape.
  Both early and late result-storage allocators consume the core view.
- Added publish_ssa_value_type to the existing SSA publication module. Result
  settlement reads recorded types and commits the retained decision before
  changing resident SSA. Linked scalar/aggregate projections, retained callsite
  results, exact-result formal propagation, canonical shape materialization and
  declared-scalar normalization now publish their changes before materializing
  them. Temporary CONCORDANCE-WRITE instrumentation was removed.

This is NOT evidence that every concordance bypass is repaired. Static follow-up
must still address the control builder's direct external_values readers and
computed fallbacks, physical_dtype/storage authority still held in accounting,
accepted-component-to-caller proof transport (foreign-book Refs must not be
treated as ambient-book sources), and the missing nested descriptor/recursive
record handling. The existing live-derivation lifecycle also needs to distinguish
retired boundaries from current ones. Do not turn these into invented roots,
scalar stand-ins, metadata fallbacks, or a claim that either full case passed.

Before the execution prohibition, two diagnostic full runs were launched and
canceled: whole-module bootstrap was an unjustified switch from the partitioned
bootstrap and exhausted available memory; the tire run was later stopped. Those
cancellations are not compiler failure receipts and prove no progress. No further
runs or cancellations are authorized. Cache removal was attempted previously but
denied by the execution policy; it did not succeed. No unique cache directory was
created, and no generated/cache file was edited to implement these repairs.

- `IdentityBook.derive_current/current_ref` record exact reads, including absent
  dependencies, and lazily recompute derived views. Historical cells remain
  unchanged. Settlement uses the existing bounded fixed-point guard.
- `StructuralValue` views reuse source aggregate layouts, SSA value facts,
  optional payload/presence identities, and record/sequence descriptor owners.
- `BoundaryTransport` requires complete path correspondence and agreeing leaf
  contracts. Exact region-feed adaptation publishes these derived receipts.
- The concordance audit reports stale live views, including accepted component
  books. Source return expansion reads the core view instead of copied leaf
  attributes. The reducer uses those views to recognize named optional tuples.

These changes are in canonical compiler source, not generated artifacts.

## Initial verification

154 tests passed across `test_structural_concordance.py`,
`test_ssa_call_input_adapters.py`, and `test_precompile_to_ssa.py`.
Core tests cover revision propagation, rejected physical disagreement,
historical receipt retention, pickle round-trip, and rejected partial mapping.
Broader optional/audit tests were not all green. In particular the existing
energy-helper emission test still refused emission and reported unposted
minted values. No expectation or audit gate was weakened.

## First paired full run

Commands (no reduced programs):

```
python -u -m tools.bootstrap_compiler_concordance --lower-only
python -u tools/build_balloon_tire_native.py --managed-dt
```

The bootstrap completed with **3 compiled-unverified, 1 partial, 0 verified,
0 failed; 4/337 admitted**. Unit 027, `IdentityPage._fork_of_scope`, is still
partial. This is NOT an advance from the original 21/331 baseline.

The source record now correctly contains one optional aggregate, width two,
with four absent arms and one two-member tail. Source layout 15 has members
30 and 31; output 32 owns return-member identities 33 and 34. But the finished
SSA exposes unknown value 33 as a public argument and returns it, with no
producer. Both 33 and 34 have `no_producer_at_use` control bindings.

The new audit also exposes two stale boundary transports: lookup aggregate
resident `2305843010213694131` was revised to `ssa.aggregate`, while the two
region consumers retain a `float64` formal 15. The previous audit of the old
artifact reported zero findings despite its rejected result layout. Current
audit: three findings (two stale transports, one unaccounted formal).

The full managed tire completed the 396-region vector-step control lowering
with zero shortfalls, linked its 369-function repository, and converged its
whole-program frame settlement after six rounds. It subsequently entered
failure-log finalization after physical-call adaptation. It failed at the same
immutable-input conflict: `step_with_dt_control_used` actual 2305843010216712756
is bool; planned region 24 formal 295 still has dtype float64, despite both
having `physical_dtype=bool` and the same `ssa_call_result_from` producer.
Neither process advanced.

## Outstanding core review

Live recipes must survive existing acceptance projection, and live operations
must participate in existing concurrent publication locking. Physical payload
views must remain distinct from the optional semantic wrapper. Boundary
receipts must retain the existing binding witness. The final detector must not
turn failed structural settlement into merely an informational detector note.
These integration points are now implemented: projection carries current live
recipes; concurrent-book operations use the existing lock; optional semantics
have a distinct view; transport carries exact CALL_EDGE witnesses; settlement
failures propagate outside the diagnostic detector's catch. Focused suite:
157 passed (fixed build/pytest directory, cacheprovider disabled).

The next paired iteration also reads the bootstrap tail from RETURN_SITE_SLOT
(tuple 15, members 30/31), not the downstream merge 32 (members 33/34).
Mapping helper projections consume the same structural layout; the existing
structural-consumed lane removes only duplicate Indexed operations whose
source, member and position exactly match that helper's published tuple.
The tire's early feed phase permits reconciliation only for returned values
whose origin and explicit physical type already agree at both endpoints.
Unsettled numerical feeds remain deferred. These are not yet full-run results.

No second iteration should be launched for only one process. Record and compare
both outcomes before further repairs; remove generated artifacts after their
evidence is captured. Keep the active goal open until BOTH processes advance.

## Second paired run: bootstrap receipt is not semantic success

Bootstrap ended 12/337 admitted: 11 compiled-unverified, 1 failed. Unit027
is now labelled compiled-unverified, but inspection of its saved SSA disproves
semantic completion: source output32 is an optional two-member tuple, while
the final Ret exports only int64 value33. Presence2305843010213700099 exists
but is not returned; member34 is absent. Aggregate-return layouts are empty.
Do not count this acceptance label as a correct repair.

The stopping exception is unit028 LiveScopeProjection.excludes: structural
publication follows its `inherited` record field to record14, for which no
descriptor exists under the table owner. Its parent descriptor has an
`inherited.__present` field. Study deferred/recursive record structure before
changing this failure; the paired tire's completed result is recorded below.

The broader existing identity spill/compression/log suite has 46 passes and
9 failures: one lifecycle-classification failure for new pages, eight raw
fixture-write refusals. The 157-test focused result does not cover these.

### Ordering study after the user's renewed instruction

No further compiler edits while the tire is running. Concrete source chain:

- Control finish creates member return Phis, but assigns the semantic output
  name to its first physical member. It does not publish the aggregate layout.
- `emit_outputs` reads semantic-output metadata and uses `(output_id,)` when
  that layout is absent. It mutates Ret to that reduced surface. It is called
  during linking, collision freshening, and final output collection.
- Optional lowering can introduce a presence result, but its aggregate-member
  branch expands only preexisting layout entries; an absent entry remains
  absent. The later output reader can collapse the export again.
- Accepted-unit result accounting permits a single published value to replace
  authored paths when the recorded container kind is not tuple/list. A named
  optional tuple therefore passes this scalar fallback despite lost members.
- Result-type settlement separately seeds `physical_storage` priority from
  either a physical-dtype marker OR a storage-kind marker, using the current
  logical dtype as its target. Equal-priority differing claims retain the
  incumbent. This proves bounded rank transitions, not preservation of the
  producer contract or independence from arrival order.

The core additions currently expose some stale views, but do not yet make
these consumer decisions depend on a complete, current producer structure.
Do not describe the three additions as fully integrated or the goal achieved.

## Second paired run: tire result and producer trace

The full managed tire finished with exit 1. It lowered 41 shells, the complete
396-region vector step, and the 369-function repository (793 final functions).
It failed in the early physical-input adapter's new boundary transport check,
before the old immutable-input check. This is not evidence that the previous
`hard_failure` failure is repaired.

The rejected edge is `step_with_dt_control_used__specialized_ac2009a74335`
to planned region11: actual2305843010216712744, formal274. Mutable operands
passed the adapter's dtype/shape equality check, but their SSA_VALUE records
are respectively float64/() and float64/(15,). Both use the owner's control
scope, as planned-region construction explicitly declares. The same namespace
is intentional; it is not evidence of an accidental scope collision.

The actual is the returned `Metrics.error_present` member, NOT a scalar field:

- The source declaration is AbstractTensor. The exact result-storage lease
  binds caller callsite262 / callee value183 to actual2305843010216712744.
- That call's callee is `balloon_tire_managed_advance__specialized_a0b2a73553f3`.
  Its shape transformation state for source183 records float64/(15,) in all
  three observed shape scopes, rooted in the source tensor descriptor.
- The caller's records264 and265 both declare this member as SPAN with that
  exact minted value ID. `coerce_metrics` forwards the Metrics record.
- The minted SSA_VALUE row nevertheless remains float64/(), with no revision.
  The caller's shape ledger separately records (15,) through projection274.

Therefore copying the scalar fact into the formal would be wrong. The new
`_ssa_structure` recipe currently treats a shape-empty SSA_VALUE fact as scalar
without consulting the registered record storage authority. Its claim to be
the complete physical producer view is not established. Publishing a transport
receipt derived from that view does not repair the upstream missing fact.
The mutable operand shapes at the point of refusal were not serialized; their
equality is known from the control path, but their exact current shape is not.

Saved full-run failure log used for this trace:
`balloon_tire_managed_python.20261021T100556.failed.log.xz`.
Bootstrap failure log:
`LiveScopeProjection.excludes.20261021T094957.failed.log.xz`.
These names identify generated evidence, not source files to repair.

### What is and is not established about order

Both full runs demonstrate incomplete physical publication: bootstrap's
complete return-arm structure is reduced by a consumer, while tire's leased
SSA fact omits an extent present in its actual producer and record contract.
This is a concrete shared preservation obligation, not proof that the two
terminal exceptions have one identical writer or that one local patch fixes
both. No further compiler edit was made during this ordering study.

The return reader `emit_outputs` is also a writer: absent aggregate metadata
falls back to a singleton and rewrites Ret. The result-type ledger ranks claims
but permits equal-rank incumbent retention; its termination rank does not prove
the surviving claim follows current source revisions. New live derivations
only track the rows their recipes read; they cannot repair omitted authorities
or mutable writes outside those rows. These are integration defects, not a
request for more user authorization or simplified cases.

The existing `_complete_propagated_frame_tails.calls_into` traverses every
function/block/instruction separately for each callee in every outer round.
For F functions, I instructions and R rounds, this call-discovery portion alone
performs O(R*F*I) instruction visits. That is a source-level bound on this
portion, not a measured whole-compiler complexity or proof of one dominant
bottleneck. The live-derivation stale detector also repeatedly scans all
registered recipes to reach dependency closure. A bounded iteration guard is
not a linear-time worklist or a proof of semantic convergence.

Required next repair remains core integration: retain semantic structure and
its exact physical transport through existing producer publications; make
consumers depend on those publications; keep acceptance independent of the
possibly reduced physical surface. Do not mark the goal complete, count
compiled-unverified as semantic proof, or silence mismatches by synchronizing
from whichever mutable object is encountered last.

### Cleanup status after this study

The generated `build/compiler-concordance`, fourteen exact second-run identity
logs, and Python bytecode directories were selected for cleanup after evidence
collection. The native PowerShell deletion command was rejected by execution
policy before it ran. They remain; cleanup is NOT complete. No alternative
deletion mechanism was used to bypass that refusal. The guestbook validator
completed with exit 0 and reported conforming filenames; its suggested archival
actions were not applied.
