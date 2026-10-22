2026-10-22 SCOPE-LOCAL IDENTITIES, RETURNED LISTS, HEAP LEASES (MEASURED):
Native corpus (`tests/native_semantics_corpus`, C lane, run against the
authored Python): 63/63 pass with 0 concordance findings at baa4d9d. New:
- c60: a list returned on two edges;
- c61: `any()` over a range generator;
- c62: an f-string inside a callee;
- c63/c64: `dict.clear()`.
At HEAD 9f1643c, c61, c62 and c63 were refused, and c60 did not compile.
Bootstrap snap7 (9f1643c tree): 369 units; 243 failed, 71 partial,
55 compiled-unverified; 7 up and 0 down versus snap6. Replays on the later
tree: unit 8 (accepted call result row), unit 24 (tag formal kept), unit 35
(accepted body not re-lowered) and unit 43 (yield after its operands) all
compile with 0 findings. Unit 34 stops at the variadic return lane.
Gate (49 files, four shards, --timeout 900) at a8ad2b0: 125 failed versus
128 at g11; the 2 new failures were fixed at 9f1643c. Focused suites:
20 failed versus 22.
New audit finding `accounting-foreign-value-id`: a scope-local accounting id
(type tag, heap owner, optional payload, iterable target, tensor metadata)
must name a value of the same function.
Open, observed and recorded in doc 33 (not worked around):
- the variadic return lane;
- `'|'.join` in f-strings;
- reductions over unrolled structural aggregates (silent; the audit flags
  them);
- `len()` of a computed tuple display (silent, clean audit);
- `dict.update()`;
- calls on Any-typed receivers (the largest partial family);
- test_forwarded_tuple_field_has_one_sequence_resident.
See docs/compiler_convergence/33_identity_repairs_and_bootstrap_chain_2026_10_22.md.

2026-10-22 SHORT-CIRCUIT, CARRIED-INDEX AND LITERAL-LIST SEMANTICS (MEASURED):
Native corpus (`tests/native_semantics_corpus`, C lane, run against the
authored Python): 48/48 programs pass with 0 concordance findings. c35-c49 are
new; each failed or trapped on base 913e4c2:
- c35/c38: a loop-carried index read element 0.
- c36/c39: a literal list read 0.
- c37: a while-test row load ran after the loop.
- c41-c47: `and`/`or` guards ran their guarded operand eagerly
  (SIGFPE on `d != 0 and 10 // d`).
- c48/c49: value-context BoolOps.
Bootstrap snap5 (HEAD 7e23934 tree): 367 units; 244 failed, 73 partial,
50 compiled-unverified; 28 units up and 0 down versus snap2. Unit 33 now
compiles on a live replay against snap5's growth. That replay uses snap5's
frozen graph, so a fresh bootstrap is still required.
Focused suites (record/sequence/optional, 13 files): 22 failed versus 24 on
base.
Gate (49 files, run as four shards with --timeout 900): 126 failed versus
128 in the previous full run. Three representation tests were updated: they
asserted the eager `LAnd`/`LOr`, and now read the operator's merge Phi, which
carries its `semantic_family`.
test_sympy_process_graph_is_registered_before_python_dependency_pursuit is
NONDETERMINISTIC on clean HEAD f4a48b3 (passed 1 of 3 runs); it is not a
regression, and it is an open determinism fault. Six earlier failures were
regressions from d10dce2. Fixed since:
- a scalar's rank 0 was not mirrored by the accounting view;
- the test stand-in posted no physical fields;
- an equal-rank refinement raised.
Still failing from that set: test_forwarded_tuple_field_has_one_sequence_
resident. Ensure's `pieces` arena row is `opaque_ref` when total's lease is
minted; the shell's reconcile sweep retypes only its occurrences to float64
later.
Open: f-strings stop at dynamic sequence growth in the C lane (task in doc 33).
See docs/compiler_convergence/33_identity_repairs_and_bootstrap_chain_2026_10_22.md.

2026-10-21 DETERMINISTIC OWNERSHIP/PUBLICATION REPAIR (UNVERIFIED):
The 12:27 bootstrap receipts were diagnosed without compiling. Source repairs
now transport accepted dependency proofs through digest-qualified IdentityBook
edges, publish dictionary-truth and declared scalar ABI refinements at their
producers, seed scalar settlement from concordance, and prohibit unresolved
publications from erasing known types. Result typing no longer silently resolves
incompatible equal-priority contracts by arrival order. Recursive records retain
typed links; physical consumers still require materialized layouts. Optional
and ordinary values share shape/storage-scoped reads. Both bootstrap and full
validator/tire paths use these changes. No compiles or tests were run; this is
NOT a pass or progress receipt. See the opening repair section in
docs/compiler_convergence/32_structural_provenance_core_2026_10_21.md.

2026-10-21 SOURCE-ONLY CONCORDANCE INTEGRATION (UNVERIFIED):
The user prohibited further compilation/test runs and further process killing.
Continue source work; do not use the older commands below as permission to run.
Canonical compiler edits now publish aggregate return expansions on the book,
read record/aggregate layouts through concordance, retain return-arm aggregate
roots, publish forwarded optional pairs, and make emit_outputs a non-mutating
reader of explicit returns. Result leases read copy-scoped shape facts; call
result typing, linked projections, marker retention, final shape materialization,
and declared-scalar normalization publish SSA_VALUE changes before applying them.
These edits have NOT been executed. Earlier test counts do not validate them.
Both full compilation outcomes remain unproved; the goal is NOT complete.
The source-only continuation and remaining integration risks are recorded in
docs/compiler_convergence/32_structural_provenance_core_2026_10_21.md.

2026-10-21 STRUCTURAL PROVENANCE CORE (EARLIER RECEIPT, NOT GREEN):
See docs/compiler_convergence/32_structural_provenance_core_2026_10_21.md.
The user explicitly authorized three provenance-core additions before further
paired compilation progress. The focused suite passes 157 tests; broader
identity spill/compression/log tests have 46 passes and 9 failures. The second
full bootstrap reports 12/337 admitted (11 compiled-unverified, 1 failed), but
unit027 is falsely accepted: its optional two-member tuple exports only one
member, dropping the second member and presence. Unit028 fails at a missing
nested record descriptor in the new structural publication. The paired full
managed tire fails earlier in the new transport check: Metrics.error_present
is a declared span with producer shape (15,), but its leased SSA_VALUE still
records (). This does NOT demonstrate that the earlier hard_failure conflict
is repaired. The new core views remain incompletely integrated with producer
authority; no additional compiler edits were made during the ordering study.
Neither process has demonstrated correct progress. The goal is NOT complete.

2026-10-21 BOOTSTRAP/TIRE ABI AND PREDICATE CONVERGENCE (EARLIER RECEIPT):
See docs/compiler_convergence/31_bootstrap_tire_abi_predicate_frontiers_2026_10_21.md.
Fresh dual-oracle runs now put the bootstrap at 23/330 admitted (22
compiled-unverified, one partial, zero failed); unit 035
`IdentityPage._virtual_fact` newly compiles. The managed tire lowers its
396-region `balloon_tire_vector_step` repository with zero shortfalls. Repairs
cover primitive-vs-record ABI identity, terminal-raise predicates, unary
predicate dtype, collapsed storage views, late tensor-descriptor rereading,
returned-record provenance, and serialized Windows progress publication. The
bootstrap now stops on unit 034's real starred `Ref.row`/tuple-mapping gap. The
returned-record descriptor writer, dead entry-record pruning, and call-table
revision now carry exact source cells. A clean full tire rerun passes all
three, lowers all 41 functions, and links the 396-region/369-function tire
repository with zero shortfalls. It now stops later because
`balloon_tire_managed_advance`'s authored `Metrics` aggregate is absent from
its physical `Ret` while still present on its semantic output surface. Neither
target compiles completely.

2026-10-20 BOOTSTRAP/TIRE DUAL-ORACLE REPAIR (SUPERSEDED):
See docs/compiler_convergence/30_bootstrap_tire_dual_oracle_2026_10_20.md.
The bootstrap topology regression from an exact `Any` local is repaired, and
fresh full receipts advance from 19 to 20 admitted units after two additional
public-identity/aggregate-provenance repairs. The tire advances past duplicate
`dt_min` parameter identity and list 243's copied-AST producer refusal. It now
stops on a bool-to-float64 shape-transformation revision with no changed source;
bootstrap unit 034 stops on a `CompilerSSAExtent` conditional arm changing its
physical ABI. Neither target compiles completely. Continue by tracing the two
physical-ABI writers; do not weaken the concordance refusals.

2026-10-20 SSA RECORD ABI AND AGGREGATE QUERY FRONTIER:
See docs/compiler_convergence/29_ssa_record_abi_and_aggregate_query_frontier_2026_10_20.md.
The isolated real `_result_layout` bootstrap unit now has complete source
regions, complete root parameter accounting, resident `any`, and canonical
conditional aggregate member Phis. Program-ABI nested table fields are
admitted as child handles. The remaining exact mechanisms are the missing
`Instr.args -> SSAValue` child-pool correlation and full-row publication for
`next(iter(layouts))`. The compiler now refuses the latter as
`first-aggregate-row` instead of silently returning only column zero. The
complete precompile-to-SSA file passes 134 tests. This is not a clean full
bootstrap receipt; continue with the isolated unit before another census.

2026-10-19 PROVENANCE CHECKPOINT AND CONTINUATION (CURRENT):
See docs/compiler_convergence/20_provenance_bootstrap_continuation_2026_10_19.md.
The user's fingerprint
`9a2048331e799d30ff626164904bb2b675184fa738ecd1f0dd21994e41fd0f71`
admitted 28/327 units: 25 compiled-unverified, one partial, two failed, zero
verified. Unit 049 (MonotonicIdSource.mint) now has complete accounting and
no shortfalls; unit 033 (IdentityPage._resolved) also remains complete.
Unit 034 still has an unresolved lookup, unexplained inputs, and no physical
return publication. Units 041 and 065 now share the physical call adaptation
repeated-state refusal. The exact writer causing that refusal is untraced.

Independent review repaired function-scoped child-pool dtype ownership,
scoped ABI authority, stale sequence-view receipts, region-feed resident
identity, topology cache invalidation, CFG entry selection, and Phi checks.
The pre-commit six-file gate completed with 211 passing and nine failing tests.
All failures are in test_concordance_audit_contracts.py: five raw-write fixture
refusals and four missing optional Metrics.dt_limit declaration-cell refusals.
All nine fail identically at the pre-checkpoint baseline 7c560b04: the five
raw-write fixtures were introduced by c0127eeb (2026-10-17) and the four
class_field_declaration refusals by de5c10e4 (2026-10-17). They are fixture
faults; the five raw-write fixtures are repaired through IdentityBook.post, and
the four declaration refusals were the first link of a chain of compiler faults
the raw-string fixtures had masked (docs/compiler_convergence/21_provenance_sweep_continuation_2026_10_19.md):
six links are repaired and the four tests now stop at later, named gates listed there. This checkpoint does not claim a
clean concordance gate or native self-host completion. No bootstrap was run
while preparing the report and commit.
The separate receiver/cache/receipt and bootstrap-policy batch passed 28 tests
with two skips in 133.73s. Combined packaging checks: 239 passed, two skipped,
nine failed. See the continuation report for exact commands.

2026-10-18 AUTHORITATIVE SCOPED TYPE-ABI HANDOFF (CURRENT):
Checkpoint `3359a18b4f0303d162aba9ba3ab7223020e6051b452fd3fdcda03dfda0a4a33b`
proved that the preceding provisional-tag implementation was loaded but did
not receive the concordance row it needed.  Unit 033's scheduled callsite 31
published result value 31, and `VALUE_TYPE_ABI` already held its exact
`NoneType | tuple` contract.  The shell nevertheless constructed the control
lowerer's type table by iterating the separate `linked_value_abi` dictionary;
value 31 was absent from that physical shell view, so the authoritative page
fact was silently omitted and no provisional tag could be reserved.

The sectional handoff now projects every `VALUE_TYPE_ABI` row owned by the
function copy's lexical scope directly from the concordance page.  The control
builder performs the same projection at its API boundary and rejects any
explicitly supplied row that disagrees with the page.  This is independent of
function, callsite, value number, union members and control shape; no Python
object, runtime type query, case list or fallback is involved.  The reported
conditional-call regression now passes with no manually supplied type table,
and the complete control-SSA/value-union focused gate passes 137 tests.  No
bootstrap run was made; the user's next normal run is authoritative.  The
older `LINKED NATIVE UNION CALL RESULT` receipt below records the provisional
tag mechanism but its claim that the real handoff supplied value 31 is
superseded by this receipt.

2026-10-18 LINKED NATIVE UNION CALL RESULT (SUPERSEDED HANDOFF CLAIM):
Checkpoint `1cd834105ead7644cbffbd892821ab162900f6f256ed56a329c63d27c33c40e6`
confirmed that the loop-carried union blocker was cleared and advanced unit
033 (`IdentityPage._resolved`) to its conditional Optional call result.  The
scheduled `_fork_of_scope` call's semantic result (value 31) already had the
authoritative `None | tuple` `VALUE_TYPE_ABI`, but control SSA was built before
call-frame linking exposed the result's physical `int64` tag.  The conditional
therefore refused value 31 rather than inventing type evidence.

Scheduled union results now reserve a typed internal tag from that exact
concordance row while the call marker is placed.  Native result propagation
then registers the caller's union descriptor from the callee descriptor and
physical result cells, rewrites consumers to the linked tag by SSA-object
identity, updates the tag ledger, and removes the provisional frame value from
the completed caller arguments.  No Python object, runtime class query,
guessed tag, or public bootstrap input is introduced.  Regressions cover the
reported conditional `None | tuple` shape and the complete provisional-to-
physical link transaction.  The complete control-SSA and value-union focused
gate passes 137 tests.  No bootstrap run was made; the user's next normal run
will establish the new frontier under a new source fingerprint.

2026-10-18 LOOP-CARRIED NATIVE UNION BACKEDGE (CURRENT):
Checkpoint `05151695257c8f13addd6da57b0e3c147011171d01285bc24f2c1ebc62056710`
advanced unit 033 (`IdentityPage._resolved`) to the loop-carried Optional
`asof` value, then refused because its body-defined update (value 71) had an
authoritative `None | int` ABI but no tag definition yet.  This was an SSA
construction-order error: the loop already reserved the future payload, while
the companion tag incorrectly had to exist before the loop body was lowered.

Loop-carried unions now reserve their exact local tag and physical arm payloads
with the backedge payload.  The body definition claims those same SSA objects;
they never become Python objects or invented public formals.  The loop header's
derived ABI is published under its minted SSA identity rather than replacing
the input parameter's descriptor, deferred backedge shape-injection evidence is
posted after the body definition becomes resident, and exact/remapped incoming
tags are emitted on their Phi predecessor edges.  A focused Optional loop
regression proves the header backedge is the body tag, is not a function
argument, and passes SSA dominance.  The complete control-SSA and value-union
focused gate passes 135 tests.  No bootstrap run was made; a normal next run
must use a new source fingerprint rather than replaying the recorded failure.

2026-10-18 COMPLETE PROVENANCE CENSUS + KEYED-REFERENCE REPAIRS (CURRENT):
See docs/compiler_convergence/19_census_2026_10_16.md, Round 9.  The fresh
`repair-20261018-0733` lower-only census admitted all 327 units: 41 compiled-
unverified, 66 partial and 220 failed.  Dependency classification leaves 104
independent non-clean roots (38 failed, 66 partial).  The most frequent exact
independent failure is the seven-unit formatted-glyph resident-producer family;
the frozen receipts prove that all seven operands lack an authoritative source
class or value-type ABI, so a generic Python `repr` fallback would be unsound.

Units 024, 038, 040 and 045 now compile in that census.  The next concrete
failures showed two real contract bypasses.  Keyed sequence membership and
delete/setdefault queries now pass through the existing descriptor-aware
native reference-to-key conversion.  The REVISE publication used by table
queries now cites the incumbent SSA cell through `revise_ssa_value`, preserving
the exact concordance history.  Comprehension aggregate results also publish
their authored builtin aggregate class from graph evidence.  These changes
have focused regression coverage; a fresh fingerprint-consistent compiler run
is still required for end-to-end status.

Provenance growth is fail-fast by default.  `--jobs` controls concurrency and
does not create a clean checkpoint after 12 units.  `--stage-units N` is the
explicit admission bound; `--continue-after-failure` explicitly requests a
complete remaining census.

2026-10-17 GENERAL SECTION TYPE-EXPECTATION ABI (CURRENT):
See docs/compiler_convergence/19_census_2026_10_16.md, Round 8.  Fresh
bootstrap checkpoint
`b21dabcf27cbe6c1f848f41ce2b4133f549ecb50b0a8a87cde83a08997f8b587`
reached 20/327 admitted units.  Unit 030 (`IdentityPage._reaches_origin`) and
unit 040 (`RowField.admits`) then exposed the census-selected missing native
type-expectation family on an aggregate projection and an `Any` parameter.

Every exact native type-predicate subject now has a sectional class ABI.
Already-settled source classes remain exact.  Otherwise the accepted source
concordance supplies the closed class vocabulary and one definition cell per
identity; the physical boundary remains one ctypes-compatible `opaque_ref`
word plus one `int64` class tag.  This covers parameters, loop inputs, field
reads and aggregate projections without Python objects, live class inspection,
dtype inference or a synthetic catch-all tag.  Authored `object`, like `Any`,
is an open accepted domain rather than an exact-runtime-class assertion.

Representative frozen replays clear the former refusal for direct `Any`,
mixed parameter/loop, known-class, ordinary loop, and large multi-predicate
audit functions.  The two reported units advance past SSA type lowering to a
later linked-artifact accounting check in the diagnostic replay; only a fresh
fingerprint-consistent bootstrap can establish their complete status.  The
focused gate passes 18 tests with 2 skips.

2026-10-17 PROVENANCE GROWTH HAS NO HIDDEN CLEAN CHECKPOINT (CURRENT):
See docs/compiler_convergence/19_census_2026_10_16.md, Round 7.  `--jobs`
limits concurrent isolated workers only.  With no explicit `--stage-units`, a
clean run continues through the remaining dependency schedule instead of
returning after an arbitrary worker-count batch.  The default remains
fail-fast at the first failed/partial receipt; `--continue-after-failure`
explicitly requests complete failure collection.
The native-provenance bootstrap file passes 11 tests with 2 skips.

2026-10-17 UNCONDITIONAL TERMINAL VALIDATION (CURRENT):
See docs/compiler_convergence/19_census_2026_10_16.md, Round 6.  Fresh
bootstrap checkpoint
`cfabb41b09a7fd246f14caf2c04888f0af59fe3df9fe181354d8e851445e34fc`
advanced through the declared-type frontier and compiled 16 units before unit
024 (`IdentityBook._admit_raw_write`) exposed a resident-glyph refusal.  The
unknown glyph was `row!r` inside the message of the function's unconditional
terminal raise.  The exact authored function body and raise node were already
retained, but validation discovery published diagnostic ownership only for
conditional raise guards.

A straight-line function ending in `raise` now uses the existing native
`ValidationBlock` contract with a compiler constant-false predicate owned by
the exact raise node.  Native control reports the typed error; the authored
message remains shell-only and therefore does not invent a runtime `repr`.
An exact frozen-unit replay completes with one validation call, zero repository
SSA self-check findings and zero concordance findings.  The complete glyph-
provenance file passes 58/58.  The full bootstrap still requires a fresh run.

2026-10-17 SHARED NATIVE TYPE-EXPECTATION ABI + CENSUS SELECTION (CURRENT):
See docs/compiler_convergence/19_census_2026_10_16.md, Round 5.  A closed
semantic class domain whose alternatives share one physical representation now
uses one `opaque_ref` payload plus one `int64` class token.  It no longer grows
one reference formal per class.  The payload remains the authored ctypes-
compatible native reference value; the source-class concordance supplies the
closed class vocabulary and subtype identities.  No Python object, class object,
PyObject or CPython ABI participates in the sectional contract.

The actual frozen unit 038 input from checkpoint
`67f802071546cf2702dd317446e4580d3b27bc1aa683756a2c8bfae3cefe3749`
now lowers `_validate_fact` with 89 semantic alternatives, one shared payload
row and one tag.  Repository SSA self-check reports zero findings and the
identity concordance reports zero findings.  The focused shared-union/API gate
passes 3/3.  This is an exact unit replay, not a claim that the full bootstrap
or native verification completed.

The requested continue-after-failure census used all 323 receipts in
`build/census-2026-10-17`: 37 compiled-unverified, 237 failed and 49 partial.
Of 286 non-clean units, 113 are independent and 173 inherit another failed
dependency.  The largest independent exact semantic family is now 19 missing
native type expectations, with 102 failed units in its reverse dependency
closure; the glyph family is next at 16.  The shared declared-type repair is
therefore the selected high-leverage lane rather than a one-function exception.

2026-10-17 NATIVE DECLARED-TYPE EXPECTATIONS + CLEAN GROWTH STAGE (CURRENT):
See docs/compiler_convergence/19_census_2026_10_16.md, Round 4.  Sectional
ProgramABI inputs now admit one declared semantic type or a closed union.  The
declaration is joined at compile time to source-class concordance and lowered
as native payload plus an int64 type token; no Python class object, PyObject or
CPython ABI crosses the bootstrap boundary.  `Page.fact_type` may itself be a
class union, and dynamic `isinstance(fact, page.fact_type)` consumes the same
native token/subtype join.  A real `_validate_fact` public-entry lowering moved
past its former structural output failure.

The continue-after-failure census was reclassified before choosing the next
lane: 170 of 234 failures were incomplete-dependency cascades; the leading
independent exact failure was 12 formatted-glyph resident-producer refusals.
The current formatter/concordance state passes all 57 glyph provenance tests.
The fresh bounded growth stage at checkpoint
`63be20e60ea4e6ad709a3468cd141653f6ba5aab186e66f89043fafd537e60bd`
compiled 10/10 selected units with zero partials and zero failures.  Its four
glyph formatter units compiled, and unit 045 (`identity_spill._replace_cells`)
advanced through the former glyph refusal.  Two subsequently exposed missing-
source errors were repaired by deriving pre-record member discovery from the
existing planner call edge and an unbound composed-callee frame lease from the
existing `linked_caller_member` decision.  Formal-storage tests pass 7/7;
declared-type/union focused tests pass 6/6.  This is a clean lower-only growth
stage, not a claim that all 327 units or native verification are complete.

2026-10-17 VALIDATION DIAGNOSTIC CONCORDANCE (CURRENT):
See docs/compiler_convergence/19_census_2026_10_16.md, Round 3.  The fresh
327-unit provenance run's two SSA failures (`_validate_fact` and
`identity_spill._replace_cells`) were exception-message glyphs with no native
consumer: their terminal raises were already compiled as `ValidationBlock`s.
Validation planning now records the exact diagnostic source-value ids and
`post_control_program` publishes their cells on the existing `control_block`
concordance fact.  Resident glyph planning reads that fact and omits only the
diagnostic `JoinedStr` owned by compiled validation; every other unknown
formatted value still refuses.  `Any`/`object` retain the established
reference/`opaque_ref` ABI; no generic repr or new handle was invented.  Glyph
provenance passes 57 tests and control-source passes 24.  A fresh provenance
run at fingerprint `70cae82e87bbb0d0a458a82ccbb04b04424e338aa9be1872db0802f2f6670528`
confirmed `_validate_fact` now completes SSA lowering and reaches finalization.
It also proved `_replace_cells` takes the retained-loop validation lane, whose
`ValidationBlock` constructor did not yet carry the diagnostic ids.  Diagnostic
extraction is now shared by ordinary and retained-loop planners; a membership-
guard regression proves the loop block -> `control_block` concordance -> glyph
reader chain.  The combined current gate passes 82 tests.  The next fresh
provenance run is the authoritative verification of unit 045 because frozen
unit plans correctly reject the latest compiler fingerprint change.
Unit 038's remaining final-accounting shortfall is separately identified as the
Boolean result of `isinstance(fact, page.fact_type)`.  `Unresolved` is already an
explicit admitted concordance fact.  The dynamic class operand is only an
`opaque_ref`; full-native lowering has no concorded class/type identity for it,
and must not reinterpret opaque storage as subtype evidence.  The existing
Python-shell-shaped regression for this case is still red, while the compiler's
full-native execution contract rejects a CPython ABI.  This is a distinct open
type-identity contract, not part of the diagnostic-glyph repair.

2026-10-16 MEMORY GRAPH TAKEOVER + PROVENANCE BLOCKERS (CURRENT):
See docs/compiler_convergence/17_memory_graph_bootstrap_repairs_continuation.md.
The public source compiler now owns canonical topology through the independent
BitTensorMemoryGraph NetworkX-shaped protocol; relabel, extraction, scoped
catalogues and checkpoint rehydration preserve MG/G pairing without embedding
an nx.DiGraph. Explicitly linked NetworkX remains a valid target, not a hidden
self-hosting store. Consecutive materialized runs have confirmed the unit
026/031 and unit 002/028/043 repairs. The latest run had no partials and advanced
unit 045 through normalization to SSA. Its two SSA blockers are repaired: a
rehydrated glyph-vocabulary reference is now validated by its exact active-book
key rather than Python Page object identity, and `_replace_cells` retains its
authored `IdentityPage` receiver so `page.name` uses the existing resident
string-token path. All declared source/glyph pages now have spill lifecycle
classifications. Final combined focused gate: 345 passed. Memory-graph storage
authority is established, but compression relative to the prior 1.68 GB
nonmaterialized observation is not. The full bootstrap remains unverified; run
a fresh provenance bootstrap next.

2026-10-16 FAILURE NEIGHBORHOODS + PURSUIT DEDUPE + BATCH SORT (CURRENT):
See docs/compiler_convergence/16_failure_neighborhood_dedupe_continuation.md.
Refused posts now carry a diffused local neighborhood of the concordance in
the exception note and the failed log.  Pursuit no longer re-ingests a
definition the submitted source already defines (re-ingested definitions
175 -> 65, units 478 -> 296, units stage 5m31s -> 2m41s on the user's run).
Optional-glyph carries are keyed by the copy's glyph scope.  The resident sort
is batch-wide (key pass, flat merge passes, batched gather) and needs an
`order` workspace.  Open: nested child-list sort in IdentityPage._reindex
(unit 031), _fork_of_scope membership (unit 026), native check of the sort
(tools/benchmark_ssa_sort.py), and the unattributed 20 GB pytest session:
tests/test_ast_parent_ingestion.py is SKIPPED until understood.  The full
bootstrap remains unverified.

2026-10-15 PROVENANCE/SORT/GLYPH BOOTSTRAP CONTINUATION (PRIOR):
See docs/compiler_convergence/15_provenance_sort_glyph_bootstrap_continuation.md.
Resident stable sort and formatted glyph sequences now lower through repository
SSA and concorded source contracts.  The latest bootstrap regression was a
cross-copy glyph recipe publishing one function copy's generated formatter
identity into another through a shared authored scope and shallow-copied nested
attributes.  Dynamic recipes now use the existing copy-local ingestion scope,
function graph views detach mutable attributes, and downstream enum/optional
readers consume exact concordance references.  The complete glyph provenance
file passes 32 tests; the mapping concordance audit has zero findings and zero
unsourced facts.  The full bootstrap remains unverified.  Bootstrap caches are
clear for the next user-owned run.

2026-10-14 PROVENANCE BOOTSTRAP TERMINAL OWNERSHIP + FAILURE ROLLUP (PRIOR):
The provenance child now exclusively owns the terminal while it is active;
the outer bootstrap opens its compositor only after provenance returns. Resolved
worker events stay structured, worker JSON/traceback stdout remains in each
unit's durable log/receipt, and a completed stage prints a deterministic list
of blocking unit, qualified function, compiler phase, exception type, concise
reason and receipt path. Provenance growth is fail-fast by default: after the
first failed/partial receipt it admits no more units, drains already-running
children, publishes every simultaneous blocking failure and exits nonzero.
The outer process propagates that status without a second traceback.

The long apparent pause at `compilation units` was observed, not inferred: the
coordinator had already divided 466 units and was serially writing roughly
10.8-MiB scoped input checkpoints. It had reached unit 459 before dispatch.
That existing durable checkpoint boundary is unchanged. It now emits source
plan messages plus a `Scoped unit inputs N/466` counter, followed by the real
child source-compiler columns only when dispatch begins. A live run correctly
refused dispatch because compiler source was edited after its plan fingerprint;
that was a toolchain-integrity refusal, not a unit semantic failure. No worker
from that run remains active. Targeted terminal/checkpoint/fail-fast checks pass;
the final directly affected gate passed 12 tests. The broader focused gate completed
with 133 passed, 2 skipped and 5 failed. Four failures reproduced unchanged in
a detached pre-change baseline (source-acceptance fixture, two existing SSA
geometry/concordance cases and the renderer work-contract case). The fifth was
the new checkpoint-progress test under suite order and passed immediately in
isolation; it does not reproduce in the directly affected gate. The detached
baseline worktree was removed after comparison.

2026-10-14 BOOTSTRAP CONTINUATION HANDOFF (CURRENT):
Start with docs/compiler_convergence/12_bootstrap_handoff.md for the user's
complete goal, retained memory constraints, implementation/review state,
observed blockers, existing shell channels, prioritized repairs, subdivision
mechanisms, validation evidence and runnable continuation commands. The work
is on draft PR #783; full bootstrap remains unverified. No diagnostic worker
is left running. Earlier timings are component/partial-bootstrap observations.

2026-10-13 NATIVE SHELL CHANNEL CHECK (CURRENT):
Shell diagnostic facilities already exist: shared log/error/profile/progress/
trace telemetry and native launch logging/bounded trace storage. The isolated
bootstrap progress(str) capture has no recorded shell endpoint binding; the
native profile logger has a different signature. See the shell-channel section
in docs/compiler_convergence/11_bootstrap_performance.md. Two C-shell device_ns
locals now match the declared unsigned-long-long pointer ABI, fixing the LP64
Linux build failure reproduced on main. Four focused checks passed; a real
native executable confirmed bounded trace overwrite/loss and launch status.

2026-10-13 BOOTSTRAP CLOSURE ACCOUNTING REPAIR (CURRENT):
Project unit completeness now consumes existing exact-source closure_formals
receipts for surviving argument IDs, matching the existing SSA signature check.
A numeric nested closure previously failed despite that receipt; it now has
complete repository SSA and passes three LLVM native input pairs. Stale capture
IDs do not explain other arguments. Fresh _report lowering recognizes progress
but still lacks message: its source Call 4 has conditional ownership but no
PlanCall in the hierarchy. All four authored bootstrap helpers remain partial;
Unbound callable-input Calls omitted from planning are now recorded in the
existing unresolved-call table with their exact source node provenance. This
also prevents a no-argument callback closure from looking complete after its
capture is correctly recognized. Final focused gate: 99 passed. Two broader
shaped native closure checks reproduced main's undefined binary_value operands.
missing object/AST/hierarchy parameter representation is not bypassed. See the
follow-up section in docs/compiler_convergence/11_bootstrap_performance.md.

2026-10-13 BOOTSTRAP PERFORMANCE INVESTIGATION (CURRENT):
See docs/compiler_convergence/11_bootstrap_performance.md for actual bootstrap
timings, bounded source-probe and progress fast paths, empty catalogue control
partition handling, scalar adoption, and validated relocation of already-LZMA
disk tiles. Adaptive memory regulation, cold compression, collection and
identity accounting remain enabled. Partial-bootstrap component timings are
not a whole-bootstrap completion claim. Authored helper receipts still expose
missing parameter/capture representation on unchanged main. Earlier scoped
unit publication is the existing route for reducing deployment's broad RAM
peak; holistic auto has not been silently replaced.

2026-10-13 LIVE DEPLOYMENT STORAGE + ELEVEN-STAGE VISIBILITY (CURRENT):
Deployment plans, loop plans, semantic loops, generated deployment-class
catalogues, and ordinary NetworkX process graphs now enter the existing live
tile arena without changing their Python identity. Initial adoption streams
the cold tail out before retaining the active front. A shared conversion memo
preserves aliases across fields. Late NetworkX promotion preserves shared edge
records and cached-view behavior. Empty tiled mappings now invalidate their
retained route before reinsertion; that was the first wrong writer behind the
observed missing loop operands.

The progress system now has an eleventh lowering stage, `compiler
finalization`. It remains live through native-gate accounting, concordance and
identity-log finalization, storage statistics, and storage release. Previously
silent compiler passes in source closure, topology reduction, deployment,
control/SSA lowering, frame and record reconciliation, native-gate checking,
identity checkpointing, and final storage collection now use the same
per-worker event stream and terminal compositor. No per-instruction polling
was added to the normal hot path.

Both user-facing compiler commands now accept `--min-workers` (alias
`--minimum-workers`) and default it to 12. The supplied-source scheduler and
each source's auto deployment-region scheduler receive that floor. The floor
is bounded by ready work and an explicit `--jobs` maximum, but it overrides
the memory estimate when enough independent work is ready. A parallel pool
that initially admits only one job no longer becomes permanently serial: it
resamples admission every 0.5 seconds while that job runs and expands when the
window opens. Project/bootstrap process admission also honors the same floor.
The decoded-front elasticity coefficient is now 75 percent everywhere (the
10-GiB base plus 75 percent of signed available-memory headroom above/below the
2-GiB reserve), replacing 25 percent.

The deliberately constrained four-source torture run used a 2-GiB aggregate
and per-source budget with elasticity disabled. It was stopped at the user's
request because that configuration continuously exercised minimum-footprint
behavior and was not a throughput benchmark; it produced no completion
receipt. A separate one-source bounded regression completed with three exact
LLVM native checks and zero structural concordance findings. Focused storage,
native call/loop, status, source-schedule, stdout-lifetime, identity-log and
admission tests passed. The latest admission/default gate passed 24 tests in
8.80 seconds. The mapping audit remained at zero identity findings; 21
unsourced facts remain visible. No full bootstrap or orbital compile was run
by the repair, and the user's running orbital process was not disturbed.

The existing strongest live-arena collection remains one-tile-at-a-time mark
and compaction. Its conservative worst-case bound is O(V*S + E + B), where V
is retained roots/values, S is the number of storage tiles scanned per root in
the current mark organization, E is traversed references, and B is stored
bytes compressed. This work makes that phase visible; it does not claim to
have reduced that bound.

2026-10-13 SUPPLIED SOURCE SCHEDULES + IN-COMPILE AUTO (CURRENT):
`lower_ast_source_to_ssa(source, entry, source_compilation="auto")` now runs
the whole compiler without a preparatory source partition or unit-worker DAG.
Automatic threads consume the flat region closures already formed by normal
deployment planning, during SSA region expansion. Their temporary value ids
come from the existing atomic monotonic issuer; their facts and provenance
publish to the same IdentityBook under one joined publication transaction.
Page reads share that transaction. The ordinary serial book/page methods have
no added lock dispatch outside an actual parallel region join.

Explicit serial/parallel schedules accept caller-supplied source-request lists.
Each submission still lowers through the public source entry with auto inside.
The bootstrap supplies its authored pieces and an explicit pieces-before-outer
barrier to the existing project publisher. Orbital supplies its missing/stale
law pieces, then the complete linked system. The scheduler does not invent a
replacement source cut. Explicit legacy `partitioned_directory` remains
available; the earlier receipt describing auto as prepartitioning a resolved
unit DAG below is historical, not the current automatic path.

Both commands default to a parallel supplied schedule. Worker admission uses
live RAM headroom rather than a fixed global worker count: 10 GiB aggregate,
10 GiB per-worker compression budgets, 2 GiB reservation/system reserve, and
128 MiB inner working reservations. Bootstrap admission can contract to its
2 GiB minimum; the existing signed decoded-cache elasticity remains active.
Budgets never terminate running compiles. Independent source submissions keep
working after a lone failure; final failures retain their original tracebacks.
Per-book specialization caches no longer clear another running book's answers.

There is one parent-owned terminal compositor for the entire command, including
native publication. Exact ten-stage states and inner-thread lanes feed that
same display. Failed lanes survive thread reuse, and joined region snapshots
cannot hide the source coordinator's later pre-native stage. The rule clock is
retained by the terminal owner so background redraws cannot remove its line.
Log producers enqueue diagnostics; console writes are batched, numeric-shaped
repeats are condensed, and oversized diagnostics have a bounded console preview.
Complete diagnostic text is retained in build/compiler-logs and its path is
printed at completion. Live standard-stream proxies checkpoint through their
existing import-owned module slots, not their ConsoleThreadLocals internals.

Verification: 32 selected small tests passed in 9.24 seconds, with a 1-GiB
compiler budget and decoded elasticity disabled for those tests. The final
single-source/after-join display regression also passed in 0.20 seconds.
The gate includes two real parallel source programs, two actual LLVM law
pieces with exact native results, whole-graph record native parity, independent
failure continuation, atomic publication, command plumbing and real Rich
rendering. The mapping audit reports 0 identity findings (its 21 unsourced facts
remain visible and were not suppressed). Syntax and both CLI-help checks pass.
No full bootstrap or orbital compile was launched; existing Python processes
were neither inspected nor disturbed. Large-run throughput and remaining
compiler semantic frontiers are not proven by this small gate.

From C:/dev/Powershell/turing, user-owned runs:
  python -u -m tools.bootstrap_compiler_whole_program
  python -u tools/compiler_probes/profile_memory_n2.py all --backend llvm

2026-10-13 ELASTIC DECODED COMPILER FRONT (CURRENT WORKTREE):
Once live compiler storage enters a compressed tier, its backing tier remains
monotonic but its decoded working front is now elastic.  Its effective process
ceiling is the configured ceiling plus 75 percent of signed system headroom
(available physical RAM minus the 2-GiB system reserve).  Thus abundant RAM
lets the temporary decoded cache exceed 10 GiB, while availability below the
reserve pushes the ceiling below 10 GiB.  The sample is refreshed by the
existing throttled pressure check.  Tiles enter a larger front only when the
active stage reads them; a falling ceiling contracts that front before storage
escalates.  The authoritative compressed backing, handles, route indexes, and
concordance identities do not change.  Receipts distinguish the declared
decoded budget, current working budget, system sample, signed adjustment, and
effective ceiling.  The percentage is a work-contract/environment value and
the bootstrap CLI exposes `--worker-elastic-available-percent` (default 75).

Final storage release now completes before the canonical compiler progress
display closes.  The pre-native row therefore remains present while the final
compressed-arena mark/sweep runs, with explicit start/end log lines; “compile
done” no longer precedes invisible compiler work.  The full live-storage,
memory-regulation, progress, and bootstrap suites pass 104 tests in 82.31
seconds, and the mapping concordance audit reports zero findings.  No bootstrap or orbital
compile was launched, and no existing Python process was inspected or
disturbed.

2026-10-13 IMPORT-OWNED TYPE CHECKPOINT IDENTITY (CURRENT WORKTREE):
The first real auto-bootstrap attempt completed whole-source closure and
topology reduction, then failed before scheduling while checkpointing the
resolved ProcessGraph.  The compiler source has long retained both CPython
code instances and the exact `types.CodeType` class used in authored
`isinstance` checks.  The graph checkpoint contract already serialized code
instances through `marshal`, but stock pickle named the class itself as the
nonexistent `builtins.code` slot.  The private resolved-graph pickler now uses
the existing import-owned-module-slot contract for any type whose declared
module/qualname slot does not own that exact class.  Normally importable types
still use stock pickle; this is a general identity rule, not a `CodeType`
exception.  The regressions checkpoint the class and an instance together,
require exact class identity after loading, and pass.  This is not a scheduling or memory-
admission failure: the run stopped before auto measured plan width or launched
a worker.  The failed `build/compiler-partitions` attempt tree was removed so
the next user-owned run starts without its partial checkpoint.  No full
bootstrap was launched by the repair.

2026-10-13 AUTOMATIC RESOLVED-GRAPH SCHEDULING AND ADMISSION (CURRENT WORKTREE):
The canonical source compiler accepts `source_compilation="auto"` alongside
serial and parallel operation.  It still resolves the complete ProcessGraph
before making any scheduling decision.  Auto measures the widest dependency-
ready wave in that resolved SCC/unit DAG, then admits at most that many workers
and never exceeds the explicit worker maximum.  A real three-unit compiler
repro measured width two, ran two workers concurrently, and reproduced the
native result.

The partitioned compiler can now lower its aggregate worker admission ceiling
from a configured maximum as physical RAM becomes unavailable.  The bootstrap
CLI defaults to a 10 GiB maximum, 2 GiB minimum, and 2 GiB system reserve.
Only new-worker admission changes; a running worker is never terminated.  The
selected schedule, graph width, active ceiling, and sampled availability are
published in progress receipts.  The disposable `build/compiler-whole-program`
and `build/compiler-partitions` attempt trees were removed;
`build/compiler_foundations` was preserved.  The universal frontend gate passes
22 tests in 158.34 seconds, the whole-bootstrap gate passes 7 tests in 51.80
seconds, focused scheduler/default tests pass 4 tests, and syntax, CLI-help, and
diff checks pass.  No full bootstrap or orbital compile was launched, and no
existing Python process was inspected or disturbed.

2026-10-13 CANONICAL PARTITIONED BOOTSTRAP (CURRENT WORKTREE):
The whole-program bootstrap no longer calls the separate project bootstrap-
creep coordinator.  It submits the authored
`_lower_process_graph_to_compilation` source through the sanctioned
`fortran_c_shell.lower_ast_source_to_ssa` entry with
`partitioned_directory`, exactly as ordinary partitioned program compilation
does.  Its versioned wrapper now owns only source/toolchain version selection
and the final manifest.  The bootstrap-only worker-overlay projector was
removed; terminal output uses the shared compiler telemetry.

Partitioned source now carries the public compiler's existing `source_module`
and `python_package` context into source ingestion.  This resolves the
compiler source's authored relative imports during source pursuit; it does not
admit Python runtime fallback, and public partitioned compilation still forces
native interface tiles.  Worker memory settings now feed
`TURING_MEMORY_BUDGET_BYTES`, the existing WorkContract compression policy.
No project, resolved-unit, subdivision, creep, or exponential-bootstrap
coordinator terminates a compiler worker for elapsed time or memory usage.
The whole-program defaults are 10 GiB aggregate admission, 10 GiB per-worker
compression budget, and 2 GiB worker reservation.

Focused whole-bootstrap, public-partition routing, memory-contract,
project-product, and exponential-supervisor gates pass 99 tests.  Syntax and
diff checks pass.  No bootstrap or all-rules compile was launched, and no
existing Python process was inspected or disturbed.

2026-10-12 PARENT-OWNED COMPILER PROGRESS (CURRENT WORKTREE):
The whole-program bootstrap still owns exactly one terminal compositor.  Its
workers do not create terminal displays.  Each worker now publishes the exact
ten-stage state emitted by `CompileProgress` into its durable
`compile-progress.json`; the parent renders those stages as rows and concurrent
workers as columns.  A failed worker remains visible while independent workers
continue.  Only the outer bootstrap scope closes the display.  Stage
publication occurs at stage and substage transitions, not at work-item count
updates, so this adds no callback to the per-item lowering path.

The telemetry, live-display, project-product, and whole-bootstrap gate passes
136 tests in 66.85 seconds.  Four focused structured-state tests pass in 1.30
seconds.  The outer bootstrap was not launched; the user owns the next
observable run.

2026-10-12 COMPILER BOOTSTRAP FRONTIERS (CURRENT WORKTREE):
The current worktree retains the exact authored identities exposed by the
latest bootstrap frontier: returned static Python values are materialized as
their existing `StaticReference`; generic formals accept multiple record class
variants without traversal-order stamping; and authored aliases retain their
lexical aggregate kind.  Resolved graph checkpoints now use the existing
import-owned identity and AOT-checkpoint contracts for cached descriptors,
context variables, standard streams, and synchronization handles.  Native
build commands use a short transient working directory while keeping every
artifact path absolute.  Runtime-catalogue dependency ordering recognizes an
authored `recursive_backedge` on its private semantic dependency projection
and orders that projection through its condensation graph.  It does not
rebuild or annotate the shared ProcessGraph.  A catalogue with no executable
numerical nodes returns the exact empty shader dispatch plan before unrelated
catalogue topology is validated.

Seven focused identity, checkpoint, progress, and native-path tests pass in
2.02 seconds.  The scheduling and loop-composer gate passes 107 tests in 9.06
seconds, including refusal of an unowned cycle and of a residual cycle merely
adjacent to a declared backedge.  The persisted 109,621-node failed catalogue
graph was also loaded directly: dependency ordering returned all 109,621 unique nodes and the
empty shader projection returned no dispatches while its physical edges,
recursion metadata, and level table remained unchanged.  No outer bootstrap
was launched.

2026-10-12 PROJECT SOURCE MODULE IDENTITY (CURRENT WORKTREE):
Project compilation no longer discards the module identity it already resolved
from the authored source path.  The sanctioned source compiler now receives
that exact module and package, records the module on source spans, and supplies
the package to Python's relative-import resolver.  Before this change the
compiler bootstrap's `aot_compile.py` slice recorded repository calls such as
`build_map_dependency_regions` as external references whose entries had
`python_callable=None`; their conditional results therefore had no possible
SSA producer.  The calls now enter ordinary repository-source pursuit and the
existing compilation-unit/call-frame machinery.

A focused package regression proves that `from .helper import imported_step`
becomes a real `callee_ref`, not an unresolved `external_callee_ref`, through
the public source compiler.  A second regression proves the project worker
hands that resolved module/package pair to ingestion.  The project-product
gate passes 74 tests in 1.56s.
A plan-only run of the real `_lower_process_graph_to_compilation` slice moved
from its former four-second seven-unit truncated graph into imported compiler
source pursuit; it was deliberately interrupted after 90 seconds because the
user owns the next observable bootstrap run.  No native build or outer
bootstrap was launched.

2026-10-12 INDEXED MAPPING VALUE IDENTITY (CURRENT WORKTREE):
The resident mutation extractor no longer flattens a tuple-valued mapping
item into extra mapping columns.  An authored `mapping[key] = tuple_value`
already supplies the exact two identities required by `mapping_items`: its
key and its value.  Tuple expansion remains in place for sequence rows, but
mapping keys and values now retain their own aggregate identities.

A focused regression proves an `IndexedStore` with a tuple value remains one
key/value pair.  The loop-composer gate passes 72 tests in 8.78s.  A bounded
rerun of only `aot_compile.py::_lower_process_graph_to_compilation` reduced
the structured frontier from 7 to 6 in 24.6s; the fixed-aggregate mapping
`update` shortfall is gone.  Remaining: four carried-name identities, one
carried-field version, and one loop-carried producer.  No outer bootstrap or
all-rules compile was launched.

2026-10-12 SCALAR MAPPING SETDEFAULT SSA (CURRENT WORKTREE):
The mapping mutation lowerer no longer assumes every `dict.setdefault` value
is a nested child table.  For an exact flat mapping schema with key columns
and one scalar value column, it now performs the existing table lookup, reads
the lookup status, stores the authored default only on the missing edge, and
publishes the call result as the Phi of the existing and inserted values.  The
nested child-table handle path remains unchanged.

The focused scalar regression proves lookup, conditional store, and result
Phi ownership.  A bounded rerun of only
`aot_compile.py::_lower_process_graph_to_compilation` reduced the structured
frontier from 10 to 7 in 24.0s; all three scalar `setdefault` shortfalls are
gone.  Remaining: four carried-name identities, one carried-field version,
one loop-carried producer, and one fixed-aggregate mapping update.  No outer
bootstrap or all-rules compile was launched.

2026-10-12 TRY-COMPARTMENT / PREDICATE IDENTITY REPAIR (CURRENT WORKTREE):
The real compiler partition's 48 `sequence-mutation-guard` shortfalls had one
cause.  `_branch_compartments` correctly records lexical compartments for
`Try`, `If`, `IfExp`, and short-circuit expressions, but the sequence-mutation
installer treated every compartment role spelled `body` as a boolean arm.  It
therefore asked the outer `Try` node for a predicate identity which does not
exist.  The installer now admits a predicate guard only when the retained
source-control owner is exactly `If` or `IfExp`; Try/dataflow and resource
scope handling remain on their existing paths.

A focused regression proves a mutation in `Try.body` is placed lexically
without an invented predicate.  Branch-compartment and control-to-SSA gates
pass 126 tests in 2.17s.  A bounded rerun of only
`aot_compile.py::_lower_process_graph_to_compilation` reduced the structured
SSA frontier from 58 to 10 shortfalls in 24.55s: all 48 false mutation-guard
failures are gone.  The remaining five carried-state, one loop-carried, and
four mapping-operation shortfalls are separate iterative repair targets.  No
outer bootstrap or all-rules compile was launched.

2026-10-12 STRUCTURED BACKEND-NEUTRAL SSA FRONTIER (CURRENT WORKTREE):
The whole-object lowering aggregate is no longer raised as
`FortranEmissionError`.  It occurs before target emission and now raises the
distinct `SSALoweringIncompleteError`.  The error retains every exact
`SSALoweringShortfall` and publishes `frontier_kind=ssa-lowering-incomplete`,
`lowering_stage=backend-neutral-ssa`, the total count, counts by mechanic, and
the full symbol/domain/name/location/reason records through worker failure,
project product, creep frontier, and bootstrap continuation receipts.  Other
exceptions still use the general `compiler-unit` failure path.

A bounded rerun of only
`aot_compile.py::_lower_process_graph_to_compilation` reached this new type in
23.96s with 58 shortfalls: 48 sequence-mutation guards, four carried-name
arms, three `setdefault` operations, and one each of mapping `update`, a
carried-field arm, and a loop-carried producer.  This is the same real
backend-neutral frontier previously mislabeled as a Fortran emission failure;
no outer bootstrap or all-rules compile was launched.  Project-product and
whole-bootstrap gates pass 79 tests in 59.91s.

2026-10-12 EXACT AUTHORED CONTROL-SITE RETENTION (CURRENT WORKTREE):
The compiler no longer collapses a direct control-nesting edge to a bare
parent/child relation when the child is a region-less comprehension loop.
Planning now retains the exact authored AST slot and path.  An inner loop in
an owning comprehension's iterable executes as a prelude; one in an `ifs`,
`elt`, `key`, or `value` executes inside the owning iteration.  Multiple exact
children retain authored evaluation order.  Ordinary `body`/`orelse` placement
continues to use the existing exact-arm proof and still refuses ambiguity.

The same real partition exposed three downstream consumers that had also
discarded exact identity.  Conditional containment now treats `IfExp` arms as
their AST-valued expressions rather than splatting them as statement lists;
an already-recorded unresolved field arm retains its precise reason instead
of attempting a causeless revision; and one authored parameter name publishes
one control ABI slot even when later lexical versions are separate region
arguments.

Bounded verification compiled only
`aot_compile.py::_lower_process_graph_to_compilation`, not the outer bootstrap.
It now lowers all 44 regions and reaches the intended aggregate
`FortranEmissionError` listing the remaining unsupported mapping/update,
carried-value, and mutation-guard frontiers.  It no longer fails at
`declared nested loop has no exact authored arm`, `IfExp` arm iteration,
causeless concordance revision, or duplicate `class_navigation`/`map_ir`
parameter names.  Focused control, ownership, overlay, and SSA gates pass 47
tests in 5.21s; the user retains ownership of the next observable all-rules
bootstrap run.

2026-10-12 NON-BLOCKING COMPILER BOOTSTRAP FRONTIERS (CURRENT WORKTREE):
Project bootstrap no longer converts one failed authored compiler unit into an
immediate project-wide `NativeInstallationRequiredError`.  The catalogue's
existing dependency rule remains authoritative: callers wait for a dependency
to become terminal; a verified complete dependency may be linked, while a
partial or failed dependency remains its exact authored Python body.  Verified
independent units are still activated only after automatic native verification
and an exclusively native probe receipt.  Their installed names are now taken
from successful activations rather than verification alone.

Each failed compiler unit is persisted in the round's
`native-installation-frontier.json` and `continuation_failures`.  Subdivision
creeps run afterward, and an exception from one subdivision is recorded on
that subdivision without preventing later independent plans from running.
The CLI still exits nonzero for a frontier, but only after all available work
has reached a receipt.
The user's next outer-bootstrap run proved this continuation policy: the root
failure remained a frontier while all four nested authored units reached their
resolved-plan/subdivision receipts.  It also exposed terminal ownership: each
authored worker inherited the same TTY and could start its own Rich Live, so a
failed worker's cursor restoration removed the display while siblings kept
working.  Authored-worker stdout is now captured by a bounded 4096-line parent
queue, child progress overlays are disabled, resolved/subdivision worker stdout
is durable per-worker output, and their stage/RSS snapshots feed one parent
overlay.  Only that parent owns the stationary status/table and bottom work
bars; completing or failing one row cannot close the display.

Focused project-product gate: 69 passed in 1.13s.  After the display change,
project-product, whole-bootstrap and live-Rich display gates pass 90 tests in
65.01s under the already provisioned system Python.  PyYAML is declared in
`requirements.txt`; the repository venv had omitted it even though extraction
contracts import it.  The agent did not launch the outer bootstrap compile.

2026-10-12 VERSIONED OUTER COMPILER PARTITION (CURRENT WORKTREE):
The abandoned whole-program bootstrap no longer frames compiler self-hosting as
one replacement DLL.  Its plan now composes the existing topology, dependency,
dead-metadata, broadcast and whole-module storage-effect native semantics around
the Python coordinator/IdentityBook authority.  The measured losing metadata
copier remains in the receipt as a rejected hot boundary, not an active piece.

Every boundary version hashes its exact authored source, declared input/output
contract and the compiler toolchain fingerprint.  A source or toolchain change
therefore selects a new side-by-side product directory; only the existing
receipt-verified bootstrap activation may feed a product into a later compile.
The next target is the existing complete `_lower_process_graph_to_compilation`
seam: one built ProcessGraph enters; reduction, unit planning, map/class tables,
deployment planning and capture remain inside; one AOTCompilation leaves.
Python-AST and SymPy ingestion are recorded as later sibling front ends to that
same seam.  No inner graph kernel was added and no outer bootstrap compile was
launched, per the user's request to run and observe it.  Non-compiling manifest,
version invalidation, delegation and catalogue tests: 4 passed in 0.27s.

2026-10-12 INHERITED DUAL-IR PRESSURE / COMPRESSION TELEMETRY (CURRENT WORKTREE):
The symbolic compiler's pressure arena now remains the active physical storage
arena when the public source compiler resumes the SAME IdentityBook.  The
non-owning source regulator may continue the coarse-to-fine pressure ladder and
its throttled in-stage checks; ownership still gates lifecycle decisions about
which semantic book pages are cold, not physical regulation of already-adopted
storage.

* `graph_work.source_and_ssa_paged` is no longer hard-coded false.  It is true
  only after ProcessGraph, ControlProgram or SSA instruction storage has
  actually joined the book's arena.  The same report exposes pressure level,
  disk/decoded bytes, hot payload count, route hits/misses, payload cache
  hits/reloads/evictions and compression jobs/time/workers.
* Every pressure-tier transition now logs its action, elapsed wall time, RSS
  before/after and those physical-storage counters.  Resuming inherited
  pressure emits an explicit line, so a long source stage no longer hides that
  it is operating on the symbolic compiler's evicted material.
* Maximum-tier strong disk codecs run through a bounded worker pool.  Semantic
  serialization remains ordered because it discovers exact shared leaves;
  only independent tile codecs are parallel.  At most W raw tiles and W
  <=1MiB codec dictionaries exist for W workers, rather than a whole-arena
  future/input queue.
* Let B be live payload bytes, T tile count, S largest tile and W bounded codec
  workers.  Maximum conversion remains O(B+T) total work and overall worst-case
  elapsed time because ordered serialization is O(B+T).  Within each serialized
  bounded batch the codec span is O(B/W+S+T).  This is a wall-clock reduction,
  not a false claim that the required full-storage traversal became sublinear.
  Subsequent alternating payload accesses retain the existing O(A*S) term;
  retained route indexes prevent repeated trie rediscovery but cannot remove a
  real requested payload read.
* Focused live-storage, regulation and graph-work gate: 76 passed, 1 deselected
  in 78.26s.  The deselected native-worker bootstrap test fails at clean
  7e892712 with the same pre-existing `carried-name-arm-missing` SSA error, so
  it is recorded rather than attributed to this change.  The full orbital
  all-rule receipt is intentionally pending the user's requested run.

2026-10-12 RETAINED LIVE-MAP SEARCH INDEX (CURRENT WORKTREE):
The earlier maximum-tier design repeatedly evicted the mutable TiledDict hash
trie itself.  A logical lookup therefore reread several routing tiles before it
could read the requested bucket.  That was the source of the improper read
storm; compiler-unit interfaces were not its cause.

* Each live mutable map now retains a compact open-addressed pair of uint64
  columns mapping full Python hash to its authoritative trie bucket handle.
  Keys and equality remain in the tiled bucket, so the index cannot invent
  equality and full-hash collisions still traverse their bounded exact chain.
  Trie splits rewrite routes for the affected bucket; clear drops the index;
  physical mark/sweep drops indexes whose owning dictionary is unreachable.
* Identity-header tiles remain decoded, but bucket payloads do not.  Maximum
  pressure therefore means one decoded PAYLOAD tile plus compact search state,
  not one arbitrary tile that continually evicts its own addressing metadata.
  Configured live_resident_tiles remains the sourced policy; a separate
  working_hot_limit records the final payload allowance.
* For H discovered hashes, the retained index is O(H) fixed-width storage.
  First route discovery follows the authoritative trie, O(D*S) worst-case for
  trie depth D and tile decode bound S.  A retained route is expected O(1)
  (strict open-addressing worst case O(H)) plus at most the requested payload
  decode O(S).  Thus repeated lookup no longer pays O(D*S); actual alternating
  payloads can still pay O(A*S), which is real data traffic rather than index
  rediscovery.
* The synthetic maximum-pressure regression proves the discovery pass performs
  more reads than the next pass, then records64/64 route hits with at most one
  payload reload per key. Collision, split, compaction and GC ownership tests
  are included. Current storage/regulation gate excluding the long forced
  source case:53 passed,1 deselected in11.23s. Fresh forced-disk public source,
  LLVM execution and concordance audit0:1 passed in61.25s.
* The formerly pathological forced-disk dual-IR Jacobian now PASSES in399.31s
  (6m39s). The rejected version was manually interrupted unfinished after
  1251.88s (20m51s). At five minutes the repaired run had207,587 read
  operations versus4,193,142 observed by sixteen minutes in the rejected run.
  This is a large measured correction, not a claim that maximum compression is
  fast or that all remaining payload-locality work is complete.

2026-10-12 AUTOMATIC COARSE-TO-FINE STORAGE LADDER (CURRENT WORKTREE):
The default compiler policy is now dormant below a 10 GiB process-RSS
threshold.  It does not create tiled accessors, post compression policy facts,
flush at boundaries, compress IdentityBook pages, or run tile mark/sweep while
below that threshold.  This is a default capability, not an instruction to
compress every compile.

* On the first over-budget observation, the regulator late-adopts registered
  live compiler roots and IdentityBook pages into the existing live-tile
  mechanism.  It then escalates monotonically only while RSS remains over the
  limit: (1) disk-back the cold decoded LRU tail, (2) compress all remaining
  resident tile contents in RAM and collect dead physical records, (3) put all
  tile backing on disk and reduce the decoded payload working set to one tile
  while retaining compact search metadata.
* ProcessGraph/NetworkX, ControlProgram, IRModule, FusedProgram, deployment,
  control-SSA builder and FunctionTable roots opt into weak late registration.
  Ordinary NetworkX graphs remain ordinary dict-backed graphs until pressure;
  adoption changes their documented map attributes in place so extant graph
  identity survives.
* Complexity vocabulary for the slowest tier: let B be live encoded payload
  bytes, T tile count, V reachable storage records, E storage edges and S the
  largest decoded tile.  Full disk conversion and strong compaction each read
  and write every tile once: O(B+T) time and O(S) arena workspace with the
  configured fixed <=1 MiB codec dictionary.  Mark/sweep visits each record and
  edge once, but its intentionally one-tile DFS cache can re-decode on every
  adversarial cross-tile visit.  Its strict parameterized bound is therefore
  O(V*S+E+B), followed by O(B+V) sweep/compaction.  Because tile rows, codec
  dictionary, 64 KiB mark window and 4096-handle frontier block are fixed in
  this implementation, the configured operational form is O(B+V+E), with a
  potentially severe locality constant.  Do not describe maximum compression
  as constant-time or as merely O(number of tiles).
* The maximum-tier transition is not the whole wall-clock model.  Once only one
  decoded tile remains, A later accesses that alternate tiles can each incur a
  decode and dirty eviction, adding O(A*S) codec work and disk traffic.  If K
  declared dead-stage boundaries run collection while at maximum pressure, the
  complete storage overhead is bounded by O(B+T + A*S +
  K*(V*S+E+B)) in the parameterized model.  Fixed S makes that formally linear
  in accesses and graph size, but does not make its compression/I/O constant
  small.  This separates the cheap pressure decision, the one-time transition,
  and the real steady-state wall-clock penalty.
* Focused current receipts: memory-regulation unit gate 19 passed in 1.77s;
  memory-regulation/live-tile gate excluding the known long forced-source case
  52 passed, 1 deselected in 37.56s; ordinary dual-IR/splicing source gate
  20 passed, 1 deselected in 14.81s.  The ordinary gate confirms that the
  automatic default does not activate storage below the limit.
* HISTORICAL RED PERFORMANCE RECEIPT, superseded by the retained-index result
  above: the isolated forced-disk
  dual-IR Jacobian was manually interrupted after 1251.88s (20m51s), still in
  control-SSA finalization.  At an earlier 16-minute sample it had issued
  4,193,142 reads / 1.65 GB read while RSS held near 518 MiB.  The interrupt
  stack was TiledDict update -> tile load -> decoded owned-row measurement,
  direct evidence for the old routing/cache-miss term.  That run had no pass
  result. Counters for cumulative
  tile reloads, recompressions and hot peak are now retained in the live report
  so future runs can distinguish transition cost from steady-state thrash.

OPEN: late adoption necessarily has a transient migration allocation and only
roots using the declared registration mechanism are covered.  Opaque leaves,
native/backend buffers, allocator slack and a single indivisible value remain
outside the tile working-set bound.  The 10 GiB threshold is not a hard RSS cap.

2026-10-12 PAGED DIRECTORY / ACCESSOR FOLLOW-UP (CHECKPOINTED, INCOMPLETE):
Concurrent "Review dual IR math groundwork" work completed; its edits were
preserved. No full/all-law run was started. The user subsequently requested
the command for their own all-law run, not an agent launch.

* The SAME physical DiskTile headers now use DiskTileDirectory: fixed-width
  disk entries and at most one64KiB metadata window. Live arena directory,
  kind and row-length columns share that index. Cold CompressedTable and its
  DiskPackedIndex also use it once offloaded; cold functional point updates
  clone scalar directory entries on disk, not list(N headers) in RAM.
* Ordered-map deletions tombstone their existing entry ordinal and compact
  amortized. Sequence slice edits rebuild once into paged storage; sorting
  uses bounded stable runs and pairwise merges. Mutable set update operators
  no longer make tuple(self). Explicit exports/copies can still materialize.
* Original Python data descriptors remain canonical field writers/readers;
  setter-only NetworkX descriptors must still read the paged instance field.
  Rebinding graph maps invalidates the weak-header cached views at that writer.
* Immutable concordance value facts retain their original classes and exact
  registry leaves. Large owned fields can use the SAME paged containers; byte
  accounting now follows owned fact fields instead of charging only56 bytes.
  Existing decoded-size accounting caches one scalar per decoded row and drops
  it on eviction, avoiding a second traversal of the old row on every write.
  This is bounded physical cache metadata, not another semantic/provenance book.
* Unread legacy graph-build stdout now goes to NUL rather than accumulating
  a whole StringIO. The existing compiler fingerprint covers live storage,
  regulation, native boundaries and the completed symbolic/linking files.
* Storage/live/regulation gate67 passed12.12s,3 deselected. Separate cold
  storage gate including paged cold directory update22 passed1.30s. Byte
  measurement/fact repro2 passed0.24s. Initially red descriptor/cache-view
  and owned-fact accounting repros now pass at the exact accessors.
* Public16-add source + LLVM O3/Python oracle, audit0,4MiB decoded cache,
  forced disk, unchanged compiler fingerprint:3 passed236.16s (before the
  final cache-local measurement change). Serialized live payload4,037,609B,
  disk624,030B, resident directory66,041B. The forced threshold1B is NOT RSS
  acceptance and payload was slightly smaller than the declared cache.
* Current source-control LLVM/C/Fortran oracle gate + forced-disk Jacobian
  + writable book +800-node graph footprint:16 passed347.06s. Tiny two-tile
  graph stress alone239.738s: selected raw storage1,419,460B; RAM compressed
  tiles plus counted headers215,099B; disk headers86,033B plus128,292B file.
  This selected component reduction is NOT a whole-process RSS bound or speedup.
* Same one-expression profiled forced-disk source:22.836s ->18.420s; function
  calls91.01M ->67.14M; byte-accounting cumulative12.880s ->8.755s. Both
  source fingerprints remained unchanged during each run. No orbital timing
  or controlled multicore speed inference; profiled timings include overhead.

Still OPEN: exact live opaque AST/foreign objects, whole copy-memo/native
boundary/export/sort construction buffers and allocator floor; headroom-aware
end-to-end RSS acceptance through all source stages. The dependency_levels
source kernel remains red as recorded below. Do not assert that10737418240B
is a hard process cap: it is the current disk-escalation threshold.

2026-10-12 LIVE STORAGE OWNERSHIP UPDATE (CHECKPOINTED, INCOMPLETE):
This opening update supersedes the older retention/foundation claims below.
The user's stop on all35 remains in force. No large run was restarted.

* Live managed headers are weakly cached. Cold containers and original
  compiler objects have physical layout descriptors in the SAME compressed
  tile store, not a second IdentityBook. The descriptors hold type/layout and
  storage addresses; existing graph value IDs, book clocks, registry objects
  and provenance remain authoritative. These descriptors DO add physical
  metadata overhead; do not describe them as free or a second semantic ledger.
* NetworkX cached views no longer enter the strong opaque-leaf registry. Original
  Meta/OpStep/FusedProgram and the control builder use the same optional store.
  Builder exit contexts must be rebound to their published managed container;
  mutable sets need the explicit TiledSet protocol, not an empty builtin set
  subclass (set(view) silently read its empty C backing store).
* collect_unused traces actual physical references using disk-backed mark bits
  and a bounded disk-backed traversal frontier, then sweeps unreachable rows
  and opaque leaves. It now runs at existing dead-working-set stage_end points
  and owned outermost compile cleanup. It does not add book pages/identities.
  Shared live views, cycles and handles remain valid across reclamation.
* Decoded cache admission now checks both tile count and conservative owned-row
  bytes. WorkContract defaults: 4096 tiles and 128MiB decoded bytes; override
  TURING_LIVE_DECODED_BUDGET_BYTES. A single decode workspace/opaque allocation
  is not a hard process cap. Physical tile-directory entries, opaque live
  objects, scratch/export/native marshalling buffers remain open memory floors.
* Current memory_regulation + live_tiles gate:42 passed304.58s. Includes
  forced-disk PUBLIC SOURCE compilation, LLVM O3 execution against authored
  Python for three inputs, zero concordance findings, and dead-stage/compile
  cleanup tests that assert no extra IdentityBook pages. Cleanup-only gate:
  17 passed1.41s. The intentionally two-tile test configuration still thrashes;
  this is NOT an all-law timing or hard-RSS acceptance receipt.
* The topology indegree formal2/75 failure was resolved in the exact source-read
  resolver: resolve the authored loop generation before consulting its durable
  planning alias, never the temporary cyclic body spelling. Topology foundation
  plus LLVM accumulator/held-seed source gate:3 passed7.80s (tiling off).
* RED: dependency_levels foundation cannot compile: conditional arms63/21 and
  34/11 name inner-loop body generations unavailable on their loop exits. The
  same graph-work native-workers test FAILS on untouched C:/dev/tg-parity-base
  (12.92s), so this is not attributed to the live-storage cleanup. Observed
  source builder bindings:63 Load in while_body.3,34 Load in while_body.2;
  consumers are while_exit.3 and while_exit.2, which those producers do not
  dominate. Find the canonical loop/conditional exit-port writer; do not bypass
  dominance or insert late IR repairs. Prior current gate:52 passed1 failed33.42s.
* Stale hot-book gate PID37008/session21466 (old code, many hours CPU) was stopped
  after its exact command was checked; its interrupted exit is not a test result.

Acceptance still requires bounded physical indexing/opaque and pass buffers,
real payload larger than declared headroom through every public source stage,
current C/LLVM/Fortran oracle parity, and the dependency-level kernel repair.
Do not declare fully out-of-core compilation or restart all35 from these micros.

2026-10-11 LIVE WRITABLE TILE CHECKPOINT (CHECKPOINTED, INCOMPLETE):
The all35 LLVM probe PID27420 was stopped at the user's explicit request after
verifying its exact command. Its exec session90567 ended exit1 from that stop;
there is NO completed all35 receipt. Do not restart the large run before the
remaining out-of-core work is ready. Only small regression tests are authorized.

* live_tiles.py extends the SAME CompressedTile/DiskTileStore to writable
  NetworkX dict factories, original ProcessGraph/control/deployment/SSA object
  fields, and the IdentityBook's hot tables. Shared views survive eviction,
  mutable edge aliases preserve identity, and book clocks/registry references
  remain authoritative. --ram-tiles enables this path; the existing explicit
  budget triggers its disk escalation during storage accesses as well as stages.
* A compile-wide decoded LRU bounds tile COUNT. Disk mode writes active dirty
  tiles with zlib9; cold offload/compaction choose bounded-workspace LZMA2/zlib9.
  Streaming compaction reclaims superseded file frames and retains the original
  index on failure. This is temporary storage, not a durable restart facility.
* Forced eviction exposed real source-writer bugs: MutableMapping.setdefault
  returned an obsolete raw default; a region's final Ret was appended to its
  raw construction list instead of its published BasicBlock; identity-only
  copy memos allowed recycled Python addresses to alias decoded source records;
  inherited list.__reversed__ read empty subclass backing storage and lost
  book as-of reads. Fixes are at canonical writers/accessors, not IR repairs.
  Sequence ordering and full-hash collision overflow now have bounded regression
  tests. Failed initial regulation setup restores both active contexts.
* Latest storage/live/regulation gate:54 passed19.07s. Includes the setup-cleanup
  test (earlier standalone regulation gate16 passed1.13s). Fresh public source
  root(x)=x*x+1 compiled under forced disk, tile_rows8/hot_limit2, produced a
  complete LLVM artifact and concordance audit0; it now also builds O3 and
  executes three inputs against the SAME authored Python source oracle.
* Same cProfile fixture before/after the codec primitive fast path:56.48s ->
  21.18s pytest wall;139.35M ->35.54M calls. persistent_id17.79M visits cost
  16.83s self/37.06s cumulative before,17.35M cost3.26s self/4.33s cumulative
  after. Large numbers of ABC isinstance checks were avoidable. No all-law
  speedup or whole-process bound is inferred from this measurement.
* The800-node real-NetworkX metadata fixture earlier measured raw selected
  reachable storage1,419,460 bytes; tiled RAM plus counted headers1,214,878;
  disk headers1,118,156 plus85,976 disk bytes; hot_peak2,3.725s conversion.
  This is about14.4% selected owned-storage reduction in RAM, NOT an RSS cap.
  Opaque leaf payloads, allocator slack and some DiskTile scalar/header fields
  are outside that accounting. The raw comparison graph was also retained.

RED / OPEN -- DO NOT DECLARE FULL OUT-OF-CORE COMPILATION:
* Strong identity-header and opaque-leaf registries still retain every managed
  object until the arena dies. Logical dead rows are not reclaimed. Compaction
  removes superseded disk frames, not unreachable compiler subgraphs.
* Tile-row/cache-count bounds are not byte bounds. Giant leaves, ASTs, builder
  locals, sorting/export buffers, and native-boundary materialization can remain
  whole resident allocations. Ordered map deletion and sequence shifts can be
  quadratic. Backend emission/build has not been proved bounded.
* Source-native topology foundation currently FAILS even with tiling disabled:
  function_parameter for compiler_graph_topology__topological_order@control:0,
  indegree recorded2/proposed75.75 is provisionally minted through _region_feed
  ->_read->external_value despite a planning alias75->2. The writer
  _bind_loop_result_ports_inside_body posts LOOP_BODY_SPELLING2->75 under loop118;
  initial2 still has Meta(shape=(),dtype=int64) and no accounting at that point.
  Why the explicit span contract did not reach that writer remains unresolved;
  do not hide the disagreement in finish metadata.
  The untouched C:/dev/tg-parity-base and pre-live copied child pass this test.
  Disabling inactive TiledObject hooks did not remove the canonical failure.
  Thus attribution to the live implementation is NOT yet established. A stale
  native copier cache no longer loading is only an unproven hypothesis.
* The earlier forced source-control gate was11 passed1 failed304.19s, failing
  only PyYAML export of tile-backed metadata; explicit ABI export was fixed.
  A hot-book source-control plus math gate PID37008/session21466 remains active
  as of this checkpoint; it loaded older code and is not a final current gate.
  A separate hot-book loop+math gate passed12/12 in498.16s before the codec fix.
  No claims of complete C/LLVM/Fortran parity or all-law success follow.

Acceptance before any full run: resolve the red source foundation; establish
live-unit ownership/reclamation from existing graph/lifecycle contracts; bound
pass working buffers and decode bytes; test real payloads larger than declared
headroom through the public source compiler and Python numerical oracle; audit
the existing sourced storage receipts. Keep the user's large-run stop in force.

2026-10-11 STRONGER DISK TILE TIER RECEIPT (CHECKPOINTED):
User: "nono, we need to be able to dump tiles to heavier compression on disk"
This supersedes the earlier RAM-only/no-disk description for the tile mode.
It does NOT turn on legacy whole-page disk spill beside compressed tables.

* CompressedTable.offload_to_disk moves independently encoded data tiles AND
  packed hash/ordinal index chunks into one shared temporary tile file. LZMA2
  preset6 with <=1MiB dictionary (bounded encoder workspace); zlib9/raw is
  retained when smaller. Same lookup/iteration/select/map/reduce/point-update
  interfaces, no whole-page decode. Index reads retain only one8KiB chunk each.
* File writes commit before releasing the RAM tile. A simulated partial disk
  failure preserves every value in the mixed RAM/disk table. CRC/length checks
  detect damaged records. Shared views retain the file until their last owner
  releases it; whole-book serialization explicitly promotes a mutable copy and
  does not serialize a live file handle. These are temporary caches, not durable
  restart checkpoints. Incompressible data and metadata have irreducible size.
* WorkContract.compressed_tile_disk=True enables escalation for --ram-tiles
  WHEN memory_budget_bytes is set and exceeded. --budget-bytes supplies that
  threshold; no threshold is inferred by the compiler if it is absent. Old
  pure-RAM manual regulators can leave tile_disk_policy=False. The probe now
  prints resolved RAM/disk flags and budget in RULE_SUBMISSION.
* Pressure checkpoints are inside function-catalogue instantiation and callsite
  specialization, throttled to one RSS query/.5s and independent of UI. The
  predeclared disk/budget policy refs license a sourced runtime offload receipt
  retained on the SAME book; posting new private edges during pressure would
  thaw their whole page and undo the release. Concordance audit checks these
  exact source cells; missing/disagreeing causes produce tile-storage-policy.
* 20,000-fact real-book fixture:2,993,542 RAM tile/index bytes ->157,464 resident
  bytes,1,703,012 disk bytes,1112 data tiles;94.74% further retained-storage
  reduction. Offload2.587400s. RSS60,628,992 ->56,233,984 bytes (4.19MiB decrease).
  Lookup/registry identity exact, no whole-page reload. One-byte forced budget
  deliberately reports target_met=False: not a plausible process-RSS target.
* Final storage/regulation/telemetry/callsite/math/concordance gate119 passed,
  28.20s. Includes a fresh public source Jacobian compile with forced disk tier,
  structural audit0, and LLVM O3 execution versus authored analytic Python.
  Earlier all10 math tests under forced one-byte disk pressure passed9.09s.
  Existing cffi imp deprecation warning only. No all-law compile launched or
  existing process killed, restarted or dynamically reconfigured.

OPEN: this tier covers compressed identity tables, NOT arbitrary live
ProcessGraphs, instantiated shell state or final SSA bodies. Writers/full-book
exports can still promote a whole page. RSS allocator slack/other processes
can defeat the declared target even after all eligible tiles are on disk.
Do not call this a hard process/system memory cap or full-main out-of-core
compilation. The supplied35-rule log stopped during deployment instantiation
at770/1253,20.74GB RSS; it contains no completed main-build/parity receipt.

2026-10-11 COMPRESSED-IN-RAM / DUAL-IR MATH RECEIPT (CHECKPOINTED):
Canonical C:/dev/Powershell/turing; earlier repairs below remain in place.
User correction: "did you not understand then why I said we'd make an object,
we would make the accessors and operators to work compressed IN RAM"

OBSERVED:
* compressed_tiles.CompressedTable keeps keys and values in independent resident
  compressed tiles with two uint64 hash/ordinal index arrays. Lookup verifies
  actual equality, including collisions; identity-sensitive/duplicate keys refuse.
  Missing hashes decode no tiles. items/select/map/reduce stream; with_value
  encodes only one tile and shares untouched tiles. No disk or whole-map decode.
* IdentityBook.compress_pages installs those mappings behind existing accessors.
  cell/latest/history/stamp/edge/mint reads remain tiled and registered Page,
  Stage/Transform/Reason references resolve against the SAME registry. Canonical
  writes and whole-book copies explicitly promote a mutable page; no invisible
  read promotion. Held dictionaries/unserialisable facts refuse, not disappear.
  Private provenance tables can compress at compile-end after admission receipts.
  This is cold-book storage, NOT live ProcessGraph/SSA tiling or parallel lowering.
* WorkContract.compress_cold_pages and TURING_COMPRESS_COLD_PAGES select RAM
  storage; disk spill remains separate and mutually exclusive. The existing
  budget supplies headroom, with measured recompression and an explicit unmet
  result. No arbitrary guaranteed ratio or process-RSS cap. --ram-tiles added to
  profile_memory_n2.py; its report now distinguishes uncompressed cells from
  resident compressed buffers/indexes. No long orbital compile launched here.
* Hand-posted REAL book fixture: 20,000 facts/40,000 cells before receipts.
  Estimated reachable selected Python table footprint:21,833,969 bytes before;
  compressed buffers/indexes/headers:2,994,252 bytes after (7.292x,86.29% saved).
  Registry/codec shared objects and allocator slack excluded from retained
  buffer accounting. Compression .290646s; lookup .000153s/ONE decoded cells
  tile. No refused pages. 14 additional cells are compression-policy receipts.
  RSS59,600,896 ->62,685,184 bytes: rose3,084,288 bytes. Smaller table storage
  does NOT demonstrate smaller whole-process or full-compile peak RSS.
* Windows RSS sampler previously declared a new ctypes Structure/POINTER for
  EVERY sample; 1000 reads retained1000 classes and grew pointer cache1017,
  also reproduced on untouched HEAD8eb40eae. Binding is now created once under
  a lock; regression proves pointer cache grows0 across1000 warmed reads.
  The ~67k ctypes type counts in the supplied log match this leak's shape,
  but attributing their entire byte total to it remains an inference.
* Storage/spill/regulation/telemetry gate:82 passed,6.61s. Math/DEC gate:
  37 passed,4.32s earlier; RAM-enabled full math gate10 passed,3.75s, including
  fresh source-to-LLVM O3 execution against authored analytic Python/DEC oracles
  and structural concordance0. One existing cffi imp deprecation warning.

DUAL-IR MATH / DEC SCOPE:
dual_ir_math.py retains actual ProcessGraph + ControlProgram + explicit region
ownership, original source shell/map/hierarchy and concordance book by reference.
Pure scalar-component symbolic ops include composition, exact seed coefficient
extraction, Jacobian/Hessian/divergence/curl/Laplacian/biharmonic and heat/wave/
Poisson/Helmholtz residual graphs. These are not PDE solvers or Riemannian
covariant operators. General control/effects, opaque calls and unspecified
tensor components refuse. Existing controlled reverse is delegated to, not
reimplemented. Existing DECSystem.laplace0_diagram composes oriented local edge
and joint functions; source glue retains those call boundaries, uses existing
star0/star1 and matches existing eager DEC. No dense DEC matrix stand-in.

SUPPLIED 16-RULE LOG: build success, NOT target numerical acceptance.
2570 functions; elapsed2827.058s; sampled peak19672MiB/19.21GiB;
3,062,484 cells all RAM and0 spill pages at sampled peak. DLL exists in copied
native-fallback-link child; LLVM SHA256 matches supplied log:
94570d8cf04e341aed4a5642e7f42a4d461781d2ffc55db4273dfa222c1f042c.
No target execution in the probe; revision not established by log. New math and
RAM storage did NOT cause that historical run's build success.

ACTION / AUDIT PLAN, WITH ACCEPTANCE CONDITIONS:
1. Land and verify resident compressed lookup/operators/registry identity:
   DONE on small fixtures and fresh source/native math execution. Measure cold
   table retained bytes separately from conversion peak RSS and latency.
2. Audit live graph/planning/SSA retained owners at real boundaries. Reuse exact
   graph identities and existing region tiles; do not duplicate graph authority.
   Extend compressed column access to those owners only with a proved read/write
   contract. Required: actual peak-RSS reduction on unchanged source workload,
   no disk dependency and no hidden whole-graph rehydration. NOT DONE.
3. Bound writable hot tiles and whole-copy/export admissions. Current book
   writes/copies promote a whole page; giant individual facts are indivisible.
   Compression targets can fail when live graphs/indexes/held objects exhaust
   the budget. Never label this a hard RAM bound. NOT DONE.
4. Extend math audit beyond current pure scalar-component lane: controlled
   reverse wrapper, general registry VJP Call/Indexed bodies, exact tensor port
   extents, source control/map cause edges and every missing source identity.
   Gate each with symbolic identity + fresh public source lowering + independent
   authored Python oracle. C/Fortran math execution not yet covered.
5. Finish encapsulated DEC suites using existing DEC code: local law overrides,
   orientation/cochain degree/Hodge contracts, bounded batched composition, then
   higher-degree operators. Current laplace0 is a small unrolled proof, not a
   general large diagram scheduler.
6. User launches unchanged N4 acceptance when desired, now optionally
   python -u tools/compiler_probes/profile_memory_n2.py 4 --backend llvm --ram-tiles --save-ssa
   No timeout. Build success alone is not parity; compare source trajectory to
   authored Python/SymPy truth before8/all. Main parallelism remains unfinished.

2026-10-11 PI / PRIVATE RECORD / SOURCE PARITY REPAIR RECEIPT (CHECKPOINTED):
User requested: "God fucking damn it fix the pi and the uninitialized private record and FUCKING DONT STOP UNTIL YOU ARE FINISHED REMOVING ALL BUGS"
Canonical source: C:/dev/Powershell/turing. Changes below are in the compiler
source, exercised through lower_ast_source_to_ssa, not saved/emitted IR edits.
Older receipts below are preserved as history and superseded where stated.

OBSERVED FIRST WRITERS AND REPAIRS:
* _schedule_loop_callsites chained hierarchical ranks across a nested
  Conditional terminal return. It placed accept-tail PI before the no-save
  guard and then executed no-save PI too. Terminal-edge detection now includes
  nested Sequence/Conditional edges, not unrelated callee/nested-loop returns.
  Fresh 235-function real dt-step source compiled to LLVM O3 matches authored
  Python for rollback False/True, dt .01/.025, two repeated executions each.
  At the .01 case, next dt is .013027846332858448 and controller accumulator
  .029389333245105932; the wrong values were .013416407864998741 and
  .058778666490211864. Return, accumulator, state.x and pub_contract agree.
* materialize_program_abi_record_literals looked up raw SPAN member16 although
  the exact record-storage alias resolved it to resident1. It omitted the
  pub_contract member, leaving a private root input without a producer.
  Constructor members now consult the existing storage/concordance identity.
  Parameter SPAN descriptors also use exact formal storage, not a competing
  alias. Provisional returned-record projections now bind physical ownership,
  not merely shape/dtype accounting; unused conceptual containers are omitted.
  Fresh written-span source returns15,16,17 on repeated calls in C/LLVM/Fortran,
  matching authored Python, with no private root input and structural audit0.
* The scalar loop builder seeded an already-reserved body result a second time
  in the preheader and confused a reservation with a produced definition.
  The header Phi owns entry; reservation and produced-update sets are distinct.
  All six previously recorded loop-carried failures are now passing.
* _post_passthrough_rebinding rewrote exact operands/initial values but left
  dependent shared-planning aliases pointing at pruned output ids6656..6664.
  It now advances those aliases with the existing passthrough proof and book
  cells in the same transaction. Real-step structural findings9 ->0.
* _loop_carried_storage_aliases equated scalar LoopResult versions merely
  because they were connected. Storage identity now requires a dependency
  cone anchored in an actual IndexedStore; reverse adjacency/BFS is O(V+E).
  Scalar versions and unanchored cycles are not storage-equivalent.
* LLVM's reference-identity cone ignored the existing
  ssa_storage_identity_backedge proof on an unranked arena Phi. It allocated
  a one-double snapshot, then Jacobi row stores wrote beyond it. The proof
  now seeds the existing reference cone, including exit Phis. Ordinary scalar
  Phis retain snapshot semantics. Untouched HEAD8eb40eae reproduces the crash;
  all seven compiled-linalg tests, including native Jacobi eigh, now pass.
* Public entry returning a record now demands its declared scalar output
  fields through the existing activation-root and record ABI contracts,
  even if unread or only written in a dead branch. Internal helpers stay lazy.
  The old static_branch strict xfail is now a passing parity test, with an
  additional wholly-unmentioned field case. No generic whole-record projection.
* A first conditional field write's initial GetAttr reused its already-owned
  root ABI scalar slot. Region capture required an alias to another id and
  missed that exact ownership. It now accepts only exact formal, field,
  receiver and descriptor proof and consumes that projection without self-Cast.
  Four source/native field-reader variants and six regional proof tests pass.
* id_scale counted intentional bit61 MINTED ids as raw memory addresses.
  It now checks serial_of and rejects undeclared bits; flagged address-sized
  serials remain rejected. Three optional-record false failures are cleared.
* Python-shell trace test hardcoded id7 rather than the actual carried header
  Phi. It now follows the parameter binding and asserts the unchanged exact
  history2,1.9,1.805,1.71475. No widened numerical tolerance.
* Fortran uses its existing direct rank1 scalar indexed-store and reference
  mutation contracts. Returned inout spans retain caller storage even when
  their returned value is ignored. Source span parity covers all three backends.

FINISHED VERIFICATION (overlapping suites, do not add counts together):
* Final focused compiler/dt/concordance/native/Fortran gate:335 passed,
  0 failed,0 xfailed,178.78s. Includes fresh real dt-step source LLVM execution
  versus authored Python and a zero-structural-findings audit.
* LLVM backend/Python shell/loop-carried/optional-record gate:80 passed,
  3 existing skips,29.51s. The skips share the pre-existing disabled whole-XOR
  capture fixture (unbounded child-process hazard); they are NOT acceptance.
* Compiled linalg:7 passed,11.10s, including full native Jacobi.
* Earlier Fortran-focused gate:75 passed,1.70s.
* Graph-native reverse follow-up:4 passed,20 deselected,19.39s, covering
  ProcessGraph and AbstractTensor reverse acquisition, shared-use accumulation,
  and a real abstract_nn linear-loss adjoint compiled and executed natively.
* Fresh controller520rows/27functions, energy490rows/22functions,
  oscillator332rows/22functions: structural0 and unposted0 each.
  Unsourced facts642/272/232 remain OPEN respectively. Real-step structural0
  does not mean all minted identities/facts are fully posted/sourced.
* All failures encountered in these gates were followed up; baseline comparison
  used untouched C:/dev/tg-parity-base HEAD8eb40eae, never stash/reset.
  No new skips/xfails, no test deletion and no numerical tolerance relaxation.

CORRECTION / LIMITS:
The prior energy-presence buffer claim was too broad: passed private bool9497
through root19948, but callee formals9419/9420 have no operand uses in
_propose/_apply. That is not evidence of an actual uninitialized numerical read.
The untyped linalg fixture still has a pre-existing unused unaccounted formal
audit finding; numeric execution passing does not close that ABI diagnostic.
Full N4/8/all35 has NOT been rerun in this turn. Older saved N4 artifacts remain
untouched. Do not claim whole-program trajectory parity or "all bugs gone".
Source/SSA paging, ordinary-main parallelism and unsourced provenance debt remain
unfinished. The latest edits are in the canonical-source checkpoint; ignored
copied child worktrees are not automatically current.
User-run acceptance command (no timeout):
python -u tools/compiler_probes/profile_memory_n2.py 4 --backend llvm --save-ssa

2026-10-11 NUMERICAL PARITY FOLLOW-UP / THREE SOURCE FIXES (CHECKPOINTED):
User requested: "Okay well whatever you elected not to do because the issue is what it is please do now because you are tasked with helping me fix these"
The previous fresh N4 build passes emission, NOT numerical acceptance.
Independent authored Python/SymPy execution (native calls forbidden in the
oracle) previously demonstrated extra controller updates, stale elapsed-time
accumulation, and reads of private record storage without a producer. The
118 independent law-output checks were within 2.173e-16 relative; isolated
PI arithmetic matched Python. Do not mistake those local successes for N4.

Repairs now in canonical source, not a saved module or emitted IR:
1. control_source.overlay_scheduled_control materializes a regionless owner
   with declared children. Previously a class surface Sequence owned an
   effect-only conditional but was dropped, so both return calls executed.
2. _ControlSSABuilder._region_feed retains exact consumer operand identities
   while resolving planning aliases. total=tensor(0) is carried as graph3;
   its canonical resident2 is the literal0. Reading2 lost the header/latch
   generation and produced .01 twice instead of .01 then .02. Resolve3 by
   the existing lexical-read and loop-entry concordance; distinct generations
   cannot silently share one regional formal. Pre-loop held bindings remain
   separate, even when they share the original seed.
3. Fortran emits direct Store(value, scalar formal) as assignment through
   its existing reference ABI. Mutation propagates on existing call edges;
   scalar rank and input identity remain unchanged.

Real public-source compilation + native execution versus authored Python:
nine parameterized cases across C/LLVM/Fortran pass, including both distinct
guard arms, .01/.037/.007 accumulation, and a pre-loop held seed.
Final focused gate134 passed /1 existing strict xfail /0 failures,55.38s.
The xfail remains static_branch declared-but-unread scalar record output;
it is not new, not removed, and not silently treated as numerical success.
Controller/energy concordance audits before and after:520/488 rows,
0 structural findings,0 unposted ids;642/272 unsourced facts remain OPEN.

Wider census: six loop-carried producer tests and one native rejection-effect
test fail identically on untouched HEAD8eb40eae at C:/dev/tg-parity-base.
Two uniform-bound assertions also failed there: updated to check the existing
int64 range Cast consumes the original uniform (not identity relabeling).
No other tests deleted, weakened, skipped, or given a new tolerance.

STILL OPEN: actual N4 accept-tail PI call executes before its no-save guard
and the no-save PI then executes too. The precise first writer of that
placement is not yet trapped; do not assert that the newly fixed effect-only
guard explains it. The actual N4 coerce_metrics/private-frame pub_contract,
exchange-time and energy-presence slots have no root producer/initializer.
Their exact return/formal identity must be resolved, not zero-filled.
An existing real-step diagnostic (rollback=False literal, original source,
no IR edits) stopped after121.24s before linking on repeated unresolved
ssa_field_version for unresolved_report; it did not verify those defects.
A separate tiny Pair constructor slice refuses a right-field member binding;
it did not execute and is not evidence of the N4 private-buffer bug.
No fresh N4/8/all35 run was launched in this repair turn. Preserve these
negative receipts; source-and-SSA paging and ordinary-main parallelism remain
unfinished. Prior saved N4 and diagnostic artifacts remain untouched.

# Test baseline and hazards — read before running any test

2026-10-11 FRESH N4 SOURCE COMPILE VERIFIED:
User requested: "Nothing is useful to us if you fix it in ir, that is never okay, so please verify the source compile works"
Ran canonical source38c2da32 through unchanged public source compiler:
python -u tools/compiler_probes/profile_memory_n2.py 4 --backend llvm --save-ssa
TraceFalse, serial source lane, O3; no timeout, job termination, saved-IR
input, diagnostic transplantation, or post-lowering repair. All four stale
law pieces rebuilt against these compiler sources. Process34060 exited0.
PROBE_ELAPSED458.310s (7m38.3s including imports/profiling/stale leaf builds).
Combined source stage times total260.172s (4m20.2s); compile boundaries
178.7->443.5s (264.8s). Peak progress-sampled6143MiB; observed OS peak6.67GiB.
Source-generated output:
C:/dev/Powershell/turing/build/profile_rules_4_llvm_d96c6ueg
repository-ssa.pkl190157673 bytes; emitted LLVM4152280 bytes,
native step_3__dt_system_over.dll282624 bytes (plus import lib/debug symbols).
LLVM sha2560dcc132c36881f6e9d53bb927c82d462abb8fc6c29efbb085afbf529e72b3e5e;
996 functions. The native build refused no shortfalls.
Read-only audit of this FRESH public-lowerer result:14725 rows/996 functions/
0 structural findings,11.719s;60383 unsourced facts/468 unposted identities
still OPEN. No compiler editing or IR changes were used for this audit.
This supersedes the earlier "fresh N4 remains unverified" receipt below.
Four SOURCE-TO-NATIVE BUILD now passes. Numerical dt execution/parity was
not run; eight/all35 have not been tested on this source state.
Source-and-SSA paging and ordinary-main worker parallelism remain unfinished.

2026-10-11 N4 COMPLETE ERROR CENSUS / SOURCE FIXES (patch35):
Authorized N4 run at aa18da80 still failed:496.725s including four stale
leaf rebuilds; source-to-SSA completed. Saved190,213,415-byte repository-ssa.pkl
at build/profile_rules_4_llvm_wfdofixz. Complete emission census SIX:
four scalar Casts wrongly promoted to rank1 at pointer-kernel call settlement,
and two child-copy append int64 Phis seeded with default int32 constants.
Prior replacement Phis are absent from the complete census.
Source fixes keep shared kernel pointer transport distinct from actual
semantic rank; child-copy indices are int64, status int and live flags bool.
Also fixed authored local bool-to-double pointer input conversion (False
previously read garbage in both C/LLVM, reproduced on untouched tg-base).
All native conversions still mint/post through existing concordance causes.

Saved-module audit first crashed on Unresolved bindings. It now reports
unresolved captures without crashing; the binding-removal writer tombstones
owned capture claims with derived provenance. N4's five structural findings
were followed up: two cross-frame result ids misrecorded as local aliases,
two duplicate sequence-descriptor findings caused by live private extent
reads not following a proven handle alias, and one retired binding capture.
Existing retirement operators now use exact return edges/descriptor roles;
no name-based suppression. Native LLVM tests prove live extent rebinding.
Adjacent keyed tensor tests no longer select hardcoded region0 and require
exact row-base/shape/rank feeds; two-axis Indexed ports publish their rank
demand on the existing lookup shape concordance (real lowering gap fixed).

In-memory DIAGNOSTIC replay on the unchanged saved module:
one descriptor and two aliases retired; structural audit5->0 on14727 rows/
996 functions. Replayed prevention of four bad rank posts and regenerated
two child-copy helpers with the corrected source builder: LLVM shortfalls0,
complete native build succeeded. Final replay23.485s, artifact directory
build/n4_saved_ssa_repair_8_ww2w6k. This is NOT a fresh source compile or a
full dt execution/physics test. Eight/all35 were NOT started. Next fresh
user N4 command: python -u tools/compiler_probes/profile_memory_n2.py 4 --backend llvm --save-ssa
Use canonical repo; no trace, timeout, or job termination. Do not claim4 passed
end-to-end until that source run completes. Main parallelism/paging remain open.
Saved-book migration remains60507 unsourced facts/468 unposted minted ids,
latch OPEN; these were not hidden or relabeled as structural success.

Focused gate172 passed84.43s, zero failures/skips; final native live-extent
regression additionally1 passed1.47s. Cffi deprecation unsuppressed.
Canonical and both copied compilers/tests synchronized; unrelated files untouched.
Matching storage-effects products rebuilt and load-only verified:
live ecd4a6b2c54ed8b047a70f0fa79dbd3a,
benchmark70daf59e20ba7b1b94d27fcabfbacfdb,
repairdd182144082e22902246e76c5075b096.
Each1690 rows/188 functions/0 structural findings/0 unposted ids,
1116 unsourced facts OPEN.

Prompt history: "All errors need to be fixed"

2026-10-11 N4 EMISSION FOLLOW-UP (patch34 sequence_phi_span_cast_extents.patch):
Observed N4 first-five negative list: four replacement helper i64 Phis fed
by i32 zero constants, plus a planned-region span Cast without runtime extent
origin. Tiny exact replacement helper reproduced the width refusal before
editing; lower_sequence_replace now mints zero/one as int64, matching its
existing index/backedge/length cells. Status remains its existing int contract.
Native C and LLVM replacement tests include alias/nonalias, repeated calls,
empty/excess/negative length, corrupt source capacity and preserved failure contents.

Separate tiny nested-call span Cast reproduced the exact missing-extents
message. LLVM's existing origin walk now follows the exact same-rank numeric
Cast SSA operand edge, preserving precision-specific origins. Cast-only regions
join existing %extents propagation. No fabricated size or single-element fallback.
Native tests cover rank1/2/3, nested call depth1/2 and positive/negative2**40;
an actually unbound extent still refuses. 18 targeted cases passed34.92s;
combined UI/telemetry/sequence/merge/span/adjacent LLVM gate135 passed41.20s,
zero failures/skips. Cffi deprecation remains unsuppressed.

Untouched C:/dev/tg-base comparison: same sequence index/seed width mismatch
exists, but its old pointer-Phi emitter never validates pointee widths and
reports no replacement shortfall. Current gate exposes that latent SSA contract
defect. The separate span Cast repro DOES refuse on untouched baseline too.
This is not a claim the entire user's N4 program now emits successfully:
no saved SSA survived in profile_rules_4_llvm_3kkhydfy and no agent N4 rerun.
Next user-run should add --save-ssa; exception lists only first5 shortfalls.
Actual N4 graph-work receipt remains worker_threads0/admission_limit1 and
source_and_ssa_pagedFalse;2,203,705 book cells RAM/0 pages spilled. Those
architectural concerns are not solved by the display or these native repairs.

Both fixes installed in canonical live, benchmark and repair copies;1975
Python/contract files compared, only preserved repair N4 diagnostic hooks differ.
Storage-effects assets refreshed/load-only verified for final source:
live e18005a989ae4399ef836f1ede43c0e3, benchmark d786ca7bc17cd9f79de4a614ea340d51,
repair fc187adb43eeb801700c4eff2c855f06;1690 rows/188 functions/0 structural
findings and0 unposted IDs,1116 unsourced facts OPEN in each.


2026-10-11 ONE-OWNER TERMINAL DISPLAY (patch33 compiler_live_display.patch):
Auto uses installed Rich14.3.3 Live; optional progress_renderer=auto/rich/tqdm
or TURING_COMPILE_RENDERER. No dependency installed; no Rich import headless.
One redraw owner composes timing table, stage status and measured work bars;
logs/stdout/stderr scroll above it. Existing heartbeat, no second refresh
thread. Resize collapses rows/compact view; active stage survives20x6. All
timings/counters remain in the existing summary even when rows are hidden.
Ctrl+C and drawing faults restore streams; failures degrade to plain logs.
N4 probe heartbeat now uses the active renderer instead of bypassing through
sys.__stdout__. Installed in canonical live and both existing copied compilers.
Baseline60 passed2.88s; final UI/telemetry/fixed-point gate73 passed3.67s;
broader UI+pythonic-control+scalar gate127 passed254.44s. Cffi warning retained.
VT hide/clear/restore and resize tested explicitly (command harness TERM=dumb
otherwise disables cursor control). No interactive Windows screen capture.
Matching storage-effects assets refreshed/load-only verified: live066b79899d6a71495dd9a34c5f59669a,
benchmark0930a4b9fc7e9cf642990f12012aefb1, repair2e13a919a0842dbc23b89e3fdda6c667.
Each audit1690 rows/188 functions/0 structural findings;1116 unsourced facts OPEN.
User N4 failure subsequently received: combined source lowering350.81s, full
probe658.981s, peak6.11GB; emission refuses four sequence-replace Phi widths
and a span Cast without extents. Error prints FIRST FIVE shortfalls only;
not proof there are only five. No saved SSA in the named build directory.
That native failure is separate from successful display/telemetry verification.


LOCAL GIT INTEGRATION COMPLETE: Turing main compiler commit a4548f89 contains
the preserved compiler foundations,32 repair/probe patches, tests and profiling.
Root coordination map committed9fc96c9; shared guestbook/todo initial receipt
committed9f424ba. No remote push. Unrelated pre-existing edits/artifacts remain
unstaged. New Turing worktrees must use its own main, not the parent's branch.
Matching native storage-effects assets also refreshed/load-only verified in
benchmark2bffc361c6871181b735265c159e5a7e and repair9df7d0019cc11278dabb98ecc246b46e;
all three have the same1690/188 audit and1116 unsourced facts OPEN.

FINAL VERIFICATION (2026-10-11): combined affected+telemetry gate653 passed
361.18s, zero failures/skips/exclusions; final profiling-only gate60 passed
3.77s (includes two tests added after combined collection). Third-party cffi
imp deprecation warning remains unsuppressed. Native storage-effects unit
refreshed/load-only verified: dcf60218adf76c0b6f38b40d6682c3f7,1690 rows,
188 functions,0 structural findings/0 unposted IDs;1116 unsourced FACTS OPEN.
32 probe_job_observation.patch removes a stale hardcoded concurrent-main claim:
null means not measured, not a claim that any main job exists or is absent.
1973 Python/contract files compared against both copies: compiler sources,
tests and contracts agree. Only repair probe_four_rule_timing.py differs with
its preserved --capture-ssa exact598 diagnostics and --piece-cache option.
No changes were dropped to force those diagnostic probes byte-identical.
The pending combined-gate sentence below records an earlier point in this turn.

2026-10-11 CANONICAL INTEGRATION / PROFILING UPDATE:

Canonical compiler: C:/dev/Powershell/turing (its own main repository).
All29 saved repair patches,30 four_rule_copier_options.patch, and31
compiler_profile_summary.patch are installed here and in the benchmark copy
C:/Users/alber/.codex/worktrees/native-fallback-link/Powershell/turing
and its build/n4-repair/turing copy. Those copies are ignored children of a
root-only coordination worktree, NOT Turing Git branches. Parent merging
cannot transfer them. Existing compiler foundation edits were preserved.
Prior claims below that live source is untouched/unapplied are historical.

Before telemetry changes: live affected gate595 passed335.79s without exclusions.
Profiler/progress/fixed-point gate60 passed3.77s. Combined gate rerun pending
at this receipt; latest completion will be appended before handoff.
Seven live concordance audits:0 structural findings/0 unposted minted IDs;
unsourced fact latch remains OPEN392/759/272/642/533/20/230.

Public lower_ast_source_to_ssa now retains module.metadata['compile_profile']
by default even headless; compile_profile=False disables headless collection.
The TTY display has scrolling logs, persistent stage wall/RSS table and work
counters below. Stage order is not a percent, fixed-point safety limits are
not work totals, and iterator counters advance after processing. Known totals
show completed work fractions; unknown totals have counts, no invented ETA.
Timings are source-to-SSA only, NOT emission/build. Partitioned compilation
keeps its existing manifest unit timings. Planner aggregates retain existing
phase durations and slowest function context without per-dispatch graph copies.
Use src.compiler.shell_telemetry.compile_profile_dataframe(module) afterwards;
TelemetryChannel.profile_dataframe() and DeploymentProfiler.summary_dataframe()
are lazy pandas views. Nested/inclusive timing rows are not additive walltime.
No new dependencies, no tracemalloc, no agent N4/main run, no killed processes.

Recovery: build/integration_recovery_20261011_29/manifest.json and original
affected files; combined29 applicability artifact build/compiler_integration_all.patch.
Remaining: user fresh N4 LLVM run; no whole-main speedup/parallel residency
claim, and the OPEN provenance writers remain work. Native whole copier is
optional; the original Python copier and faster serial route remain available.


2026-10-11 NEGATIVE-MARK FOLLOW-UP / MERGE READINESS:

Current verified repair set supersedes the process/test status in older receipts.
Repair copy: benchmark worktree build/n4-repair/turing. Combined regression gate
595 passed in 325.72s, zero failures/skips/exclusions. Includes all precompile,
dispatcher, compilation-unit, loop-composer, shell-reference and reducer tests,
native records/collections/copier contracts and C/LLVM/Fortran torture cases.
The Fortran shell file's whole-object conditional regression is included; this
is not a claim that every test in that file or the entire repository was run.

After installation into the ACTUAL benchmark copy, focused verification:
352 passed in 127.52s, zero failures/skips; all newly fixed native cases included.
21 installed files match repair source after normalizing line endings/terminal
whitespace. Patch28 negative_contract_followup.patch applies cleanly before
installation and reverse-checks afterwards. Patch29 rule_probe_cli.patch
reverse-checks the already installed N/all CLI. Prior27 remain installed.
Live C:/dev/Powershell/turing compiler source was NOT edited by this repair.

Obsolete assertions/fixtures were modernized, not used to excuse failures:
scoped synthetic graphs get existing ingestion/value declarations; independent
programs get separate identity books; output assertions use current exact
identity/storage contracts; deprecated precompile-only checks become public
whole-program lowering; mutation bootstrap checks actual authored occurrences.
All four formerly excluded dispatcher tests now pass. Full precompile/
dispatcher gate is139 green; the eight loop-composer negatives and adjacent
reference/reducer negatives have all been resolved in the combined gate.

Observed production faults fixed at their writers:
* Carried initial resolution followed its own future loop output alias.
* ParallelDeployment field traversal treated ControlBlock lanes as .body owners.
* AugAssign's Store-context target was omitted from implicit-read discovery:
  flags was minted inside an arm and discarded. Entry snapshots now include it.
* Unary-plus retirement dropped the return expression's already-posted identity.
* Static selected aggregate members were classified dynamic despite literal ids.
* Generator extend forgot its exact LoopIterationOutput materializer route.
* Deferred recursive child construction consulted an empty function catalogue.
* Lexically separate Visitor classes were keyed by bare class name.
* Unknown loop-exit field VALUE discarded known effect ordering; exact recorded
  body writes/nested exits now join the construct before its later field read.
* Unrolled tuple materialization elements were mistaken for constructor args;
  empty materializations counted their seed as an element.
* Alias folding deduplicated ordered binding versions and selected another arm
  as the final return. Histories now retain their exact ordered versions.
The last two faults reproduce on untouched C:/dev/tg-base: the two-element
generator leaves len's operand18 undefined and root returns old total7 instead
of selected result19. All four generator variants now execute exact on all
three native backends (12 tests), including repeated calls.

Audit contracts retain declared helper ABI values rather than replacing them
with root-only values. The top-level audit uses the actual five-argument
proposal API and an authored compiled distribution callable. Exact address/
cast/projection writers use the incumbent Novel/mint_ssa_value posts.
All seven concordance cases: zero structural findings, zero unposted minted
identities. Rows/functions: view493/24; top780/45; energy488/22; controller520/27;
untyped520/27; mapping27/2; oscillator334/22. UNSOURCED FACT LATCH REMAINS OPEN:
392/759/272/642/533/20/230 facts respectively. These are outstanding provenance
writer migrations, not eliminated or renamed as passing tests. The only pytest
warning is third-party cffi's deprecated imp import; it was not suppressed.

Patch27 already installed before this follow-up: scalar Phi physical types,
integer loop seeds/induction casts, Fortran API shape/extent handling, native
tuple truth and per-procedure IEEE imports. Actual Fortran steady numeric/row
micro .577us vs122.4us before the IEEE import correction; NOT a four-law speed
claim. The user's5m28s/6.11GB LLVM Phi failure is not a successful artifact.
No new N4/full-main was launched by the agent. Complete N4 parity/timing and
main source/SSA parallelism/residency tiling remain unverified/unfinished.

Existing sanctioned storage-effects builder refreshed the benchmark asset:
products/f8aec357b935ef27baf0ffc3ac7b1840/compiler_storage_effects__read_only_storage_effects.dll.
Load-only validation succeeded. Its own source audit1690rows/188functions,
zero findings/unposted ids;1116 unsourced facts/latch OPEN. No stale-fingerprint
bypass, whole-copier replacement, packages or other Python compiler.

Git integration detail verified: live turing is its own repo on main, with
existing dirty work preserved. The benchmark turing directory is an UNTRACKED
COPY beneath the parent coordination worktree (detached HEAD); Git reports
the parent Powershell root, NOT a compiler branch. A parent-worktree merge
will not transfer this compiler. Integrate the preserved ordered patches into
turing's own repository; do not force-add the ignored child to the parent.
No merge/commit/stash/reset was performed. Tested repairs are ready for that
integration, not proof of a completely green historical repository manifest.

2026-10-11 PYTHONIC CONTROL TORTURE — VERIFIED REPAIRS IN BENCHMARK WORKTREE:

Latest state supersedes the historical active-process/"no changes yet" receipts
below. No main or N4 is running. User requested a small program that breaks the
compiler, followed by iteration. Seven ordinary authored programs all FAIL on
untouched C:/dev/tg-base (7 failed/24.38s), by wrong native results or an unnamed
index-read formal. The repaired corpus passes both OUR C and LLVM backends:
14 tests/58.96s, 60 native invocations, same artifacts/buffers reused. No trace,
process deadline, interpreter fallback, package installation or alternate
Python compiler. Inputs separate population from the surrounding guard; cases
include empty/1/2/17 rows, nonempty literal contents, literal birth in a while,
enumerate, nested guards, break, and starred tuple cardinality. Expected values
are exact cardinality/effect arithmetic, not the compiler interpreter lane.

Observed causal chains and fixes:
* Patch22 newly consumed redundant while->for and while->if->for ownership
  edges, inserting the same source loop in both scopes. In-memory removal of
  patch22's overlay logic restores zero-loop old output. Transitive reduction
  preserves containment reachability and gives the retained loop one owner.
* List->literal edges survive ingestion, but ordinary resident list birth
  supplied only zero-length initialization. Existing lexical sequence effects
  now publish nonempty literal rows at their exact AST construction sites.
  Joined-byte and accepted opaque-object materializers retain their routes.
* All loop mutations were discarded as unbounded. Interval transfer over
  existing Control IR joins branches by max, multiplies bounded-loop growth,
  and proves single-entry while bodies from terminal edges. Unknown backedges
  or mutated extents remain unknown. A million-trip nested-loop bound is
  computed without unrolling. Existing capacity metadata/storage requirements
  allocate the resulting spans; no guessed capacity or new container runtime.
* enumerate was mistakenly counted as an escaping/mutating call, suppressing
  the two/17-row prefix capacity. Exact builtins.enumerate identity now uses
  the incumbent nonmutating builtin classification.
* A scalar list's appended-element index read was excluded by the multi-column
  row collector; the numeric plan retained only its feed. Existing row-load
  lowering now admits column zero and restores an otherwise-unowned graph read
  at its first consuming region, without reloading region-owned definitions.
* LLVM root-frame capacity was uninitialized even when the arena was allocated.
  Its wrapper now publishes the actual allocation capacity for private FIXED
  sequences. Borrowed and dynamic pointer-cell contracts are untouched.
* The earlier tuple-splice repair derives prefix length + row count from exact
  graph operands through existing extent/arithmetic lowering and namespace IDs.

An extra empty-list clear initially broke the existing sequence-truth prefix
test; removed the redundant entry-birth effect, keeping the incumbent empty
initializer. Regression was NOT excluded. Final repair-copy gate:127 passed,
160.26s, including native collections, source contexts, control compartments,
sequence truth/snapshot/replacement and owned-heap tests. Ordinary view/controller
audits agree before/after:493/24/0 and522/27/1, including the SAME restore-type
operand15 finding; unsourced latch remains OPEN. New torture tests also assert
zero structural findings, in addition to native execution assertions.

Applied verified core changes and four new test files to the ACTUAL benchmark
worktree C:/Users/alber/.codex/worktrees/native-fallback-link/Powershell/turing
after verification in build/n4-repair/turing. Live C:/dev/Powershell/turing
compiler source remains untouched. Patch26 tools/compiler_probes/
pythonic_control_torture.patch preserves the whole delta after existing25;
reverse applicability in the benchmark worktree is checked. Ordered26 live
apply-check passed before the final LLVM/test additions; repeat final check.
Compiler fingerprints invalidate previous accelerator assets normally: refresh
through existing sanctioned builders before another N4. No whole-N4 success,
main speedup, memory tiling, main parallelism or general-Python support claim.

2026-10-10 USER-AUTHORIZED ITERATIVE FOUR-LAW REPAIR:

User explicitly requested diagnosis/iteration until4 laws work, then the full
available law set. Subsequently ordered all Python except newest run killed.
Verified exact process identity, stopped old main33064/7196 and native N4
23236/14620; only newest diagnostic launcher/worker remained. Native N4 is
USER-TERMINATED, not completed; no final copier counters or verdict. No files
removed. Separate exact-source repair copy under snapshot
build/n4-repair/turing protects the prior compiler snapshot and preserves25
promotion patches. Four existing original-copier law assets reused for diagnosis.
Initial graph-checkpoint hooks failed after100.9899s (local callback pickle)
and77.5540s (write-only stdout pickle); these are DIAGNOSTIC TOOL failures,
not compiler outcomes. Corrected hook prints source rows without serializing
the whole graph, then captures final SSA without aborting compilation if a
checkpoint fails. Current diagnostic session78998; no deadline/trace.
No compiler implementation changed yet. Small tuple-splice controller loop
repro exposes unnamed-formal admission failure; relation to N4 remains unproven.

2026-10-10 USER-REQUESTED N4 — ORIGINAL COPIER FAILED, NATIVE STILL RUNNING:

Original-copier comparator session9750 exit1 after561.4117003s (9m21.41s).
All4 laws built by175.6213412s; combined source/SSA385.7903545s total,
begin->SSA359.9601409s. Native storage17calls/143271facts/.6692912s including
transfers. Native copier absent, original route explicitly verified; SAME
snapshot compiler fixes retained. No full-system native artifact or emitted-C
hash available: C emission refuses two extent findings on %t598 plus two
downstream unavailable operands in step_3__step_with_dt_control_used specialized
body. Both missing operands have an EXACT producer Call(%t598), respectively
if_true.15 and if_merge.15, consumed by loop_header / loop_header.1. No matching
formals; diagnostic provenance {}. These producer instructions exist, so this
is NOT evidence those two values were dropped from SSA. Unavailable extent
emission can leave its results unbound; causal writer chain remains unobserved.
Do not call it a copier fault or prove historical content loss from this raise.

Read-only comparison of the4 current law LLVM files: function counts Python /
native are6/6(properties),13/13(inverse),71/71(stage0),38/38(stage1); instruction
opcode counts agree exactly for each. Properties LLVM SHA agrees; other3 differ
(sampled inverse diff includes differing SSA numbering/placement). Counts are
NOT semantic parity. Both initial combined graphs18829nodes/21599edges.
Original copier is unchanged; generic lowering fixes remain independently of
provider selection. Earlier successful507.330979s N4 is NOT a direct successful
timing baseline for today's failed561.4117003s. Historical dropped content is
possible but unproven; no old-vs-current producer/consumer/concordance replay yet.
Native law-build milestone1229.3s vs175.6s original (~7x) is observed, NOT a full
compile speed ratio or precise boundary/kernel attribution. Native session52547
continues; worker14620 CPU increased1703.27->1762.27s during observation, RSS
~4556MiB at later sample. No intervention/restart/timeout/compiler edits. Final
native counters/outcome/peak remain pending; main33064/7196 left untouched.

SECOND comparator explicitly authorized and launched22:28:25 America/Chicago:
python -u tools/compiler_probes/probe_four_rule_timing.py --graph-workers 1 --fresh-pieces
Parent11844/worker32076, session9750; SAME current snapshot compiler/fixes and
storage unit d5d7dc1bdd73dace2db368d4e04801f6, original Python copier confirmed.
Output build/profile_memory_n4_native_storage_tixr18_u/profile_memory_n2;
independent empty law_cache forces all4 builds and avoids cross-run cache writes.
Both N4s and earlier main initially left running; no trace/deadline/intervention. Concurrent
wall times are not isolated estimates. All4 original-copier laws built by+175.6s;
combined source graph18829nodes/21599edges, SSA starts+412.3s. Native laws complete
+1229.3s; same combined graph18829/21599 observed, still deployment specialization
at+1554.1s. Native final time/artifacts/retention comparison remain PENDING;
original terminal outcome is recorded above.

User requested afterwards: reason about previously dropped content now retained,
and consider original copier with native-enabling compiler fixes kept. Do not
discard those fixes, silently disable native route, or claim content parity from
matching initial graph counts alone. User asked observation only while runs proceed.

Initial explicit user request: "so can we do a 4 rule test". Launched snapshot
python -u tools/compiler_probes/probe_four_rule_timing.py --graph-workers 1 --native-copier
at22:09:37 America/Chicago. Parent23236/worker14620, execution session52547.
NODUS_TENSOR_CORE=live nodus build/Release/nodus_tensor_core.dll;
TURING_PROBE_EXTERNAL_ROOT=C:/dev/Powershell/turing. Existing main33064/7196
untouched. No trace, deadline, orbital target execution or live-source edits.
Serial source route/one graph worker/eight resident regions, matching prior N4.
Native storage unit refreshed through OUR compiler BEFORE timed submission:
d5d7dc1bdd73dace2db368d4e04801f6. Copier LOAD ONLY0ac1ded2afe6cb7887bf3b2eb1fb44b2.
Fresh output build/profile_memory_n4_native_storage_l7gvdp8m/profile_memory_n2.
Probe counts real native batches separately from pre-execution subclass fallback
roots/batches, with boundary-inclusive copier time; no per-item native tracing.

First law rebuilt; first batch1847roots native1/fallback0 in1.6493831s. Native
storage first pass594facts/.0014769s. Inverse law reached source catalogue
function extraction at~44s. It ended+353.2s:7406nodes/8894edges,308.985s for
extract-function-subgraph. At+431.1s inverse graph1494nodes/2936edges published
6of8 regions, one-worker residency6/limit8. NOT a completed N4, final timings,
native batch totals or peak-memory result. This is large-source evidence of
substantial extraction cost; precise copier attribution awaits final counters.


2026-10-10 WHOLE original metadata copier executes natively — SNAPSHOT ONLY:

Patch25 tools/compiler_probes/native_whole_metadata_copy.patch UNAPPLIED after24.
Ordered concatenated25 git apply --check0 against live dirty checkout. This is
applicability, NOT promotion. Live compiler/source untouched. Existing main
33064/7196 (Oct9 18:25:20 local launch) untouched; no main/N4 launched, timeout,
kill, package installation or alternative Python compiler this turn.

The exact50 authored lines of isolate_metadata are taken from production AST,
wrapped once as metadata_copy_program(value, metadata_memo), submitted through
lower_ast_source_to_ssa under existing full-native/no-callback contract. Entire
source lowers/emits complete LLVM (2.722457s preliminary emission probe). Native
execution proves dict/list/tuple/set/frozenset, leaves, shared containers, self
cycle and tuple/list cycle. One outer invocation; recursion, memo, iteration,
construction, updates and identity reduction native. Original copier unchanged.

Observed losing writers and connected fixes:
* Self PlanCall has exact result bindings but no child activation shell. ABI
  settlement skipped it; it now consumes the existing recursive_scc_backedge
  receipt for the SAME function and graph, not an invented clone/signature.
* Returning Dict3 born atline4 was placed before line5 update. Actual return
  construct cell/span join, not returned container birth, owns the return edge.
* Spanless sibling comprehension signatures collided; source nesting now uses
  exact AST identity, restricting signature fallback to located nodes.
* memo.get lost valid node0 through operand-or-value truthiness; exact key lookup
  retains0. Shared storage contract and private recursive guard remain intact.
* Existing lexical stores40/85/92 replaced their source numerical work, but
  scheduled markers3/5/7 survived as undefined region calls. Only regions whose
  complete deployment node sets are installed effects retire; mixed regions stay.
* Overlay emitted list/tuple/set/frozen controls BEFORE memo guard38. Guard was
  present atroot[4], not missing. Shared list handle3 copied to8 then11. Source
  SOURCE_SPAN rows prove guard/body ordinal2, dict3, list4, tuple5, set6, frozen7.
  Conditional-only sibling runs now consume those exact concordance-owned AST
  paths/ordinals; no coordinates/name proxy, no crossing producer/loop/effect.
* Recovered contains37 consumed Cast36. Recovery ignored other recovered uses,
  then reused a stale instruction offset after removing Cast36, moving it after
  contains. Real consumer instruction identity now anchors the dependency move.

Connected operations reuse existing context/sequence storage and accepted
native_object_source facts. List extend, set update, frozenset materialization
are native helpers. Exact all(new is old for zip(...)) uses a scalar intrinsic
identity reduction, not a Boolean collection arena/general Python generator.
The existing4097-row bulk materializer test caught an unwanted member-set
dependency introduced during this batch; separate accepted frozen helper fixed
it without weakening the original bulk-copy guard. No new SSA opcode, object
registry, Python callback or generic Python runtime introduced.

Reusable deployment uses existing compiler accelerator registry and checkpoint
store. compile_and_register_metadata_copy_accelerator builds through OUR entry;
load_metadata_copy_accelerator never lowers/emits/builds. Fingerprint/platform,
source identity and BOTH library digests checked. Missing/stale assets stay
serial; matching corrupt assets refuse. Explicit load/register activates native
extraction; no automatic compile on import. Graph AND node metadata share one
native batch and memo. Memo capacity bounded by imported identity count; context
births/growth use incumbent owner/release. Builtin subclasses take original
serial copier BEFORE native execution; execution faults never silently retry.
Boundary ABI checkpoint is not a duplicate ProcessGraph/provenance registry.
Matching asset: build/compiler_foundations/metadata_copy/products/
0ac1ded2afe6cb7887bf3b2eb1fb44b2/copier/metadata_copy_program__metadata_copy_program.dll.

FINAL gate after all source changes:132pass/4 same known stale exclusions in
125.67s; no new failure. Final fresh-process LOAD ONLY provider active during
whole-copier compilation/execution AND original clean-extraction/compartment
tests:12pass/30 unrelated deselected6.63s. Guestbook dry-run exit0, no --apply.

Gates:67pass71.76s before deployment;78pass/4 known stale dispatcher exclusions
66.40s before final source-path refinement. Final connected15pass18.73s after
source-path refinement. Native loaded real-extraction4pass0.68s and whole-copier
8pass5.92s in fresh process before final source-path refinement. Installation
test forbids lowering/emission/build during reload, forbids Python item fallback
on real cyclic graph+node batch, checks subclass admission and corrupted digest.
Seven ordinary audit structural counts unchanged493/24/0;360/36/1;488/22/0;
522/27/1;525/27/5;24/2/0;331/22/0, rerun after final source-path refinement.
Full copier structural audit0; provenance unsourced latch remains OPEN. Context
release helper emission reports no attached book, an existing helper limitation,
not closed provenance. Wider gate first hit documented stale scalar comparison
test (30pass/1fail19.15s); excluded same4 previously baselined fixture defects.

PERFORMANCE IS NOT WON. Whole-section medians including import/export/release:
solo64rows Python .111400ms / native2.740800ms (24.60x slower); prior concurrent
64rows .147000/3.546200ms. After final refinement under concurrent gates:
1/4/16/64 rows .007100/.012000/.040500/.152000ms Python versus
.413500/.539600/1.131100/3.410100ms native. Build4.739261s; prior matching
load-only .137912s. Five samples per mode/size, identical original source and
alias assertions. NOT N4/end-to-end speedup. Cold single-call profile at64rows
24,776calls/12ms includes6ms export plus import costs; not warm causal timing.
Native member get/set/memo remain linear scans, potentially quadratic. Native
temporaries still survive until context release and exporter reifies them all.
NEXT: warm separate boundary/kernel/export measurements; use existing graph
identity/observability contracts to eliminate unnecessary rehydration and resident
lookup work, not another dynamic Python runtime. Main parallelism and whole-source
memory tiling still unfinished. All earlier options/investments preserved.

Repros (snapshot, NODUS_TENSOR_CORE points at live nodus Release DLL):
python -u tools/compiler_probes/probe_native_metadata_copy.py --object-context
python -u tools/compiler_probes/probe_metadata_copy_timing.py
python -u -m pytest tests/test_copier_connected_contract.py -q --tb=short --show-capture=no -x


2026-10-10 paired generators consume concordance-owned representations — SNAPSHOT ONLY:

Patch24 tools/compiler_probes/native_paired_object_concordance.patch UNAPPLIED,
after23. Ordered24git apply --check0 on live dirty checkout; applicability is
NOT promotion. Live compiler unchanged. No main/N4 launch, timeout, process
kill, package, alternative Python compiler or rewrite of the production copier.

Exact tiny source: def root(value): return tuple((key,item) for key,item in value.items()).
Pre24 source-loader replay of all nine changed compiler modules refuses two
unnamed row-column formals (source6 and minted column1),0.913166s. This is an
actual public-source lowering refusal, not an API/import or test assertion gap.
Observed source graph retains tuple3 leaves1/0, loop8 targets key1/item0 and
LoopResult11 value3/control8. The object collector refused two targets and
the dictionary-items view; lower_loop10526-10551 then invented those columns.
Composer expanded_row_arguments kept only1/0 BEFORE object admission (no
accepted constructor cell then). Mutation placement used source spans; the
comprehension has no span, so tuple3's native constructor landed at entry,
reading loop-local1/0. These exact writer/consumer chains were observed inline.

The existing book now owns native_object_source representation acceptance,
DERIVED from incumbent producer, component, receiver and loop identity cells.
Source identity/edges are not duplicated into another registry. Publication
preserves the original producer-to-projected-columns transition on that same
page. Control carries Ref cells; SSA reads their latest facts before allocating,
unpacking, publishing or updating. Emission reads the module's ATTACHED book,
including exact planning aliases, and validates actual tuple component writes,
position casts, pair projections and collection/update operands. Missing or
superseded cells and copied projection changes refuse. Page lifecycle is
LOWERING->EMISSION, explicitly classified and spillable after its last reader.

Paired dictionary iteration reuses context member key/value columns and existing
collect/get helpers. Fixed tuple producers now store their admitted reference
components in existing context storage. Original consumer/update generator is
finite and materialized through the incumbent collection contract. Native UPDATE
walks that pair collection in one native call and uses incumbent native get/set;
no Python item callback. Keys still have declared reference-identity semantics,
NOT general Python equality/hash/custom items behavior. Existing heap owner and
release cover all births. The recursive storage guard is unchanged.

Birth loop ownership comes from exact aggregate-user/generator edges, recorded
in the book, not coordinates. Stable topological placement uses the existing
dependency-cone helper and exact scoped units. It also fixes linked-call case
publication12-before-calls5/6: calls -> tuple7 -> publication. It visits relevant
cones, not whole-graph scans per birth or a persistent graph/cache. Numeric
tuple-row expansion retains its old route; admitted object handles remain one
value. Position literals use explicit existing object_context_position casts;
raw opaque Const integers had been widened and propagated into helper formals.

Native tests execute (key,item), reversed pairs, nested tuple components and
dictionary update with linked identity calls at0/1/3/64entries. Shared leaf
identities, insertion order, imported backedges, fresh output containers, one
outer invocation and explicit release are checked. New7 plus lifecycle1:8pass
10.85s. Focused new/prior/lifecycle25pass31.68s before the final attached-book
alias reader adjustment. Broad74pass73.50s before final validation strengthening;
Final75pass77.19s after all changes, including attached-book aliases, tuple
field/projection validation and page lifecycle. Guestbook dry-run0, no --apply.
NODUS_TENSOR_CORE=C:/dev/Powershell/nodus/build/Release/nodus_tensor_core.dll.
python -m pytest tests/test_native_paired_object_generators.py tests/test_native_object_contract_batch.py tests/test_native_object_collection.py tests/test_regionless_loop_ownership.py tests/test_source_object_context.py tests/test_native_memo_membership.py tests/test_identity_book_spill.py::test_every_declared_page_is_classified_and_the_table_is_consistent -q --tb=short --show-capture=no -x
Before/after ordinary audits unchanged493/24/0;360/36/1;488/22/0;522/27/1;
525/27/5;24/2/0;331/22/0. Existing findings exit1; provenance latch OPEN.
Syntax10changed sources0; added patch whitespace0; ordered24check0.

Production50-line copier now clears row97 and refuses Boolean collection98,
1.808068s. Observed98=value26/control79/materializer80/iterable70. Node26 is
new is old; node70 is zip(isolated,value); node80 is its generator expression,
consumed by the source all predicate. NEXT: retain this as its intrinsic Boolean
reduction through existing zip/all identity programs, not a private bool arena.
Simpler recursive dict-copy variants still refuse private formal6 (without memo)
or8 (with memo),dtype=None/shape()/declared_workspace=False. Losing writer there
UNKNOWN; no native recursive paired-copy success is claimed. Do not weaken the
guard. No whole copier artifact/execution, N4 timing/speedup, hash acceleration,
whole-program memory tiling or main parallelism claim. Native get/set remain
linear scans; compiling them is NOT an asymptotic improvement.

2026-10-10 connected native object contract batch — SNAPSHOT ONLY:

Patch23 tools/compiler_probes/native_object_contract_batch.patch UNAPPLIED,
after patch22. All23 ordered patches pass combined git apply --check0 against
the live dirty checkout. This is applicability only, NOT promotion. Live source
unchanged. No main/N4 launch, deadline, process kill, package, alternate Python
compiler or rewrite of the production copier. Existing main processes left alone.

Observed ownership writers, not inferred storage guesses:
* Full copier call14 captures mutation40; existing value_aliases proves40->memo6.
  by_id(40) missed its descriptor. Binder now consumes settled exact_bindings
  and incumbent caller/callee aliases for ALL members of the shared descriptor.
  Its span/rank1 column accounting was legitimate, not corruption.
* Tiny memo IndexedStore was hoisted before its branch-local [] constructor.
  Existing keyed Input ABI now admits the SAME lexical mapping mutation path.
* Composed If12 owns Loop14. Call5 has both owners; old placement chose If12,
  outside the child Load in Loop14. Actual composed depth now selects the
  deepest declared owner, including conditional-inside-loop and reverse order.
* An early Function.args pruning slice dropped the root memo's unused-before-
  expansion value column despite its program_abi_parameter owner. Later missing-
  formal restoration stripped that owner; LLVM leased one stack i64 instead of
  the public arena. First native recursion crashed with access violation.
  Watched argument removal/accounting writes identify the exact chain. Pruning
  now preserves existing ProgramABI-owned members until call expansion; native
  tests assert every descriptor cell is a real public buffer. No storage guard
  was weakened and no opaque reference was cast into an address.

Optional builtin type evidence extends SSAObjectContext's SAME identity table
with a third int64 column, one bit per explicitly admitted builtin guard.
Default two-column ABI/mapping unchanged. Boundary stamps once; native births
stamp matching evidence; source builtins.isinstance consumes existing intrinsic,
actual static class objects and exact operand edge. Ordinary Call/And/Ne SSA
reads this evidence natively; no per-element Python callback or separate table.
New helper interface TYPE_MASK is not a new SSA opcode. This is NOT arbitrary
Python metaclass/MRO behavior. Scalar bool/int guards and opaque tuple-subclass
type guards are tested without claiming the subclass has native member rows.

Payload requirements derive from incumbent IRModule.reachable_functions and
live helper/source operations. Concordance/backend validate exact requirements
and type predicate helper/mask/operand/context receipts. Boundary refuses
opaque container subclasses if a reachable pocket requires member storage;
type evidence alone never authorizes payload access. Native release clears
the third column through the existing shared heap allocation/release contract.

Final batch17pass20.66s, with recursive assertions checking exact memo keys,
exported [] values and one outer invocation. Earlier17pass22.08s preceded those
stronger assertions. Syntax10changed sources0; added patch whitespace0; patch's
205-line new test exactly matches snapshot. Guestbook dry-run0, no --apply.
Broad92pass75.09s before that test-only strengthening and final graph-receipt
unit addition. Command:
python -m pytest tests/test_native_object_contract_batch.py tests/test_native_memo_membership.py tests/test_native_object_collection.py tests/test_native_object_boundary.py tests/test_source_object_context.py tests/test_regionless_loop_ownership.py tests/test_linked_reference_kind_folding.py -q --tb=short --show-capture=no -x
Focused command: python -m pytest tests/test_native_object_contract_batch.py -q --tb=short --show-capture=no -x
NODUS_TENSOR_CORE=C:/dev/Powershell/nodus/build/Release/nodus_tensor_core.dll.
Native recursive fixtures execute depths0/1/7/32 with BOTH isinstance(tuple)
and len(item)>0 guards, shared caller-owned memo and no callbacks. Public-source
pre23 source-loader replay (FCS/precompile/resident modules, old default context)
refuses private opaque memo column at recursive call5 in0.885788s. This is an
actual lowering refusal, not a missing API or obsolete assertion baseline.
Before/after ordinary audit outcomes unchanged493/24/0;360/36/1;488/22/0;
522/27/1;525/27/5;24/2/0;331/22/0. Exit1 existing findings; provenance latch OPEN.

Unchanged50-line production copier STILL REFUSES call14: collection97 unknown
scalar row,2.311162s. Its observed graph says LoopResult/collection, loop75,
materializer76, aggregate tuple leaves14/15, iterable19. It is a paired generator
row (dict items/update), not the now-bound memo6 column. No whole copier artifact
or native execution, N4 speedup, asymptotic lookup, main tiling or parallelism
claim. Next connected group: pair-row ownership/projection, paired iteration,
mapping update and sequence extend, set/frozenset materialization, zip/all and
optional memo.get. Keep known-type/extent and recursive allocation guards.
Static tuple type targets in source currently lose their class-object receipt
upstream: (int,tuple) becomes an empty structural aggregate. Type helper admits
a real existing tuple receipt and declines empty/missing evidence; whole-source
union guard remains OPEN, not a claimed native pass. No source-name guessing.

2026-10-10 region-less loops consume declared control ownership — SNAPSHOT ONLY:

Patch22 tools/compiler_probes/regionless_loop_ownership.patch UNAPPLIED.
All22 ordered patches pass combined git apply --check0. Live compiler unchanged.
No main/N4, timeout, process kill, package, alternative compiler or source-copier rewrite.

Observed writer: _source_control_nesting_hints supplies {2:{0}} for the tiny
visit graph: catalogue conditional2/source11 owns loop0/source13. Exact AST
object13 is inside If11.body, not orelse. _overlay_control_or_require_subdivision
passes that edge into overlay_scheduled_control. nested_root ignores region-less
loops because _anchored_control admits only conditionals; the root loop list
then emits loop13 independently. Output Loop13/If11 are siblings. Later call3
is in If11 while member Load0 is in Loop13, inventing recursive formal0.

Overlay now CONSUMES that existing declared edge for a region-less loop.
The same graph source objects choose the exact arm/statement slot; incumbent
dispatch deployment nodes place it among existing arm statements. Missing or
cloned identity and a region crossing that slot refuse; no source-coordinate
or dtype guess. Nested children cannot also be emitted at root. Zero-region
parents retain composed children. Existing control_block_placement publication
and concordance reused; no new ABI/schema/registry/opcode or allocation change.

New9tests plus linked-reference12:21pass8.71s. Native conditional tuple generator
with a linked identity function executes both body/orelse variants at lengths
0/1/17/257, both predicate outcomes. One outer invocation, leaf identities
preserved, no new object when the arm is unselected, one native birth when
selected, explicit native release. Both source modules have zero concordance
findings. Same public-source native test on exact pre22 sources refuses an
unaccounted loop-target formal1 (4.29s), not a syntax/API-only baseline failure.
Broader gate75pass70.22s (before the final two negative tests were added):
python -m pytest tests/test_control_source.py tests/test_overlay_scope_refusal.py tests/test_regionless_loop_ownership.py tests/test_native_object_collection.py tests/test_source_object_context.py tests/test_native_memo_membership.py -q --tb=short --show-capture=no -x
Final21case command:
python -m pytest tests/test_regionless_loop_ownership.py tests/test_linked_reference_kind_folding.py -q --tb=short --show-capture=no -x
NODUS_TENSOR_CORE=C:/dev/Powershell/nodus/build/Release/nodus_tensor_core.dll.
Before/after seven ordinary audits unchanged493/24/0;360/36/1;488/22/0;522/27/1;
525/27/5;24/2/0;331/22/0. Exit1 existing findings, provenance latch OPEN.
Syntax5files0; added patch whitespace0; combined22apply-check0.

Tiny recursive tuple repro no longer refuses private target0. It reaches the
full-native gate and refuses undefined isinstance predicate10 instead; graph
preservation is still NOT native recursive execution success. Full unchanged
50-line copier lowering still REFUSES at callsite14,4.570708s; no artifact or
speedup. Exception-frame observation identifies its current opaque formal as
compiler_frame_sequence_id6/member1, program_abi_parameter=metadata_memo,
sequence_arena=True, program_abi_storage=span, rank1, shape(), provenance
exact_call_argument_sequence_member. It is a memo column, not a fresh opaque
object. The writer losing its shared keyed binding is NOT yet observed.
NEXT: trace that memo-column owner/binding at the writer; retain the recursive
storage guard. Runtime isinstance admission is also unfinished; existing KIND
helper is closed-vocabulary, not arbitrary Python MRO/metaclass behavior.
No full copier execution, main speedup, hash indexing, memory tiling or parallelism claim.

2026-10-10 linked reference facts preserve dynamic type control — SNAPSHOT ONLY:

User: extend the EXISTING contract with details needed by interior native
elements; proceed carefully, future-proof, in elegant steps toward running.
Patch21 tools/compiler_probes/linked_reference_kind_fact.patch UNAPPLIED.
Twenty-one ordered patches pass combined git apply --check0; live compiler
unchanged. One source change in glsl_deployment_strategy.py plus new tests.
No main/N4, deadline, process kill, package, alternative compiler or copier rewrite.

Actual observed cause chain in an11-line public-entry recursive tuple repro:
1. ABI fixed point admits visit.item Input1 and child Input0 as
   storage=reference,dtype=opaque_ref,rank0,shape(). Collection17 initially
   retains parents3(value)/13(control), materializer15, target0 and iterable1.
2. _fold_callsite_structural_values.evaluate(Input) ignored the linked receipt
   when parameter_value_abi was absent. Its tensor fallback RETURNED
   _ProgramABIValueFact(python_type='opaque_ref',storage='scalar',dtype='opaque_ref').
3. evaluate(isinstance(item,tuple)) RETURNED False against that fabricated fact.
   Conditional pruning at22538 removed the body nodes. Instrumented parents
   assignment caught _set_operands/remove_node deleting Collection17's value
   edge after its control edge was removed; its parents became[].
4. source_object_collections then returned{}; lowering used tensor extent on
   the opaque input and mixed i32 counters with i64 backedges. This was NOT
   merely missing batch layout: structural folding had deleted live source.

Fix consumes the SAME linked reference ABI at the Input evaluator BEFORE its
weaker tensor fallback. Existing python_type is retained when present; unknown
dynamic type stays unknown, not the spelling of physical dtype. Existing
_ProgramABIValueFact and positive reference/subtype rules reused. No new
contract/schema/page/opcode; existing structural-fold concordance owns the fact.

New12cases2.64s: five builtin target kinds remain runtime, exact value identity
distinct from node key, positive declared type, generic base not negative proof,
numeric scalar evidence unchanged, weaker tensor hint cannot override reference,
forwarded native id executes1/17/2**60+3 with zero concordance findings, and
recursive collection retains exact value/control edges through ABI settlement.
The final test proves graph preservation ONLY; its existing recursive-frame
refusal is NOT counted as native execution success. Exact pre21 replay of the
first regression fails Constant versus Call1.63s. Broad58pass58.55s (before the
12th graph-preservation test was added); then final new12pass. Command:
python -m pytest tests/test_linked_reference_kind_folding.py tests/test_native_memo_membership.py tests/test_native_reference_boundary.py tests/test_native_object_collection.py tests/test_source_object_context.py -q --tb=short --show-capture=no -x
NODUS_TENSOR_CORE=C:/dev/Powershell/nodus/build/Release/nodus_tensor_core.dll.
Additional hierarchical gate20pass then old _sequence_concat_ops two-result
unpack fails1.68s; exact pre21 replay SAME1.56s. No relaxation or unrelated fix.
Before/after seven audits unchanged493/24/0;360/36/1;488/22/0;522/27/1;525/27/5;
24/2/0;331/22/0, exit1 existing findings, provenance latch OPEN. Syntax2files,
patch-added whitespace and ordered apply-check0.

Corrected understanding of patch20's source gate: passing it was NOT proof
the copier was semantically sound. Incorrect reference kind folding had
removed live container branches. Preserving them reveals earlier blockers.
Final exact50-line copier probe1.876627s refuses recursive invocation storage
at callsite14 for opaque private formal2305843010213694858, declared_workspace=True.
This is a lowering REFUSAL, not a compile or performance result.
Tiny tuple repro refuses private target0. Observed Loop13 and owning If11 are
SIBLINGS in composed control; call3 is placed in If11 while Load0 is in Loop13.
A trial deepest-call-owner change did not repair that source control ownership
and was fully removed (fortran_c_shell normalized content equals pre21).
Also observed forwarded runtime isinstance conditional has no source producer;
existing object_kind_matches is explicitly closed-vocabulary membership, NOT
arbitrary Python MRO/metaclass semantics. Do not silently equate the two.

NEXT: repair loop/conditional ownership at the existing composition writer using
the source/control graph records, then connect admissible runtime type predicates
to the existing object contract and helpers. Keep recursive-frame refusal intact;
do not allocate opaque scratch or cast handles to pointers. No main speedup,
hash indexing, bounded object batches, whole-RAM tiling or parallelism claim.

2026-10-10 integer memo membership/capture — SNAPSHOT ONLY:

User: "This the in operator we're needing? Should we make a hash cache find/sort
in the best algorithms"; then "Proceed".

Patch20 tools/compiler_probes/native_memo_membership.patch UNAPPLIED. Twenty
ordered patches pass combined git apply --check0; live source unchanged.
Implementation remains native-fallback-link/Powershell/turing. No main/N4,
process kill, timeout, package, alternative Python compiler or rewritten copier.

Observed exact tiny failures, fixed at their writers:
1. key in memo rejected membership-field because structural recovery required
   GetAttr/base/record despite the parameter's existing sequence descriptor.
   It now consumes that exact descriptor. Earlier control pruning had removed
   its two column arguments; recovery restores the SAME admitted parameter
   arenas, not anonymous storage. Late scalar uses update the incumbent
   declared_parameter/function_parameter pages (shared with id(value)).
2. contains used int32 induction with int64 length; its bool Phi received
   int32 true/false. Native LLVM emission now uses int64 counters and bool
   constants. The tiny native test actually executes hits/misses, empty,
   deleted-row flags, zero, negative and 2**60+3 keys.
3. The EXISTING call-edge ABI fixed point carried storage=keyed to captured
   memo Input6, but omitted mutability/key encoding/owner, and row declaration
   read only top-level receipts. Those exact contracts now propagate; row
   declarations consume linked INPUT identities and refuse conflicting
   declared/linked layouts. Complete descriptor members retain the same caller
   ownership, including columns/length/capacity/status/live flags. No parallel
   registry or integer-to-object-reference key substitution.
4. Existing builtins.id contract explicitly means stable identity-table handle,
   never a CPython address. --object-context copier probe now declares the
   incumbent integer-key/opaque-value memo ABI. Default probe remains unchanged.
   This is a lowering boundary declaration, NOT Python memo import/export.
5. New structural_membership_parameter receipts name exact graph target/query
   and descriptor. Concordance validates helper, arity and ordered operands;
   corrupt storage/query/target tests diagnose membership-parameter-contract.

Final47pass52.38s, one existing cffi warning:
python -m pytest tests/test_native_memo_membership.py tests/test_native_reference_boundary.py tests/test_native_object_collection.py tests/test_source_object_context.py -q --tb=short --show-capture=no -x
New test file has10 cases (five source/native modes, two exact-contract cases,
three corrupt receipts). Earlier13pass7.73s; broader gate61pass49.20s then
test_keyed_tensor_lookup_feeds_tensor_property_region_as_span StopIteration.
Exact pre20 source-loader replay reproduces SAME failure2.76s; not waived/fixed.
Seven ordinary audits unchanged493/24/0;360/36/1;488/22/0;522/27/1;525/27/5;
24/2/0;331/22/0. Audit exit1 for existing findings, provenance latch OPEN.
Syntax6files/added-whitespace0; combined patch apply-check0; guestbook dry-run0
(no --apply cleanup or archive moves).
Use NODUS_TENSOR_CORE=C:/dev/Powershell/nodus/build/Release/nodus_tensor_core.dll.

Exact unchanged50-line copier now CLEARS full-native source gate: no undefined
predicate37, unmaterialized values, unresolved calls or unaccounted formals.
Final normal lowering+LLVM attempt2.154057s. LLVM_COMPLETE=False: five loop
Phi i32/i64 disagreements (loops58/63/67/71/75) and four unhandled extent Calls.
Observed body still issues GetElementPtr/extent against opaque iterable handles
11/56; another extent input is projected dictionary-items19. Its collection
receipt is empty. These must be admitted through existing graph-owned object
iteration/collection contracts, not retagged as tensors or cast to pointers.
Entire copier has NOT executed natively. This duration is not a native timing.

Hash/direct indexing and a shared find(found,row) are NOT implemented. Current
contains/lookup/store remain linear; repeated growing-memo probes can remain
quadratic. No compiler speedup, general Python dict equality/hash, main
parallelism or whole-memory tiling claim. NEXT: tiny exact captured-object
iteration repro, observe loss of existing collection ownership, then close
those producer contracts; integer memo row indexing still needs growth and
mutation-validity contracts, not an answer cache or sort on every lookup.

2026-10-10 native graph-owned generator collections — SNAPSHOT ONLY:

User: "Is this the need to make a native generator for irreduceable generators"
User: "Let's proceed to fix it"

Patch19 tools/compiler_probes/native_object_collection.patch UNAPPLIED. All
nineteen ordered patches pass combined git apply --check0; live source unchanged.
Implementation: native-fallback-link/Powershell/turing. No main/N4, process kill,
package, alternative Python compiler, rewritten copier or live installation.
No full-source probe deadline added. The existing conditional-call regression
does retain its own existing20s subprocess parity deadline.

The actual generator LoopResult/materializer/iterable/target/member edges now
own a native collection handle. Dense single-iterable, unconditional unit-step
tuple/list generators use the existing context and SSA operators. COLLECT is
an object helper operation (ordinary Call/Phi/Load/Store), NOT a new SSA opcode
or suspended Python generator runtime. It shares materialization lowering but
always creates fresh rows, even tuple->tuple. One native reservation/stamp pass
pair prepares contiguous destination rows; existing authored body calls then
replace their values directly. Stable row indices survive recursive/member
arena relocation: each GEP reloads the incumbent heap pointer CELL after calls.
The extent is the admitted member-count delta, not a static tensor shape of1.
No per-element ctypes/Python/get/set/member-search dispatch. Shared mutable
context loops are not advertised as parallel candidates. Filtered/nested or
non-unit-step collection control is not newly admitted. No general lazy
generator, scalar boxing, equality/hash, GC, whole-memory tiling or main
parallelism claim. Preparation remains O(GLOBAL members); body stamping O(N),
excluding authored body work and existing allocation/table scans.

Observed losing writers, not patched guesses:
1. Tiny generator initially emitted an unknown private array and stop1. Exact
   graph collection ownership now reaches its native producer and loop extent.
   Direct-input lowering carries append in loop.sequence_mutations; the other
   retained route places it in the body. Both existing locations are honored.
2. fresh(item) returning [] was typed only at final linking, after its caller
   created three useless array bookkeeping formals. Native graph births/handles
   now participate in the EXISTING bounded ABI fixed point and exact result
   edges before frame declarations. Scalar/unknown leaves are not guessed.
3. Emitted LLVM reread the length address for BOTH end and base after collect,
   computing zero and silently skipping the body. Exact scalar int64/uint64/
   opaque_ref loads now capture their SSA values, like existing float/int32.
   Aggregate/address representations are not classified by LLVM i64 alone.
   New isolated pre-edit-backend reproduction returns0 instead of expected12;
   fixed native test returns12. No on-disk compiler swapping/revert was used.
4. Member-growth repro copy(item):return list(item) had source result binding
   (0,2), live native Ret2, formal input0. Observer on Instr.args caught linker
   line35359 rewriting publication2 to0 based on the stale source result id.
   Aliased-return selection now consults the live callee output identity, so
   fresh native results remain fresh. Existing structural identity receipts
   separately resolve genuine identity(item) without inventing an identity.
5. Concordance/emission now validate collection births, live kind Consts,
   source ownership, exact row column/base/index, dense induction/extent and
   opaque member producers. Corrupt column/index/kind/member cases refuse;
   definition inventories are reused, not rescanned for every row operation.

New15 tests passed19.06s: tuple/list, direct/identity/native fresh allocation/
native child-copy bodies; N0/1/17/257, identity cases also4097; one invocation,
order/aliasing, distinct new objects, member arena growth, int64 snapshot, and
four stale-contract refusals. Earlier31-test batch passed38.14s. FINAL broad
125 passed98.65s, one existing cffi deprecation warning:
python -m pytest tests/test_native_object_collection.py tests/test_native_object_materialize.py tests/test_source_object_context.py tests/test_ssa_object_context.py tests/test_native_object_boundary.py tests/test_shell_external_references.py tests/test_ssa_owned_heap.py tests/test_ssa_object_operations.py -q --tb=short --show-capture=no -x
Set NODUS_TENSOR_CORE=C:/dev/Powershell/nodus/build/Release/nodus_tensor_core.dll.

Additional backend/call gate:20passed/3existing disabled-XOR skips, then existing
conditional tuple test fails KeyError tuple_merge__tick (11.36s). Exact pre19
source-loader replay has SAME KeyError3.14s; untouched C:/dev/tg-base SAME2.70s.
Separate root-frame test1passed1.20s. In-place-store gate's first test hits an
existing i32/i64 Phi mismatch; exact pre19 sources reproduce it2.68s. Clean
tg-base has3passes then its existing old IR-symbol assertion fails. These
failures were not waived, changed or attributed to this patch. Seven ordinary
audits remain493/24/0;360/36/1;488/22/0;522/27/1;525/27/5;24/2/0;331/22/0,
exit1 for existing findings; provenance latch OPEN. Syntax/added-whitespace0.

Unchanged50-line copier passes collection93 and private recursive frame guards.
Final normal retry2.353027s: no unmaterialized values, unresolved calls, or
unaccounted formals; ONE undefined predicate37 at if_merge.4 CondBr. Observer
identifies node37 EXACTLY as identity in metadata_memo, parents36(lhs)/6(rhs),
not a guessed generator/tuple failure. Observed retry with source tracing8.24s.
NEXT: bind memo membership/lookup under their actual identity/key contract and
the incumbent table/query route, then retry the unchanged copier. Integer
identity keys are NOT interchangeable with reference-identity member keys.
Full copier native execution and end-to-end compiler speedup remain unproved.

2026-10-10 bulk native materialization and reusable ctypes boundary — SNAPSHOT ONLY:

User: "Move forward and recognize the delays from dispatch stack up extremely quickly right on boundaries we're trying to use native pockets to fix the speed of, so in the ideal world this may involve making a special wrapper object to sit on the option side and use ctypes to have fast object stamping"

Patch18 tools/compiler_probes/native_object_materialization.patch UNAPPLIED;
all eighteen ordered promotion patches pass combined git apply --check0. Live
compiler unchanged. Existing SSAObjectContext now emits a MATERIALIZE helper
using ordinary Call/Phi/Load/Store, incumbent reserve/construct and shared heap.
tuple/list conversion preserves shallow member identities; dict iteration
yields keys, sequence positions are preserved even if table rows are interleaved,
and tuple(existing exact tuple) returns the same handle. Extent/position bounds
refuse before invalid address calculation. Zero remains failure, not an object.
Stamping counts in one member-table pass, reserves once, then copies in one
pass with no per-item lookup/store/constructor dispatch or Python callback.
Both passes still scan GLOBAL member rows: O(total members), not per-object
adjacency indexing. All native allocation remains owned until context release.

Our public source entry lowers ordinary tuple(value), list(value), and their
composition when the source is admitted as an opaque object reference. Exact
builtin extraction identity, argument edges and materialized-source ledger
are the authority; generator-produced unknown collections are not guessed.
Native source effects replace the old sequence replace effect at the SAME
source identity. Observer caught that old replace winning for tuple(list(value)).
Another observed intermediate sent the already-defined kind-token SSAValue
through source aliases and invented an untyped formal; exact occurrences now
remain exact. Source materialization receipts validate live kind tokens/helper
contracts at concordance/emission. The pre-change tiny root(value):tuple(value)
even emitted LLVM_COMPLETE with a phantom unknown output formal and lost its
real input. Final source tests actually execute and check identity, not just emit.

NativeObjectBoundary.bind_execution returns NativeObjectInvocation over the
EXISTING LLVMExecution/ctypes ABI. It binds context/input buffers once, rejects
overridden context cells/missing feeds, verifies real shared addresses, caches
the native entry and guards the exclusive lease. Native-to-native calls need
no intervening export/import. No PyObject pointer ABI, second registry or
alternative runtime; export remains explicit and context ownership is unchanged.

Actual observer:10 native calls triggered20 substring searches through71,996
bytes of LLVM IR in reset/take validation. LLVMFunctionArtifact now caches that
capability against exact text identity AND declared flag (replace resets it).
Validation calls/fault checks remain; test executes real native error7 twice.
The dispatch micro uses1000 calls to the SAME source tuple identity operation:
first51.4574us prepare-each vs23.2019us reuse; after capability cache, final
31.7249us prepare-each vs0.6329us reuse (50.126x boundary-only comparison).
Native stamping4097 elements1call0.0312ms; import1.48ms, bind0.0858ms,
export6.0894ms. Cold first invocation costs are reported separately. These are
micros, NOT eager-Python or whole-compiler speedup, parallelism or memory tiling.
Command: python -u tools/compiler_probes/probe_native_object_dispatch.py
Final artifacts: C:/Users/alber/AppData/Local/Temp/turing-object-dispatch-c7mst6ci.

Earlier broad107 passed77.22s; post-cache backend/heap/shell batch59 passed,
3 existing disabled whole-XOR skips24.26s; final new materialization11 passed9.50s.
One test-command typo named nonexistent test_external_functions.py: no tests ran.
An initial single-block validation fixture hit the existing fast emitter's
unhandled validation symbol; the real multi-block control fixture verifies the
guard lane without changing emission. Seven ordinary audits unchanged; latch
OPEN. Syntax/addition-whitespace checks0. FINAL broad rerun110 passed78.45s.
Command: python -m pytest tests/test_native_object_materialize.py tests/test_source_object_context.py tests/test_ssa_object_context.py tests/test_native_object_boundary.py tests/test_shell_external_references.py tests/test_ssa_owned_heap.py tests/test_ssa_object_operations.py -q --tb=short --show-capture=no -x

Unchanged50-line copier now passes members=tuple(value)56, refuses collection93
at recursive call14 (final retry2.1322359s). Trace identifies93 as LoopResult
collection, source29/control58, materializer60, iterable56: the authored
tuple(isolate_metadata(item) for item in members) generator collection. NEXT:
connect that existing graph loop's item/result/extent identities to native
collection storage and fuse its stamping in the native caller, not per-item
ctypes. Dict equality/hash, generic opaque leaf operations, indexed adjacency
and full copier execution remain unfinished. No main/N4/timeout/kill/package/
alternative source compiler/copier installation this turn.

2026-10-10 ordinary source object births execute — SNAPSHOT ONLY:

Patch17 tools/compiler_probes/source_object_context.patch staged UNAPPLIED.
All seventeen ordered patches pass combined git apply --check0. Live compiler
source unchanged. The public source entry now accepts native_object_context,
the existing SSAObjectContext descriptor. Ordinary {} and [] birth at their
authored control positions and produce opaque_ref handles, not private unknown
columns. Shared context cells cross existing exact call-frame bindings through
forwarding and recursion. The original codec boundary exports their results.
No Python callback, second registry, new opcode or substitute copier.

Observed intermediates: context owner/counter initially missed formal storage
accounting; filtering stale survey declarations preceded the actual field-slot
writer and did not prevent unknown columns; [] reduction retained aggregate
birth metadata but removed its AST. Fixed those producers, not formal guards.
Dictionary IndexedStore was unclaimed after private declarations were removed,
so a planned region emitted GEP(opaque_ref,key) and Store; returned dict empty.
Exact handle identities now use the existing lexical table-operation route and
native object helper. len(item) mapped to the correct object but finish only
materialized descriptor-owned lengths; it now admits declared object lengths.
Phi/LoopResult aggregate metadata also created unnecessary unknown arenas.
Graph-edge handle propagation requires both branch alternatives or exact loop
initial/update identities; zero trips keep the initial handle. It is a bounded
dependency worklist, not a name/dtype guess. Loops with break bindings are not
newly admitted by that rule.

Sixteen source tests execute direct/forwarded/recursive/loop/conditional dict
and list construction (depths0/1/7/32), reference-key dict set/get/length, and
refuse corrupted context roles/kind tokens/module acceptance. Concordance and
emission validate source receipts against live helpers/tables/formals; definition
and helper inventories are reused, not rescanned per birth. Native dictionaries
STILL use reference-key identity, not general Python equality/hash. Imported
opaque-container iteration, tuple materialization, generator update/extend and
general leaf/key semantics are NOT complete.

Regression gates:66 passed39.72s (source/context/boundary/shell/heap), then final
66 passed53.26s (source/object-operations/index-store/scalar-promotion), one cffi
deprecation warning. Seven ordinary audits retain exact493/24/0;360/36/1;
488/22/0;522/27/1;525/27/5;24/2/0;331/22/0; provenance latch OPEN.
The audit command returns1 for the existing findings, not a clean gate.

Snapshot commands (set NODUS_TENSOR_CORE to the existing live Release DLL):
python -m pytest tests/test_source_object_context.py tests/test_ssa_object_context.py tests/test_native_object_boundary.py tests/test_shell_external_references.py tests/test_ssa_owned_heap.py -q --tb=short --show-capture=no -x
python -m pytest tests/test_source_object_context.py tests/test_ssa_object_operations.py tests/test_native_scalar_index_store.py tests/test_mixed_domain_scalar_promotion.py -q --tb=short --show-capture=no -x
python -u tools/compiler_probes/probe_native_metadata_copy.py --object-context

Exact unchanged50-line copier with --object-context now passes workspace12 and
refuses private_formal56/call14. Trace observer identifies56 as members=tuple(value)
(aggregate_materialization, materialized_source_value_ids=(11,)), still lowered
to unknown private sequence storage. Final retry2.0368875s, no native copier
execution or speedup claim. NEXT: wire iteration/materialization from the
accepted object context at the existing source producer, preserving actual
copier memo/aliases/backedges and dictionary equality obligations. No main/N4
started, timeout, kill, package, other compiler or live installation this turn.

2026-10-10 Python/native object conversion boundary works — SNAPSHOT ONLY:

User authorized the API/ABI boundary for piecemeal Python/native replacement.
Patch16 tools/compiler_probes/native_object_boundary.patch is staged UNAPPLIED.
All sixteen ordered patches pass combined git apply --check0. Live compiler
source remains unchanged. Implementation is in native-fallback-link/Powershell/turing.

ExternalReferenceValueCodec.bind_native_objects(context, roots) leases the
EXISTING codec handle domain and projects builtin dict/list/tuple/set/frozenset
graphs into the EXISTING SSAObjectContext's exact object/member tables. It does
not create another registry or expose PyObject pointers. Multiple roots share
one identity traversal. Leaves/subclasses remain retained opaque identities.
Borrowed NumPy columns live through the lease; native geometric growth copies
them into the incumbent caller-owned heap. Native execution has no per-item
Python callbacks. Export synchronizes imported mutable objects only after
graph validation, preserving their original Python identities; newly native
handles get fresh representations posted to the SAME codec so repeated exports
return the same objects. Imported immutable handles must retain exact members;
changes require a new handle. Mutable shells and an immutable dependency
worklist preserve cycles/backedges without recursive Python conversion.
Explicit native context release precedes close when native arenas exist;
close drops retentions and zeros ABI cells. No finalizer frees native memory.
This is a trusted in-process ABI, not a parser for untrusted native addresses.

Dictionary insertion order/key identities are retained. Native member lookup
STILL compares reference identity, not Python equality/hash. At export, real
Python dictionaries/sets apply their own semantics; invalid distinct refs that
collapse to equal keys refuse before publication. This is NOT generic native
Python equality/hash support. Traversal/worklists are graph-linear excluding
host hashing/equality costs. Whole imported graphs remain resident during the
lease; this is not the requested whole-compiler memory tiling or parallelism.

Observed source-link defect: native member-store's explicit i32 status Loads
were rewritten to int64 at ir_indexing._propagate_scalar_dtypes line436; its
Phi remained i32, and LLVM correctly refused. A setter observer recorded
initial four int incoming statuses and the two Load int->int64 writes at that
pass. Raw declared memory widths now remain declared, with existing width
decision records and a dedicated missing-Load-identity reason. Abstract
Call->GEP->Load return projections are distinguished by their exact existing
call-result edge, not names. The first broader fix incorrectly treated the
oscillator's output69 projection placeholder as raw memory and caused a new
book/width refusal; that intermediate is retained here. Exact aggregate edges
keep the incumbent projection rule; oscillator returns to331/22/0 findings.
Clean C:/dev/tg-base confirms raw int8/int16/int32/int Loads previously widened
to int64. It also confirms release(retain(None)) falsely raised unknown handle;
codec release now checks membership instead of treating None as absence.
An initial baseline script reused numeric IDs and properly hit concordance;
the corrected baseline uses the incumbent monotonic issuer.

PUBLIC lower_ast_source_to_ssa compiles a whole loop using accepted native
constructor/store links. One invocation creates1/17/257 fresh dictionaries,
links every dictionary back to the imported list, grows resident tables, then
exports exact cycles and original leaf/list identities. Pointer/counter feeds
are checked to share actual boundary buffers BEFORE execution. Both native
context and source module have structural0. Provenance latch remains OPEN.
Other tests cover2500-deep immutable graphs, aliases, sets/frozensets, actual
dict key identity, subclass opacity, equal-key refusal, immutable mutation and
impossible cycles, bad extents, borrowed-array preservation, ownership release
and repeated export identity. These are explicit-boundary source tests, NOT
automatic ordinary-container lowering or a substitute production copier.

Final broad gate100 passed/62.89s; after final collision-check placement,
boundary/shell gate17 passed/4.36s. One existing cffi deprecation warning.
Command: python -m pytest tests/test_native_object_boundary.py tests/test_shell_external_references.py tests/test_ssa_object_context.py tests/test_ssa_owned_heap.py tests/test_ssa_object_operations.py tests/test_native_scalar_index_store.py tests/test_mixed_domain_scalar_promotion.py -q --tb=short --show-capture=no
Seven ordinary audit outcomes unchanged after aggregate-edge distinction:
493/24/0;360/36/1;488/22/0;522/27/1;525/27/5;24/2/0;331/22/0;
unsourced counts unchanged. Syntax and diff whitespace checks exit0.

Exact50-line production copier retry1.7746843s still refuses dictionary
workspace12 at recursive call14. User correction: "of course {} is both an
object and an unknown recursive workspace, that's the nature of the syntax
as it interacts with the user". UNKNOWN CONTENTS ARE NOT UNKNOWN STORAGE:
{} supplies a known dictionary kind and initial extent zero. The missing
producer must publish its native handle/context representation before recursive
leasing; no requirement to statically enumerate future Python contents.
Next: connect actual source object birth/inspection to this accepted context
and preserve dictionary equality/hash obligations, aliases, memo/backedges.
No production copier installed, main/N4 run, process kill, package install,
alternative source compiler, full-main parallelism or speedup claimed.

2026-10-10 explicit native object context executes and crosses source calls — SNAPSHOT ONLY:

User: "okay, this sounds like a subtle thing but like a backbone architectural
object that would help us to do it right, please proceed". New patch15:
tools/compiler_probes/native_object_context.patch, UNAPPLIED. All fifteen
ordered promotion patches pass combined git apply --check0. Implementation is
in native-fallback-link/Powershell/turing, not the live compiler. No N4/main,
package, alternative source compiler, process kill or production installation.

SSAObjectContext is a backend-neutral ABI view of TWO EXISTING typed dynamic
sequence tables (object handle/kind; owner handle/key handle/value handle),
one shared caller-owned heap and a handle counter. No additional Python
registry, pointer masquerading as a handle, new opcode or generic object model.
The kind vocabulary is explicit. Member keys compare by REFERENCE IDENTITY,
not Python dict equality/hash; this is not permission to lower arbitrary
Python dictionaries as this domain. Counter is positive, zero invalid, and
construction refuses invalid kind, collision or exhaustion without advancing
the counter or overwriting the incumbent kind. Allocation failure retains
the existing fail-fast allocator behavior. Initialization is for an empty
caller-supplied context, not reset of live storage. Mutation requires exclusive
ownership of one context; different contexts have separate storage.

ssa_object_context.lower_object_context emits real initialize, construct,
kind, get, set, length and release helpers through incumbent sequence/table
and object-operation interfaces. Growth, insertion/store and typed transfer
use the previously implemented native paths. Release frees shared ownership
and clears pointer cells/counters; it never GEPs through freed payload.
Each function owns its SSAValue occurrences, sharing only immutable context
descriptors, not mutable cross-function accounting. Context receipts validate
against live descriptors and actual formals at emission/concordance.

OBSERVED source-link defects, fixed at their writers:
- Final sequence dtype settlement at fortran_c_shell.py formerly wrote
  ptr -> opaque_ref/int64 from COLUMN ELEMENT dtype. It now preserves the
  pointer-cell contract when the descriptor has an explicit heap owner.
- The public source caller retained raw ptr without the callee's heap-cell
  accounting, so its buffer ABI said double and preparation copied pointer
  cells. First invocation could work, then retained length with lost addresses
  caused an access violation on a later invocation. Exact callee_input_ids now
  transport cell/pointee/shared-owner contracts over source-linked call edges.
  A monotonic reverse-call worklist covers forwarding and SCCs; conflicts or
  missing owner bindings refuse. Normal modules without heap contracts skip
  the instruction inventory. The projection receipt is audited against the
  LIVE exact call edge and formal contract, not names or pointer spelling.
- Existing _storage_values published opaque_ref whereas Alloca published its
  canonical int64 layout. Observed later formal accounting propagation replaced
  that canonical layout, creating four allocator findings. Storage publication
  now uses the shared layout table's canonical name at its original producer.
- LLVM Cast(bool,int64) returned i1 unchanged, then stored it as i64. Observed
  invalid LLVM is fixed with exact zero-extension (0/1, not sign-extension).
- Key-column auditing now consults the shared integral-layout authority;
  opaque_ref is exact i64, not a float key. Float/unresolved keys still fail.

Tests: real native construction/growth/access/release, 17 distinct objects,
references >2**60, overwrite, self-edge, backedge, missing references,
invalid kinds, handle collision/exhaustion and release/reinitialize. Actual
source programs use the PUBLIC lower_ast_source_to_ssa with accepted native
signature declarations, not compiled Python callback bodies. Whole 1/17/257
object loops run in ONE native invocation, repeatedly on the same context,
directly and through an extra relay. Outputs equal kind*count exactly;
pointer buffers are asserted ptr and share caller storage BEFORE execution.
Both source modules have zero structural findings. The provenance latch is
OPEN (including helper IDs born outside the source book), not fully closed.

Broad gate168 passed/5 SAME PREVIOUSLY RECORDED failures80.77s; final gate
after per-function occurrence isolation66 passed45.33s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_context.py tests/test_ssa_owned_heap.py tests/test_ssa_object_operations.py
Seven ordinary audits unchanged:493/24/0;360/36/1;488/22/0;522/27/1;
525/27/5;24/2/0;331/22/0 (rows/functions/findings). Unsourced counts unchanged.
The earlier native source test access violation is retained above as a real
failed intermediate, not a green run; subsequent repeated execution passes.

Production exact50-line copier retry1.6916952s STILL refuses unknown dictionary
workspace12 at recursive call14. The explicit native context now exists and
links, but {} and the copier's ordinary Python-object inputs are NOT yet
bound to it. No compiled production copier, native replacement installation,
N4 speedup, main tiling or main parallelism claimed. Next: connect admitted
source constructors/inspection and the existing Python-object boundary to
this context without silently changing dictionary semantics or leaf identity.

2026-10-10 typed owned dynamic storage path completed; source birth remains — SNAPSHOT ONLY:

User: "okay let's clean everything around the dynamic allocation contract and
see if we can make that the last rivit". Patch14 dynamic_owned_sequence_storage.patch
staged UNAPPLIED; all fourteen ordered patches pass combined git apply --check0.
Snapshot changes: ir_sequence_tables.py, ssa_heap_allocation.py, ssa_llvm_backend.py,
identity_concordance.py, sequence_storage_binding.py, transmogrifier/ssa.py,
test_ssa_owned_heap.py. No live compiler/copier install, main/N4, process kill,
package or alternative Python compiler. Native LLVM emission uses our incumbent lane.

PRE-EDIT OBSERVATIONS:
Exact50-line copier rejects sequence12, actual {} producer, columns(12, secondID),
column_dtypes unknown/unknown, scalar rows, FIXED policy, heap_owner=None.
Graph retains Dict/aggregate_kind=dict and two-column writable/key0 shape, but
provides no native object-domain/owned allocation descriptor. Existing allocator
already implements runtime size + caller-owned lifetime. Existing typed owned
DYNAMIC descriptor still gives dynamic-growth-unavailable from insert; initializer
also refused DYNAMIC policy. Separately, heap length/capacity aliases were accepted
and int64 heap columns bound to float64 heap columns through ptr ABI unchanged.
All three gaps were observed before their respective changes, not guessed.

IMPLEMENTED in the EXISTING descriptor/storage path:
- Empty typed owned columns can initialize with FIXED or DYNAMIC capacity policy.
- lower_sequence_reserve validates nonnegative live length/required count,
  length<=capacity and allocator byte-overflow bounds BEFORE copying. Chooses
  max(required, checked double capacity), capped at the shared safe row maximum.
- Allocates/zeroes all new columns, byte-copies live rows, then publishes exact
  pointer cells and capacity together. Length unchanged. Same-cell column aliases
  reuse one allocation; incompatible aliased element layouts are refused.
- Existing fixed insert/table-store bodies remain the mutation authority. A shared
  dynamic wrapper validates extents before scanning, tries the incumbent operation,
  reserves ONLY on full, then retries. Duplicate keys and existing-key updates do
  not grow. Existing statuses retained, no Python callback.
- Extend includes exact destination owner, binds row formals by their declared
  IDs and retains the ENTIRE insert/fixed/reserve helper closure. Heap element
  contracts match; shaped rows/liveness/child pools without supported replacement
  are still named shortfalls, not silent truncation.
- Native typed element transfer is ordinary Call to LLVM memmove through the
  existing heap ABI. Integer element counts checked before byte multiplication;
  scalar count width limited to64bits (no truncating a wider count). No opaque
  handle becomes an address. Byte-copy preserves float32 bits without numeric
  widening/narrowing and supports overlap. Allocation operation itself validated.
- Owned descriptors reject overlapping length/capacity/pointer cells, owner/status
  collision and pointee-layout ambiguity. Cross-call heap binding checks exact
  layouts and normalized row shapes, not merely ptr ABI equality.
- Concordance audits heap-copy-contract beside existing heap-allocation-contract.
  No new object registry, datatype, source opcode or allocation-guard relaxation.

LIFETIME / COMPLEXITY:
One caller-owned heap cell retains current AND retired arenas until explicit
release. Pointer-CELL identities stay stable; interior addresses are arena-version
views and must be reacquired after growth. No claim that an old interior view
magically follows replacement. Geometric growth amortizes row-copy work; retired
payload is geometrically bounded for monotonic insertion. Owner release frees
the chain and clears its cell. Unique-key scanning itself remains linear, NOT
a new hash table, and no main compile speedup/parallelism is claimed.

NATIVE VERIFICATION (SSA allocation-contract tests, not automatic source binding):
Five layouts int64/opaque_ref/float64/float32/bool reserve through1/2/7/4097 rows,
preserve byte patterns/retired views and zero new capacity; no-op reserve and
invalid counts preserve state. Native insert AND table-store loops run whole
1/257/4097-row jobs in ONE native call, for int64 and opaque_ref, exact values
above2**60; re-inserting keys preserves capacity/owner/columns. Native extend
copies257 borrowed rows twice without modifying source or duplicating rows.
Invalid extents reject before touching null columns. Overlap copy preserves
memmove semantics. Explicit release resets owners. Valid dynamic module audit
observed124 rows/6functions/0 findings; standalone SSA module has no attached
book, so this is not proof of source-born emission provenance.

Final targeted gate163 pass/5 PREVIOUSLY BASELINED failures73.98s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_operations.py tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py tests/test_control_branch_compartments.py tests/test_native_conditional_call_result.py tests/test_ir_sequence_tables.py tests/test_sequence_contract_concordance.py
Same three stale ID/helper-name expectations and two real int32/int64 exact-region
feed conflicts, no exclusions. Final seven audits unchanged0,1,0,1,5,0,0 with
identical unsourced counts, latch OPEN. Guestbook validation dry-run only.

NOT THE LAST RIVET YET:
Final production exact copier retry2.2399622s still refuses unknown dictionary workspace12
at recursive call14/SCC0. New code does not pretend that {} has a native object
handle/domain, element layout or shared allocation owner just because malloc exists.
Remaining SOURCE bridge: constructor materialization must publish its accepted
object-domain representation, runtime capacity and shared-owner descriptor before
recursive leasing, and schedule the real allocation producer. Existing opaque
construct helper interface alone is not a generic object implementation.
Previously observed early-return missing recursive Call remains unresolved.
No compiled/installed production copier, source-constructor success, N4 gain,
whole-main tiling or parallelism claim.


2026-10-10 reducer-owned result Phis retained; declared recursion executes — SNAPSHOT ONLY:

User: "okay, let's proceed". Patch13 conditional_result_phi.patch staged UNAPPLIED;
all thirteen ordered patches pass combined git apply --check (exit0). Changes
are snapshot glsl_deployment_strategy.py and test_ssa_object_operations.py only.
No live compiler/copier install, main/N4 launch, process kill, packages or new
compilation route. Existing public source entry and native LLVM lane used.

OBSERVED CAUSAL CHAIN, before editing:
Graph Phi18 has test4/body17/orelse9, binding result, source_conditional_id5,
and NO initial_value_id. _ordinary_conditional_control_programs reaches its
initial-value check and skips this Phi. lower_conditional receives result_aliases
empty while sequence_length_values is already {17:14}, descriptors14/6 present.
No one requests len-result17, so no Load or Phi is emitted; Ret reads call9.
The graph knew the complete merge; this was planner loss, not missing type data.

Fix: reducer Phis without an entered version use the EXISTING conditional-result
path with their exact arm IDs. No invented merge from names, no new SSA opcode,
no backend repair or allocation-guard relaxation. Existing control lowering
requests len17 on its true edge, publishes Phi18 and the concordance derivations;
existing linked result-type propagation settles recursive result to int64.

A second minimal case ("if index: result=23; else: result=41") exposed the earlier
retention guard: no numerical predicate/arm regions, calls or field-state merge,
but graph Phi4(test0/body2/orelse3) owns source conditional1. Observed guard drops
that conditional despite this exact Phi. Retention now admits reducer-owned
body/orelse merges as well as the prior field-state merges. No-Phi no-invention
regression remains green.

Actual source common-return recursive table-length program now lowers under the
unchanged full-native gate, has zero concordance findings, and executes LLVM
native at depths0/1/7/32 for three objects: lengths2/0/1, one handle above2**60.
All twelve executions return the base-case length; all caller columns remain
unchanged. Recursive feeds share exact declared keys/values/length and member
column pools/lengths/stride; only int lookup-status scratch is invocation-local.
Fortran text emission complete. C module text still explicitly reports missing
zeros spelling for the two invocation-local status producers; NOT C completeness.

Four additional source assignment programs execute nine branch cases: bare
predicate, comparison predicate, nested merge, and an entered carried version.
Five focused tests pass6.31s. Saved pre-edit strategy replay gives three failures/
one pass7.26s for the comparison/nested/recursive/entered set; the bare-predicate
failure was observed separately before changing its retention guard. Thus both
losses are deterministic and reproduced, not inferred from a post-edit success.

Final targeted gate148 pass/5 PREVIOUSLY BASELINED failures69.15s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_operations.py tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py tests/test_control_branch_compartments.py tests/test_native_conditional_call_result.py tests/test_ir_sequence_tables.py tests/test_sequence_contract_concordance.py
Same three stale ID/helper-name expectations and two real int32/int64 exact-region
feed conflicts, no exclusions. Earlier reference/control subset93 pass62.94s
before adding bare-predicate regression. Seven final audits unchanged0,1,0,1,5,0,0
and identical unsourced counts; provenance latch OPEN. Guestbook dry-run only.

Production exact50-line copier retry2.1632375s still refuses dynamic dictionary
workspace12 at recursive call14/SCC0. Borrowed declared contexts and ordinary
fresh branch results are now demonstrated, NOT generic native dictionary
construction, Python object-domain binding or installed copier acceleration.
Separate early-return kind variant rechecked: recursive Call remains absent,
exported Ret dtypeNone; losing writer still unobserved, never executed as native.
Remaining frontiers: that call-retention defect, and the dynamic constructor's
actual element/extent/owner contract. No main speedup, tiling or parallelism claim.


2026-10-10 recursive declared keyed storage settled before leasing — SNAPSHOT ONLY:

User: "how is our gap now", then "proceed".
Patch12 recursive_keyed_storage_binding.patch staged UNAPPLIED; all twelve
ordered patches pass combined git apply --check (exit0). Only snapshot
fortran_c_shell.py and test_ssa_object_operations.py changed. Live compiler,
copier and processes unchanged. No main/N4, package, timeout or process kill.

Pre-edit observation at record-ABI completion: authored GetAttr6 book.objects
still has anonymous unknown keys/value arenas and descriptor, but the lookup
already names its owner and the exact declared length/keys/values formals exist.
The incumbent late keyed binding pass is now a single reused closure. It runs
after record ABI materialization BEFORE source-call discovery/recursive leasing,
then again at its original post-link stage. Its existing remap retires anonymous
storage and derives descriptor revisions from old descriptors and physical cells.
No new logical sequence-field claim, allocation fallback or guard relaxation.

First relocation also typed opaque queries too early: six native reference cases
hit opaque_ref/int64 physical input disagreements. The early invocation now
settles STORAGE only; query producers/formals retain their incumbent types until
the original post-link query binding. Saved pre-edit compiler replay confirms
those six pass (5.73s), revised current reference/object batch45 pass29.69s.

Common-return recursive source now passes the storage guard. Diagnostic witness
asserts recursive feeds share exactly the same declared keys/values/length and
pooled columns/lengths/stride identities; descriptor keys/values are int64 and
capacity=length. Only one int lookup-status scratch is invocation-local, no
borrowed table arena. Regression1 pass2.27s is NOT recursive native success.

Next observed blocker: full-native structural_outputs reports missing Phi18
(test4, body17, orelse9). Rejected SSA has base-case selected table length address
but no Load/result17 or Phi18; if_merge Ret reads recursive Call9 alone with
dtype=None. Recursive call DOES survive (unlike the previously recorded
early-return variant). This malformed module was inspected from the refusal
traceback, never emitted/executed; full-native refusal remains enforced.
The original writer losing the length-result/control merge is NOT yet traced.
Next repair must observe that producer/control reduction, not patch the backend.

Final targeted gate128 pass/5 PREVIOUSLY BASELINED failures56.18s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_operations.py tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py tests/test_ir_sequence_tables.py tests/test_sequence_contract_concordance.py
Same three stale ID/helper-name expectations and two genuine int32/int64
exact-region feed disagreements, no exclusions. Existing NODUS_TENSOR_CORE used.
Seven audits unchanged0,1,0,1,5,0,0, same unsourced counts, latch OPEN.

Original50-line copier was NOT rerun this step; prior measured dictionary12/
call14 refusal remains the last production evidence. No installed native copier,
main speedup, automatic object binding, dynamic constructor completion, main
tiling or parallelism claim. Storage ordering gap closed; recursive control and
the broader copier/native object gaps remain.


2026-10-10 complete declared member-table context through source calls — SNAPSHOT ONLY:

User: "continue", following the industry-pattern discussion of reference,
descriptor, runtime extent and owner contracts. Patch11
object_table_source_context.patch staged UNAPPLIED; all eleven patches in
order pass combined git apply --check0. Only snapshot fortran_c_shell.py and
test_ssa_object_operations.py changed. No live compiler/copier/accelerator
installation changed, main/N4 launched, process killed, timeout or package added.

Observed original source root->inspect chain: materialize_declared_row_columns
published metadata.Object with kind span only and deferred objects[].members
table. _linked_caller_member then refused callee members' capacity/stride formal
against that incomplete caller record. Callee table columns/length vector also
had no record claims and were being leased as unrelated private storage.

Both row materializers now share materialize_row_table_context over the existing
declared ABI. It publishes caller-owned column pools, lengths and stride as real
record fields. Its per-function cache keys the exact owner record ID + declared
field path; this is materialization reuse, not an object registry. Mints derive
from existing schema/owner cells. Selected row GEPs stay the incumbent sequence
view. The existing record_field_decomposition maps only row capacity to stride;
it does NOT confuse pooled lengths/columns with row-selected addresses/pointers.
Call linking itself is unchanged and follows existing exact record pairs/claims.

Second forwarding level exposed an older duplicate-storage-across-call finding.
Observed materialize_parameter_record_abi: relay has field_read_ids=(), no writes,
propagated read demand, incoming_slot=None, mapping=None. Its fallback minted a
logical mapping as an extra input beside existing length/keys/values. That ghost
input had no record claim, so root leased it despite resident field storage.
An unread keyed field now publishes its physical decomposition without minting
the nonexistent input. Reuse also consults the existing exact decomposition row.
Saved pre-change compiler reproduced the same duplicate finding before this fix.

Eight real source-call cases, one/two forwarding levels, execute kind/length/get/
set with exact reference values above2**60, empty/nonempty lengths and writes to
caller buffers. All assert zero structural findings and one unleased copy of
each table-context formal. No manually constructed SSA caller in these cases.
C and Fortran text emission complete for all eight; Fortran uses the public
lowerer's COMPLETE outputs map, not a root-only map that omits callee results.
Only LLVM artifacts were compiled/executed, through our existing compiler lane.

Final focused gate74 pass43.83s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_operations.py tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py
Existing NODUS_TENSOR_CORE=nodus/build/Release/nodus_tensor_core.dll.
Separate final sequence/concordance gate54 pass/5 old failures7.04s:
python -m pytest -q --tb=short --show-capture=no tests/test_ir_sequence_tables.py tests/test_sequence_contract_concordance.py
Same three ID/helper-name expectations and two real int32/int64 exact-region
feed conflicts; no exclusions. Earlier combined run127 pass/5 fail51.58s before
adding the explicit recursive refusal regression. Seven audits before/after
retain0,1,0,1,5,0,0 and identical unsourced counts; provenance latch OPEN.

RECURSION IS NOT CLOSED. New explicit refusal regression uses:
def inspect(book,handle,index):
    if index<=0:
        result=len(book.objects[id(handle)].members)
    else:
        result=inspect(book,handle,index-1)
    return result
def root(book,handle,index):
    return inspect(book,handle,index)
It refuses private_formal6 (authored GetAttr book.objects), unknown shape(), at
recursive call9. Observed sequence_member claims (6,column0)/(6,handle), while
Book record only owns length/keys/values and rows: no record->sequence6 claim.
The existing physical decomposition is established too late to prevent recursive
frame leasing. Saved pre-change compiler gives the identical refusal. A late
sequence-field claim attempt preserved provisional unknown key-column contracts
and did not fix recursion; it was fully removed, not layered onto the working fix.
The passing refusal regression protects the guard, NOT successful native recursion.
Next: settle the authored keyed view from its existing owner/decomposition at the
producer BEFORE recursive lease selection, with exact descriptor/type receipts.

Also observed a separate early-return version:
def inspect(book,handle,index):
    if index<=0:
        return book.objects[id(handle)].kind
    return inspect(book,handle,index-1)
def root(book,handle,index):
    return inspect(book,handle,index)
Saved pre-change and current forms lower without the recursive Call; exported
Ret values have dtype=None. No native execution of that malformed candidate;
backend completeness is NOT semantic success. Its original losing writer is
not yet observed. Preserve this diagnostic for control/call retention repair.

Original50-line metadata copier retry1.6806848s still refuses unknown dictionary
workspace12 at recursive call14/SCC0. Ordinary Python object->descriptor binding,
builtin runtime dispatch, recursive result/storage constraints and constructors'
runtime extent/owner lifetime remain. No copier install, speedup, main tiling or
main parallelism claim. The nonrecursive SOURCE forwarding gap is now closed;
the broader opaque-reference/recursive copier gaps are not.

2026-10-10 opaque object SSA/backend contract — SNAPSHOT ONLY:

User: "what it sounds like to me is we want an opaque object type and helpers for ssa and the backends"
User: "can you realize this"

Patch10 opaque_object_operations.patch staged UNAPPLIED. All ten patches in
order pass combined git apply --check0; no promotion applied. Implementation
is in native-fallback-link/Powershell/turing. Live compiler/copier and any
previously running main process untouched; no main/N4 launched or process killed.

SSAOpaqueObject is a semantic view of the existing scalar opaque_ref plus an
explicit admitted storage domain, NOT a new physical dtype or object registry.
ssa_object_operations declares kind/length/get/set/construct/release interfaces
on incumbent SSA function bodies, partitioning exact formal IDs into reference,
context and operands. Calls remain ordinary native Call instructions. Helpers'
actual Ret operands provide result identities/layouts; existing return_value
metadata supplies Fortran's scalar result contract. No signature-name/position
guess, opaque-to-pointer cast, per-item Python callback or alternate compiler.
LLVM/C/Fortran validate the live helper, actual bindings and result contract at
emission; concordance diagnoses stale contracts and dangling field-sequence IDs.
Kind membership uses an explicit one-based owner vocabulary, NOT arbitrary
Python isinstance/MRO/metaclass semantics. Identity/equality are native Cast/Eq.
Bodyless imported helper interfaces are still refused by this new helper API;
existing native linking mechanisms remain unchanged.

Actual native KIND/LENGTH/GET/SET helper bodies compile through the sanctioned
lower_ast_source_to_ssa entry against existing Book.objects keyed-record ABI.
The test caller is SSA API composition, NOT automatic source dispatch. Tests
execute caller-owned kind/member/length columns, exact handles above2**60,
read-only mutation refusal, native identity/equality and kind membership. C and
Fortran emit all four helper/caller programs complete; neither was compiled by
an external C/Fortran compiler. LLVM products execute using our existing toolchain.
Construct/release interface shapes exist, but no generic object constructor or
lifetime implementation is supplied by this layer. Incumbent heap ABI is separate.

Observed/fixed writer chains:
- literal {} already had Dict producer + aggregate_kind=dict but unresolved
  payload; isinstance Call discarded that exact kind. Known constructor kind
  now survives without requiring mutable payload constant folding.
- nested positional row-column Store12 had Indexed11 as base, yet row ops were
  empty. Old lowering loaded row0 as a float and wrote through a temporary.
  _sequence_row_operations follows the two authored index edges; sparse row
  stores write only the selected column, retain row alias, and reject read-only
  descriptors. No execution of the old malformed store was used as a baseline.
- caller-owned table field was classified local and initialized before row
  pointer selection. _authored_source_sequence_ids now follows existing ABI
  graph identities/owner edges, preserving borrowed storage and lengths.
- opaque id Cast was tagged tensor_identity, rejected by C/Fortran. Its writer
  now marks opaque_reference_identity; genuine tensor feeds retain their tag.
  Existing regression assertions now require this correct contract and still
  execute unchanged identity/parity checks.
- keyed rebind at former38557 overwrote _BookRows with dict (observed before/
  after), bypassing book descriptor/member updates. Both keyed table rewriting
  sites now revise incumbent rows from old descriptor + exact physical cells.
  Dead-sequence pruning retires exact dependent record field claims in the same
  pass, retaining historical book revisions and authored schema. Fortran length
  had refused metadata.Object.members' absent sequence6; it now emits complete.

Final focused gate66 passed37.47s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_operations.py tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py
NODUS_TENSOR_CORE uses existing nodus/build/Release/nodus_tensor_core.dll.
Extended gate adds test_ir_sequence_tables.py/test_sequence_contract_concordance.py:
119 pass/5 prior failures43.90s before last additional contract/claim detector test
(separately1 pass2.04s, included in final focused gate). Same five previously
reproduced on clean tg-base: three obsolete ID/helper-name expectations and
two real exact-region-feed int32/int64 conflicts. No exclusions or masking.
New object test file has21 cases. Seven audits retain0,1,0,1,5,0,0 structural
findings and unchanged unsourced fact/identity counts; provenance latch OPEN.

Remaining observed source-composition gap: inspect(book,handle) reading members
works directly, but root(book,handle)->inspect receives a matching Object row
with only kind span; caller materialize_declared_row_columns defers table leaf.
It refuses metadata.Object.members absent, NOT a different Book/Object identity.
This expected refusal test is not a claim that automatic binding is solved.
Ordinary Python object entry still has no accepted descriptor-domain conversion.
Exact original50-line copier retry3.0152095s remains blocked at recursive workspace
12/callsite14/SCC0, unknown type/extent. No copier installed or speedup measured.
Next: preserve complete table-field context along exact source call binding,
then native runtime builtin dispatch, recursive result/storage constraints and
constructor extent/owner lifetime. Do not fabricate a registry or opaque ABI
simply to bypass unresolved payload storage. No main tiling/parallelism claim.

2026-10-10 steps1-3 started: native descriptor inspection — SNAPSHOT ONLY:

User: "Go ahead and start on 1-3"
Patch9 native_reference_inspection.patch staged UNAPPLIED. All nine patches
combined git apply --check0. Only snapshot fortran_c_shell.py,
precompile_to_ssa.py and new test_native_reference_inspection.py changed.
Live source/copier/accelerator installation and running main7196 untouched.

Observed chain: keyed row4 already has metadata.Object identity, but members
GetAttrs5/6/10 were not table declarations. _field_slot_ops recognized root
parameters or Sequence[Record] rows only. New _graph_record_abi_by_value is a
DERIVED view of existing published graph identities + declared ProgramABI
schemas, indexed once (nodes+schemas), not a new object registry or name guess.
Detached accepted parameter-record receipts remain valid authorities. This view
feeds both table declarations and column dtypes before SSA planning handoff.

Direct len already had sequence_length_values {6:4}, yet finish dropped it
because no region published it. Control finish now materializes a required,
already-declared resident length; no provisional ABI formal. Nested record
materialization previously accepted projected-loop formals only; it now binds
direct resident descriptor members to the same existing column/length/stride
ABI. It keeps the exact GetAttr identity as the native first-column pointer.
Repeated reads of one exact owner/field publish planning aliases and retire
duplicate provisional descriptors with derived book edges. Intermediate audit
caught a dropped handle producer and shared-member claims; both are fixed.

New native public-source cases: len(item.members), both indexed reference
columns, zero-trip/nonempty traversal, and combined repeated-field reads.
All use incumbent keyed value_record/table ABI, caller-provided columns, one
compiled program invocation, no host callback in traversal. Payloads above2**60
survive exactly in indexed and combined tests. Each native case asserts zero
STRUCTURAL concordance findings. The unsourced provenance latch is still OPEN.
Traversal count is explicitly floating; it does NOT prove integer accumulation.

Observed resident length int64 vs hard-coded int/i32 loop header caused invalid
Phi while latch settled i64. Header/latch counter now follows range operands;
preheader seeds widen by explicit Cast. This fixes this table traversal, NOT
the inherited simple-indexed-loop/pair-update Phi failures, still1 fail/3 errors
in3.19s. Integer accumulator probes also found loop_composer float(literal)
seed conversion and int64 carried seed vs float64 planned body result. A direct
reference carried assignment/return also exposed an unnamed externalized output.
Their age has not been established by this turn; both remain unresolved, not
papered over. No compiler-wide integer/alias completion claim.

Step3 now derives an empty dictionary's row from ALL exact IndexedStore operand
constraints: declared scalar/reference parameter ABI, linked scalar/reference ABI,
or literal operands. Unknown/conflicting stores do not become known from one
write. Public source copied={}; copied[key]=value; return len(copied) has
(int64,opaque_ref) descriptor. This is layout proof only: source constructor
capacity/owned allocation is still step4 and that local-dictionary artifact was
NOT executed. No invented bound for dynamic keys or memo population.

Final focused gate45 passed/29.47s:
python -u -m pytest -q --tb=short --show-capture=no tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py
NODUS_TENSOR_CORE remains the existing nodus/build/Release/nodus_tensor_core.dll.
Extended gate97 pass/5 existing failures37.33s before adding two binding-authority
unit tests. Same five previously baselined failures; no exclusions. Seven audits
before/after retain0,1,0,1,5,0,0, unsourced latch OPEN. Saved pre-edit modules used
read-only in fresh process for the before audit; no checkout/stash manipulation.

Exact original50-line copier final retry2.1708728s: same recursive workspace12,
callsite14/callee_reference1/SCC0, unknown dtype/extent. Generic opaque input
still lacks an accepted runtime descriptor binding; builtin runtime kind dispatch
and recursive result row constraints are NOT finished. The explicit descriptor
inspection gap is closed, not all of steps1-3. No replacement copier installed,
no N4 run or end-to-end speedup inferred. Capacity/arena placement remains separate.

2026-10-10 reference-valued native boundary — SNAPSHOT ONLY:

User: "Can we take care of everything and try to leave the where and how do we allocate an arena ideally"

Incremental patch8 native_reference_boundary.patch is staged, NOT applied.
All eight promotion patches combined git apply --check returned0. Compiler
implementation/test edits remain in native-fallback-link/Powershell/turing;
live main7196 and incumbent copier are untouched. No copier accelerator installed.

Observed public source memo[key]=value; return memo[key] with explicit keyed
ProgramABI initially emitted flat GEP/Store/Load, despite existing row receipt.
No unsafe generated code was executed. _field_slot_ops required aggregate_kind
on Input, weaker than the exact committed boundary. It now derives parameter
row declarations/dtypes from incumbent ProgramABI/annotations and recognizes
their exact initial SSA identities. object annotation means opaque_ref SLOT,
not inspectable object payload. Conflicting row contracts refuse.

Next observed native root ignored caller buffers and allocated six singleton
uninitialized private frame cells. _ControlSSABuilder._sequence_descriptor now
publishes borrowed parameter ownership for explicitly declared known row
layouts at creation. An initially broader ownership change misclassified a
provisional unknown text-key view, redirecting keyed tensor payloads; observer
identified exact writer, and explicit-layout requirement fixes that regression.
lower_table_store also publishes length/index/latch arithmetic as int64, matching
descriptor cells. Storage-member binding carries owner identity and refuses
direct-arena/heap-pointer-cell mismatch before mutating bindings.

Native source tests through lower_ast_source_to_ssa execute identity-key memo
insert/lookup/update with handles above2**60, exact length/rows, caller capacity4.
Both explicit ABI and dict[int,object] annotation paths pass. Existing keyed
value_record Book.objects -> Object.kind ABI executes native kind lookup using
caller-provided descriptor columns. This is NOT an automatic opaque-handle
object registry, generic Python dictionary equality, or full copier completion.

Focused new boundary/heap/public-source gate:35 pass/18.37s.
Repro (set NODUS_TENSOR_CORE to nodus/build/Release/nodus_tensor_core.dll):
python -u -m pytest -q --tb=short --show-capture=no tests/test_native_reference_boundary.py tests/test_ssa_owned_heap.py tests/test_native_structural_producers.py
Extended gate adds test_ir_sequence_tables.py and
test_sequence_contract_concordance.py:89 pass/5 fail/26.96s. Same five previously
reproduced on untouched tg-base: obsolete fresh-ID expectation, two fixed helper
name expectations, two real exact-region-feed int32/int64 conflicts. No exclusions.
Inherited earlier native loop Phi regression remains OPEN (see prior receipt).
Seven concordance cases retain0,1,0,1,5,0,0 findings; unsourced latch remains OPEN.

Exact original copier retry:2.0398794s, refuses isolate_metadata callsite14,
callee_reference1/SCC0/private_formal12, dtype unknown/shape(), workspace=True.
The empty dictionary's row layout and extent are still unresolved. A nested
Object.members sequence experiment also still rejects len(item.members)
structural Call result16/argument6. Neither was hidden by weakening full-native.
Bare opaque input has no automatic descriptor-column binding; generic container
construction, iteration, immutable publication, and memo growth are not closed.
Therefore arena placement is NOT the only remaining problem. No N4/main speedup
is inferred from these small correctness tests.

Ownership requirement, independent of allocation policy: shared memo and copied
results must outlive recursive calls. Existing caller-owned heap owner can express
that lifetime; allocation backend/placement/capacity policy has not been chosen.
Continue from exact source identities and existing reference/sequence/record ABI,
not a substitute copier or a Python callback per member.

2026-10-10 owned runtime-sized heap allocation — SNAPSHOT ONLY:

Uses incumbent Alloca/Call vocabulary and strict dtype_layout authority.
An explicit caller-owned native pointer cell holds the allocation chain;
arenas survive helper returns and later allocations until explicit native
release. Opaque reference handles are not treated as dereferenceable pointers.
LLVM guards negative/overflow counts and malloc failure with trap. This is
owned storage, not generic Python object semantics or a compiled copier.

Existing SSASequenceDescriptor now carries optional heap_owner_value_id;
existing sequence book member rows retain it, and concordance checks malformed
allocation contracts. Runtime-sized FIXED capacity is distinct from dynamic
growth. lower_sequence_allocate initializes typed scalar columns; existing
insert/lookup consume pointer cells by exact formal SSA identity. Unsupported
growth, shaped-row initialization, child pools and live-flag initialization
remain explicit refusals. No ordinary source constructor automatically selects
this ABI yet. No per-item Python callback or alternative source compiler.

Native allocator tests cover 0,1,7,4097 elements and int64/int32/float64/float32/
opaque-handle layouts. Returned allocations persist through subsequent helper
calls. One native caller allocates columns, inserts at the final runtime row,
looks up a key above2**60, releases storage, and returns the exact int64 result
for capacities1,7,4097, without host materialization between those operations.
These are internal SSA ABI tests, not public-source copier acceptance.
Final focused heap + real public-source producer regression:29 passed /16.51s
(13 allocation/sequence tests and16 public-source tests). Guestbook dry-run0.

That chain exposed an incumbent LLVM insertion bug: a ptr-only length operand
overrode the declared int64 cell, storing double1.0. Untouched C:/dev/tg-base
reproduced length4607182418800017408 (bits of1.0). The helper now derives the
length operand from its storage formal; insert/lookup lengths and scan induction
values use matching int64 contracts. Direct-column native regression passes.
Heap-address Load pins its value before release; heap loads/stores honor actual
element alignment and refuse incompatible load layouts.

Final extended batch:83 passed,5 failed /26.77s. All five fail on untouched
C:/dev/tg-base: obsolete exact fresh-ID assertion, two helper-name assertions,
and two genuine keyed tensor feed dtype conflicts. Earlier wider batch:
79 passed,6 failed,3 errors /27.73s. Native loop controls fail on the EXACT
saved pre-heap snapshot too (1 fail,3 errors /3.31s), though two native numerical
tests pass on clean tg-base: an earlier staged-patch regression remains OPEN,
not attributed to this heap change or dismissed as an outdated test.
Audit case counts remain0,1,0,1,5,0,0; unsourced latch OPEN. No audit suppression.

Unchanged50-line full-native metadata copier retry:2.015371s, still refuses
callsite14/private formal12, ordinary empty-dict arena, unknown element dtype
and no proven runtime extent. This ABI removes an allocator/backend absence;
it does NOT resolve source element layouts, runtime object inspection,
memo growth or source-to-heap constructor/frame wiring. No copier installed;
no N4 speedup or main parallelism claim. Main7196 observed alive, untouched.

Promotion patch7:tools/compiler_probes/owned_heap_allocation.patch, UNAPPLIED.
All seven staged patches pass combined git apply --check. Live compiler source
and installed accelerator remain unchanged. Snapshot source:
C:/Users/alber/.codex/worktrees/native-fallback-link/Powershell/turing.

2026-10-10 reference predicates / invocation storage / indexed effect repair — SNAPSHOT ONLY:

Declared builtin reference types now prove positive isinstance predicates through
the existing ProgramABI fact/fold producer. Unknown dynamic subtypes remain
unresolved; no negative fold, suffix-name type guess, Python callback or object
layout assumption. Real native dict/dict, list/(dict,list), object/object micros
return True. custom.dict is NOT assumed to be builtins.dict.

An admitted SCC edge now leases typed, fixed-size private workspace into a real
local zeros SSA producer instead of caller.args. Existing mint/lease cells and
LLVM function-frame allocation are reused; descriptor/accounting authority must
prove workspace ownership. Unknown/dynamic/opaque storage still refuses before
minting. Shared caller-bound storage is not made invocation-local.

Mutable repro exposed two exact structural-fold writers: IndexedStore's authored
ast.Subscript(ctx=Store) was evaluated as a read of [0.0], erasing the effect; after
preserving the write, folding its initializer still reconstructed distinct region
arenas. The fold now retains stores and follows their exact base edges to protect
initializers using the incumbent mutation rule. Untouched live nonrecursive repro
had zero Stores and lost input n. Snapshot native writes/readbacks exact0,1,4,12;
recursive parent value survives child call and returns0,1,10,78 at those depths.
New audit assertion then caught root-only pruning orphaning live capacity/status
descriptor members; root cleanup now uses the existing callee member-liveness rule.

Final targeted native_structural_producers:16 passed /14.85s; root-prune units:
3 passed /2.02s. Intermediate broad gate125 passed/1 failed /136.49s caught the
descriptor defect above (fixed, not excluded). Final focused cross-module gate:
59 passed/86 unrelated deselected /73.63s. Seven final audits retain counts
0,1,0,1,5,0,0. Existing unsourced migration remains OPEN; no whole-book-clean claim.
Combined six-patch apply-check0; guestbook dry-run0, no --apply cleanup.

Exact unchanged50-line copier retried AFTER small success: refuses in1.940700s,
isolate_metadata/callsite14/callee_ref1/SCC0/private arena12, dtype unknown,
shape(), declared_workspace=True. Generic object/container operations and dynamic
arena layout/lifetime remain required. NO native copier installed or measured
four-rule speedup. Source stays isolated; cumulative native_structural_producers
promotion patch staged only. Full-main7196 and live compiler/accelerator unchanged.

2026-10-10 structural producer / recursive activation cleanup — SNAPSHOT ONLY:

Corrects the historical identity diagnosis below: id(value) already had a
native Cast. Its parameter was finalized CALL_ONLY before late materialization,
and private leaf(id(value)) feeds bypassed that producer altogether. One shared
producer now handles both, promotes the exact declared parameter through existing
book cells, and publishes its adopted SSA value. Unused parameters stay unused.
The existing graph descriptor rule now declares id's int64 result before caller
planning: native helper forwarding previously rounded handles above 2**53.
Actual native public/private results are exact at1,4660,2**60+3; no CPython
object/address parity claim. Public identity concordance has0 gating findings.

The existing SCC activation record was posted, but hierarchy freezing required
a callsite clone and dropped recursive PlanCalls. Freeze now uses the catalogue
only for the signature, retaining the actual call without unfolding its body.
Runtime dependency walking admits those exact catalogue targets, not all registry
entries. SCC owners are visited once with an iterative worklist; their completed
plans are frozen after activation. Actual OUR-compiler/LLVM native direct/nested
recursion returns0,1,4,12; mutual recursion returns0,1,4,11. Full-native/no callback,
runtime_closure_only=True. No replacement compiler/package or Python dispatcher.
Original public/private/self/mutual regressions reproduce on untouched LIVE.

Full copier still NOT native. After call recovery it reached6 unresolved calls,
14 undefined operands,7 structural shortfalls (previous19/13). Inspection found
another real defect: allocate_result_storage cloned private callee frame slots
into caller.args along the same SCC; later recursive sites cloned those leases
again. This produced thousands of extra formals, not a missing-frame-input list
(unresolved_frame_value_ids was empty). Now refuse BEFORE mint/signature mutation
on that exact admitted recursive edge. Exact50-line copier rejects in1.918846 s:
caller isolate_metadata, callsite14, callee reference1, SCC0, private formal12;
requires per-invocation native storage, not caller-signature propagation.
This SAFETY REFUSAL is not implementation of invocation-local scratch allocation.
Dynamic isinstance over opaque_ref and opaque memo membership remain unimplemented;
their no-native-substitution refusal tests stay explicit. Earlier integer-return
recursion also exposed an LLVM Phi int64/double mismatch; not repaired here.
Next: native invocation-local frame ownership plus the accepted object operations
ABI, keeping shared memo explicit and distinct from each invocation's scratch.

New targeted test_native_structural_producers:8 pass /7.53 s;5 native positive
cases plus3 explicit capability refusals. Pre-final-safety extended gate154 pass,
1 skipped,2 documented hierarchy cases deselected /129.24 s. One native-loop
execution timeout reproduces unchanged on BOTH live and snapshot /23.68,23.67 s;
do not hide it or remove its existing execution bound. Final-source extended gate:
155 passed,1 skipped,2 documented hierarchy cases deselected /132.22 s. Seven
final-source audits retain0,1,0,1,5,0,0 findings; OPEN-latch unsourced migration
debt still exists. Guestbook dry-run validator0, no --apply cleanup performed.
Six staged patches, including native_structural_producers.patch, pass combined
git apply --check; NONE promoted live. Native source fingerprint changed again;
rebuild normally before future accelerated timing. Full-main7196 left untouched;
no long benchmark or main restart launched in this cleanup.

2026-10-10 requested N4 completed; exact metadata copier native pursuit BLOCKED:

User requested four-rule test. Snapshot (ABI sharing + dependency cone) ran
existing probe_four_rule_timing --graph-workers1,8 resident regions,no trace or
deadline, matching726431cbf86656cc6973da89e78cf980 native storage unit, original
external examples. All4 stale laws rebuilt; OUR snapshot compiler path printed.
Exit0 /507.330979 s (8m27.33):47.617967 s /8.5806% below previous554.948946,
but20.951189 s /4.3076% above older486.379790. Single runs, full-main concurrent;
no causal attribution or orbital execution/parity claim.996 functions.
Combined begin172.714214;begin->SSA300.786486;combined whole334.614719 s.
Remaining delta vs older:13.032525 s pre-combined,7.918600 s combined; prior
combined355.558314 ->300.786486 recovers54.771828 s. Native17 calls/143043 facts
cost0.622484 s INCLUDING transfers, not the hundreds-of-seconds bottleneck.
Output root snapshot build/profile_memory_n4_native_storage_rmf6odxo/profile_memory_n2.
C SHAd73885274cde3e219393c8bdb73b7f3c5a94a611732cc7e9b6d9cd499fe1ef0c;
not identical to older artifacts. Max printed stage RSS6157 MiB (prior6460),
progress peak6141 is a different sample. Book2203575 resident cells/0 spills
(prior1590383,38.56% more); edge/dependents715970 each. Cause of book growth NOT
attributed. Final concordance detector reports its own Unresolved.value_id
exception; compilation exit0 is NOT a clean concordance audit.

User challenged full-main provenance: actual worker7196/parent33064 created
Oct9 18:25:20 America/Chicago under earlier explicit full-source lower request
and timeout-removal request; NOT restarted this turn. Today N4 worker22236/
parent30192 created Oct10 07:35:52, now finished. Main remains untouched.

Read-only follow-up tiny profile (with metadata/caller-work observers):
21.35M calls /9.198 s profile;165 folds/2.048 cumulative seconds,18625 descriptor
calls/1.251 s,515987 copier calls/.860 s,24372 book.post/1.042 s. These costs
OVERLAP; observer itself adds~.452 s, wall9.910813 s is instrumented.33 whole
copies traverse213733 ssa_identity_tokens entries,60 region copies39054; other
receipt copies remain. This does not attribute N4 timing or show a native win.

User then explicitly requested fully compiling the~500k-call dispatcher.
Submitted EXACT incumbent isolate_metadata nested function (50 authored lines)
inside one root taking value+shared memo, through sanctioned public lowerer,
existing reference ProgramABI and full-native/no-callback contract. No copy
algorithm substitution, packing walk, per-item callback or production route
switch. Initial~1 s refusal: predicate comprehension for `if all(new is old
for ...)` assigned to branch body by full AST.walk containment. Existing
ordinary-conditional nesting uses arm slots instead. Same source-slot policy
now applied to heterogeneous loop/conditional nesting in SNAPSHOT ONLY.
Fail-first predicate test fails on snapshot ANDlive;body test already passes.
After fix24 ownership/control tests pass /1.58 s. This is NOT a native copier.
Whole exact source now reaches SSA but full-native gate reports19 undefined
operands plus recursive call-result-unavailable/type/membership structural
results; no LLVM artifact or installed replacement. Minimal reference-input
`return id(value)` rejects unnamed/unaccounted formal; `isinstance(value,dict)`
has unavailable result. tuple(value)/list(value.items()) emit complete LLVM in
minimal compile-only checks, NOT object execution/identity parity. Do not
claim all dynamic container primitives absent or those emissions correct.
Actual blocker is runtime object identity/type and recursive result producers
under existing reference ABI; not performance of integer graph arithmetic.
Need native object-operation lowering/accepted exact object ABI before this
entire copier can run compiled. A wrapper forwarding to Python does NOT meet
the request; no such wrapper installed. Existing copier stays unchanged.
Repro snapshot tools/compiler_probes/probe_native_metadata_copy.py, no timeout.
metadata_copy_control_ownership.patch preserves fix/test/exact repro; all5
staged patches pass combined apply-check, NONE promoted live. Snapshot glsl
change intentionally invalidates the7264 native storage fingerprint; refresh
normally before a later native-accelerated benchmark, never force stale load.
Final combined gate169 passed /41.74 s, same4 audited stale exclusions, actual
nodus DLL env enabled. Seven-case concordance audit after ownership fix retains
0,1,0,1,5,0,0 findings. Guestbook validator dry-run exit0, no --apply cleanup.

2026-10-10 audit of the four original dispatcher failures:

Reproduced on LIVE:4 failed,29 deselected /3.77 s. No compiler or test edits
for this audit. These four are test-maintenance defects, not demonstrated
compiler shortfalls. Do not silently turn the exclusions into passing tests.

- tensor_accessor_and_its_scalar_comparison_stay_with_coordinator: accessor
  verdict COORDINATOR_ACCESSOR still holds; comparison verdict NUMERICAL_WORK
  contradicts the test's old metadata expectation. This is the explicit
  Sept30 scalar-expression-deferral removal, documented in
  docs/DECISION_scalar_expression_deferral_2026-09-30.md. The scalar must have
  one recorded producer, not deferred recomputation by selected consumers.
  Existing public-entry annotated-scalar probe rerun:all3 forms pass.
- a_lookup_nobody_reads_is_not_a_region_boundary:fixture has no ingestion
  scope; raises while deriving output Store identity, before boundary assertions.
- dispatch_exports_a_value_routed_by_identity:same missing scope/identity rows.
- dispatch_exports_an_internal_value_used_outside_the_region:same fixture flaw.

Diagnostic-only wrapper declares ingestion/lexical/operand scopes and posts
INGESTION_VALUE NodeFacts exactly as the existing scoped dispatch-cut fixture
does; actual _dispatch_subgraph and original test assertions unchanged. All3
then PASS: lookup outputs(1,3) ->(3,) when inert ignored; identity export(1,2);
external consumer export(1,2) with2 Store nodes. No production fallback, identity
guard suppression, weakened provenance or altered boundary expectation.
Next authorized test-maintenance pass should modernize these fixtures and the
scalar expectation, preserving all boundary assertions and identity refusal.
This does not establish whole-compiler correctness or GPU dispatch efficiency.

2026-10-10 reusable dependency cone — isolated, prior authorized work:

src/compiler/dependency_cone.py accepts exact identities plus existing relation
readers, FIFO O(Vcone+Econe), or incumbent sweep order to preserve invalidation
cause/event order. parent_record_dependents derives one phase-local O(V+E)
index from authoritative operand records, including absent producer keys;
caller owns rebuilding after writes. No new graph/book/descriptor cache.
Descriptor withdrawal and call-result descendant re-settlement now reuse it.
Reverse64 chain fail-first2016 reads ->126; sizes32/64/128/256 ->62/126/254/510.
40 deterministic mixed-graph oracles retain exact final data/cause/event order;
FIFO re-evaluation retains old visit order. Final focused gate145 pass /44.72 s,
same4 audited exclusions; this gate ran without nodus DLL env and warned about
pure-Python tensor backend. It is a correctness gate, not a timing comparison.
Cold fixed-PYTHONHASHSEED=0 DT original-cone3.9488375 s vs new3.7688789 s:
same730 ABI publications/24106 summed sizes,40 functions,0 findings; both LLVM
hash570695ad11b99133402dc4ffe833b8f658a147584dc611d74538c39e621e7b34.
One sample pair does NOT establish wall gain. Global165 prefolds unchanged.
Optional descendant-wave publication batching was removed; final code retains
original per-query publication policy. No N4 rerun/speedup claim.
dependency_cone.patch plus updated serial_graph_scaling.patch diagnostic and
previous ABI/native-link patches pass combined git apply --check; unapplied
live. Full-main7196 left alone, last read46907.62 CPU seconds/1193.6 MiB RSS.
Snapshot native storage rebuilt/load-verified through OUR public lowerer/LLVM
under726431cbf86656cc6973da89e78cf980:0 findings,1306 unsourced facts/93 identities,
latch OPEN. Native DT actual outputs exact0.9/.25 /11.7404743 s total probe;
0 findings,888 unsourced facts/0 identities, OPEN. Seven existing audit counts
unchanged0,1,0,1,5,0,0. Guestbook validator dry-run exit0, no cleanup --apply.

2026-10-10 shared declarative ABI — isolated, measured work reduction:

User: "Proceed" after the slower N4 handoff. No new long orbital run launched.
Small real _propose_dt_pen public-entry profiling:93 graph extractions,575688
isolate_metadata calls,165 structural-fold calls /13428 fold descriptor reads.
Shape settlement/folding remains a larger candidate, not replaced here.
Exact field counter identifies59460 recursive ABI metadata visits across60
proper regions. Public lowerer additionally deep-copied all schemas per function,
before AND after canonical reduction. Writer inspection: program_abi installed
by whole assignment; readers derive mutable parameter_record_abi views and do
not mutate the declaration in place.

Snapshot now shares that compilation-owned receipt across catalogue functions
and proper compartments, via existing _COMPARTMENT_SHARED_METADATA ownership.
Whole-callsite specialization still deep-isolates its containers. No provenance
disabled, graph identity guessed, native shim, new compiler or package. Schema
assignment work O(functions*schema size) ->O(functions+schema size), and region
receipt copies eliminated; NOT a whole-compiler asymptotic/memory bound.
Fail-first2 tests ->2 pass: actual public catalogue pointer sharing,32 singleton
regions/128 schema records with0 schema traversal; nested specialization edits
cannot alter source. Full gate97 passed /41.32 s with same4 known exclusions.
Tests now split across test_plan_descriptor_publication and
test_shared_program_abi_receipt; new file2 pass /1.38 s after separation.

Metadata profiler now516168 calls (59520 fewer, ~10.34%). Region ABI counter0;
whole-specialization ABI visits remain32703. Cold DT LLVM hash still
570695ad11b99133402dc4ffe833b8f658a147584dc611d74538c39e621e7b34.
Actual native execution via OUR lowerer ->existing LLVM emitter returns exact
0.9/.25;0 concordance findings,888 unsourced facts/0 identities, latch OPEN.
All seven existing audit cases rerun: finding counts remain0,1,0,1,5,0,0.
Before unprofiled DT times3.258604/3.147716/3.400177 s; after metadata-observed
3.072286/3.346539/3.699394 s overlapped regression gate and full-main. NO causal
wall gain established. Repeated same-process LLVM hashes still differ; only
cold identity is asserted. Profiling itself costs~8-10 s vs~3 s unprofiled.
io.open also varies greatly (.776/1.970/.358 s exclusive across samples); one
string-table write accounted for1.584 s in second pre-change profile. Do not
attribute elapsed variance to graph changes. Pool still92 writes/220 reloads
in the slice. No spill format/policy/cache invalidation weakened.

Live promotion patch tools/compiler_probes/shared_program_abi_receipt.patch,
independent core hunks; serial_graph_scaling.patch diagnostic updated. All3
staged patches pass one combined git apply --check, NOT applied live. After
promotion native asset must be rebuilt under its actual fingerprint as usual.
Snapshot storage unit rebuilt and load-verified at fingerprint
718d64f3130504f1fc1db1746f5016ba through OUR public entry/LLVM. Asset audit0
findings, same1306 unsourced facts/93 identities, OPEN latch. Live full-main
PID7196 remains active and untouched, last read45512.125 CPU s/1273.5 MiB.
Next: repeated fold/descriptor work and remaining immutable receipt copies,
using exact writer/reader boundaries and tiny repros. N4 benefit of THIS ABI
change has not been measured; previous554.948946 s remains the large-run receipt.

2026-10-10 remaining receipt scans — isolated, N4 COMPLETED, NO wall gain:

Fail-first real48-add lowering found97 settlement and49 dispatch-carving ABI
publications per fixed graph. Existing settlement walk now publishes once at
the original first read; existing immutable dispatch index owns the same lazy
first-read boundary. Each value still gets a fresh _DescriptorQuery. No shared
descriptor answers, disabled provenance or altered transfer rules.
Gate95 passed /40.29 s; same four known dispatcher failures explicitly excluded.
Real128/256/512 arithmetic:45 publication calls at EACH size, summed graph sizes
14731/29323/58507, vs partial implementation113547/423563/1634445. This receipt
work now doubles rather than quadruples; whole compilation is NOT proved linear.
Observed times1.584164/3.011236/6.743024 s with caller-work observer and concurrent
full-main: no additional wall gain established against partial512 time6.570869.
All three LLVM hashes below still match; zero arithmetic concordance findings.

Existing storage-effects native unit rebuilt through OUR public lowerer/LLVM in
snapshot, fingerprint619c4b55579500dfada7e845e9fab6bf. Asset source audit1690 rows,
188 functions,0 findings, but latch OPEN (1306 unsourced facts /93 identities).
N4 ran from the snapshot with that matching asset, --graph-workers 1,
resident regions8, no trace/deadline, compile only. External original examples
selected via TURING_PROBE_EXTERNAL_ROOT=C:/dev/Powershell/turing; compiler source
path explicitly verified as snapshot. Output root build/
profile_memory_n4_native_storage_mo1powht. Full-main PID7196 still concurrent and
untouched: timing is NOT isolated. Promotion patch refreshed
and git apply --check passes; NOT applied while live source pursuit continues.
Remaining mutable-fold restarts/global work are not fixed by receipt batching.
The Oct9 next-step notes below are historical; the asset and snapshot benchmark
no longer need to wait for promotion. Live promotion still waits for main exit.

Completed exit0:554.948946 s (9m14.95), vs prior486.379790 s (8m06.38):
68.569156 s /14.097863% SLOWER. Both1 graph worker/8 regions/no trace/deadline,
both rebuilt all four laws, both concurrent with full-main. Snapshot additionally
includes staged native-link fix and all serial graph cuts, so this is NOT an
isolated attribution to either one.17 native calls/143043 facts cost0.646459 s
INCLUDING transfers: still not the minutes-long bottleneck. Combined begin
165.077906 s; begin->SSA355.558314 s (old296.812716); combined whole389.868437 s
(old326.696119). About58.75 s of68.57 s extra is in combined source-to-SSA.
996 functions. Max printed stage RSS6460 MiB vs6437; progress peak6180 MiB is
a different sampling measure. Book1590383 resident cells /0 spills. NO memory
gain established. C SHAea56fd8317c754affc0b502870d15262ad67a2ea209ea58c4ae00b7f924b5bb7
differs from prior hash below: value numbering plus storage/call ordering differ.
Not byte parity, and no orbital execution/numerical-parity claim. Small arithmetic
LLVM parity remains established. Do not promote this as a four-law optimization.
Next: attribute combined planning/instantiation and SSA cost with tiny targeted
repros before another long benchmark; fold whole-graph restart/descriptor snapshot
work remains a candidate, not a measured cause of this timing difference.
Full-main last observation PID7196 CPU23235 s /1260.4 MiB; still active, no new
output on session poll. No timeout, source mutation or interruption.

2026-10-09 serial asymptotic reductions — isolated, NOT live:

User: "we need to be cutting down the minutes long processes, we need to be making powers of O reductions".
Implemented in current-source snapshot C:/Users/alber/.codex/worktrees/
native-fallback-link/Powershell/turing; no live compiler source/asset changed
under the no-deadline full-main pursuit. Promotion patch:
tools/compiler_probes/serial_graph_scaling.patch (git apply --check passes).
This patch is independent of the staged native-fallback link patch.

Existing graph/ABI/concordance rules remain authoritative. Changes:
1. ABI identity publication: dirty round worklist over exact parent records,
   with loop-target writes invalidating their actual consumers. Same transfer
   and insertion/round ordering, bounded fixed-point guard retained. Reverse
   record chain changes O(N^2) node work to O(N); NOT a general convergence
   guarantee of O(V+E) for arbitrary conflicting/cyclic identity rules.
2. Region identity-token projection: keyed lookups for region members, not
   scan all function tokens per region. O(R*V) -> O(sum of region sizes)
   for THIS projection only, not all metadata copying or whole compilation.
3. Read-only plan freezing publishes ABI at first descriptor read once.
   Fresh existing _DescriptorQuery per value, no shared descriptor answers.
4. Structural folding batches publication between actual operand/topology
   writes and resets at each sweep. Literal replacement, removal, aliasing
   invalidate the boundary. All descriptor/provenance writes remain active.
   48-add public-entry repro observed388 checks with0 operand writes BEFORE;
   now <=4 per fold invocation. Plan freezing49 ->1 per invocation.

Gate:94 passed /44.29 s, four known dispatcher tests explicitly deselected.
Full dispatcher module: those same four failures reproduced on untouched LIVE
sources (metadata comparison and three ingestion-scope-less synthetic graphs).
Hierarchical gate45 passed /2 failed; both failures also reproduce live:
legacy _sequence_concat_ops unpack count; packet function_address disagreement.
Formal-shape gate16 passed /1 failed; same empty history failure reproduced live.
No fixes/relaxations for those unrelated failures. Original seven-case concordance
finding counts 0,1,0,1,5,0,0 are the comparison baseline; latch remains OPEN.
The graph worklist is compared against frozen original transfer code on48
cases incl40 mixed-rule graphs, loop targets and exact span shape book facts.

Small real public-entry arithmetic scaling (single runs, concurrent unchanged
full-main job; NOT a controlled four-law timing):
128 ops old/new1.994243/1.911884 s; publication calls12019/429.
256 ops old/new3.314352/2.910634 s; publication calls23923/813.
512 ops old/new8.141574/6.570869 s (~19.3% lower observed wall);
publication calls47736/1582; publication time1.256325/.090277 s.
Summed graph sizes checked60064147 ->1634445 at512. Remaining checks still
grow quadratically: whole compiler is NOT O(N), and minutes saved NOT proven.
All three corresponding emitted LLVM texts are BYTE-IDENTICAL (SHA256):
128 b8aed9007c1e95d0a2c76f2075b32c8c0533ea988cc0fdbb157f3c948e48bdbe
256 3b52979e2f0893230363a2b072301465b4ce718911ef0aeae27508fb2512ce97
512 422dcd034394410e30364950594ee7b2409d802127bb16a7e5308bab2dc2e7fb
All three have0 concordance findings. Existing DT proposal native execution
also passes exact outputs0.9 and0.25, public lowering -> existing LLVM emitter.
Cold DT lowering old/new3.806376/3.780554 s: NO meaningful tiny-DT wall gain.
Publication calls15922 ->1372; summed sizes1780234 ->54288. Cold emitted LLVM
SHA570695ad11b99133402dc4ffe833b8f658a147584dc611d74538c39e621e7b34
is identical. Repeated same-process DT emissions are NOT consistently identical
across old/new (second emission differs, third agrees); cause unestablished.
Do not claim universal byte parity from the cold proof or native examples.

Repro in snapshot: python -u tools/compiler_probes/probe_serial_graph_scaling.py --arithmetic-scaling --llvm-proof
Add --original-passes for frozen publication + original per-query policy.
Diagnostic baseline keeps one extra cached publication at each batch boundary;
it is NOT a byte-for-byte historical compiler. No trace or lowering deadline.
For DT probe, set TURING_PROBE_EXAMPLES to live turing/examples in this snapshot,
and NODUS_TENSOR_CORE to the already-built live DLL. No packages/foreign source
compilers installed; no new native compiler unit claimed for these Python passes.
Next: remaining unbatched receipt scans and mutable-fold restart/global scans;
measure exact writer/read boundaries before batching. Promote both staged patches
AFTER main job exits, rebuild native assets with existing fingerprint gate, then
the user can launch the four-law comparison. Do not edit under source pursuit.

2026-10-09 user-authorized N4 native-analysis benchmark — COMPLETED:

Repro: python -u tools/compiler_probes/probe_four_rule_timing.py --graph-workers 1
No trace/deadline; explicit serial source route; 8 resident regions; fresh output
build/profile_memory_n4_native_storage_xl7u1you/profile_memory_n2. Existing
unmodified full-main job runs concurrently: NOT an isolated timing. New native-
link patch is still staged, NOT applied; compiler sources/assets untouched.
Load-only matching native storage-effects asset observed. First actual call
594 value facts /1.4493 ms including transfers. Completed exit0, 486.379790 s
wall (8m06.38), all four stale pieces rebuilt. This is113.068210 s /18.8620%
below latest599.448 s, but21.839790 s /4.7014% slower than older464.540 s.
NOT an isolated causal accelerator-only comparison; see changed settings below.
Installed native analysis:17 calls,143043 value facts,0.510050 s total INCLUDING
packing/unpacking. Combined begin159.681689 s, begin->SSA296.812716 s,
combined end-to-end326.696119 s. Module996 functions, C source SHA256
561cc7e1061aa4f454206e5d0db3308cd910d346ef5c7ae37abf9ae59aae910b.
Maximum printed stage RSS6437 MiB (old5606 MiB); sampled progress peak6255 MiB
is a DIFFERENT sampling measure. No memory improvement. Peak book1623974 cells
resident, zero pages spilled. Compile only, no orbital execution/parity claim.
User has now prioritized RAW SPEED, not parallel/memory-tiling prerequisites.
Use phase timings to select the next substantive SERIAL native conversion.

Initial same-config 4-worker attempt FAILED before its first law: public entry
ensure_graph_foundation -> compile_and_register_dependency_levels_accelerator
-> OUR public lowerer -> _class_surface_ssa_program refused exact carried arms
63/initial21 and34/initial11 as name_arm_version_missing. This is a regression
in the native graph-foundation bootstrap; do not hide it or claim 4-worker gain.
Retry explicitly selects preserved graph-workers=1 lane; comparison with old
599.448 s (4 workers) is confounded by that setting, overlapping full-main job,
and reduced per-AST-node console printing in this new diagnostic. Existing
profiler's stage/RSS/histogram work remains; law staleness/rebuilds remain real.
No compiler fix/change to evade refusal and no other source compiler/packages.

2026-10-09 native-link / authored-fallback boundary — isolated, NOT live:

The no-deadline main self-lowering is still running. Its compiler sources and
installed assets remain unchanged. A current dirty-source snapshot under
C:/Users/alber/.codex/worktrees/native-fallback-link/Powershell/turing holds
the fix and tests; it is NOT the clean historical baseline or a new compiler.
Promotion patch: tools/compiler_probes/native_fallback_link_boundary.patch.
git apply --check succeeds on the live dirty tree; DO NOT apply under the
running main source-pursuit job. Promote afterward, then regenerate the native
storage-effects asset under the existing changed-toolchain fingerprint gate.

Observed: installed.__turing_deployed_callable__ IS the real LLVMPiece, but
llvm_piece_of(installed) returned None. Public lowering reached graph source
pursuit -> ExtractionContract.decide -> subject -> inspect.getsource on the
installed fallback; the source-read trap failed there. Fix follows only that
existing deployed edge and validates the ordinary piece ABI; no graph/ABI copy,
name inference, foreign source compiler, or new package. Native verification
flags alone remain insufficient; unrelated __wrapped__ decorators and cyclic
deployment references do not become native links.

Further trapped consumers: declare_external discarded the resolved piece and
compiled a new signature; retained mapping/sequence factories checked .module
on the wrapper and published no native bindings. They now consult the resolved
piece. External slots reuse exact buffer/value IDs and the native entry address,
not a Python callback. Retained-container tests stop at the actual source graph
boundary; they are NOT whole-record native execution proofs.

Explicitly rebuilding a submitted root must still compile its authored body.
The old auto-link writer attached the existing bodyless native module to that
submitted definition; deployment yielded None and the tail failed on .metadata.
The existing selected/whole-source root election now excludes its native
binding from dependency auto-linking. Other native dependencies stay linked.

14 boundary regressions pass in 12.30 s. Actual caller native execution agrees
exactly: [5,1,13.25]; direct/wrapped links contain just caller plus one bodyless
declaration, 0 fallback source reads and 0 concordance findings. Direct/wrapped
lowering .549430/.540208 s, single overlapping-run samples NOT a speed claim.
Existing source/activation/extraction and payload retention gates also run in
the isolated snapshot: final combined gate 59 passed /46.25 s. Repro there:
python -u -m pytest tests/test_native_fallback_link_boundary.py tests/test_source_realization.py tests/test_compiler_bootstrap_runtime.py tests/test_extraction_contract.py tests/test_llvm_piece_runtime_payload.py -q --tb=short
A native runtime adapter lacking a declared link ABI is
NOT covered by this change; do not bypass source ingestion based on a DLL flag.
Initial fresh-snapshot tensor imports automatically built the existing CFFI
bridge; that is incidental backend initialization, not the compiled test unit.
Later gates point NODUS_TENSOR_CORE at the existing live DLL and reuse cached
bridge state. No package installed. All tested program compilation uses the
sanctioned lower_ast_source_to_ssa entry and existing LLVM emitter/toolchain.

2026-10-09 entire main compiler submitted to its own public lowerer:

User explicitly removed the diagnostic timeout. The current repro below has
NO parent deadline and no periodic faulthandler "Timeout" dumps. A new run
was launched without either; the45/90-second results below are historical
terminated attempts, not the current run or semantic compiler refusals.

Repro: python -u tools/compiler_probes/probe_main_compiler_lowering.py
Unchanged complete fortran_c_shell.py (45,000 lines), selected public entry
lower_ast_source_to_ssa; full-native execution overlay on the existing
program_extraction contract, actual module bindings, normal reachable closure.
Source SHA256 ef4ba61f2107de15b3ec51a9aa7a0f6fdf4b035fc70cf98577bf4b24f7d74036.
No extracted helper, substitute compiler, target execution or compiler edits.

Initial45-second and stack-observed90-second children reached source closure
but hit their diagnostic deadlines BEFORE ProcessGraph build/SSA completion.
This is NOT a semantic refusal or proof of an unsupported construct/ABI.
The90-second run discovered the real main implementation at work item33 and
ProcessGraph at438, then graph linking at443. At25 and50 seconds both stacks
showed deepcopy(cached) in graph_express2._source_ast_definition line788,
called by class-field/constructor reference resolution. One route was
_ast_definition_bindings -> _class_field_reference; both ran beneath
_expand_unresolved_ast_parents, invoked by main entry line44277 and impl42848.
At75 seconds the stack was traversing AST children. These samples identify
observed front-end work, not a complete timing attribution or infinite loop.
No main SSA module/native artifact produced; no concordance-success claim.
Probe streams progress through sys.__stdout__ because discovery internally
redirects incidental stdout. Repeating25-second stack dumps are CPU-stack
diagnostics only, NOT malloc tracing. Parent kills only its own timed-out child.
Installed fused analysis component remains unchanged and compatible.

2026-10-09 FUSED native effects + pointer-storage inference — latest:

Both whole-module analyses now execute in ONE invocation of the same authored
~230-line program through OUR compiler. Pointer inference preserves the exact
local GEP/PtrCast relation (NOT View/call edges), Load-result/Store-value dtype
occurrences, no-evidence/exact-type/conflict joins and 'ptr' publication veto.
Known physical types/accounting survive. Imported effect facts override body
effects while original local typed-memory inference still reads the body.
Native owns both fixed points, CSR/queue work and final tables. Host only packs
existing SSA facts and publishes final formal accounting. Scratch is reused;
native zeros allocate directly, without duplicate Python zero lists.

Matching default asset: build/compiler_foundations/storage_effects/products/
e2727d33f52be72e48858609d2d2a5ff/compiler_storage_effects__read_only_storage_effects.dll.
Same build-once/checkpoint/fingerprint/digest LOAD ONLY deployment. No alternate
source compiler/packages. The historical asset and numbers below predate fusion.
62 modules / 61 native calls, exact all-value effects and complete formal
accounting on 27 pointer modules. Fresh-process real serial lowering reuses
the component in 3 rounds / 2090 facts, binary outputs .9/.25 unchanged.
Native/audit gate: 4 passed / 32.59 s. Focused branches/adapters: 17 / 17.66 s.

SOLO combined-section medians, packing/unpacking/publication included:
25 values .059400/.134200 ms Python/native (tiny loss).
648 values 10.843000/1.061175 ms (10.218x).
5400 values / 384 queries / 5040 instructions / 9970 operand occurrences:
471.711250/8.369925 ms (56.358x), saved 463.341325 ms per section pass.
Separate Python stages on the largest input: pointer inference 14.593350 ms,
read-only queries 454.328100 ms. Independent medians need not sum; 56x is NOT
incremental pointer-only or end-to-end/N4 speedup. Cold lookup/load .070959 s.
Repro: python -u tools/compiler_probes/probe_storage_effect_timing.py
35-second child bound; public source entry forbidden during section benchmark.

Exact trapped faults fixed at their writers: true-only store-version 462
rebound Phi319 -> Phi321, then used on parent false edge in Phi322(321,321).
Carried readers now require existing exact-object CFG dominance, then use the
existing concorded alias/snapshot rules. Separately incumbent/proposed shared
366; only proposed merges to 373/374. Blanket merge publication changed366
to374 and compared proposed with itself. Existing name_binding facts protect
the other binding's source. Tiny native branch regression trapped the later
reconcile_conditional_phi_continuations writer repeating 0->Phi18; it honors
the same Phi receipt. Audit verifies the receipt against scope_rows of its
declared read scope, including copy-on-read scope_origin projections.
Observers: build/observe_storage_phi.py, build/observe_shared_conditional.py.
0 structural findings; unsourced latch OPEN, not a closed provenance audit.
Legacy whole-object conditional test still returns None module and also fails
untouched C:/dev/tg-base (2.66 s); excluded from final green gate, not hidden.
Baseline cannot lower the whole larger source, so its Phi fault's age is unknown.
No N4 launched. Main refinement/planning is NOT converted to native yet;
whole-source/final-SSA memory tiling and main compile parallelism unfinished.

2026-10-09 whole-module native storage-effect component — historical before fusion:

OUR public compiler compiled graph_refinement_kernels.read_only_storage_effects
as a complete program: dependency construction, reverse CSR, definitions and
write seeds, alias/call fixed point, queue and final Boolean table are native.
No Python callback/per-value native call inside that analysis. SSA marshalling
and consuming its results remain host boundaries, not a replacement compiler.
The physical-call adaptation round now uses one table per whole module. Missing
or stale installation retains the existing _read_only_feed route, never auto-
compiles this component. Existing ArtifactCheckpointStore saves a stripped
artifact under a content-addressed immutable product directory; exact compiler,
authored/host source and platform fingerprint plus DLL digest guard load-only.
Explicit builder/load APIs live in native_compiler_accelerators.py.

Default installed asset: build/compiler_foundations/storage_effects/products/
6d37c031b07e725a640a6ef274295df1/compiler_storage_effects__read_only_storage_effects.dll.
Changing fingerprinted compiler files intentionally invalidates the installation;
explicitly regenerate it after such edits, never weaken the matching gate.

Latest native gate: 2 passed in 26.41 s. 35 SSA modules (one empty), 34 native
calls, every value agrees with the original recursive analyzer; pure recursion,
recursive writes, alias definitions vs address effects, Store value/index roles,
imported read-only facts and exact callee formals included. Fresh child loads
without compilation, actual serial compiler performs 3 whole-module calls /
2090 value facts and native outputs .9/.25 exactly. DT is only an integration
input, not what was compiled. Structural findings 0, unsourced latch OPEN.

Section benchmark is NOT a DT or whole-compile benchmark. Repro:
python -u tools/compiler_probes/probe_storage_effect_timing.py
35-second parent; public compiler entry is forbidden, proving LOAD ONLY.
Five-sample medians include input packing and complete output table unpacking:
1 function/4 formal queries/21 values: Python .03475 ms, native .09485 ms.
8 functions/64 queries/584 values: 8.78095 / .95705 ms (9.175x).
24 functions/384 queries/5016 values, 4656 instructions/9586 operands:
434.70860 / 7.904625 ms (54.994x), saves 426.80398 ms. Native supplies ALL
value facts in one call, Python is timed only on actual formal-query usage.
Fresh-process asset lookup/DLL load separately .068812 s. Synthetic SSA call
chains, not N4 or an end-to-end compiler speedup. Tiny inputs remain slower.

This larger authored program exposed invalid nested conditional pointer Phis:
arm 215/216 (entered loop versions) was rebound to child Phi145/Phi169 by
lower_conditional->_rebind_view->_bind; parent false edges then consumed the
child Phi. Observed stacks/old-new writes in build/observe_storage_phi.py.
lower_conditional now snapshots already-bound true/false arm identities through
the existing carried_snapshot rows, not only source initials 37/38. No future
branch-local value is materialized. LLVM accepts the unchanged large program.
The clean baseline cannot lower this whole program (overlapping-control-region
refusal), so it did NOT establish the age of the Phi defect. No baseline claim.

The earlier native dead-metadata worklist is preserved but not the active
component: 81 graph cases (80 native calls) match exact removal order; real
energy lowering's 26 passes match graph/book rows, 25.24 s. Its Python graph/
book publisher boundary was rejected by the user as insufficient. Keep that
history distinct from the now-complete native storage-effect analysis.

2026-10-09 first native serial-unit checkpoint:

Descriptor broadcast integer law now has SAME authored Python eager/native
body in graph_refinement_kernels.py. Explicit existing-registry builder:
native_compiler_accelerators.compile_and_register_broadcast_shapes_accelerator.
Only admitted proven operand extents cross the ABI; original unknown/conflict
guards, dtype/Precision and concordance publication remain. No automatic
compilation/registration or foreign source compiler/packages. Per-invocation
scratch/feeds; two overlapping native invocations matched 407 exact cases.
Int64-inadmissible extents use SAME eager body before execution, not coercion
or native-fault rescue. 2**63/2**64-1/2**100/non-integer cases gated.

Real SERIAL DT proposal Python/native/Python: exact .9/.25 every time, 0
structural findings, unsourced latch OPEN. Only 53 admitted broadcasts, .170-
.175 ms eager TOTAL vs 4.326 ms native. Public lowering 3.765/3.812/3.907 s;
whole check 11.131/10.970/11.104 s: NO compiler speedup. Final solo bulk ABI
check (packing/unpacking included): 4096 rank8 7.504/6.753 ms eager/native;
rank32 19.926/17.303 ms. Single rank2 1.435/36.367 us, 256 rank2 .271/.305 ms.
Native availability is verified; performance is workload-dependent. Keep
eager/default serial and both preserved routes. Do NOT launch N4 for this.

Final targeted gate 16 pass/1 deselected in 12.58 s. The excluded projection
history test ALSO fails unchanged on clean C:/dev/tg-base (0.60 s):
test_callsite_formal_shape_concordance.py::test_existing_call_projection_is_enriched_by_later_exact_descriptor.
It reads () instead of two callsite_projection_specialization rows. No fix
or identity-key weakening. Earlier combined gate: 13 pass/that 1 fail,11.13 s.
Syntax/whitespace pass. Bounded probe tools/compiler_probes/probe_native_shape_transfer.py
has 55-second parent and ten-second native children; --benchmark-only skips
DT runs. Artifacts build/native-shape-transfer-checkpoint / -solo. No N4 run.

New CPU-only serial profile 20,292,830 calls/7.850 s. Fold 1.901 s cumulative
but .083 self; descriptor 1.447/.309, shape publication .314, ABI identity
publication .199, book post .903 s. These overlap and are not normal timings.
Next native selection must measure an actual larger structural/dataflow
island, not treat whole-query cumulative time as extent arithmetic cost.
See sibling speaktome/todo/turing_graph_residency_workers.stub.md.

2026-10-09 preserved-route / native-serial roadmap checkpoint:

User redirected further architectural tiling work toward identifying native
units in the reliable serial process. Nothing was broadly reverted. Both DT
routes are explicit: lowered_system(source_compilation="serial"|"units").
The previously enabled units default is unchanged; serial omits partitioned
source options entirely. Public lower_ast_source_to_ssa remains serial when
partitioned_directory is absent. Full orbital compatibility of units is NOT
established. The current roadmap is in
speaktome/todo/turing_graph_residency_workers.stub.md (workspace sibling).

Real DT proposal now lowers/links/executes through four units (84/24/135/10
nodes), peak two workers, exact outputs 0.9 and 0.25. Clean serial/native control
also returns both exactly. Unit/native total 25.943 s vs clean serial 10.801 s:
NOT a speedup. Artifacts: build/dt-proposal-native-owned-q7cwmwj3 and
build/dt-proposal-native-baseline-24cl0x44. Worker peak RSS 510-550 MB, private
1.55-1.59 GB; these small slices do not prove total/compiler-wide memory bounds.
Structural findings zero; the unsourced latch remains OPEN, not fully closed.
Distribution=None/dx/dt_current facts use the existing specialization operator
before the source cut. Trapped writers: literal-formal prune at ~29690;
conceptual-record prunes at ~26613/~30046/~37391. All now respect imported
native ownership. Accepted authored/physical formals, not old source positions,
drive native argument binding after specialization.

Canonical expanded/planning instruction snapshots share one BookSpill cache
per control lowering, default 8, with audited policy receipts. Existing graph
region partitioning uses work-contract graph_region_node_limit (default 256).
TURING_GRAPH_REGION_NODE_LIMIT / TURING_SSA_RESIDENT_REGIONS provide the bounds;
WorkContract.ssa_resident_regions=None retains eager snapshots. These bounds
do NOT page final SSA bodies/source ingress or automatically subdivide an
oversized source SCC. Do not describe that unfinished work as complete.

Receipts: 7 new cache/literal/specialization checks passed in 28.31 s; two-worker
two-chain conditional, graph limit 8/cache limit 1, passed exact native
64701152/-4 in 19.48 s. 10 cache/audit/counted/while/carried-alias checks passed
in 16.45 s under 8/1. Three record/default/nested-return regressions passed in
48.43 s. Final route/dispatch/hierarchy selection: 33 passed, 4 failed in
4.81 s. ALL FOUR were reproduced on untouched C:/dev/tg-base tests in 3.12 s:
tensor_accessor_and_its_scalar_comparison_stay_with_coordinator plus
a_lookup_nobody_reads_is_not_a_region_boundary / dispatch_exports_a_value_routed_by_identity /
dispatch_exports_an_internal_value_used_outside_the_region. The latter three
construct graphs with no ingestion scope; do not suppress their identity
requirements or blame this checkpoint. Earlier typo in a selected test name
ran no tests and is not a passing gate.

Small serial CPU profile, no allocation tracing: 20,293,257 calls / 7.694 s
instrumented. Structural fold 1.852 s cumulative, tensor descriptors 18,627
queries / 1.378 s, call metadata 0.439 s, result type settlement 0.098 s.
Times overlap and are NOT uninstrumented benchmarks. Metadata isolation
575,688 calls / 0.879 s; book posting 0.889 s. SCC 0.000079 s and effect walk
0.005353 s are not first priorities on this evidence. No new native conversion
was implemented; existing topology/levels providers remain available. Observer
build/profile_serial_dt_slice.py is 35-second bounded and prints inline.

Final isolated route selection gate: 3 passed in 1.49 s (default units,
explicit units, explicit serial). Syntax/whitespace checks and guestbook
filename validation passed; validator was dry-run only. No Python probe remains.

2026-10-09 deterministic native-frontier contraction checkpoint:

A record-identity-changing root(Actual)->relay(Declared)->leaf(Declared)
micro returned native 16 rather than 59 under separate generic headers,
despite zero structural concordance findings. Untouched C:/dev/tg-base serial
compiler returned exact 59 (5.40 s). The interface planner now computes finite
reachability over exact record-parameter forwarding edges. Those indivisible
edges enter the EXISTING SCC operator as reverse planning edges; convex
components stay together, unrelated scalar branches remain separate, original
semantic recursion remains unchanged. No physical field-name/type matching was
added. Worker validation re-derives the same relation from its sealed input;
the source-node cap still refuses oversized components before admission.

Record-only callsite variants were also absent from native symbol identity.
The existing variant scope/record ABI now supplies that key. A separate exact
writer fault was found: the child bytes-join scan overwrote enclosing
function_reference, stamping relay with leaf's reference and root with relay's.
The child variable is separate now. Native imports join accepted source refs,
not positional exports or name proxies; only canonical function entries are
headers, while specialized bodies remain in their own object/diagnostic SSA.

The exact native 59 case now passes with two worker processes; a later caller
also imports the contracted component and returns 59 (20.83 s). Final checks:
3 record/serial/import cases in 45.52 s; 4 tuple/tensor/law regressions in
58.68 s; 5 record/default/convexity/node-cap/acceptance checks in 28.80 s.
All 12 passed. Syntax, whitespace and guestbook-name validation passed. This is a small
real lowering/ABI correction, NOT an end-to-end bounded-RAM/multicore N4 claim.
The last actual DT proposal cut has separate 84-node energy, 24-node scalar,
135-node record component and 10-node root inputs. Record forwarding proceeds;
it remains PARTIAL at source call 8 / untyped distribution formal 0, and its
caller refuses the incomplete dependency. No execution or speedup claimed.
The default DT unit route remains unverified for complete orbital programs.
Initial source closure and the worker's fixed tensor basis remain resident.

tests/test_compilation_units.py has two book-lifetime failures on both the
dirty compiler and pre-imported clean baseline: function_address disagreement
in plan_identity_uses_qualified_names_not_reference_order and
portable_plan_receipt_is_recorded_on_the_project_graph. Do not attribute these
to contraction or silence them. The new convexity/recursion constraint test
passes separately. An earlier combined micro regression hit its outer 80 s
bound; that interrupted run was NOT a suite pass and left no compiler process.

2026-10-09 main unit-boundary completion checkpoint:

Normal `examples/llvm_dt_system.lowered_system` link-mode C/LLVM compilation
now requests the existing public source-unit runner, with at least two jobs
and a fresh `source-units_*` directory. Inline mode is unchanged. A plumbing
test verifies the actual `dt_system_over` entry requests this path. This is
NOT yet an end-to-end N4 compatibility or performance verification: full DT
can still hit the explicit sequence/reference/nested-record/runtime-extent,
runtime-slot/text-sink and oversized-SCC refusals. No long compile was launched.

Record boundaries carry existing boundary-closed SSARecordDescriptor rows.
The existing record-field forwarding graph now consumes bodyless callee
descriptors, rather than treating required fields as uninitialized scratch.
Authored return slots/aggregate edges supply native output paths; authored
and native calls use one shared projection operator. Nested tuple tests link
and execute exact 189; flat tuple 105; record caller 13; retained native law
42. Accepted linked pieces reuse their existing native IR, never lower an
empty declaration as an implementation. Scalar Return/Phi boundaries use
their actual published SSA value. Late unused private root inputs are pruned
with the existing root prune. Unit conditional accounting reads the existing
source receipt, not a source graph already mutated by deployment.

The actual DT-helper slice first exposed a broken archive boundary:
ReturnBlock.source_cell retained an ingestion Ref without its fact cell.
`IdentityBook.project_cells` now transports the exact predecessor closure:
unchanged Ref keys, facts, stamps, mint rows, edge/dependent order and unsourced
receipts. The source archive keeps that acceptance view so legacy later cuts
also work after the active book closes. Each worker gets only its selected
acceptance subgraph; the complete source diagnostic book stays separately
stowed. Exact cell/index transport and exclusion of unrelated siblings are
tested. This is a graph restriction, not a second provenance schema.

`tools/compiler_probes/probe_dt_unit_interfaces.py FRESH_DIRECTORY` compiles
the actual unmodified `_energy_time_limit`, `_no_exchange_observed`, `_scalar`
and a small calling program through the public compiler. One-participant
contract, four units, peak two distinct processes, 84/24/34/16 source nodes
and 15/15/10/5 acceptance cells; no dependency bodies imported. Latest bounded
run completed lowering/link/native check in 20.464 s, four compiled-unverified
receipts, zero findings, exact native output 0.0. Its participant extent proves
the two early returns: this is NOT broad DT numerical/optional-field coverage.
Artifacts: build/dt-unit-micro-6knmgfqx. Native execution is isolated at 10 s;
compile workers at 20 s; outer probe at 60 s in this session.

Initial ingestion still holds the source closure. A supplied tensor repository
is serialized once, but each worker presently loads that complete fixed basis;
it is not yet reduced to the worker's operator dependency closure. Admission
limits and sampled per-worker limits do not make arbitrary growth a hard RAM
bound. Do not call this the complete architectural conversion or an N4 speedup.

Regression receipt: 119 passed in 140.62 s across identity-book, main boundary,
scoped-input and project-product suites. A subsequent one-tensor-element test
exposed overbroad projection ownership: `leaf(x)[0]` must remain a numerical
Indexed region when leaf returns one tensor, not an authored tuple. The native
projection set now consults the accepted return kind/arity. Final six main
boundary checks passed in 72.41 s, including exact tensor element result 43.
Syntax and whitespace checks pass. Guestbook validator reports conforming names.

2026-10-09 four-law link-interface measurement and same-Python kernel probe:

`lower_ast_source_to_ssa(..., retain_linked_piece_bodies=False)` imports each
retained LLVMPiece as a bodyless, copied ABI/effect declaration instead of all
private SSA functions. The full piece/archive remains unchanged. The existing
`llvm_dt_system.lowered_system` now requests this in link mode; inline mode and
the public compiler's default retain bodies. `linked_piece_interface` receipts
preserve actual SSA IDs, types, shapes, argument effects, native buffer/extent
order and LLVM digest; concordance detects disagreement. Aggregate-layout
interfaces unsupported by this seam refuse rather than lose their descriptors.

User-authorized N4, no trace, graph workers 4/resident regions 8, four stale law
rebuilds, fresh TEMP/TMP output directory: 599.448 s, exit 0, versus preceding
697.227 s (14.0% lower in one sample). Combined-system begin -> SSA lowered:
146.1 -> 550.4 s (404.3 s), versus 175.1 -> 608.4 (433.3 s). Module functions
1066 -> 996. Maximum RSS in printed stage rows 5632 -> 5606 MiB: essentially
unchanged. The book still has 1,828,426 resident cells, zero spills. Observed
CPU remains approximately one core. Earlier 464.540 s is STILL faster than
this run. Do not describe this as end-to-end bounded-memory/multicore success.
C SHA a0c6ae19fe4fd577830bf7d17ee6b167e5650d39e90576790fbfc87fc469f1e7;
artifacts: build/profile_memory_n4_link_headers_20261009_133215/profile_memory_n2.
This benchmark compiles; it does not execute orbital physics or prove parity.

New bounded probe `tools/compiler_probes/probe_graph_kernel_modes.py` directly
runs the existing authored `graph_refinement_kernels.dependency_levels` eagerly,
uses an already registered native provider / explicit trusted native-worker
payload, or `--compile FRESH_DIRECTORY` builds through the existing public
source compiler accelerator API. Fresh builds save a native-worker payload
with source/payload/library digest receipt for `--native-payload PATH` reuse.
No package or other source compiler is introduced. Invocation-local CSR and
scratch spans support `--workers 2`; packing/setup is timed separately.
4096 nodes/8186 edges, 16 ASAP + 16 ALAP repeats: eager kernel sum 138.34 ms,
native sum 0.8374 ms; wall 179.90 vs 12.53 ms. Two workers wall 13.52 ms,
not faster at this size. These are one-sample kernel timings, NOT compiler
speedup; the production Python registry fallback still uses ILPScheduler.
Focused checks: 13 piece-retention/accelerator tests in 35.75 s; 7 probe tests
in 12.78 s, including optional compile/reuse, two owned arenas and stale-source
refusal. Native child execution is bounded to 10/15 s; compilation check 30 s.
LLVM arithmetic and direct C linked buffer returns are exact. An unrelated C
publication defect for adding two unpacked tensor results reproduces both with
full/compact imports and on untouched C:/dev/tg-base at 54c67fbd (8.51 s).
Do not silently classify it as fixed or add a numerical tolerance.

2026-10-09 native interface/body separation (supersedes the limits of the next checkpoint):

The canonical `lower_ast_source_to_ssa` now accepts `partitioned_directory`
and `partitioned_options` (jobs, source-node bound, worker memory/time limits).
This opt-in lane plans through that same public entry, releases the planning
heap, and compiles SCC bodies in bounded separate processes. Worker inputs
contain ONLY their SCC, not transitive authored dependencies. Callers import
digest-checked bodyless declarations and already-compiled native objects;
dependency SSA/graphs are not reloaded. Final C/LLVM emission links object paths,
not all SSA bodies. C's existing standalone host also links these objects.
CLI product lane: add `--interface-tiles` to `--partitioned-source`.
Use a fresh directory. Reusing a completed source-product directory refuses
before overwriting its inputs; resume the existing frozen resolved plan instead.
Requested entrypoints are selected by their recorded function reference, not
by reconstructing an export name. Unit/plan/source-SSA digests are cross-checked.

The interface is a derived view of accepted SSA values, preserving their IDs,
dtype/shape/accounting, actual native buffer/extent order, and per-formal
read/write summaries. The concordance detects declaration/ABI disagreements.
Both emission and link acceptance refuse stale contracts or object hashes.
Headers are created AND unpickled under isolated declaration-book scopes:
SSA table unpickling otherwise reattaches onto the caller's active book and
accumulates unrelated rows. Source and complete unit histories remain archived
for diagnostics. Units remain compiled-unverified, NOT installed native compiler
replacements; the existing replacement verification gates are unchanged.

Real micro: left/right/root/unrelated/top, two admitted workers, maximum eight
source nodes per SCC. Root imports two callee declarations with no blocks;
top contains only its own body and links top/root/left/right objects. Scalar
and four-element AbstractTensor span contracts reproduce exact native 105 and
210 outputs. The tensor micro caught shape discovery still requiring a callee
graph: it now reads the accepted result descriptor before the provisional
scalar domain. This fixes the writer/query, not the final output extent.
No new packages or substitute source compiler. No long orbital run launched.

IMPORTANT LIMITS: opt-in, not an automatic conversion of the orbital benchmark.
Initial ingestion/reduction still holds the source project. Native tile imports
currently require static scalar/span boundaries with one result per call.
Record/sequence/reference layouts, multi-result projection planning, runtime
extent transport and runtime-slot/text publication imports explicitly refuse.
Precompiled SSA imports must already be bodyless; a hidden old piece body is
not admitted as though it were a compact header. This lane currently emits
C/LLVM objects. Fortran refuses these declarations instead of printing empty
function bodies; its ordinary source/SSA lane is unchanged.
Oversized SCCs refuse before admission; a smaller proven region cut is still
needed. Aggregate RSS is still an admission budget, not a hard cap on later
process growth. Standalone validation-bearing objects explicitly refuse until
that host checks the shared validation runtime; C/LLVM managed executions
reset/take the existing runtime's errors. This is a real separate-compilation
path, but NOT the complete graph/compiler conversion or an orbital speedup claim.

Expanded aggregate-call regression: six failures also reproduced on clean
`C:/dev/tg-base`, with its compiler pre-imported before loading the live tests.
Do not attribute those pre-existing failures to native interface work.

API (supply the program's existing extraction/ABI contract):

```python
module, outputs, exports = lower_ast_source_to_ssa(
    source, entrypoint, extraction_contract=contract,
    partitioned_directory="build/fresh_native_tiles",
    partitioned_options={"jobs": 2, "max_unit_source_nodes": 4096},
)
artifact = emit_ssa_function_to_llvm(module, exports[0])
compile_artifact(artifact, directory=Path("build/fresh_native_tiles/image"))
```

Final focused batch: **88 passed in 76.87 s**, with crash-isolated native
executions bounded at 10 s and compilation workers bounded at 30 s in the micro:
`python -u -m pytest tests/test_scoped_compilation_inputs.py tests/test_project_compilation_product.py tests/test_llvm_piece_runtime_payload.py tests/test_native_call_input_receipts.py -q --tb=short --show-capture=no`.
Compilation/syntax and whitespace checks pass. No workers remain after the run.

2026-10-09 actual source-unit execution/input cuts:

The existing `compile_resolved_process_graph_plan` worker runner now publishes
per-SCC/dependency input archives BEFORE launch. Workers read only their indexed,
SHA-matched cut, not the entire resolved project. Function addresses/contracts
remain in their original slots; excluded graph bodies and catalogue topology
are absent. Selected graph nodes/edges/read scopes and table aliasing survive.
Source planning still uses the canonical public compiler. New
`compile_partitioned_source` ends that pass before deployment, streams cuts,
stows the source concordance intact in `source-concordance.pkl` with a digest,
and collects the original planning heap before bounded worker admission. Worker
artifacts retain a diagnostic reference to that source book, not a live copy.
Worker stdout is inline and unbuffered. Admission counts at least the configured
reservation for every live worker, including just-launched/unmeasured workers.

The source-unit lane requires an explicit source-node bound (default 4096).
An over-bound dependency closure is neither serialized nor admitted. It produces
the existing resource-failure/subdivision records. This is a bound on GRAPH NODE
WORK INPUT, NOT a theorem about RSS. Existing sampled private-memory/time limits
remain; aggregate RSS is an admission ceiling, not a hard cap on later growth.

New failing micro: left/right/root/unrelated, requesting left, formerly loaded
all four bodies. The same failure was confirmed with the compiler pre-imported
from clean `C:/dev/tg-base`; an earlier absolute-test-path attempt imported the
live tree and is NOT baseline evidence. The repaired worker sees only left.
Actual two-process test: three independent source functions, at most 5 source
nodes per cut, peak 2 admitted processes, 3 distinct worker PIDs. Weak references
prove all original source graph wrappers are gone before any worker starts.
All three unit modules have zero structural concordance findings and produce
exact LLVM native outputs 43, 62, 3 for input 21 (children bounded at 10 s).
Units remain `compiled-unverified`; no native compiler replacement is installed.
Source-book archive round-trip, exact node/edge/scope/address preservation,
refusal of digest drift, no whole-archive worker read and oversized-cut refusal
are tested. Final focused batch: 73 passed in 23.30 s:
`python -u -m pytest tests/test_scoped_compilation_inputs.py tests/test_project_compilation_product.py -q --tb=short --show-capture=no`.
The compilation-unit planner suite separately retains two function-address
disagreement failures, reproduced on the clean short baseline (3 pass / 2 fail).

CLI entry: `python -u -m tools.compile_project_catalogue --partitioned-source
--source <program.py> --output <new-directory> --jobs 2 --max-tile-nodes 4096
--extraction-contract <contract.yaml>` (one line when actually invoking it).

LIMITS: this is the existing catalogue/unit PRODUCT lane, NOT a replacement for
the ordinary monolithic source-to-single-module return. The orbital benchmark
still uses that ordinary lane. Source ingestion/reduction remains whole-project.
Unverified callees still require their authored dependency closure, so an
interface-only bounded tiler and streamed final image assembly are NOT delivered.
An explicit bound now prevents silently expanding those closures without limit;
it does not make an over-bound program successfully compile. The next central
change is fixed type/shape/storage/effect/precision interfaces at call boundaries,
so workers need body + boundary summaries rather than transitive bodies. No
physics, numerical substrate, lowering equations or concordance gate was weakened.

2026-10-09 user-requested real four-rule orbital benchmark, after bounded
graph-work integration (dirty `54c67fbd`):

`python -u tools/compiler_probes/profile_memory_n2.py 4`, with
`TURING_GRAPH_WORKERS=4` and `TURING_GRAPH_RESIDENT_REGIONS=8`, completed
with exit 0 in **697.227 seconds wall** (11m 37s). No `--trace`, no packages,
no compiler edits during the run. TEMP/TMP pointed at a fresh output directory;
old artifacts and the user's dirty probes were preserved. All four orbital
pieces were stale and rebuilt. Authored graph foundations were also compiled.

The previous dispatch-indexing N4 run was 464.540 s, also rebuilding all four
pieces. Current total wall is about 50% worse. Combined lowering began at
175.1 s (previous 121.3), repository SSA finished at 608.4 s (previous 398.4),
deployment released at 614.1 s, and compiler end occurred at 640.8 s.
Combined-start to SSA completion is **433.3 s versus 277.1 s** (about 56%
worse), so piece rebuilding alone does not explain the difference. These are
single profiled runs, not an isolated attribution to any one compiler change.
The probe's GC object histograms/book reports remain enabled even without
tracemalloc; the result is not bare compiler time. The remaining wall includes
artifact emission/build, final reporting and process teardown.

Maximum printed stage-table RSS: **5,632 MiB**, versus the previous 5,600 MiB.
Sampled Windows process peak: 5,655 MiB. The probe's separate 5%-threshold peak
report says 5,449 MiB and is NOT the table maximum. Its retained book at that
report held 1,828,277 cells, ALL resident, with 0 book pages spilled. Histograms
included 2,607,842 dicts and 7,107,443 lists. Topology spilling does not bound
this whole-book/source/SSA residency. Sampled process CPU tracked wall within
a few seconds through roughly ten minutes: about one core overall, not four.
The completed piece reports used one participating graph worker despite the
four-worker cap; the 64-region piece kept at most eight topology units, spilled
56, and reloaded zero. The existing orbital wrapper does not expose the final
combined module's complete graph-work counters, so no combined worker-count
or topology-peak assertion is made.

The result contains 1,066 functions. C SHA256:
`cac028f5d0b6fc878be426ba4ed7c522782e12b8a94b00a956d21bff5e7acda1`.
Output: `build/profile_memory_n4_graph_work_20261009_114435/profile_memory_n2/`.
The C hash differs from the previous run; this probe compiles but does not
execute orbital numerical parity checks. Exit 0 is not a physics-parity claim.
This benchmark shows NO whole-process memory improvement or compile speedup.
Read-only timing attribution and a failing performance micro are the next
steps, not another large run or an unmeasured compiler patch.

2026-10-09 integrated dispatch work/residency checkpoint (dirty `54c67fbd`):

`graph_work.py` is now wired into the canonical source entry and real dispatch
extraction. Carving/finalization stays on the coordinator. Dependency-only
snapshots enter a bounded window; results publish in region ordinal order,
after checking the snapshot still agrees with the live cut. Completed results
count against admission, so an early slow unit cannot grow a reorder buffer.
Topology slots are reserved BEFORE carving, and cached plus in-flight dispatch
topologies share one compile-wide limit across all shell specializations.

Regions keep their actual ProcessGraph headers and NetworkX API. Node/edge
topology and level tables spill through the existing `BookSpill` segment codec
in a separate transient store. AST and other semantic leaves retain identity;
the book, Python bindings and source catalogue are not recursively pickled into
each segment. Cached NetworkX views route through the current resident payload.
Read containers are immutable; writes use generation-checked region leases.
Nested dict/list/set mutations and escaped old-generation writes are guarded;
cycles and current-generation container aliases survive spill/reload. Attribute
wrapping is deferred until a read/write actually needs it. This source-SSA path
normally consumes headers only and does not reload the spilled topology.
Deep compilers and ephemerals use their existing constructors lazily, share the
same headers, and retire with the existing dispatch-copy release boundary.
The transient store is removed at compile end, including failure cleanup.

Worker policy is explicit (`WorkContract.graph_workers`, default 1;
`TURING_GRAPH_WORKERS`). Dispatch residency defaults to 8 regions
(`graph_resident_regions`, `TURING_GRAPH_RESIDENT_REGIONS`; None is the eager
compatibility policy). More than one worker bootstraps both authored integer
foundations through `lower_ast_source_to_ssa`, before the program book opens:
lexical Kahn/min-heap ordering AND the existing ILPScheduler ASAP/ALAP equations.
Code and per-call arenas are shared/isolated respectively; neither kernel has
process-global validation-error state. This is not threads around a Python
replacement compiler. No foreign source implementation or package installation
was used. Worker pools are reused across shells within a compile. Foundation
builds get unique directories so separate processes never overwrite loaded DLLs.

`graph_work_publication` records the snapshot digest, method, counts, providers,
admission and exact carved read scope, derived from region/policy/scope/origin
cells. The concordance checks its schema, region method, resource policy and
exact provenance; lifecycle permits cold spill after repairs. Work/residency,
spill bytes, participating worker count and coordinator/worker/wait/publication
timings are returned in `module.metadata['graph_work']`. Resource policies are
also on the book. A worker cap is NOT a claim that all capped threads ran.

Reproductions (no tracing; runtime children bounded at 10/15 seconds):

- `python -u tools/compiler_probes/probe_graph_work.py --matrix`
- `python -u tools/compiler_probes/probe_graph_work.py --calls 32 --workers 8 --resident-regions 8`
- `python -u tools/compiler_probes/probe_graph_work.py --mechanics --unit-nodes 4096`
- `python -u -m pytest tests/test_graph_work.py tests/test_memory_regulation.py tests/test_identity_book_spill.py -q --tb=short --show-capture=no`

The first 2/4/8-worker source matrix passed exact native output at 4/8/16 calls.
It used different residency limits at 8 workers; do NOT attribute its timings
solely to threads. The matrix command now fixes all points to the same residency
limit (at least 8). A subsequent 32-call/8-cap probe lowered in 4.0298 s,
produced 35 functions/34 region publications, spilled 26 topologies, kept cached
plus in-flight topologies at most 8, and returned exact native
`[11944, 8528, 13946, 14864]`. Only 3 worker threads participated. This is NOT
the orbital N32 or N4 timing. Latest 16-call/4-cap/8-resident probe lowered in
1.7316 s, with 19 functions/18 publications, 10 spills, 3 participating workers
and exact output. Foundation setup and native build are excluded from lowering;
setup has been ~5.4 s alone and 12.27 s alongside tests. Child import/startup is
included in total probe time, not in lowering time.

Sixteen larger DAG units through the SAME admitted scheduling operation, with
8 resident slots and exactly 8 spills at EVERY point, matched the authoritative
scheduler exactly. These are graph-mechanics probes, not source/physics timings:

| Nodes / edges per unit | 1 inline | 2 cap | 4 cap | 8 cap |
| --- | ---: | ---: | ---: | ---: |
| 1,024 / 2,976 | 0.763 s | 0.692 s | 0.626 s | 0.618 s |
| 4,096 / 11,904 | 2.790 s | 2.726 s | 2.724 s | 2.661 s |

The larger 4-cap and 8-cap runs both used only 4 worker threads. Timings are
single-sample smoke measurements, not a statistically demonstrated speedup.
They locate the remaining cost: coordinator graph construction/packing, snapshot
validation and spilling; more worker capacity alone barely helps. Worker wall
time is summed and can overlap coordinator time; it is not CPU utilization.

Verification receipts: the graph/kernel/spill/adapters batch passed 63 in
61.92 s; graph-work/memory/copy-on-read/loop-scope passed 60 in 11.68 s before
the final extra audit/alias micros; native providers passed 7 in 22.74 s,
including 60 ASAP/ALAP random-DAG checks in two synchronized native threads.
Graph-work/spill/memory passed 70 in 17.11 s before the last alias micro.
The current focused graph-work suite also passed 20 in 18.86 s before adding
the alias identity micro; that micro failed on copying an already-managed
container, then passed alongside cycle/set/generation checks (4 in 0.24 s).
Final graph-work/spill/memory rerun: 71 passed in 17.69 s (21 graph-work tests,
including the repaired alias micro). The declared-page/lifecycle consistency
check passed separately after assigning the new auditor's repair-stage read.

LLVM/layout/dispatch regression batch: working 51 passed, 6 failed, 3 skipped
in 8.99 s; clean short baseline 46 passed, the EXACT SAME 6 failed, 3 skipped
in 8.81 s. Five extra passing tests are our existing LLVM/dispatch regression
micros. The six failures are the four historical dispatch failures listed
below plus `test_graph_only_reshape_builds_region_and_emits_wasm` and
`test_graph_only_tobytes_builds_region_and_emits_wasm` (legacy AOT programs
contain bit operations, not the expected reshape/tobytes). Do not repair those
as part of this work or call them new regressions. The earlier eight
loop-composer failures and separate historical native timeout remain recorded.

Limits: this bounds dispatch topology units, NOT all process bytes. Mutable
source graphs, graph metadata/semantic leaves, the source catalogue and whole
module SSA still reside in memory; most refinement remains coordinator-serial.
It is not a finished full out-of-core/parallel compiler. Region spill files are
transient storage, not resumable deployment/checkpoints. A first kernel variant
returning the conditionally used mutable `levels` reference exposed a dominance
finding; the final kernel publishes through its declared mutable output span
and returns processed count. No detector was suppressed; conditional arena-return
lowering remains a separate micro-test/repair task.

2026-10-09 self-hosted graph-mechanics checkpoint (baseline `54c67fbd`):
`compile_and_register_topology_accelerator` now compiles authored integer CSR
Kahn/min-heap mechanics through `lower_ast_source_to_ssa`, then uses the existing
LLVM artifact route. The hand-written Fortran descriptor remains historical;
it is not the active provider build. Registration is explicit, never an import
side effect. The existing canonical reducer consults this provider when it is
registered. A bounded child probe runs 100 deterministic small DAGs/multigraphs
through two synchronized worker threads, with invocation-owned arenas and equal
lexical-key ties; all results exactly match the registry's NetworkX oracle
(7.52 s including compilation/imports). This proves independent concurrent
invocations, NOT compiler-wide parallel refinement or measured CPU speedup.

Meaningful public-entry heap/frontier micros exposed and repaired lost RHS
read receipts, captured-value expression re-expansion, indexed-element/result
type disagreement, Store/GEP read-only classification, signed integer floor
division, loop-entry/latch selection by binding, unchanged break-edge state,
and scalar Phi value capture before a reused buffer is overwritten. A scalar
Phi previously selected 17 but returned 31 after its selected slot was reused.
LLVM now captures scalar merges simultaneously into owned homes, preserves
ranked/reference Phi arenas, and refuses incompatible incoming physical element
representations. Heap tests cover zero trips, both heap directions, cycles,
empty graphs and differing outgoing-neighbour order. Native failures are run
in child processes with 10/15-second bounds, not in pytest's process.

New `name_binding_capture` rows connect exact source/target binding versions
and their AST-owned loop; canonicalization carries those rows, the concordance
checks them, and page lifecycle permits cold spill after pre-native repairs.
One sealed-scope index is built per SSA builder, not a full capture-page scan
for each read. The compiler does not infer capture ownership from line numbers.
The existing spill reader's load registry is now context-local: the forced
two-book reload interleaving failed on clean HEAD and now resolves both books'
registry objects independently. This is NOT permission for concurrent writes
to one IdentityBook or BookSpill.

Focused LLVM/adapters/spill/graph micros: 82 passed, 3 skipped in 63.91 s,
including the final one-index-per-builder micro. That new micro and the real
inside-loop capture proof also pass alone (5.87 s). Memory regulation, copy-on-read,
loop-scope concordance and self-hosted provider group: 47 passed in 17.33 s.
Result-type/transformation/control/operand/loop-composer group: 100 passed and
the same eight failures on this tree (9.91 s) and the clean short worktree
`C:/dev/tg-base` (9.66 s). Adding five existing native binding-read cases gives
105 passed, the same eight failures, one deliberately deselected historical
`call-argument-carried` case (30.36 s). That case's 30 s native timeout was
separately reproduced on clean HEAD; it is not a new regression.

The four-call PUBLIC micro (not the orbital four-law benchmark) remains exact:
lower 1.344 s, total 1.683 s, 10 functions, zero structural findings; emitted C
SHA256 `3f43760e36b1f3aec4cad2b11008b03d1a32325bc6c84359b81e13a8ea41af69`,
byte-identical to its clean baseline. Its 148 unsourced facts remain OPEN.
No new long orbital compile was launched. No packages were installed.

REMAINING: the all-in-one overhaul is not complete. Source graphs, expanded
regions and whole-function collections still have resident/serial ownership.
The self-hosted primitive has compact per-invocation storage, but a true bounded
region lease/admission path and parallel worker proposal/deterministic book
publication path have NOT been implemented. Do not report this checkpoint as
breaking the whole-graph RAM requirement or making the whole compiler a swarm.

2026-10-09 four-law benchmark after dispatch indexing: untraced
`python -u tools/compiler_probes/profile_memory_n2.py 4` completed with exit 0
in 464.540 seconds wall. The prior run was 531.749 seconds, including two
piece builds; the new compiler stamp caused all four pieces to rebuild.
The new profile's combined-lowering start is at 121.3 seconds, repository
SSA completion at 398.4 seconds, and deployment release at 403.2 seconds.
Approximately 343 seconds of the wrapper wall remained after the recorded
combined-start event (including profiling and emission/build overhead),
versus approximately 377 seconds previously. These single runs have
different cache preparation and do not isolate the change's speedup.
Highest RSS in the printed stage table: 5,600 MiB. The profiler's separate
5%-threshold peak report says 5,403 MiB and is not the maximum of that table.
There are 1,066 functions. Emitted C SHA256:
`0ba0512e8bb5712c23c17955b0be562915a65eb5b1e498322c361f4ce1e67b65`.
This differs from the prior N=4 hash after rebuilding all input pieces;
no semantic equivalence or runtime parity claim is made for this benchmark.
Output: `build/profile_memory_n4_graph_frontier_temp/profile_memory_n2/`.
The run still used the serial, fully resident lowering path. It does not
satisfy bounded graph residency or parallel compiler refinement.

2026-10-09 dispatch frontier checkpoint (baseline `54c67fbd`): dispatch
carving now derives one immutable edge/operand index per planning phase and
queries each region's members and boundary neighbours. The index is released
at phase completion. It changes no identity ownership or region membership.
`python -u tools/compiler_probes/probe_graph_region_frontiers.py 2` (also
`4` and `8`) exercises the public whole-source compiler entry and the existing
native artifact route. All three have exact numerical parity and zero
structural concordance findings. Unsourced facts remain OPEN, unchanged in
the four/eight-call before/after comparisons (148 and 248 respectively).
The four-call case goes from 164 per-region source adjacency queries to 31
index-build queries and zero per-region queries; eight calls go from 556 to
47 and zero. Both emit byte-identical C to the baseline. These are traversal
counts, not a claimed large-program timing or memory improvement. Full
region copies and whole-function residency remain.

`tests/test_scheduled_process_graph_dispatches.py`: 29 passed / 4 failed on
the changed tree, versus 27 passed / the same 4 failed on the clean short-path
worktree at `54c67fbd`. Both new tests pass. The unchanged failures are
`test_tensor_accessor_and_its_scalar_comparison_stay_with_coordinator`
(classifier returns False) and `test_a_lookup_nobody_reads_is_not_a_region_boundary`,
`test_dispatch_exports_a_value_routed_by_identity`,
`test_dispatch_exports_an_internal_value_used_outside_the_region`
(hand-built graph has no ingestion scope). Adjacent copy-on-read, kernel
deduplication and repository SSA dispatch tests: 18 passed in 1.42 seconds.
The N=35 memory probe was stopped under explicit user authorization after
2,417.556 seconds wall, in deployment instantiation, to permit tiny architectural
proofs; it produced no completed combined artifact. Its completed piece
caches were retained.

2026-10-03 orbital archive reconstruction update (`7d984b78`): the
`ArgumentBindingFact` pickle defect mentioned in the return-site port below
is fixed at its existing declaration. Nine focused archive tests pass, and
`tests/test_native_record_return_state.py::test_child_record_conditional_write_reaches_return`
now passes in 4.94 s, including its four native child-process combinations.
The actual archived orbital RK4 SSA book also round-trips with identical
concordance audit and argument/output identities. This closes that specific
serialization failure; the rest of the port matrix was not rerun.

2026-10-03 return-site identity proposal (written against 3559a8f, ported onto
854e145c): the focused `test_record_return_site_identity`,
`test_ssa_record_return_state`, `test_control_source`,
`test_pruned_loop_return`, `test_native_record_return_state` batch is 58 passed,
1 xfailed (`static_branch`, HELD: an untouched declared field is not a record
output), 1 failed in 140.8 s. The failure,
`test_child_record_conditional_write_reaches_return`, is pre-existing at
854e145c: unpickling `ArgumentBindingFact` (a tuple subclass whose `__new__`
takes two arguments) in the native child raises TypeError. The new file's
native matrix (core O0/O2, seven O0 variants, 20 s child bounds) passes except
the held variant. Port record and every resolution:
`docs/concordance_census/CONTINUATION_return_site_port.md`; the proposal's own
account (its baseline, not this head): `docs/RECORD_RETURN_SITE_IDENTITY_FIX_2026-10-03.md`.

The proposal also reports, on its baseline, `test_native_loop_arm_return`
stopping at `run_all` with 10 id-scale findings, and shared identity-book
`function_address` collisions in combined loop-composer (6 passed / 1 failed)
and reducer field-state (3 passed / 1 failed) batches whose failing tests
pass alone. Those were not re-measured during the port.

2026-09-15 shaped-view and integer-operator repairs: four compiler defects were
found by comparing compiled output against eager `AbstractTensor` execution,
and each is now covered by `tests/test_native_shaped_view_lowering.py`
(5 passed, 1 xfailed, ~22 s). None of them raised: every one produced a
complete, compiling program that read the wrong elements.

1. A reshape view occurrence lost its extents. `b.reshape((-1, 1, 2))` and `b`
   share one storage identity, so whole-module ABI propagation and formal-use
   interning both restamped the view with the allocation owner's shape. The
   broadcast kernel then conformed the wrong axes. View occurrences now carry
   an `ssa_storage_view` receipt that both passes retain when the element
   count proves they describe the same bytes.
2. Tensor `%` and `//` had no opcode spelling, so they fell through to the
   scalar emitter, which computed element zero and left the result buffer
   untouched. Both are catalogued binary kernels and now lower as such;
   native output matches eager for all four operand-sign combinations.
3. A dtype spelling (`to_dtype("int64")`) counted as a non-numeric constant
   operand, which made the whole cast coordinator metadata and left its result
   an unproduced region feed reading as zero.
4. A slice over a tensor call result was treated as a call-boundary
   projection. Only a literal integer index is such a projection; a slice is a
   numerical view an owning region must compute.

The dispatch classifier cache schema moved 2 -> 3 so saved graph replays
recompute these classifications. `reduce_scheduled_shader_regions` also lost
two superlinear scans (per-candidate iteration over every coordinator edge,
and per-quotient-edge iteration over every direct edge and region member);
the fusion identities and legality rules are unchanged.

Adjacent focused batch on the repaired tree: `test_ssa_fusion_regions`,
`test_region_kernel_dedup`, `test_deployment_outlining`,
`test_repository_ssa_dispatch`, `test_scheduled_process_graph_dispatches`,
`test_native_call_input_receipts` and `test_tensor_ssa_call_metadata` pass
71 tests in 12.54 s, with no failures.

`tests/test_process_graph_function_linking.py` reports **14 failed, 57 passed**
on the repaired tree against **16 failed, 55 passed** in a clean worktree at
`3af7d206`. The repaired tree's failures are a strict subset of HEAD's:
`test_callsite_shape_discards_padded_scalar_result_descriptors` and
`test_shape_constant_waits_for_authoritative_callsite_descriptor` now pass and
nothing new fails.
`test_structural_boolean_call_feed_is_materialized_before_linking` fails in
both, so it is pre-existing despite sitting in the same recovery area.

`tests/test_precompile_to_ssa.py` plus `tests/test_ir_sequence_tables.py`
report **9 failed, 125 passed** in the working tree. The identical nine
failures reproduce in a clean `git worktree` at `3af7d206` (9 failed, 123
passed), so they are pre-existing and none belong to these repairs. They
supersede the older per-file rows below for these two files. Two of the nine
are the previously documented `test_native_fortran_ops_keep_mean_and_span_fill_in_ssa`
and `test_index_dtype_propagation_is_scoped_per_function_identity`; two more
(`test_sequence_helper_fresh_ids_reserve_non_argument_descriptor_identities`
and `test_nested_iterable_row_is_handle_stride_and_child_address`) assert
locally derived "next id" values that the central `GLOBAL_MONOTONIC_IDS`
issuer no longer produces.

5. A short-circuit reduction operand was never produced. Structural recovery
   rebuilds coordinator-only `and` operands from their exact graph edges but
   had no reduction case, so `pressure.isfinite().all()` failed, its enclosing
   boolean chain failed with it, and the owner called its own region with a
   value nothing defines. This is the single undefined operand that stopped
   the validator build at `validator_simulation_advance` planned region 44.
   Recovery now emits the same instruction a region body carries for one-operand
   `all`/`any`/`sum`/`prod`/`mean`/`min`/`max`; an explicit axis is left
   unrecovered rather than guessed.

2026-09-15 validator build v13: the fresh balloon-validator build with every
repair above reports `undefined_operands=0` (was 1), with unmaterialized
boundaries, unresolved calls, optional merges, structural outputs and
non-native boundaries all empty. `unaccounted_formals` fell 47 -> 27 and is the
only remaining gate family: 13 in `validator_simulation_advance`, 6 in each of
the two `_wrench_force` specializations, and one each in
`step_with_dt_control_used` and `vehicle_tire_recurrence`. Command:
`python -u tools/build_vehicle_validator_simulation.py --output build/validator_frontier_20260915_v13 --lanes 8`;
it reached the gate in about 45 minutes. No native artifact exists yet.

Captured bindings of a nested function are now recorded as a `closure_formals`
ABI receipt and accounted by `check_formal_parity`. The receipt is keyed on the
source graph's `closure`/`external` binding kind, never on a name: a name-keyed
first attempt claimed `balloon_tire_reduced_vector_step`'s own `r0`/`z0`
tuple-unpack temporaries, which are escaped locals and must keep announcing
themselves. `_wrench_force` drops from 14 unnamed formals to 6. Execution of nested calls
that capture enclosing values is still wrong and is pinned as two xfail
regressions: the result reaches the caller as a scalar occurrence, and an
enclosing parameter consumed only through the closure is dropped from the
enclosing signature. Both reproduce identically at `3af7d206`.

Open and pinned, not fixed: a declared `int64` output published as the double
working representation whenever a tensor kernel also consumes it. Returned
alone it is correct; consumed as well, the root wrapper still declares
`int64_t` while the publication copies raw doubles. The call edge has
physical input adapters but no output adapter. It is marked xfail in the file
above so it announces itself when repaired.


2026-09-14 native managed DT snapshot: the late conditional-continuation pass
now preserves an explicitly owned raw loop update. The return-state/control
batch passes 42 tests (72 deselected). Current replay of source checkpoint
v147 converges in six frame and three result rounds, with zero structural
findings and incompatible contracts; its explicit `-O0` one-step executable
matches 48/48 native/eager buffers for both one and two short outer calls.
The full `1/120` window completes in both executions (169 native / 167 eager
successful substeps, no critical/nonfinite attempts), but strict pointwise
parity remains 38/48. Full-window status and exact commands:
`docs/CONTINUATION_2026-09-14_NATIVE_DT.md`.

Adjacent shell checks found two existing failures:
`test_whole_object_if_assignment_lowers_to_conditional_phi_merge` expects an
obsolete function name; `test_nested_if_threads_inner_phi_into_outer_phi`
supplies no extraction contract. Both reproduce with HEAD versions of
`fortran_c_shell` and `ssa_record_return_state` loaded into an isolated
process, without replacing working-tree files. Their assertions remain intact.

2026-09-14 full-cycle perforated engine training: the complete focused LLVM
file passes 4 tests in 63.38s. Its three-minibatch regression exercises
gradient accumulation 2, a 0.15 global-norm clip, and a trailing partial group,
matching an independent reference for loss, five parameters, ten moment
tensors, beta powers, iteration, and pre/post-clip norms. The actual headless
LDT path (`423 -> 212`, batch 2) compiled an 18-batch cyclic bank and executed
72 forward/loss/generated-VJP/Adam motions in one native call in 0.999s;
training loss moved 1.11127 -> 0.02979. Adam is presently a backend-native
cycle around the pre-LLVM-composed forward/loss/VJP motion; it is not yet a
linked functional-Adam ProcessGraph.

2026-09-14 adaptive engine sample refresh: a forced-trigger real LDT headless
run completed 2 native epochs / 36 motions with accumulation 2 and clipping
1.0. After epoch 1, a widening 0.26246 validation gap captured 54 new
real-simulator transitions, grew the training pool 33 -> 87 rows, refilled the
same 18-batch LLVM bank without recompilation, and completed epoch 2. Training
loss moved 0.89212 -> 0.13489 and validation improved 0.73089 -> 0.64798.

2026-09-14 perforated startup cache: the real-dataset NPZ round-trip regression
passed in 28.25s, and the native cold-then-cached forward/VJP regression passed
in 22.68s. A direct 50-transition LDT preparation measured 9.537s cold and
0.037s cached. These checks do not cover a full-catalogue capture or long
training run.

2026-09-14 engine capture neutral/cooling/friction: the full
`tests/test_perforated_multifuel_engine.py` passes 11 tests in 250.73 s.
The exact Camry allocation (256 named samples per fuel, 104 random rows,
seed 3747) makes this file take about four minutes; it is not a quick gate.
Before the neutral fix that regression failed with load speed
78,539,816,339.74483 rad/s. The engine_toy friction/stand/oxidation group
passes 16 tests in 11.15 s; the station cooling/stand group passed 8 in
9.97 s. Validation covers real teacher capture and energy accounting, not
an all-profile 30-epoch training run or complete per-bearing calibration.

2026-09-08 sequence-arena/dominance settlement: compiler-owned sequence arenas
now claim storage provenance when created, before result reconciliation can
replace them with a loop-local scalar. Unique operand-free constants whose
original block does not dominate an exact-object use are hoisted to entry with
priority/tie provenance. The focused sequence/dominance batch passed 7 tests
in 2.68s. Saved pre-frame replay v122 converged in 6 frame rounds and 3 result
rounds with zero incompatible contracts; structural findings fell 7 -> 2.
Only `run_superstep` call-result-unavailable IDs 228 and 298 remain. No native
compile, optimization, or execution deadline was used.

2026-09-08 exact energy-limit helper closure: optional scalar returns now use
an explicit payload/presence pair through callee Ret and linked caller Call.
ProgramABI field discovery treats `getattr(obj, "field", default)` as the same
field read as `obj.field`; structural recovery replaces anonymous predicate
formals with their exact membership/isfinite producers and places each private
dependency closure before its unique consumer. `_energy_time_limit` now has
zero formal-parity findings, zero optional-merge findings, and zero C-emission
shortfalls. Its exact-source artifact compiled successfully as a DLL at `-O0`.
The focused optional/extraction batch passed 3 tests in 19.46s; an adjacent
structural/call/optional regression batch passed 10 tests in 8.44s. Neither
used an execution deadline. No helper execution or full-controller parity
claim was made in this chunk.

2026-09-08 Targets sidechain ABI repair: the shared ProgramABI now declares
`energy_exchange_fraction` and `shadow_growth_max` as optional float64 scalar
fields. Before the change, the default structural audit folded
`getattr(targets, "energy_exchange_fraction", None)` to `None`, reducing
`_energy_time_limit` to an unconditional `return None`; afterward the default
audit retains both record inputs, explicit presence slots, all three guards,
keyed channel reads, and the final multiply/divide result with zero structural
shortfalls. Contract/presence checks passed 2 tests in 4.45s, without an
execution deadline. An explicit `-O0` native attempt now reaches the next
honest gate: `_energy_time_limit` needs optional return payload/presence
representation and caller accounting for three recovered structural values.
No optimized build was used.

2026-09-08 scalar return snapshot settlement: focused batch passed 5 tests in
17.74s, followed by a 2-test placement check in 3.64s, without execution
deadlines. Replay v111 replaces managed advance's returned Metrics velocity
positions 194 with the unique dominating write version 295, records two
priority/tie-policy receipts, and has zero structural findings. Its explicit
`-O0` native build exposes snapshot output 295, writes caller slot 1741, and
keeps all three downstream consumers on 1741. One-frame parity remains 54/78;
this is a verified alias-boundary repair, not full controller parity.

2026-09-08 returned-record argument settlement: focused record identity/call
batch passed 3 tests in 4.66s with no execution deadline. Saved full-controller
pre-frame replay v107 converged in 6 frame rounds and 3 result rounds, with zero
incompatible result contracts and zero structural findings. `coerce_metrics`,
`_propose_dt_pen`, and `_apply_energy_sidechain` now receive the produced
`Metrics.max_vel` aggregate slot 1741 instead of stale input field 1654. The
later v107 `-O0` baseline completed at 54/78 one-frame parity.

2026-09-07 starred-generator maximum repair: one compiled native artifact
covered the complete and filtered forms across empty-filter, multi-value,
suffix-wins, signed-zero incumbent, incumbent-NaN, and candidate-NaN cases.
The adjacent generator/dict/control batch passed 8 tests in 28.95s; the final
focused lowering/native batch passed 3 tests in 16.24s. No execution deadline
was used. Saved full-controller replay v15 completed in about 87s without source
extraction or native compilation: `_propose_dt_pen` no longer has a formal
parity finding, and the pre-gate structural count moved 45 -> 44. The controller
still fails later native gates; this is not a full-controller execution proof.

2026-09-07 checked sequence replacement: native keyed copy/self-alias/empty/
capacity-failure/source-bounds regressions plus conditional lowering:
4 passed15.68s (outer timeout90s, native subprocess20s).
Broader sequence-table/conditional-tuple group:42 passed,1 failed41.39s.
`test_compiled_retained_loop_mutates_caller_sequence_record` fails at its
line811 empty-C-file assertion because the generated ABI shim is nonempty;
that fixture uses append, not replacement. The assertion was retained.
See docs/REPAIRS_2026-09-07_KEYED_REPLACEMENT.md for the full diagnostic and
missing keyed constructor/store/owned-field binding frontier.
Fresh lower-only diagnostic `artifacts/compiler_evidence/keyed_replace_full_formals_20260907.log`:
19 formals, zero undefined operands/unresolved calls, scalar publication
1482 ->9564 ->593 and earlier read1378 preserved. Terminal exit1 at strict
gate; no full native build/parity, no live jobs.

2026-09-06 scalar return publication: focused return-state, reachability,
and authored child-record tests: 18 passed, 1 xfailed in 26.38s, external
timeout90s. Both Boolean and provisional-float64 native variants invoke the
actual final publication pass, preserve output IDs, check idempotence, and
exercise repeated branch outcomes/buffer reuse (native subprocess timeout20s).
The authored child-record case still fails the preexisting ABI provenance
gate and is not an end-to-end native proof. Static-reference cache eviction
reproduced the missing-node crash; the guard fix and related reducer checks
passed 4 tests in 2.26s. See docs/REPAIRS_2026-09-06_RECORD_RETURN_STATE.md
for the fresh whole-source result and remaining keyed/optional-field limits.
Fresh source diagnostic `artifacts/compiler_evidence/record_return_publication_full_formals_20260906.log`:
four scalar return publications, early hard_failure read1378 unchanged,
returned1482 -> Cast9564 -> ledger593; 19 formals, zero undefined operands or
unresolved calls. Terminal exit1 at strict gate, no full native build/parity.
All launched processes are terminal.

2026-09-06 effect-order repair: targeted scheduling/alias-order gate 12 passed
2.67s; optional-field admission + wrapper checks 6 passed 3.47s. New native
two-append/conditional-clear truth test passed 12.87s across all eight flag
combinations after retaining non-loop clear. Native executions bounded at20s.
Two narrowly detected expected failures (2 xfailed5.32s):
`test_scalar_field_capture_precedes_conditional_write` loses scalar write/guard;
`test_rejected_attempt_never_calls_accept_mutation` loses call-only for-loop/guard.
These are not native proofs. Saved whole-step control replay has zero shortfalls.
Fresh whole diagnostic19 formals, zero undefined operands/unresolved calls;
terminal exit1 at strict gate. Linked field read1378 now precedes laterPhi593;
accept calls follow continue. No full native build/parity; all sessions terminal.
See docs/REPAIRS_2026-09-06_EFFECT_ORDER.md for intermediate failing runs and limits.

2026-09-06 second-opinion reachability repairs: conservative CFG/Phi checks,
formal-provenance gate, and existing paired caller/callee pruning regression:
13 passed, 1 native test deselected in 4.62s. New native return-path test plus
existing conditional-call-result and sequence-truth native checks: 3 passed
27.88s, execution subprocesses bounded at 20s. Saved-SSA replay removed five
private formals (23 -> 18), undefined operands zero. Fresh full diagnostic
completed with 19 formals, zero undefined operands and unresolved calls; exited
1 at the strict gate. It includes the existing field ledger, unlike the old
saved artifact. No full native build or parity run. All sessions terminal.

2026-09-05: latest whole-validator lowering failed at Pygame draw loop effects
after611.818s, session91926 terminal. Subsequent full-native extension ABI admission
tests and existing gates:5 passed2.82s. Pygame->Abstract UI AST ingestion:5 passed
2.66s; transformation only, not geometry backend or native execution proof.

2026-09-05: authored support-stage placement/binding, component placement/law,
and vehicle state feedback: 5 passed in3.01s. First attempt caught an editing
indentation error in the constructor, corrected before the passing run.
The support-stage check executes the actual AST block with19 supports/2 lanes;
it does not initialize tires, run the whole validator, or execute native code.
Corrected full dually allocation/ABI test completed successfully in106.92s;
session50509 is terminal, superseding the active-run note below.

2026-09-05: `test_dually_program_allocates_and_declares_requested_rig_point_count`
requires expensive full dually preparation: first run 575.12 s, failing on a
test-only ABI accessor typo after buffer assertions passed. Correct accessor is
`binding.field.shape`. Corrected rerun active (see continuation); do not treat
this as a seconds-long test or rerun for cosmetic verification.

2026-09-05: variable rig-point law regression failed before generalizing the
hard-coded 16-point reshape (3.54 s), passed after (2.95 s). Existing declared
rig SSA/C/library compilation test passed with 19 points in 9.99 s; no native
execution in that test. Eager checks 0/1/3/19 points and balanced reactions.

2026-09-05: common player/validator placement action and stale revision regression
passed 0.60 s; existing IssuedAction table count check passed 0.52 s.

2026-09-05: general four-port attachment placement regression passed 0.58 s.
Graph edits and routing validation only; not mechanical constraint execution.

2026-09-05: authored dually vehicle feedback regression reproduced missing state
carry (0.69 s), then passed after fix (0.59 s). Expanded actual mapping/rollback/
replay check passed 0.60 s. Deterministic test plant, not native vehicle solver.

2026-09-05: fixture mechanics test passed in 1.89 s after correcting pillar
reaction to the actual scalar-y output shape. Previous vector fixture was not
representative of the numerical graph output.

2026-09-05: identified fixture command/reaction/pose test and authored viewer
acknowledgment regression passed together in 2.17 s. No native physics execution.

2026-09-05: selected fixture/graph geometry regression passed in 0.57 s after
converting graph edges to identified AbstractUI objects with visibility state.

2026-09-05: three selected vehicle geometry checks passed in 0.70 s:
Abstract UI object/child identities and graph relationships across poses,
fixture/graph geometry, and existing seven-primitive geometry after relocation.

2026-09-05: `tests/test_vehicle_part_geometry.py::test_tire_geometry_preserves_winding_material_depth_and_center_surface`
passed in 0.57 s. Numerical authored geometry only; no native/pixel parity.

2026-09-05: `tests/test_vehicle_part_geometry.py::test_authored_part_geometry_preserves_shape_width_and_world_coordinates`
passed in 0.54 s. Covers seven authored shape primitives and skipped membrane
records; no GPU rendering or compilation involved.

2026-09-05: two focused `tests/test_geometry_consumer.py` checks passed in
0.91 s (frame completion/failure/camera/empty layers and persistent mesh buffers).
Mesh check passed again in 0.84 s after adding explicit color-presence handling.
GL calls mocked; actual shader output remains unverified.

2026-09-05: geometry packet ownership/invalid-native-span regression in
`tests/test_geometry_display.py` passed in 0.52 s. Geometry capability opt-in
and rejection by pixel-only shell test plus existing shared-mailbox ABI test
passed together in 1.91 s. These exercise codec/contracts, not rendering.

2026-09-05: `tests/test_vehicle_viewer_frame_ack.py::test_viewer_does_not_acknowledge_revision_published_during_draw`
reproduced premature revision acknowledgment (0.70 s) and passed after fixing
the authored loop (0.57 s). Executes the production loop AST with controlled
publication during drawing; excludes model startup and native compilation.

2026-09-05: `tests/test_glrenderer_host.py::test_external_context_renders_without_pygame_and_propagates_present_failure`
passed in 1.19 s. Mocked GL, denied Pygame import: external context host,
overlay ordering, resize dimensions, core-context enables and presentation
failure propagation. Does not verify pixels or native validator parity.

This document exists because the same expensive mistake kept being repeated:
an agent sees a failing test, assumes it broke it, and spends two full suite
runs plus a stash/pop to find out the failure was already there. That costs
hours and has damaged working state in this tree. **The baseline below is the
answer to "is this mine?" — consult it before running anything.**

## The rule

1. **A failure in the manifest below is NOT yours.** Do not re-derive it. Do
   not run a second time to confirm. Say "pre-existing, see
   TEST_BASELINE_AND_HAZARDS.md" and move on.
2. **Never baseline with `git stash` or `git checkout -- <path>`.** Both have
   destroyed uncommitted work in this tree. If you genuinely must compare
   against a commit, use a worktree, which touches nothing you have:
   `git worktree add /c/dev/turing-head <commit>` — and remove it when done
   with `git worktree remove /c/dev/turing-head`.
3. **Never run a test purely to collect names for a report.** If you already
   have a pass/fail count or a progress string that answers your question,
   that IS the measurement. Re-running for cosmetics is not free here.
4. **Update this file** when you legitimately run something and learn a name
   or a count the manifest is missing. Filling a gap as a side effect of work
   you were doing anyway is right; running to fill it is not.

## Hazards, measured

| hazard | detail |
|---|---|
| `pytest tests` (whole suite) | **Does not finish.** Observed 2026-08-19: 5 h wall, **zero bytes** of output, then decayed to ~3 CPU-seconds per 5 minutes and had to be killed. Do not run it. |
| Output is buffered | pytest writes nothing until it exits when its stdout is a pipe. `-u` / `PYTHONUNBUFFERED` do not help — pytest buffers its own progress. A silent run is not a hung run, and you cannot tell them apart by watching. |
| `pytest-timeout` is **not installed** | There is no per-test timeout. One hanging test eats the whole run. Bound it from outside instead: `timeout 100 python -m pytest tests/<one_file>.py -q --tb=no`, one file at a time. Exit code 124 means it hung. |
| Killing a run | Several tests spawn native toolchains (gfortran, zig cc). Killing the pytest process can leave children; check for stray `python`/`zig` processes afterward. |
| Cached artifacts | A stale `control_repository_ssa.pkl` presents a program built by a compiler that no longer exists, so your change appears to do nothing. `_cache_is_stale` guards the fluid path; other paths may not. |

## The manifest — known-failing at `af00599`

Verified 2026-08-19 by running each file in the working tree AND in a clean
`git worktree` at `af00599`. Identical results in both, so these are
pre-existing and independent of the namespace/indexing fixes on top.

| file | result | notes |
|---|---|---|
| `tests/test_ast_indexing_aot.py` | **13 failed, 10 passed** | progress string `FF.F.FFFF.F.F.....F.FFF`, byte-identical in both trees. Individual test names NOT captured — collecting them costs a full run of an expensive file, which rule 3 forbids doing for its own sake. If you run this file for your own reasons, paste the `-rf` names here. |
| `tests/test_index_set_scatter.py` | **2 failed, 7 passed** | `test_index_set_emits_a_complete_scatter_module`, `test_index_set_scatter_runs_correctly` |
| `tests/test_extraction_contract.py` | **2 failed, 14 passed** | `test_default_contract_draws_python_native_and_decompile_lines`, `test_print_host_boundary_uses_existing_stream_operator` — both assert `print` decides `python_host_call`; it now decides `use_native`. Verified 2026-08-20 in a clean worktree at `6cb148b`: identical there, so pre-existing. |
| `tests/test_process_graph_autograd.py` | **1 failed, 21 passed** | `test_real_abstract_nn_xor_has_exact_native_adjoint_and_training_loop` — `NotImplementedError: SSATensorOperations must implement tolist_()` (`abstraction.py:752`). The three former aggregate-linking xfails were fixed and promoted to passing regression tests on 2026-08-20; two multi-output/contract tests were added. The remaining failure was verified 2026-08-19 in a clean worktree at `2ee2fd1`. |
| `tests/test_process_graph_function_linking.py` | **7 failed, 17 passed** | `test_record_field_assignment_is_a_real_inout_value`, `test_direct_source_lowers_declared_record_literal_and_bool_return`, `test_record_return_call_refreshes_completed_physical_field_surface`, `test_record_parameter_call_uses_fields_without_python_receiver_handle`, both comprehension publication tests, and `test_returned_record_fields_feed_structural_call_argument`. Verified 2026-08-24 with identical results in the working tree and a clean worktree at `57b5e25`; independent of class-emitter receiver wiring. |
| `tests/test_site_bundle.py` (focused bundle pair) | **2 failed** | `test_program_bundle_owns_page_source_wasm_manifest_and_inventory`, `test_one_shot_bundle_packages_the_discovery_numeric_trace` — both reach WASM fidelity and fail with `ValueError: a compiled program needs at least one output`. Verified 2026-08-20 in a clean worktree at `735409d`; unrelated to the shader-region deployment stage. |
| `tests/test_webgpu_ssa_backend.py` (deprecated-AOT quartet) | **4 failed** | `test_ast_generated_float32_program_emits_wgsl_compute`, both `test_ast_generated_loop_uses_structured_wgsl` cases, and `test_float64_is_a_named_webgpu_core_shortfall`. The deprecated `compile_ast_aot` path no longer presents the captured numerical program/cycle shape these tests expect. The first was verified 2026-08-20 in a clean worktree at `ab9a078`; the other three were exposed by a focused run the same day and do not touch direct repository-SSA GEMM/WebGPU emission. |
| `tests/test_machine_target_languages.py` (focused pair) | **2 failed** | `test_existing_backend_operator_lists_are_exposed_without_a_fifth_copy` hard-codes 40 C operators but the shared inventory now has 50; `test_fortran_and_desktop_glsl_print_the_same_numeric_program` expects `cos(` in a Fortran artifact that no longer retains it. Verified 2026-08-20 in a clean worktree at `ab9a078`; unrelated to WebGPU benchmark emission. |
| `tests/test_precompile_to_ssa.py` (current tree, 2026-08-30) | **2 failed, 64 passed** | `test_native_fortran_ops_keep_mean_and_span_fill_in_ssa` now sees the span fill lowered to a `Call`; `test_index_dtype_propagation_is_scoped_per_function_identity` sees a `float64` load instead of `int64`. Discovered during the vehicle call-shape propagation regression gate; neither exercises call metadata propagation, and no clean-worktree ancestry comparison was commissioned. The older 2026-08-23 repeat/API-publication failure no longer appears in this run. |
| `tests/test_wasm_class_modules.py` (current tree, 2026-08-24) | **1 failed, 27 passed** | `test_describe_process_graph_api_resolves_the_real_source_parameter_name` sees no logical input from the deprecated `compile_ast_aot` path. Discovered while testing state-feedback inventory wiring; the new focused regression passes and does not exercise that compilation path. No clean-worktree ancestry comparison was commissioned. |
| `tests/test_wasm_class_coordinator.py` (current tree, 2026-08-24) | **1 failed, 10 passed** | `test_wasm_coordinator_calls_cards_internally_and_honors_latched_ranges` broadcasts the first element (`seam=[4,4,4]`, `result=[16,16,16]`) instead of preserving the three-element input. The passing count includes the new control-region state-naming regression. No clean-worktree ancestry comparison was commissioned. |
| `tests/test_symbolic_equation_compiler.py` (current tree, 2026-08-25) | **2 failed, 4 passed** | `test_compiled_scalar_equation_can_run_in_fortran` expects integer literal `2` but the backend emits `2.0_c_double`; `test_fluid_equation_compiler_builds_the_full_default_model` expects 30 inputs but receives 28. Discovered while adding the independent float32 SSA-to-WebGPU vehicle contact path. No clean-worktree ancestry comparison was commissioned. |

| `tests/test_ssa_c_aggregate_constants.py` (current tree, 2026-09-01) | **6 failed, 29 passed** | The six failures (`const`-qualified aggregate constants, invocation-local `frame_N[...]` storage instead of malloc'd root storage, imported-LLVM literal decode, planned-aggregate caller-storage binding, physical-dtype preservation across planned outputs, static extent shape emission) are the prior session's SPEC for the still-unfinished native activation-storage contract. They are the same defect family as the managed-window emission shortfalls (`dynamic temporary %t96 has no native activation-storage contract in ...balloon_tire_managed_window__planned_region_2`) that currently block re-emitting `balloon_tire_appendage_step.c` at ANY window rate (verified identical at 1/120 and 1/1024). Not regressions from the 2026-09-01 deployment-outlining work: the failing arms (Const, root storage) were not touched, and the identical shortfall set reproduces on the un-outlined module. |
| `tests/test_deployment_native_emission.py` (current tree, 2026-09-01) | **1 failed, rest passed** | `test_frame_plan_workers_and_chunk_are_literal_in_native_source` dies inside `shader_region_pipeline` ("typed shader hole lacks shape/dtype metadata for boundary values") while constructing its plan — a dirty-tree interaction in plan construction, not in pooled C rendering, whose compile-and-run proofs in the same file pass. |

| `tests/test_compiled_linalg.py` (current tree, 2026-09-01) | **`test_jacobi_rotation_arithmetic_computes_natively` fails numerically; a later test in the same file hard-crashes the process** | The test runs the LLVM lane (`compile_artifact`), which the 2026-09-01 deployment/extent work does not touch (its only LLVM edit is an IndexError->shortfall guard). The signature — a compiled rotation kernel with partially-wrong elements — matches the long-open "native re-reads a load across an in-place store" aliasing defect already pinned for rotation/swap kernels. The 2026-08-19 "6 passed, 1 xfailed" row above predates substantial dirty-tree movement. No clean-worktree ancestry comparison was commissioned. |
| `tests/test_ir_sequence_tables.py` (current tree, 2026-09-01) | **1 failed, 35 passed** | `test_compiled_retained_loop_mutates_caller_sequence_record` asserts `artifact.c_source_path.read_text() == ""` — an in-progress/spec expectation (source file consumed?) from commit 839a40d. Unrelated to deployment outlining (which never runs without an explicit pass call or `deployment=auto` contract). |

## 2026-10-03 NumPy scatter source-precision repair

At `3559a8fdcae91ea773dbb433a23f9bad9b9c117a`, the integer-tensor scatter
fast path cast `src` to the destination dtype before `np.add.at`. This lost
the representable `0.5` residual of float32 `16777216` plus float64
`-16777215.5`, and returned `-1` for int64 `-1` plus float64 `0.8` where
NumPy and the list-index path return `0`. Source values now retain their
dtype until NumPy casts each accumulated result into the destination.

`tests/test_numpy_scatter_precision.py` was added and run before changing
the implementation: **54 failed, 46 passed**. After the one-line repair:
**100 passed**. The matrix compares exact values, shapes and dtypes with
`np.add.at` and the public list-index path, covering integer/float and
float32/float64 inputs, integer sources into float32, repeated and unique
destinations, every axis in 1D–3D including a negative axis, empty indices,
scalar/singleton broadcasts, same-dtype controls, and unchanged operands.

Adjacent files, run separately: `test_scatter_reduction` **13 passed**,
`test_abstract_tensor_indexing` **2 passed**, `test_tensor_basic_ops`
**18 passed**, `test_dtype_promotion` **4 passed**,
`test_tensor_list_coercion` **2 passed**, `test_tensor_int_cast` **4 passed**,
and `test_tensor_grad` **3 passed**. Total final targeted coverage:
**146 passed**, with no skips or expected failures.

Environment: Linux, Python 3.12.14, NumPy 2.3.5, pytest 9.1.1. Command from
the worktree root, replacing the test path for each adjacent file:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TENSOR_FACULTY=NUMPY /tmp/turing-fix-venv/bin/python -u -c 'import sys; sys.modules["torch"] = None; import pytest; raise SystemExit(pytest.main(sys.argv[1:]))' tests/test_numpy_scatter_precision.py -q --tb=short
```

The optional `nodus_tensor_core` library was absent and emitted its normal
warning. These are Python/NumPy backend checks; no native-arena, Windows,
accelerator or full-suite parity is claimed. PyTorch was not imported or
installed. No tolerances, reduction policy, dispatch or promotion rules changed.

## The manifest — known-good at `af00599` plus the current working tree

2026-09-05 focused threading/loop checks: `test_literal_seeded_counter_executes_native_iterations`
passed in 9.17 s after repairing carried scalar region capture and C singleton
publication. Native execution is isolated in a subprocess with a 20-second timeout;
earlier versions hung, so keep that bound. `test_resource_wait_retains_loop_scope_and_unique_ssa_definitions`
passed in 2.71 s with current-phi and nonidentity increment assertions. These are
bounded regression proofs, not full-validator native parity.

2026-09-05 native Event addition: `test_native_event_broadcast_clear_and_future_waits`
passed in 2.95 s (Windows, subprocess timeout=20). The selected-loop diagnostic
regression in `test_process_graph_call_diagnostic.py` passed in 0.55 s.

2026-09-05 source pursuit fixes: focused registered-method, constructor and
module-field receipt regressions in `test_ast_parent_ingestion.py` passed
together (3 passed, 2.55 s). The existing reachable-root exclusion regression
passed in 2.12 s after repairing lexical-body activation; nested-helper plus
parameter-shadowing coverage passed in 2.27 s. No full-suite run.

2026-09-05 compiler-ledger coverage: focused fingerprint regression passed in
2.34 s after adding shared AST context, deployment, C emission and dispatcher
runtime sources. Saved-call extraction-parameter diagnostic passed in 0.55 s.

These passed on 2026-08-19 and are the cheap, high-signal set. Prefer them.

| check | result | cost |
|---|---|---|
| `tools/translation_scorecard.py` | 18/19 journeys equivalent; level 18 stops at materialization | ~6 s |
| `tests/test_precompile_to_ssa.py` | Historical baseline: 34 passed; current-tree result is recorded above | ~7 s |
| `tests/test_symbolic_fluid_native_runtime.py` | 1 passed | ~17 s |
| `tests/test_abstract_tensor_indexing.py` | 2 passed | ~1 s |
| `tests/test_ssa_fusion_regions.py` | 1 passed | ~3 s |
| `tests/test_region_kernel_dedup.py` | 2 passed | ~3 s |
| `tests/test_compiled_linalg.py` | 6 passed, 1 xfailed (strict) | ~9 s |
| `tests/test_ir_sequence_tables.py` | 23 passed | ~3 s |
| Full-native vehicle contract/link helpers (focused, 2026-08-30) | 3 passed: contract boundary rejection, ordered-view rebinding, allocator namespace filtering | ~4 s |
| Canonical balloon full-native lowering + C emission (batch 8, 2026-08-30) | 52 linked functions; gate complete; no unresolved/unmaterialized/non-native boundary; no id-scale finding; C complete with 0 shortfalls | ~48 s |
| `tests/test_aggregate_call_identity.py` (new, 2026-09-02) | **5 passed** (~8 s) | Synthetic ~1 min reproductions of the dually call-boundary rules: tuple-of-span-parameters stays dataflow (was folded to a Constant of ProgramABI facts -> `aggregate call binding for 'tire_history' ... 4 != 0`), member formals bound by exact index, nested-tuple results/returns correlated by structural path, unpacked multi-result call carried inside a retained loop (the "carried update value 251 has no producer" shape), non-recycled value ids, identity-keyed dependency-order cache. Run these before any ~8 min full dually lowering. |
| Planner/linking/loop batch on the 2026-09-02 aggregate-identity tree (`test_hierarchical_plan`, `test_loop_composer`, `test_vehicle_python_graph_source`, `test_process_graph_function_linking`, `test_loop_carried_producers`, `test_state_loop_deployment`, `test_precompile_to_ssa`) | **31 failed, 221 passed** (4.5 min) | All 31 reproduce identically on a pre-patch copy of `glsl_deployment_strategy.py` (verified by swap for the 11 `test_loop_carried_producers` failures -- `loopresult value-source identity conflicts with its semantic edge`, the dirty-tree `value_source_id` port work -- and for the 4 callsite-descriptor/nested-loop tests; the remaining 16 are the unpack-arity, `NameError: plans`, source-text and record/loop identity failures whose messages do not touch the patched paths). Not regressions from the aggregate-identity work. |
| Full dually repository-SSA lowering (`lower_vehicle_python_graph_ssa(inputs=dually_vehicle_python_compilation_inputs())`, 2026-09-02, ~7 min) | **plan + control deployment complete; SSA emission stops on 2 shortfalls** | The `tire_history` arity error and the `vehicle_tire_recurrence` loop-carried `251` shortfall are gone. Remaining: `balloon_tire_vector_step__specialized...__planned_region_12::where` operands `((8,4,1,1,1),(8,1,1,1,2),(8,4,800,2,3))` and `planned_region_13::add` `((8,4,1,2),(8,4,1,3))` -- shape specialization inside the balloon step under the dually profile (present in every run today, independent of the identity fixes). Known pre-existing ABI leak still visible in synthetic programs: loop variable `step` and loop-body scalars appear as `linked_call_frame_storage` root arguments (same in a fully positional program). |
| Full dually repository-SSA lowering with strict `tensor_data_descriptors` (2026-09-02, later, ~7 min) | **plan + control + region SSA complete; frame linking stops** | The two balloon `where`/`add` shape shortfalls above are GONE. Linking now stops in `_prune_unused_callee_formals`-style cleanup: `call has fewer operands than the callee signature while pruning 'vehicle_tire_recurrence__specialized...': operands=85 formals=501`. Probe (`scratch: probe_recurrence_formals.py`): of the 501 formals, 402 are `linked_call_frame_storage` slots propagated outward from the `balloon_tire_vector_step` callsite (463 of 501 are rank-0 float64; 13 carry `unbound_variant_source_id`/`variant_column=row`), i.e. the balloon step's own frame scalars become the recurrence's public formals and the tick->recurrence call was linked before that growth. This is the activation-storage contract family (`dynamic temporary ... has no native activation-storage contract`), not the aggregate-identity family: `tire_history` members `(8,4,3)`, `(8,4,3,3)` arrive as proper formals. Next identity source to fix: why the balloon step's loop-body scalars are frame storage instead of locals. |
| `tests/test_deployment_outlining.py` (2026-09-01) | 6 passed — includes a COMPILED pooled loop linked against `turing_pool.c` matching serial numerics bit-for-bit, the invariant-append guard, and the ordered-join refusal | ~7 s |
| `tests/test_repository_ssa_dispatch.py` (2026-09-01) | 3 passed — planner treats outlined single-lane iteration regions as launchable; un-outlined ones name the pass | ~3 s |
| Managed vehicle deployment region outline (2026-09-01) | `step_with_dt_control_used...` region 0 outlines (19 live-ins, guarded append block), plans launchable+parallel, and emits `turing_pool_deploy_span` + effect locks in module C | ~4 min (lowering dominates) |
| Managed vehicle COMPLETE module emission (2026-09-01, later) | `complete=True` after three `ssa_c_backend` repairs: requirement-backed dynamic temporaries, the `resolve_span_origin` extent walk (output edges, `ssa_call_result_from` hops, cast_like/broadcast shape transfer, binary broadcast-combine, local-def-before-edge), and flat `max/min/all/any` scan spellings. Unresolvable extent origins refuse again instead of registering unfillable slots. The six `test_ssa_c_aggregate_constants.py` spec failures above remain as the storage-contract *style* spec (const qualifiers, local frame arrays) — the emission-blocking subset is fixed. O3+pool compile of the monolithic TU takes ~27 min. | ~25 s emit |
| Compute-shader lane selection (2026-09-01) | `deployment_compute_selection.select_compute_lanes` — vehicle lane refuses with 4 named reasons; synthetic straight-line lane judges eligible (`tests/test_deployment_outlining.py`, 7 passed) | ~6 s |

**That table is the recommended regression gate for compiler changes.** It is
~40 seconds total and it caught nothing false in this session. Reach past it
only when your change plausibly touches something it does not cover, and then
reach for single files with an external `timeout`, never the whole tree.

## Marking expected failures in code

The manifest is the cheap fix. The better fix is `@pytest.mark.xfail(reason=
"pre-existing at af00599, see TEST_BASELINE_AND_HAZARDS.md", strict=False)` on
the known-bad tests, so a green run means green and nobody has to cross-check
a document. That needs the 13 names from `test_ast_indexing_aot.py`, which
per rule 3 should be collected the next time someone runs that file for a
real reason — not by a run commissioned for this purpose.

`strict=False` matters: these should announce themselves as XPASS when
somebody finally fixes them, rather than failing the suite for being fixed.

2026-09-05 DT control fixes: targeted native None/callable(None)/variadic Boolean
checks3passed29.70s; independent-flag sequential branch native check1passed11.98s;
existing equal-region nesting/shared predicate/nested conditional overlay3passed
2.72s. Native subprocesses20s timeout. Actual DT energy-sidechain C/DLL parity
passed for3exact dyadic dt values with optional limits disabled and emptychannels.
Broader call with conditionally rebound input remains broken; see continuation.
Full managed build session91655 active13:53:35, do not overlap heavy operations.

2026-09-05 terminal correction: managed session91655 exited1 at C emission.
No live heavy build. Current saved SSA/C/shortfalls in
build/managed_dt_control_fixes_20260905. Four native call-predicate regressions
passed30.22s after catching/fixing zero versus None equality (session15880terminal).

2026-09-05 native input receipts: 2receipt+4nativepredicate checks6passed30.88s.
Existing loop-carried record-return plus2receipt checks3passed2.89s afterlate
record-merge refresh. Known baseline test_returned_record_fields_feed_structural_call_argument
failed asalreadydocumented above; no testchanged. ActualDT proposal helper C
emissionclean1.08s. Fullmanagedsession83504 TERMINALfailed receiptconsistency;
fixedunupdatedIDs inkeyed-storage removal. Newfullbuildsession78547 PID5848 live
14:26:24; do notoverlapheavywork; latestcontinuationhascommand/log.

2026-09-05 14:37: three tests/test_native_call_input_receipts.py tests passed in2.17s, including tensor-output identity without field-name metadata. Prior full managed session78547 terminalexit1, three emission shortfalls. New full managed session90068/PID16520 active; see continuation before launching another build. Plain conditional append then bool(list) reproduces region0 arity0/1; no native execution performed.

2026-09-05 14:49: tests/test_native_sequence_truth.py plus three native call
receipt tests and two existing query scheduling tests passed6 in11.33s.
New test first failed emission0/1, then failed native assertion due private
capacity0; both corrected. Native test subprocess timeout20s. Session90068
terminalexit1 (only bool(reasons) region arity failure); replacement full build
session41778/PID19476 active, managed_dt_sequence_truth_20260905. See continuation.

2026-09-05: full managed session41778 TERMINAL exit1, now zero C emission
shortfalls. Stops before native compilation on unnamed public47/57. No live
build. tools/managed_dt_parity.py syntax checked only. Minimal completed Boolean
passed to float becomes extra public input12; direct return of the same Boolean
has correct produced Select12. See continuation for exact source and next seam.

2026-09-05 15:07: native Boolean intermediate regression passed12.65s; after
native result-contract refresh, record-return physical-surface baseline test
(run explicitly while fixing it), scalar-result-type test, and Boolean test
passed3 in12.79s. Full managed replacement session8623/PID2844 active, directory
managed_dt_result_contract_refresh_20260905; inspect continuation before builds.

2026-09-05 15:15: test_native_scalar_index_store.py passed13.39s (actual repository
tensor provider, C/DLL native writes and unchanged neighbor, subprocess20s).
Fullmanaged session8623 terminalexit1: aggregate result types fail convergence;
failed-link snapshot has coerce/advance Loads defining same fieldSSA objects.
No full build active. See continuation before starting next expensive build.

2026-09-05 15:24: native exact-forwarded-record-fields test plus existing
record-return physical-surface test passed2 in12.31s. Native test includes
callee mutation retained through void/inout call and downstream result.
New fullmanaged session86224/PID15832 active, managed_dt_forwarded_fields_20260905.

2026-09-05 15:39: fullmanaged session86224 terminalexit1, passed typefixedpoint,
only Cshortfall was restoredforwardedCallres. Clearingexactone suchres in saved
fullSSA givesCshortfalls empty. Source marker fallback corrected. No new native
test/executable in this turn. Freshfullbuild session76288/PID16400 active,
managed_dt_forwarded_marker_20260905. Seecontinuationbeforestartingwork.

2026-09-05: tests/test_managed_native_output_contract.py: 2 passed in3.81s.
Wrapper-only checks: allocate actual scalar return storage; reject missing root
inputs before invoking native toolchain. Not a native execution/parity test.
Fresh DT source build session76288 is terminal with C shortfalls=[]; wrapper
output44/45 corrected. Native toolchain continuation session44475 remains live.

2026-09-05 17:01: all earlier compile/test sessions terminal. Native cap/physical
storage checks: 7 passed, 1 obsolete typed-C-signature assertion failed (45.84s).
Control source tests: 20 passed (2.04s). Full fresh source DT build session43311
active, O0, managed_dt_predicate_partition_20260905, microstep runtime fixture.
No overlapping native jobs. See continuation before starting builds or timing.

2026-09-05 17:10: full source43311 terminal0. Parity67705 terminal1: native60s
timeout; eager setup15.446909s, stepping0.276230s, exit0. No native/build job live.
Repeated-bounds native regression58263 passed10.22s. Remaining full cap producer
still after loop; see continuation. Predicate AST helper filtering untested.

2026-09-05 17:12: final predicate syntax refinement verified by49988 terminal0:
25 passed,3 inapplicable parameter combinations skipped,39.05s. No live jobs.

2026-09-05 17:22: atomic control dependency regression and prior control/native
cap cases:28passed3skipped39.52s,24126terminal0. Capture15882terminal0. Full fresh
build11416 active in managed_dt_atomic_control_order_20260905, no overlap.

2026-09-05 17:34: fullbuild11416terminal0, cap producer now beforewhile in C.
Parity45565terminal1 native60stimeout/eager0; trace48147terminal0(copy only).
Carried-call regression86189failed then fixed;39229terminal0,5passed17.68s.
No live jobs. Remaining fullstep returns use unproduced locals before producers
in unreachable_return_control. See continuation before next build.

2026-09-05 17:55: return-slot/planner and post-conditional call fixes:7passed
39.92s (73613), direct-truth variants2passed17.61s (20540). All earlier jobs
terminal. Full fresh source build75251 ACTIVE, managed_dt_return_order_20260905,
O0 microstep fixture; no overlap. See continuation before starting another job.

2026-09-05 current terminal correction: build75251 passed; parity6606 finished
(native0/eager0, no timeout) but11 mismatches. managed_dt_return_order_20260905
is latest full DT diagnostic; no parity/performance claim. Dominance tests3passed.
Active lower-only trace9550 writes artifacts/compiler_evidence/coerce-placement-trace.log; no concurrent
heavy work. See continuation report for call460 loop-ownership contradiction.

Trace9550 TERMINAL0: missing coerce460 marker/anchor precedes fallback insertion;
late ABI refresh keeps already-wrong entry placement. New focused scheduler
capture86652 ACTIVE (lower only), artifacts/compiler_evidence/step-call-schedule-capture.log.

18:45 scheduler capture86652 terminal0; no live jobs. Conditional call-order
regression passes. Scheduler test file has5 pre-existing failures verified by
in-memory baseline disabling only the new change (75pass5fail; with new test
76pass5fail). Full saved DT plan still orders advance too late; no fullbuild or
parity rerun. See continuation for exact evidence and replay artifact.


2026-09-07 patch-sequence implementation: mapping copy/empty/overwrite native
cases pass; scalar capture/first-arm/nested/increment plus identity-child native
cases passed 5 in 44.72s. Fixed call-only rejection loop now passes zero/one/three
accepted attempts, rejection and early break on reused buffers in one compile.
Snapshot copy has separate length and contents; the private-frame watch API now
publishes its arena without changing private initialization. Latest mixed batch
passed 9 with one saved-bound-method return failure under repair. Earlier
structural Boolean/control/provenance/receipt batch passed 31, with two test
placement errors subsequently corrected; Boolean cases and control rendering
then passed. The append fixture validates the packed shim rather than expecting
an empty C file. Three former native xfails have passing implementations.

IMPORTANT USER OVERRIDE: “YOU CANNOT KEEP USING TIMEOUTS”. Do not impose further
execution time limits on this work. Two lower-only runs were previously killed
at 300 and 600 seconds and produced no fresh formal report; that workflow was
wrong for this task. The no-deadline full diagnostic is session42101 / PID11756,
log artifacts/compiler_evidence/patch_sequence_diagnostic_v3.log (started 07:06:28). It predates the
latest bound-method and explicit-split persistence repairs. Do not mistake
microtest passes for completion of the all-19 series or full controller parity.
See docs/IMPLEMENTATION_2026-09-07_PATCH_SEQUENCE.md for the implementation ledger.

2026-09-07 priority repair update: session42101/PID11756 is TERMINAL, explicitly
stopped after read-only stack samples and rule inspection diagnosed a join/split
allocation cycle (next value 70510 -> 71570). This was not an elapsed-time cutoff.
Frame rule priorities now retain incumbents on ties and log transformation
provenance. Broad native/priority batch: 18 passed in 108.71s. Latest focused
priority + bound-method + rejection batch: 9 passed in 20.57s. Saved dependency
closure regression: 1 passed in 3.91s without native compilation.
Saved graph replay v4 terminated on the missing dt_limit_hint deployment shell;
method normalization now precedes dependency closure for persisted graphs.
Current lower-only replay is session85685 / PID11580, log
artifacts/compiler_evidence/patch_sequence_replay_v5.log. It attempts a pre-frame-link checkpoint.
It uses an older reduced graph, so even successful lowering cannot establish
parity for the latest source reduction edits. Full all-19 implementation remains open.

2026-09-07 09:03 update: v5 through v10 are terminal. The reusable v6
pre-frame-link checkpoint is available; `tools/replay_ssa_checkpoint.py` skips
extraction/planning. Frame linking converges in six rounds. Result types now
use finite proof priorities with incumbent ties; incompatible physical layouts
remain hard findings. Explicit scalar/span input and fresh-output conversions
preserve physical storage. Four transient scalar-read regressions are repaired.
Latest combined batch: 24 passed in 55.69s. Current lower-only replay v11 is
session18036, log artifacts/compiler_evidence/patch_sequence_replay_v11.log, without an execution
deadline. This cached graph still cannot establish fresh-source controller parity.

2026-09-07 09:30 update: v11 is terminal; v12 completed repository SSA and saved
build/patch_sequence_replay_v12/repository-ssa.pkl, with 47 strict findings.
Result provenance rejection and physical metadata propagation are repaired;
latest priority/scalar/storage batch passed 24 in 46.54s. Dictionary producer,
overwrite, schema, capacity, source ordering and filter-ownership repairs pass a
6-test native collection batch (46.16s), including one reused DLL for constant and
runtime comprehensions. Fresh controller lowering is session43484; logs and
checkpoints are under build/patch_sequence_fresh_v13. No execution deadline.
Full all-19 implementation and controller parity remain open.

2026-09-07 final v13 checkpoint update: session43484 is TERMINAL. Fresh source
lowering completed and saved its pre-frame-link and repository-SSA checkpoints.
Frame/result convergence: 6/3 rounds, two optional-result conflicts. The strict
pre-gate scan reported 45 findings; subsequent gate pruning reports 18 unaccounted
formals across five functions. The controller is still rejected, not native-ready.
Follow-up structural collection/storage tests: 12 passed in 3.39s. No test or
lowering process from this batch remains active.

2026-09-07 region-view accounting update: saved pre-frame replay v16 is
terminal and wrote `build/patch_sequence_replay_v16/repository-ssa.pkl` plus
`artifacts/compiler_evidence/patch_sequence_replay_v16.log`. An internal reshape/view now reaches an
accounted fixed point only when one owned planned region supplies an exact
tensor descriptor, its resident storage descriptor and SSA producer agree,
and byte extent and dtype match. The receipt records region, storage identity
and target shape. Public outputs, wrapper-visible identities, missing storage,
and equal competing region claims retain the incumbent open finding. This
closed the 17 `balloon_tire_vector_step` reshape findings and the one
`balloon_tire_managed_advance` reshape finding: the strict replay scan moved
from 44 to 26 findings without source extraction or native compilation. The
focused receipt test passed. In the adjacent three-test batch, two tests passed
and the existing structural-boolean opcode-spelling assertion failed even
though its unresolved-call and structural-shortfall checks passed; that
unrelated behavior was not changed. Full controller native parity remains open.

2026-09-07 scalar/keyed ownership update: scalar `.item()` identities and exact
`mapping or {}` keyed reads now retain proof receipts. Lookups consumed only by
a linked source call use that call's exact argument binding as local ownership;
the result no longer becomes a public formal. Focused batch: 3 passed in 7.06s,
without execution deadlines. Replay v17 removed the final three structural
outputs (26 -> 24 strict findings but exposed `_no_exchange_observed` value 11);
replay v18 materialized that lookup at its source-call owner and reports 23
strict findings. Artifacts:
`build/patch_sequence_replay_v18/repository-ssa.pkl` and
`artifacts/compiler_evidence/patch_sequence_replay_v18.log`. Remaining: four formal-parity groups,
one optional merge, sixteen dominance findings, and two call-result conflicts.
No full native controller compile or parity claim.

2026-09-07 structured while-predicate update: computed numerical regions under
a structural predicate are evaluated at while entry and latch; captured leaves
keep their existing owner. Persisted plans use the same exact unique-owner rule
and record `(loop, region, deployment nodes, rule)` provenance. Focused batch:
5 passed in 5.80s. Replay v19 was unchanged at 23 because it proved the saved
control needed reconciliation as well as fresh composition. Replay v20 then
removed both `run_superstep` `%71` dominance findings and reports 21 strict
findings: four formal groups, one optional merge, fourteen dominance findings,
and two call-result conflicts. No execution deadlines or native parity claim.

2026-09-07 PlanCall loop-ancestry update: lexical source-call placement now
prefers the hierarchy's exact `enclosing_loop_ids` and uses source spans only as
a fallback. This keeps calls inside persisted comprehension loops even when the
comprehension AST has no line span. Focused batch: 2 passed in 4.60s. Replay v21
moves `coerce_metrics` callsite 9 after its loop-target projection, removes its
sole dominance finding, and reports 20 strict findings. The remaining three
unnamed `coerce_metrics` formals were not reclassified or hidden. No execution
deadline, full native compilation, or parity claim.

2026-09-07 defensive keyed-iteration update: exact `mapping or {}` wrappers now
reuse the declared mapping length/key/value ABI slots. Projected iteration keeps
its GEP and indexed Load; only the arena base changes. `str(key)` is an identity
only for a key projected from a declared `string_token` mapping, with separate
provenance receipts. Final focused batch: 2 passed in 4.01s; adjacent direct
mapping, defaulted lookup, and loop-call checks also passed. Replay v22 removes
all three `coerce_metrics` unnamed formals (25 -> 22 function formals) and its
formal-parity finding. Strict frontier: 19 findings—three formal groups, one
optional merge, thirteen dominance findings, and two call-result conflicts.
No execution deadline, full native compilation, or parity claim.

2026-09-07 region in/out ownership update: a planned region's explicit
capture/output intersection now proves that its result and incumbent formal are
the same writable storage, even if later record linking has no field label to
repeat. Mutable scalar fields that are only written now retain their exact
`SetAttr` source identity when scalar control lowering cannot represent the
producer. Focused batch: 8 passed in 11.09s. Replay v24 removes managed advance
`%182`, reduces the converged frame from 4,330 to 4,290 formals, and reports 18
strict findings: two formal groups, one optional merge, thirteen dominance
findings, and two call-result conflicts. Artifact:
`build/patch_sequence_replay_v24/repository-ssa.pkl`. No execution deadline,
full native compilation, or parity claim.

2026-09-07 total scalar dt-limit hint update: the managed tire returns its
positive declared integration step or `0.0` when inactive. `run_superstep`
already admits only finite positive hints, so this removes an unneeded
payload/`None` union without changing which bound is applied. Focused batch: 4
passed in 68.62s. A fresh source build was required because v24's pre-frame
checkpoint contained the old graph. Fresh v25 reports 17 strict findings—two
formal groups, thirteen dominance findings, and two call-result conflicts—and
no optional merge. The repository SSA was saved before the expected remaining
full-native gate rejection at
`build/patch_sequence_fresh_v25/repository-ssa.pkl`. No execution deadline,
native code compilation, or parity claim.

2026-09-07 concrete managed-tire Metrics update: the tire advance now reports
`advanced_dt=dt` and uses the largest finite float as the neutral `dt_limit` for
the controller's minimum clamps. Focused batch: 2 passed in 15.18s. Fresh v26
reports zero incompatible result contracts, an empty structural-output gate,
and 15 strict findings: two formal groups plus thirteen dominance findings.
Artifact: `build/patch_sequence_fresh_v26/repository-ssa.pkl`. No execution
deadline, native code compilation, or parity claim.

2026-09-07 predicated-continue edge completion: focused loop lowering passed 4
tests in 2.55s. Replay v27 removes both while-header definition-dominance
findings and reports 13 strict findings: two formal groups plus eleven
function-exit dominance findings. Artifact:
`build/patch_sequence_replay_v27/repository-ssa.pkl`. A broader three-file run
reported 125 passed and 16 failed. Two failures are the recorded
`test_precompile_to_ssa.py` baseline above; the other 14 reflect unresolved
dirty-tree suite drift and are not treated as green validation for this chunk.
No execution deadline, native compilation, or parity claim.

2026-09-07 constant-while CFG update: exact Boolean while predicates now use a
direct header branch while preserving real break exits. The focused
while/return batch passed 8 tests in 2.48s; the branch-compartment file passed 3
tests in 2.05s. Replay v28 removes all eleven function-exit dominance findings
and reports only two strict findings, both formal-parity groups. Artifact:
`build/patch_sequence_replay_v28/repository-ssa.pkl`. No execution deadline,
native compilation, or parity claim.

2026-09-08 scalar-field conditional update: the final focused conditional slice
passes 6 tests in 1.96s. Replay v35 completes without an execution deadline,
converges at 4,280 frame formals, reports zero incompatible result contracts,
and removes anonymous `%347` (`ctrl.clamp_events`) from the raw controller
formal-parity group. The raw surface is now 17 unnamed values across the same
two groups. Artifact: `build/patch_sequence_replay_v35/repository-ssa.pkl`.
The production pruning gate and native execution were not rerun, so this is not
a parity claim.

2026-09-08 exact record-field-state precedence: the combined focused batch
passes 7 tests in 3.68s. Fresh v40 completes source extraction, planning, and
repository SSA lowering without an execution deadline. It retains 37
controller regions, converges at 4,274 frame formals, and reports zero
incompatible result contracts. Exact reducer field-state Phis now suppress
duplicate flat attribute histories, removing controller formals `%428` and
`%429`. The production-pruned frontier is `run_superstep` `[124, 126]` and
`step_with_dt_control_used` `[477, 458]`. Artifact:
`build/patch_sequence_fresh_v40/repository-ssa.pkl`. The gate rejects those
four remaining anonymous values, so no native compilation or parity claim.

2026-09-08 stable controller diagnostic tokens: the dt controller graph now
contains zero `JoinedStr` nodes. The focused controller batch passed 21 tests in
1.26s and specialized source-provenance/formal recovery passed 7 tests in
4.08s. Fresh v32 reduces controller regions 44 -> 37, converged formals 4,320 ->
4,283, and final-gate anonymous slots 12 -> 7. Two formal-parity groups remain:
`run_superstep` `[124, 126]` and `step_with_dt_control_used`
`[428, 347, 429, 477, 458]`. Artifact:
`build/patch_sequence_fresh_v32/repository-ssa.pkl`. No execution deadline,
native compilation, or parity claim.

2026-09-08 invocation-site structural BoolOp feeds: call linking now
reconstructs an exact source Boolean expression after provisional-formal
creation, normalizes graph node keys through semantic `value_id`, and retires
the unaccounted placeholder only after every ordered operand resolves. ABI or
frame storage cannot be reclaimed, and a resident same-ID producer wins. The
focused batch passes 5 tests in 3.13s. Fresh v41 completes source extraction,
planning, and repository SSA lowering without an execution deadline, converges
at 4,273 frame formals, reports zero incompatible result contracts, and has no
dominance finding. It removes controller `%458`.
The raw surface is 14 unnamed values across two formal-parity groups; production
pruning leaves `run_superstep` `[124, 126]` and
`step_with_dt_control_used` `[477]`. Artifact:
`build/patch_sequence_fresh_v41/repository-ssa.pkl`. The gate rejects those
three remaining values; its unmaterialized-boundary, unresolved-call,
undefined-operand, optional-merge, structural-output, and non-native families
are empty. No native compilation or parity is claimed. The broader
call-linking file reports 47 passed and 9 failed in 16.97s; the failures are in
record aliasing, keyed mapping/default selection, span propagation, and
callsite shape behavior rather than the new structural BoolOp case. This is a
recorded dirty-tree suite frontier, not a whole-file pass.

2026-09-08 total unresolved-report sequence: `Metrics.unresolved_report` now
has an empty `list[str]` default, and exact column dtype evidence propagates
through matching materialization and aggregate-Phi edges to a finite fixed
point. Existing contracts retain precedence. The focused batch passes 3 tests
in 2.45s. Fresh v45 completes repository SSA lowering without an execution
deadline, converges at 4,276 formals, and removes controller `%477` from the
production-pruned gate. That gate now contains only `run_superstep`
`[124, 126]`; all other gate families are empty. Raw self-checking exposes one
separate controller dominance defect where `if_merge.11` reads `%478` from a
`while_exit` definition. Artifact:
`build/patch_sequence_fresh_v45/repository-ssa.pkl`. The gate stops before C
compilation, so this is not a native parity claim.

2026-09-08 resident report-tail and table ABI update: constant nonnegative
tail slices over resident sequences now lower as loop starts over the base
arena. ProgramABI table fields allocate complete resident sequence storage
when specialization removes their aggregate producer, and record-frame
linking correlates the complete sequence while retaining the caller incumbent
on equal-priority identity matches. The focused batch passes 5 tests in 6.22s.
Replay v54 converges after six frame rounds and three result-type rounds, with
zero incompatible result contracts and seven raw structural findings. It
removes `run_superstep` `%126`; `%127` plus the missing physical `Metrics` row
descriptor remain the direct `run_superstep` frontier. Artifact:
`build/patch_sequence_replay_v54/repository-ssa.pkl`. No execution deadline,
production-pruning rerun, native compilation, or parity claim.

2026-09-08 tail-domain ownership update: regions consumed by structured loop
bounds are excluded from flat scheduling independently of complete loop-body
emission. Deferred record rows resolve exact compiler aliases before layout
checking. The exact focused regression passes in 4.37s. Fresh v56 converges at
4,287 formals with zero incompatible result contracts and six raw findings.
It removes the `run_superstep` formal-parity group entirely; `%127` is gone.
The remaining `run_superstep` issue is a measured 11-versus-15 `Metrics` row
layout, requiring the three `error_channels` columns and a nested report-table
handle. Artifact: `build/patch_sequence_fresh_v56/repository-ssa.pkl`. The
broader loop-filtered batch remains non-green at 65 passed / 3 failed. No
execution deadline, C compilation, or native parity claim.

2026-09-08 late returned-record surface update: call linking follows exact
caller result aliases and merges callee fields discovered in later fixed-point
rounds into the resident caller descriptor. It adds only missing flat storage;
aggregate fields require real resident descriptors, and equal-priority alias
ties keep the incumbent. Replay v57 converges after six frame rounds and three
result-type rounds with zero incompatible contracts. The `run_superstep`
`Metrics` append improves from 11/15 to 14/15 columns; only the nested
`unresolved_report` handle remains. The fuller record surface also exposes 23
anonymous `step_with_dt_control_used` formals and raises the frame total to
4,365, so the raw finding count remains six. The two focused flat-record tests
pass; the existing full `Metrics` row test remains non-green at the known
nested-table boundary. Artifact:
`build/patch_sequence_replay_v57/repository-ssa.pkl`. No execution deadline,
production-pruning run, C compilation, or native parity claim.

2026-09-08 optional-control/local-mutation update: pre-planning presence
rewriting, scalar C `logical_not`, exact synthetic return merges, distinct
optional payload storage, adjacent mutable presence stores, canonical
ProcessGraph allocation, and idempotent aggregate-leaf ledger repair pass 17
related tests in 8.12s. The two executable optional tests compile only at
`-O0` and cover absent, present-zero, present-nonzero, and absent-to-present
local mutation. No execution deadline was used.

Fresh v71-v74 are diagnostic failures at a stale `snapshot` aggregate ledger;
do not use them as SSA baselines. Replay v76 from the fresh v74 resolved graph
passes that planner boundary, converges frames in six rounds and result types
in two, and saves `build/patch_sequence_replay_v76/repository-ssa.pkl`. Its raw
audit has six findings: four formal-parity groups (1, 3, 24, and 388 unnamed
values) and two controller function-exit dominance errors. The linked
`update_dt_max` specialization still drops optional-presence provenance, so
the full controller is not ready for C emission or native parity. Do not run
an optimized compile while these correctness findings remain.

2026-09-08 zero-structural/zero-emission checkpoint: recursive late record
materialization closes the window `Metrics.hard_failure` boundary. Replay v68
and fresh production v69 report zero structural findings. C emission initially
reported 18 shortfalls, fell to two after scalar `item` and finiteness support,
and reaches zero after exact initial record-field projections become explicit
planned-region scalar captures. Replay v70 records both `clamp_events`
projection receipts against incumbent `%124` and removes the obsolete record
receiver from the region ABI.

The first full compile exposed repeated declarations of mutable scalar SSA
identities (`t5` and `t11`). Later loads now assign the first C local rather
than redeclaring it. The focused executable regression and the related record
capture/item tests pass: 3 tests in 3.55 seconds, following a 5-test batch in
4.53 seconds. The full zero-shortfall v70 source compiles successfully at
`-O0` to a 1,650,688-byte DLL. No execution deadline was used. Do not use an
optimized full compile until correctness execution and native parity are
complete; parity is not yet claimed.

2026-09-08 optional ABI checkpoint: attempting to construct real v70 native
feeds stopped on `controller.dt_min=None` before the DLL ran. This is a useful
guard failure, not a native result. ProgramABI now represents optional scalar
fields with separate Boolean presence and typed payload slots; feed packing
rejects absence unless the pair exists and preserves present zero. The default
controller schema marks `dt_min` and `dt_max` optional. Contract receipt,
lowering materialization, absent/present packing, and the legacy rejection path
pass 9 focused tests in 4.32 seconds. No deadline was used. Do not treat this
as optional-control correctness: source predicates and mutable presence writes
still need to be wired and verified in a fresh build.

2026-09-08 controller formal-closure update: exact resident-field projections,
retained mutation literals, and whole-object unreachable-CFG/Phi pruning remove
all 23 anonymous controller formals. Replay v67 converges after six frame rounds
at 4,344 formals and three result-type rounds with zero incompatible contracts;
the pass records 18 CFG/Phi changes. The raw checker now reports one finding in
the entire saved module: window `%47`, the `hard_failure` field read from
returned `Metrics` receiver `%46`, whose record descriptor is absent in the
window caller. The focused reachability/ABI suite passes 6 tests in 8.57s,
including native execution of both retained return paths, without an execution
deadline. Artifact: `build/patch_sequence_replay_v67/repository-ssa.pkl`. No C
compilation of the full controller or native parity is claimed.

2026-09-08 resident returned sequence update: exact one-output planned
projections of a descriptor-backed record sequence field become accounted
resident arenas after frame linking. Replay v58 proves `%478` has no local
definition, remains a live consumed argument, and carries provenance naming
planned region 34. Both dominance findings disappear; raw findings fall from
six to four, while the controller anonymous-formal set shrinks from 23 to 20.
The new fixed-capacity child-copy append helper passes its direct CFG test and
the combined focused batch passes 3 tests in 4.78s. Deferred record rows do not
yet invoke that helper, so `run_superstep` remains 14/15 and the controller row
still lacks `unresolved_report`. Artifact:
`build/patch_sequence_replay_v58/repository-ssa.pkl`. No execution deadline,
production-pruning run, C compilation, or native parity claim.

2026-09-08 deferred child-row update: whole-record and embedded-record rows
now expand in ProgramABI order and snapshot one nested leaf sequence into a
caller-owned fixed-stride child pool. Replay v64 removes both record-row
findings, leaving two raw formal-parity groups only: window `%47` and 23
controller values. Sequence 69/66 uses handle column 14 and a 37-argument
helper; sequence 39/478 uses handle column 15 and a 41-argument helper. Every
source-linked call has exact callee arity, every child-pool identity names a
resident value, and deferred row markers are absent. The focused batch passes
4 tests in 5.60s. Artifact:
`build/patch_sequence_replay_v64/repository-ssa.pkl`. No execution deadline,
production-pruning run, C compilation, or native parity claim.

2026-09-08 optional cross-call checkpoint: never treat equality between a
ProcessGraph presence ID and an SSA instruction-result ID as ownership. The
presence slot must be a real formal; otherwise its Store may survive while the
call frame silently omits the target. Materialization now allocates a distinct
physical ID on that collision, records requested and physical IDs, and uses
payload/presence roles in frame keys. Equal-priority formal evidence keeps the
incumbent.

Replay v80 saves `build/patch_sequence_replay_v80/repository-ssa.pkl` and
proves requested presence `%24` is physical `update_dt_max` formal `%37`, the
presence Store targets `%37`, and the source-linked call carries the matching
fifth operand. It converges at 4,353 formals in six rounds and reports the same
six raw findings as v76: formal groups 1/3/24/388 and dominance `%263`/`%264`.
Focused optional tests compile only at `-O0`. No full compile, optimized
compile, execution deadline, or parity claim was used.

2026-09-08 return-edge dominance checkpoint: exact direct return expressions
from pure planned regions are recomputed on a synthesized physical return edge
when their original path-correlated definition does not dominate it. The rule
requires matching return-slot provenance, allocates fresh SSA identities,
retains a dominating incumbent, records priority and tie policy, and dead-ends
after one accepted repair.

Replay v81 records `%263 -> %3390` (three operations) and `%264 -> %3395`
(five operations). Definition-dominance findings are zero; the raw checker now
reports four formal-parity groups only, sized 1/3/24/388. The focused
return-state and dominance batch passes 14 tests with the two pre-existing
native subprocess tests deselected. No compilation or execution deadline was
used. Artifact: `build/patch_sequence_replay_v81/repository-ssa.pkl`; audit:
`build/patch_sequence_replay_v81/return-edge-audit.txt`.

2026-09-08 missing-Phi-incumbent checkpoint: when reduction removes the value
named by `initial_value_id`, control lowering must not externalize its integer
identity. The graph repair accepts only one nearest same-binding common
ancestor of both Phi arms, leaves equal candidates unresolved, records the
incumbent tie policy once, and projects accepted repairs into already-planned
Control IR.

Replay v84 records `run_superstep` `%262: initial %260 -> %256`; final operands
are `(%261, %256)`, formal `%260` is absent, and its only source-linked caller
still has exact arity. Frame formals fall 4,353 -> 4,351. The raw checker now
reports three formal-parity groups sized 3/24/388. The focused control file
passes 7 tests. No compilation, optimization, execution deadline, or parity
claim. Artifact: `build/patch_sequence_replay_v84/repository-ssa.pkl`; audit:
`build/patch_sequence_replay_v84/phi-initial-audit.txt`.

2026-09-08 post-reachability pure-feed checkpoint: scalar `item()` may consume
a unique dominating resident producer when no loop-carried Phi exists; multiple
carried candidates remain unresolved. Repeat exact pure-expression recovery
after static reachability pruning and before atomic call-signature pruning so a
dead CFG predecessor cannot preserve an internal expression as an ABI input.

Replay v86 removes controller formals `%147`, `%149`, and `%231` together with
their caller operands. Their definitions are exact source operations rooted at
resident `%43`, and controller/caller arity is 1,061. The raw checker reports
two remaining formal-parity groups sized 24 and 388. The focused recovery file
passes 5 tests. No compile, optimization, execution deadline, or parity claim.
Artifact: `build/patch_sequence_replay_v86/repository-ssa.pkl`; log:
`artifacts/compiler_evidence/patch_sequence_replay_v86.log`.

2026-09-08 static-slice checkpoint: a source slice used only as an index may
become its integer lower-bound address selector only when its bounds are static
and non-negative and its stride is one. The Const receipt retains complete
bounds and source identity; unsupported slice semantics stay unresolved.

Replay v87 recovers 24 managed-advance and 388 vector-step selectors, giving
exact arities 438 and 26. Final formals fall 5,619 -> 5,207 and `run_all`
reports zero findings. Focused result: 4 passed, 81 deselected. Artifact:
`build/patch_sequence_replay_v87/repository-ssa.pkl`; log:
`artifacts/compiler_evidence/patch_sequence_replay_v87.log`.

The subsequent fresh `-O0` build reached frame convergence and stopped before
C emission/compilation at one physical-input mismatch: caller `%109`
`float64` versus Boolean optional-presence formal `%94` in `pi_update`. No
deadline or optimized compile was used. Log:
`artifacts/compiler_evidence/patch_sequence_fresh_v88_o0.log`.

2026-09-08 optional-presence physical-priority checkpoint: ProgramABI presence
materialization now writes physical `bool`, retaining the incumbent tie policy
and recording any displaced provisional physical dtype. This prevents an
earlier numerical-region capture from reclassifying the optional-presence
storage as `float64`.

Fresh pre-frame artifact: `build/patch_sequence_fresh_v89/pre-frame-link.pkl`.
Replay v90 gives controller `%109` and `pi_update` `%94` matching physical
Boolean contracts, exact call arity 12, zero result conflicts, and zero
`run_all` findings. Both focused optional tests pass and compile only at `-O0`.
Saved-module C emission reports zero shortfalls; the `-O0` DLL is 1,669,632
bytes. No runtime parity or optimized build was used. Replay log:
`artifacts/compiler_evidence/patch_sequence_replay_v90.log`; compile log:
`artifacts/compiler_evidence/patch_sequence_replay_v90/native-o0-build.log`.

2026-09-08 standalone checkpoint: optional-field linked aliases and private
linked result-record workspace are now packed deterministically. The focused
native material-contract file passes 11 tests. Saved v90 emits and builds a
78-buffer standalone executable at `-O0`; the parity tool reads the checkpoint
reference from its manifest and runs both processes without a deadline.

One-frame eager execution completes. Native exits with a Zig misalignment
panic before producing final buffers because root planned region 3 declares
its input `%47` as `float64` while the caller supplies Boolean
`Metrics.hard_failure` result `%308`; generated `cast_double_to_bool_values`
therefore reads byte storage through `double *`. This exact call-edge dtype
conflict is the current runtime frontier. Report and logs:
`build/patch_sequence_replay_v90/standalone-o0/managed-dt-parity.json`,
`parity-native.log`, and `parity-eager.log`. No optimized build or parity claim.

2026-09-08 physical-edge checkpoint: formal/capture occurrences sharing an
unshadowed ID are interned to the incumbent formal, and physical input adapters
run again after final result/aggregate/precision settlement. The transaction
is finite: replay v94 adds five late adapters and an immediate repeat adds
zero. Keyed ProgramABI members and generated lookup helpers carry explicit
physical dtypes; the prior two string-token conflicts are zero. Focused
adapter/keyed results are 7 passing tests (one small native compile at `-O0`)
plus 3 passing lookup tests.

Replay v94 converges in 6 frame rounds and 3 result-type rounds with zero
structural findings. Its 78-buffer standalone builds at `-O0`. Native and eager
both complete the one-frame run with return code 0, closing the misalignment
panic frontier. Numerical comparison has 54 matches and 24 mismatches; native
material state contains 3,456 non-finite elements. `advanced` and `dt_next`
also diverge (`0.000244140625`/`0.0` native versus
`0.008333333333333333`/`1.856126911235753e-05` eager). The next gate is the
earliest native producer of that numerical divergence. Report:
`build/patch_sequence_replay_v94-standalone-o0/managed-dt-parity.json`. No
optimized build or parity claim.

2026-09-08 collapsed-proposal checkpoint: `run_superstep` now stops before
calling physics when its fully bounded proposal is zero or negative. This
closes the traced zero-step division cycle while preserving the existing
incomplete-window report. `tests/dt_system/test_dt_superstep.py` passes 16
tests, including both collapsed cases. Fresh v96 reports zero structural
findings after six frame rounds and three result-type rounds.

The v96 78-buffer standalone builds only at explicit `-O0` and both one-frame
processes return 0. Parity is still 54 matches / 24 mismatches. Native state has
2,056 non-finite elements, all in batch lane 4, versus 3,456 in v94; the first
substep's earlier numerical divergence remains open. Report:
`build/patch_sequence_fresh_v96-standalone-o0/managed-dt-parity.json`.

2026-09-08 indexed-store RHS checkpoint: tensor SSA lowering now computes
`index_assign_double`'s `value_count` from the RHS shape. The old selection-size
value made scalar broadcasts overread adjacent memory. The 22-test focused
batch passes, and fresh v97 audits all 37 indexed stores with zero RHS-count
mismatches. Frame linking converges in six rounds, result typing in three, with
zero incompatible contracts and zero structural findings.

The v97 78-buffer standalone builds at explicit `-O0`; native and eager return
0. Native state is fully finite (27,648/27,648) and all eight batch lanes now
agree, eliminating v96's 2,056 non-finite elements. Comparison remains 54
matches / 24 mismatches: native advances one `0.000244140625` substep and then
produces `dt_next == 0`. The active gate is the inflated native first-substep
metric/reduction result. Report:
`build/patch_sequence_fresh_v97-standalone-o0/managed-dt-parity.json`.

2026-09-08 same-pass shape-settlement checkpoint: tensor lowering now fills a
missing SSA occurrence shape from an already resident static descriptor of the
same identity. It does not replace shaped incumbents, and records provenance
plus the incumbent tie policy. The 24-test focused batch passes. Fresh v98 has
zero structural findings after six frame rounds and three result-type rounds;
the formerly scalar gas reciprocal is audited at shape `(8, 4)` and count 32.

The v98 78-buffer standalone builds at explicit `-O0`; native and eager return
0. Native first-step displacement and velocity now match the eager direct-step
reference. One-frame parity remains 54 matches / 24 mismatches because native
still stops at `advanced == 0.000244140625` with `dt_next == 0`; worst state
error improves from `3.8615154563` to `0.00890879321`. The controller
continuation/result path is the active frontier. Report:
`build/patch_sequence_fresh_v98-standalone-o0/managed-dt-parity.json`.

2026-09-08 v112 checkpoint: settled identity-forward results replace 37 stale
record-field uses with exact caller residents and leave zero structural
findings. The focused combined batch passes 5 tests. The `-O0` one-frame run
still has 54 matches / 24 mismatches, but native `max_vel_ever` is now
`0.0023950195322857593` and native `dt_max` is `1.252599387837501`, proving
that the corrected metric result reaches controller state. Report:
`build/patch_sequence_replay_v112-standalone-o0/managed-dt-parity.json`.

The next zero-`dt_next` producer is a region/effect ordering error: the later
scalar-field scheduler placed an atomic region by its earliest source member,
which can move a region containing early casts and a late field update before
the field write it reads. Atomic regions now use their completion position.
The regression is included in the same 5-test batch. Because this affects
pre-SSA planning, it has not been claimed against v112 and requires a fresh
controller replay. No optimized build or process deadline was used.

2026-09-08 v116 checkpoint: fresh v113 first confirmed the atomic-region
completion repair by moving native controller `acc` from `0.0` to
`0.42714935345574623`. Exact optional `is not None` controls are now recovered
from matching ProgramABI presence slots. Restoring those branches exposed four
non-dominating stale values; object-identity Phi continuation and exact
same-physical-type cast-result forwarding reduce them to zero. Replay v116
records eight and one repairs respectively, with provenance and incumbent tie
policy.

The focused batch passes 7 tests. Replay v116 converges in 6 frame rounds and 3
result-type rounds with zero incompatible contracts and zero structural
findings. Its 78-buffer standalone builds at explicit `-O0`, and native/eager
both complete. Numerical parity is still 54/78: native controller
`acc=0.42714935345574623`, `dt_max=1.252599387837501`,
`advanced=0.000244140625`, and `dt_next=0.0`. The active frontier is the
controller value/presence propagation that still yields a zero next proposal.
Report: `build/patch_sequence_replay_v116-standalone-o0/managed-dt-parity.json`.
No optimized build or process deadline was used.

2026-09-08 v118 checkpoint: exact reuse of one `SSAValue` object by an earlier
Phi and a later branch projection is now freshened per definition. Dominance
selects the uses of each fresh version, including separate incoming Phi edges;
the first definition remains incumbent. The C backend also preserves
exact-object bindings for formals, aggregate projections, and Phi locals.
Replay v118 records eight freshenings, six Phi continuations, one identity-cast
forwarding, and zero structural findings. The focused batch passes 9 tests.

The 78-buffer executable builds at explicit `-O0`. One-frame parity improves
from 54/78 to 55/78: `advanced` now matches eager at
`0.008333333333333333`, while `dt_next` moves from zero to
`0.013326566940131533`. Native `max_vel_ever=0.3159734017361334` still differs
from eager `0.009045569831777825`, so the active frontier is the earliest
multi-substep numerical divergence. Report:
`build/patch_sequence_replay_v118-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

2026-09-08 v125 checkpoint: speculative structural recovery findings are now
settled only when their identities are absent from every final live SSA
surface. Static address-index analysis ignores nonfinite/nonintegral floats.
Proven rank-zero `item()` identities rebind stale operand objects to a unique
incumbent, and linked BoolOp call feeds reconstruct from exact ordered source
operands rather than accepting an operand-only placeholder. Proven changes
record priority, source identity, and the incumbent tie rule.

The focused batches pass 9 tests and 7 tests. Replay v125 converges in 6 frame
rounds and 3 result-type rounds with zero incompatible contracts and zero
structural findings. Its 82-buffer standalone builds successfully at explicit
`-O0`. Native and eager one-frame processes both return 0; parity is 59/82.
The 23 mismatches are material telemetry/state, controller state, and
`dt_next`. Native/eager `max_vel_ever` are `0.3159734017361334` /
`0.009045569831777825`; native/eager `dt_next` are `0.013326566940131533` /
`1.85612691e-05`. The active gate is the earliest material/telemetry producer
divergence inside the adaptive substeps. Report:
`build/patch_sequence_replay_v125-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

2026-09-08 v127 checkpoint: returned keyed record descriptors and exact static
capacity bounds now close transitively over linked call frames. Record
constructors preserve their literal mapping universe because schema packaging
is nonmutating. The propagation is a monotonic finite fixed point and records
exact ABI provenance plus incumbent tie policy. A focused batch passes 4 tests.
Replay v127 converges in 6 frame rounds and 3 result rounds with zero
incompatible contracts and zero structural findings. Capacity 3 is present on
the full advance 213 -> step 1668 -> superstep 733 -> root 349 chain. The
82-buffer standalone builds at explicit `-O0`, and generated C initializes the
root capacity cell to 3. Live parity remains 59/82 with the same 23 mismatches
and values, so the next measured frontier is the runtime alias/store/length
path across that chain. Artifact:
`build/patch_sequence_replay_v127/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v127-standalone-o0/managed-dt-parity.json`. A
separate synthetic three-entry record fixture exposed a control-effect
scheduler cycle and was left for a later focused chunk. No optimized build or
process deadline was used.

2026-09-08 v128 checkpoint: structural recovery of `not in` now emits unary
`LNot` over the exact native contains result. The old recovered form used
`contains != False`, inverted the source meaning, and made
`_energy_time_limit` return absent when both energy keys were present. Both
recovered negations record exact-source priority and incumbent tie policy. The
combined capacity/membership batch passes 5 tests. Replay v128 converges in 6
frame rounds and 3 result rounds with zero incompatible contracts and zero
structural findings; its 82-buffer standalone builds at explicit `-O0`.
Native and eager both return 0. Parity remains 59/82, but native
`max_vel_ever` improves from `0.3159734017361334` to `0.017680656115454992`
against eager `0.009045569831777825`, and native `dt_next` moves from
`0.013326566940131533` to `6.104353894230951e-06` against eager
`1.85612691e-05`; `advanced` still matches. The newly live many-substep native
path completes naturally in about 12 minutes 22 seconds. The next frontier is
the traced store-before-clear loss of `maximum_substep_displacement_m`.
Artifact: `build/patch_sequence_replay_v128/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v128-standalone-o0/managed-dt-parity.json`. No
optimized build or process deadline was used.

2026-09-09 v129 checkpoint: dynamic mapping literal rows are anchored to their
key AST nodes inside the literal instead of to possibly much earlier value
producers. Value dependencies still run first; authored key positions preserve
the literal's clear-before-row order. Lexical mutation installation now uses
the shared atomic-region completion-position rule. A focused batch passes 3
tests, including one native `-O0` artifact reused for three executions. Replay
v129 converges in 6 frame rounds and 3 result rounds with zero incompatible
contracts and zero structural findings. Managed advance sequence 213 contains
`clear, displacement, energy, power` in one block, and the 82-buffer standalone
builds at explicit `-O0`. Both parity processes return 0. The result remains
59/82 and controller values are byte-for-byte unchanged from v128, proving the
restored displacement row is not the active limiter in this fixture. Artifact:
`build/patch_sequence_replay_v129/repository-ssa.pkl`; report:
`build/patch_sequence_replay_v129-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.

2026-09-09 v130 checkpoint: the module C emitter no longer prebinds a locally
defined returned SSA object to caller-owned output storage. Internal consumers
now read the producer's local `tN`; `Ret` publishes it afterward. Formal
aliases and aggregate projections retain exact-object bindings. A focused
batch passes 4 tests, including an explicit `-O0` compiled regression for a
returned Boolean consumed before publication. The clean v129 SSA checkpoint
was reused because SSA is unchanged. Its 82-buffer v130 standalone builds at
explicit `-O0`, and the production criticality condition emits `!t307` instead
of reading `out307`. Both parity processes return 0. Parity remains 59/82, but
native telemetry moves from 0 successful / 158 critical substeps to 158
successful / 0 critical; advanced time remains exact. The next frontier is the
loop completion boundary: native performs 158 successful substeps and leaves
completion false, while eager performs 157 and completes. Report:
`build/patch_sequence_replay_v130-standalone-o0/managed-dt-parity.json`. No
optimization or process deadline was used.
# 2026-09-14 native frontier continuation

See `docs/NATIVE_FRONTIER_REPAIRS_2026-09-14.md` for commands and artifacts.
Targeted native adjoint/recurrent/optional/managed-output coverage passes
17 tests in 117.35 seconds; the optional record/scalar group passes seven in
19.07 seconds. The validator v7 replay produces repository SSA but still fails
the executable gate: 13 formal-parity groups, three dominance findings, one
optional result contract. The fresh v8 source build and real-engine 64-transition
training run must be checked for their concrete results; neither is declared
successful here. No full-suite run or baseline suppression was performed.
2026-10-18 RESOLVED-UNIT FINAL CONCORDANCE + DIAGNOSTIC RECIPE CLOSURE (CURRENT):
Checkpoint `dba86e34f1a9db226fb537b0e5b73e41a75aec7dc48fe81fc4270ff44e4fede1`
exposed two independent finalization-boundary faults.  Unit 032 serialized
`IdentityPage._reindex` with temporal sequence descriptors 41/44/48 sharing
member 53 and the exact resident alias target 2305843010213700385, but without
the `concordant_sequence_views` receipt.  The publisher existed only on the
outer whole-source return path.  It now runs at the common class-surface
finalization seam used by resolved units.  Applying that publisher in memory
to the frozen unit 032 artifact changes its one `descriptor-member-shared`
finding to none and records `(53, (41, 44, 48), 2305843010213700385)`.

Unit 048's validation-owned JoinedStr correctly named formatter sequence 36
through its `source_glyph_sequence` recipe.  Recipe closure then iterated its
own growing result and incorrectly demanded that formatter sequence 36 own a
second top-level glyph row.  Closure now expands only the original
ValidationBlock-owned glyph IDs; derived formatter/projection/union sequences
remain recipe members.  Focused compiler regressions pass 5/5 across the two
repairs.  No bootstrap run was made; the next user run is authoritative.
2026-10-18 CONDITIONAL UNION ARM PROVENANCE JOIN (CURRENT):
Checkpoint `9b424c23baea9fc36d8901ea10690311dd72da8e38c8d32060c5b6fbe3c0db9b`
confirmed the preceding descriptor/glyph blockers cleared: unit 048 compiled
and unit 033 reached SSA lowering.  Its value 63 is the authored expression
`self._fork_of_scope(row[0]) if ... else None`.  The call arm is declared
`tuple[Any, int] | None`; the else arm is literal `None`.  Both establish the
same payload-free `builtins.NoneType` physical ABI from different exact source
cells.  Conditional settlement incorrectly compared `source_cells` as part of
physical layout.  It now compares identity/storage/dtype/shape/rank/storage
identity/metadata/row shape and, on agreement, retains the union of both source
proofs.  A true physical disagreement still refuses.  Value-union and
concordance-audit focused tests pass 11/11.  No bootstrap run was made.
