# Memory graph takeover and provenance blockers: continuation

Date: 2026-10-16

This continues
[16_failure_neighborhood_dedupe_continuation.md](16_failure_neighborhood_dedupe_continuation.md).
The first fresh user-owned provenance run reached 296 units and exposed two
blockers: unit 026 (`IdentityPage._fork_of_scope`) was partial at resident
membership, and unit 031 (`IdentityPage._reindex`) rejected a nested-list sort
whose row dtype was `unknown` against an `int64` mapping key. A second fresh run
with those repairs reached SSA completion for both units and exposed their
deeper finalization boundaries, described below.

## Graph ownership contract

The canonical public entry,
`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`, now creates a
materialized `ProcessGraph`. Its topology is owned by
`BitTensorMemoryGraph`. `NetworkxEmulation` is the existing NetworkX-shaped
protocol surface over that physical graph; it no longer imports, subclasses,
or contains `networkx.DiGraph`.

NetworkX remains a valid explicitly linked execution target when the compiling
system supplies it. It is not the canonical store and is not an undeclared
self-hosting fallback. The independent adapter implements the graph protocol
used by NetworkX algorithms, so those algorithms can consume the memory graph
without changing its storage authority.

The ownership invariant is also retained across the places that previously
reintroduced a second graph:

- canonical value-id relabeling builds one new `BitTensorMemoryGraph` and
  writes mapped nodes and edges into it directly;
- extracted materialized function/subgraphs receive one owned arena and keep
  `ProcessGraph.MG is ProcessGraph.G.bit_tensor_memory_graph`;
- the empty scoped worker catalogue is memory-backed rather than an
  `nx.DiGraph` shell;
- dependency-only worker scheduling accepts the representation-independent
  sealed snapshot without requiring a non-materialized graph;
- the physical byte arena and exact integer-allocation intervals pickle and
  rehydrate, restoring process-local callbacks and graph backreferences.

Topology and allocation are now physically memory-graph-owned. Semantic node
and edge payload attributes remain Python mappings in the established adapter;
this work does not claim that every semantic payload has already been encoded
inside the byte arena.

Unconditional memory-allocation/write debug prints were removed because making
this path canonical otherwise floods the bootstrap output.

## Fresh blocker repairs

### Unit 026: nested set membership

The exact expression combines a module-level `frozenset[str]` and
`book.registry.private_pages`, where `Registry.private_pages` is `set[str]`.
The module-level frozenset already lowered on clean HEAD. The nested record was
incomplete because authored field receipts admitted list/tuple sequences but
not set/frozenset sequences. `_authored_annotation_field_receipt` now admits
`set`, `Set`, `frozenset`, and `FrozenSet`, retaining mutable semantics only
for the mutable spellings. The exact expression now emits two typed resident
`contains` operations.

### Unit 031: nested mapping-value sort

The first attempted repair matched a loop target spelling against the graph's
source-wide annotation table. The frozen graph proved that was not identity
evidence: the apparent `columns: list[int]` came from an unrelated authored
field with the same name.

The retained repair follows the exact graph path
`mapping -> values() -> For target`. It consumes the reducer's existing
`mapping_value_aggregate_kind` and `mapping_value_dtype` receipt on that
mapping identity, then declares and types the child collection represented by
the parentless loop `Input`. The regression deliberately uses the unannotated
target name `resident`, so a name collision cannot make it pass. The frozen
unit 031 graph now derives target 2 as one writable `int64` column from mapping
value 28. The existing strict sort dtype check is unchanged.

### Second fresh run: finalization boundaries

The materialized run identified by toolchain fingerprint
`a80ebe7097a52d644314741fa9dfc4f5e8bea8876689fabe0326e70cda1d130e`
completed topology and planned 296 units. Neither earlier error recurred:
unit 031 completed SSA lowering, and unit 026 reached final structural
validation. Both receipts were partial rather than failed.

Unit 026 still lacked the nested receiver and aggregate facts at finalization.
The scoped source intentionally omits nested class definitions, but its retained
ProgramABI already states `IdentityPage.book -> IdentityBook`,
`IdentityBook.registry -> Registry`, and `Registry.private_pages -> span set
int64`. The reducer now indexes those authoritative record-field receipts when
building its class-field contracts. The exact nested attribute chain therefore
retains the receiver classes and set storage needed by membership without
duplicating the omitted ClassDefs.

Unit 031 exposed a keyed-descriptor transition rather than another sort bug.
ProgramABI minted canonical `column_positions` keys, values, and length parts,
but replacement of a provisional descriptor deleted it before rebinding its
resident members. Calls retained orphaned part IDs, and the direct predicate
`self.__dict__.get("_compressed_ram")` survived as an unknown public formal.
The existing descriptor transition now binds those members to the declared
parts with their declared dtypes, advances exact field aliases through the
normal provenance mechanism, then removes stale bookkeeping. The scalar-source
contract also recognizes only the exact direct-test form
`self.__dict__.get(static_key)`, types it as bool, and the public wrapper reads
that exact authored object's dictionary key. This is an authored projection,
not a general truthiness guess.

## Utility evidence and present limit

The second run proves that the canonical compile actually uses the memory graph:
the topology phase completed with the materialized graph, and subsequent scoped
units consumed its products. It does not yet prove compressed storage utility.
Observed topology RSS was about 3.12 GB, compared with about 1.68 GB in the
preceding nonmaterialized run. The compilation-units stage improved modestly,
from about 2m42s to about 2m28s, but one pair of runs cannot attribute that
difference to the graph representation. The defensible conclusion is therefore
that storage authority and rehydration work, while payload packing and memory
efficiency remain measurable optimization work rather than established wins.

### Third fresh run: causal shape and boundary receipts

Toolchain fingerprint
`e9e4c4d6fad4ae78cb0c3c3677e99c1447e3eb0b4a2960cf0d0690dce1340ed2`
confirmed both preceding repairs: unit 026 disappeared from the blocker set and
unit 031 was `compiled-unverified`, not partial. Topology completed in 1m20s at
about 2.56 GB RSS; compilation units completed in 2m33s at about 1.58 GB RSS
with a 2.93 GB stage peak. Fail-fast admitted 10 of 296 units and reported four
new boundaries.

Unit 002's `_GLYPH_DIGITS` constant already had an exact canonical shape edge,
but `record_proven_shape` duplicated it through an undeclared raw page. The
proven-shape page is now declared, and publications, revisions and invalidations
derive through `IdentityBook.post` from the exact shape-state cell.

Unit 045 already recorded exact loop-continuation rewire receipts, but discarded
their references before invalidating dependent shapes. Those cells now flow
through the existing invalidation path. Scalar identities without non-empty
shape proofs no longer manufacture withdrawal rows; real prior shape proofs are
invalidated causally.

Unit 028's authored `node: Any` boundary was left dtype-less because scalar
annotation lowering only recognized numerical primitives. Authored `Any` and
`object` now use the repository's concrete arbitrary-object transport,
`opaque_ref`, and finalization recognizes that concrete ABI as resolved.

Unit 043 contained three distinct identity losses. A resident row load failed
to claim the earlier provisional definition, a tuple stream publication ignored
its exact `aggregate_leaf_value_ids`, and an untyped projected Python-object
column had no concrete transport. Row loads now claim the provisional identity,
stream publication carries and consumes the retained payload member IDs, and
non-numerical object columns use `opaque_ref`.

### Fourth fresh run: rehydrated glyph references

Toolchain fingerprint
`935b795257059037bab78a5bd21a154c249705368d6129a43531604f87a30a76`
confirmed the preceding four repairs. Units 002, 028 and 043 were
`compiled-unverified`; unit 045 passed source normalization and reached SSA
lowering. The run admitted 12 of 296 fail-fast units, reported no partials, and
left two SSA failures. Compilation units completed in 3m00s at about 1.59 GB
RSS with a 3.32 GB stage peak.

Unit 038 retained the exact `string_glyph_vocabulary` reference across its
control-tile pickle round trip. Rehydration reconstructed an equivalent Page,
so the reference's Page instance was not Python-identical to the active
registry Page even though the complete canonical key and fact were present.
Validation now resolves the retained row through the active book and requires
full reference-key equality, in addition to the existing argument and
vocabulary-ID checks. No vocabulary check was relaxed.

Unit 045's failing value was `{page.name}`. ProgramABI already declares
`IdentityPage.name` as a string span, but `_replace_cells(page: Any, ...)`
discarded that authored receiver identity and classified the field as an
unsupported scalar. The parameter now carries its exact `IdentityPage` type
through a type-checking-only import, selecting the established resident
`string_token` and `glyph_token_lookup_copy` path.

The expanded test gate also exposed spill-lifecycle omissions for the newly
declared `proven_shape` page and fourteen existing lazily declared source,
glyph, constant and metadata pages. They are now conservatively classified at
their lowering or repair lifetime. This is correctness for memory release, not a test
exemption: every declared page is again represented in `PAGE_LIFECYCLE`.

## Verification

No whole bootstrap was run after the latest compiler-toolchain edits; the
frozen planned units cannot be replayed authoritatively. The earlier combined
focused gate was:

```text
python -m pytest tests/test_sequence_contract_concordance.py \
  tests/test_ir_sequence_tables.py \
  tests/test_abstract_tensor_topological_reducer.py::test_program_abi_propagates_nested_record_receiver_and_set_storage \
  tests/test_abstract_tensor_topological_reducer.py::test_bare_frozenset_field_annotation_retains_membership_storage \
  tests/test_native_reference_boundary.py \
  tests/test_scheduled_process_graph_dispatches.py \
  tests/transmogrifier/test_memory_graph_integer_allocation.py \
  tests/transmogrifier/test_memory_graph_add_node.py \
  tests/test_scoped_compilation_inputs.py::test_ssa_growth_composes_interiors_before_one_native_emission \
  -q --tb=short --show-capture=no -x

132 passed, 1 warning in 20.61s
```

The previous expanded gate added exact glyph-shape, UTF-8 loop invalidation,
dependency cone, opaque-reference boundary, provisional row-load, and aggregate
stream publication regressions. It passed 193 tests. The final gate additionally
ran the complete glyph-provenance and precompile-to-SSA modules, the rehydrated
vocabulary and typed receiver regressions, and the declared-page lifecycle
audit: **345 tests passed** with one warning in 51.72s.

`git diff --check` reports no whitespace errors.

The new memory tests prove that the adapter is not an `nx.DiGraph`, common
NetworkX algorithms accept it, canonical relabel and subgraph extraction keep
MG/G paired, physical allocations match topology, and a serialized graph
round-trips exact bytes/intervals and can grow after rehydration.

## Next authoritative observation

Run a fresh user-owned provenance bootstrap. Expected result: unit 038 accepts
the exact rehydrated vocabulary reference, and unit 045 materializes
`IdentityPage.name` through the resident string-token path. Any next blocking
receipt becomes the next compiler repair; do not infer whole-bootstrap
completion from this focused gate.
