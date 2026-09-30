# One api for the identity concordance (design, 2026-09-30)

Status: design for review. Nothing here is implemented. It rests on the six
census files in `docs/concordance_census/` (read them for evidence; this file
only decides). No compiler numberings appear anywhere: they are ephemeral.

## 0. The direction (user, 2026-09-30, verbatim intent)

Move all authority for writing the concordance to a single api that cannot be
used without either an assertion of spontaneous novel identity (which, as long
as there is an edge for the transform, can be re-identified by later stages if
topology changes) or by identifying from whence EXACTLY in the concordance the
fact came. That gives an actual causal flow graph, not a spine of definition
clusters.

Rule behind it: every value that has any known source gets an edge from that
source at the moment it is written. A fallback chosen from absence of
information is recorded as unresolved, never as a fact. A raise on
disagreement means an edge was missing; it is never the concordance's fault.

## 1. What the census found (numbers are from the six files)

- The substrate has no source slot. All eight writing primitives (`set`,
  `revise`, `concord`, `bind_alias`, `PageMapping` set/del/setdefault,
  `mint_scope`) accept a fact with no reference to where it came from. The
  only edge the book keeps is the shared clock: when, never from what.
- Write sites: ~78 in fortran_c_shell, 42 in deployment/loops, 50 in SSA
  lowering, 51 in the reducer, ~25 in ssa.py tables, 17 helpers in the book
  module. Edge-carrying (target + exact source + stage): 36 of ~265. Bare
  facts: ~120. The rest name a source id but not its row or stage.
- Minted value ids: ~100 sites (66 in the frame linker). Zero record the
  transform that produced the id. The `MINTED` flag says "novel"; nothing
  says "from what".
- Fallback recorded as fact: at least 45 sites (`"float64"`, `"unknown"`,
  `"static"`, `setdefault`, snapshot-as-phi-arm, descriptor-value-as-return
  version, argument storage minted from absence).
- Unaudited: the audit reads 32 pages; ~60 written pages have no finding.
  31 of the linker's 56 pages have no reader outside their writer at all.
- Roots are not on the book: the ingestion-to-canonical id map
  (`ssa_identity_tokens`), the reducer's field-state dicts
  (`attribute_value_nodes` / `attribute_effect_nodes`), `return_slot_values`
  / `return_record_field_states`, `planner_specializations`,
  `_dispatch_metadata_cache`, `HierarchyValueTable.correlations`, the
  control builder's `external_values` / name histories / aliases, and the
  linker's `result_storage_bindings` / `record_return_layouts`. Every later
  edge would end in one of these, so today no causal chain can reach a root.
- Worked example (`Metrics.hard_failure`, census 50): 42 facts written about
  one field, 2 with an edge, and none joining a field version to the version
  it derives from. The 2026-09-30 dt-system raise is exactly that: a version
  substituted for a descriptor value with no edge, over a phi whose branch
  write had already been lost through a private dict keyed by the wrong id.

## 2. The api

One function. Everything that writes the book calls it; nothing else can.

```
post(page, row, fact, *, stage, derived_from=(), minted=None, mode) -> Ref

    page          a REGISTERED page name (row shape declared once, below)
    row           tuple, scope-first, validated against the page's declared shape
    fact          the statement, or UNRESOLVED(reason)
    stage         the pass making the statement (a registered stage name)
    derived_from  tuple of Ref = (page, row, column): the exact source cells
    minted        Mint = (transform, operands: tuple[Ref, ...])
    mode          "concord" (must never change) | "revise" (may grow, with cause)
```

Admission: exactly one of `derived_from` (non-empty) or `minted` is present.
Anything else is refused at the call, not recorded.

- **DERIVED.** Writes the fact and, in the same clock tick, one edge row per
  source cell on the shared edge page `(target_page, target_row, source_page,
  source_row, source_column, stage)`, plus the reverse index keyed by the
  source cell (so "what was derived from this" is a `scope_rows` read, not a
  scan). The state fact stored under the row is `("resolved", fact,
  edge_ref)`. This is `record_shape_transformation`'s three-page shape,
  generalized to every page.
- **NOVEL.** Allocates the id (`compose(serial, MINTED | flags)`) and writes
  the mint edge `(row) -> ("mint", transform, operands)` in the same tick.
  `operands` are Refs, so the novel identity is joined to what it was made
  from; later stages can re-identify it if topology changes by following the
  edge. `GLOBAL_MONOTONIC_IDS.mint()` becomes unreachable except through
  here. `mint_scope` becomes a NOVEL post on `scope_registry`.
- **UNRESOLVED.** A caller that has only a default posts
  `fact=UNRESOLVED(reason)` with `derived_from` = whatever it did read. Readers
  treat it as absence. It is the ONLY legal way to record "I do not know".
  Invalidation is `UNRESOLVED("superseded")` derived from the changed source
  cell, so a withdrawal is itself an edge, never an erasure.
- **Revision with cause.** In `revise` mode a changed fact must derive from
  at least one cell that changed since the prior revision (the api checks the
  stamps). A revision with no changed source is refused. That is what turns
  the 2026-09-30 raise into a legal revision: the return-edge selection
  changes because the receipt row it derives from changed, and the edge says
  so.

Closed by this api (made non-public or deleted): `IdentityPage.set`,
`revise`, `concord`, `bind_alias`, `PageMapping.__setitem__` / `__delitem__`
/ `setdefault`, direct `IdentityPage(...)` construction, free
`IdentityBook.page(name)` creation, `current_identity_book()`'s detached sink
book (absence of a book raises), `GLOBAL_MONOTONIC_IDS.mint` outside the api.
`PageMapping` survives as a read view.

Page registry: a page is declared once with its row shape (element names and
types) and its fact shape; `post` validates both. The ~69 free-form page
names collapse into declared pages. Undeclared page name: refused.

Audit: two generic findings replace per-page checks. `unsourced-fact`: a
resolved fact on any page with neither an inbound edge nor a mint record.
`unsourced-identity`: a `MINTED` id with no mint record. The per-page checks
that exist keep working; they become redundant as pages migrate.

## 3. What changes in the failing case (census 50, section 3)

- The reducer's field write posts DERIVED(receiver row, field row, literal
  row); the conditional merge posts DERIVED(both arms, the test). The field
  state is a row keyed by (receiver, field), not a private dict keyed by
  whichever node id the writer happened to use.
- `lower_conditional` may not post a snapshot as a phi arm. An arm with no
  posted version is UNRESOLVED and stops the lowering with the missing row
  named. The lost `False` write becomes a refusal at its own site.
- `scalar_return_field_versions` posts its selection DERIVED(receipt row,
  version row) or UNRESOLVED(receipt row, reason). Its twelve silent
  `return fallback` exits become twelve reasons. The minted `Cast` is NOVEL
  with the selection as operand.
- `_concord_record_return_phi_inputs` sees a revision derived from a changed
  receipt cell and accepts it; the raise cannot occur because the "fallback"
  was never a fact.

## 4. Migration order (roots first; each step is one commit, one probe)

Every step: declare the pages, route the writers through `post`, delete the
shadow ledger or make it a read view, teach the two generic findings to see
the pages, and prove with a seconds-long probe plus the audit tool.

1. **The api and registry**, with `record_shape_transformation` re-expressed
   through it (it already has the shape). No behaviour change; the audit
   gains the two findings. Probe: existing shape probes + audit tool.
2. **Ingestion roots.** `ssa_identity_tokens` (ingestion-to-canonical map),
   `identity_table`, `class_definitions`, `function_parameter_annotations`,
   `value_kind`, `map_ir` objects. Without these no chain has a root.
3. **Reducer field state.** `attribute_value_nodes` / `attribute_effect_nodes`
   -> a field-state page keyed (receiver, field) with DERIVED merges;
   `return_slot_values` / `return_record_field_states` become DERIVED rows
   from it; `_set_operands` appends record their cause. Probe: census 50's
   field, `probe_annotated_scalar_parameter`.
4. **Planner structure.** `planner_specializations` (which lanes exist),
   `planner_tensor_descriptors`, `_dispatch_metadata_cache` (the executable
   set), `HierarchyValueTable.correlations` (global ids). Probe: the
   scalar-rule probes, the dt-system `rollback` fold.
5. **Control SSA builder.** `external_values`, name histories, aliases and
   loop rewrites, carried ports -> pages; `finish` derives `parameter_names`
   etc. from them; snapshot-as-arm refused. Probe: dt controller step lowered
   alone (census 50 path).
6. **Record materialization and return versions.** `record_return_layouts`,
   `scalar_return_field_versions` selections, minted Casts, per-field phis.
   Probe: the native dt-system lowering (the trigger).
7. **Frame linker.** `result_storage_bindings`, `frame_ledgers`, argument
   storage minted from absence, `record_storage_alias`, name fallbacks at
   the forwarding root. Probe: `probe_row_handle_record_parameter`, then
   the whole-program Woodshop compile.
8. **Book-backed tables.** `SSARecordTable.register`'s merge writes a
   supersession edge like the layout table already does; `_mint_table_owner`
   scopes join to functions by a row, not a label string.

Steps 2 and 3 are the ones that make every later edge reach a root. Steps
4 to 7 each retire one family of silent fallbacks. Nothing is migrated
before the user says which step to start.

## 5. Decisions held for the user

1. **Granularity of `derived_from`.** Cells `(page, row, column)` as
   proposed, or rows `(page, row)` with the column implied as "latest at
   post time"? Cells are exact and make the revision-with-cause check
   mechanical; rows are shorter and match how most readers think.
2. **Refuse or record.** When a writer is caught posting a bare fact during
   migration, should the api raise (stops the compile at the defect, the
   repo's stated rule) or record `unsourced` and let the audit report it
   (lets a long compile finish and lists every offender at once)? The
   proposal is raise, with a per-page allowlist that shrinks to zero.
3. **Stage vocabulary.** A closed registered set of stage names, or free
   strings as today? Closed makes stages joinable across pages.
4. **Where to start.** Step 1 alone first (api + registry + shape
   transformation re-expressed, zero behaviour change), or steps 1-3
   together so the first commit already gives the worked-example field a
   rooted chain?

## 6. Decided (user, 2026-09-30)

1. **Source reference = the full vector of the location.** `Ref = (page,
   row, column)`. Never a row with an implied column.
2. **Unsourced is a type, behind a latch.** A writer that cannot name a
   source posts `Unsourced(reason)`. While the book's latch is OPEN the post
   is admitted and recorded as unsourced, so the audit lists every offender;
   when the latch is CLOSED the post is refused. The latch closes once
   everything is fixed and stays closed. During migration every raw write
   through an old primitive is auto-tagged `Unsourced("raw primitive")`, so
   closing the latch is the proof that no writer bypasses the api.
3. **No strings for runtime decisions.** Pages, stages, transforms, modes,
   fact kinds and provenance kinds are declared objects (enums / frozen
   registry entries), never free strings. A page is referenced by its
   registry object; `book.page(str)` survives only as a read-side lookup
   into the registry and refuses undeclared names.
4. **Steps 1-3 are worked together:** the api + registry + latch, the
   ingestion roots, and the reducer field state land as one movement, so the
   first commit gives the worked-example field a rooted chain.

### 6.1 Types

```
class Mode(Enum): CONCORD, REVISE
class Latch(Enum): OPEN, CLOSED

@dataclass(frozen=True) class Page:      name, row_fields: tuple[RowField, ...], fact_type
@dataclass(frozen=True) class RowField:  name, kind  (kind in a small Enum: SCOPE, VALUE_ID, NAME, INDEX, LABEL, PAGE_REF)
@dataclass(frozen=True) class Stage:     name        (registered once, referenced by object)
@dataclass(frozen=True) class Transform: name, arity (registered once; what a NOVEL post did to its operands)

@dataclass(frozen=True) class Ref:       page: Page, row: tuple, column: int

# provenance: exactly one kind per post
@dataclass(frozen=True) class Derived:   cells: tuple[Ref, ...]          # non-empty
@dataclass(frozen=True) class Novel:     transform: Transform, operands: tuple[Ref, ...]
@dataclass(frozen=True) class Unsourced: reason: Reason                  # latched

# facts
@dataclass(frozen=True) class Unresolved: reason: Reason, read: tuple[Ref, ...]   # "I looked and could not decide"
```

`Reason` is itself a registered object, not a string.

### 6.2 The one call

```
Concordance.post(page: Page, row: tuple, fact, *, stage: Stage,
                 provenance: Derived | Novel | Unsourced,
                 mode: Mode) -> Ref
```

- Validates `row` against `page.row_fields` and `fact` against
  `page.fact_type` (an `Unresolved` is always admissible).
- `Derived`: writes the fact cell, then in the same clock tick one edge row
  per source cell on the private edge page (target ref, source ref, stage)
  and the reverse index keyed by source ref. Returns the target Ref.
- `Novel`: `row` carries the sentinel `NEW` where the id goes; `post` mints
  the id (`compose(serial, MINTED)`), substitutes it, writes the fact and
  the mint edge (target ref -> transform, operands) in the same tick, and
  returns the Ref (the minted id is `ref.row[...]`). `GLOBAL_MONOTONIC_IDS`
  is not reachable any other way once migration completes.
- `Unsourced`: admitted iff `book.latch is Latch.OPEN`; recorded on the
  unsourced page with its reason and the caller's stage; refused otherwise.
- `Mode.REVISE`: admitted iff at least one `Derived` cell has a stamp newer
  than the row's previous revision (a changed source). Otherwise refused.
  `Mode.CONCORD`: a different fact for an existing row is refused (as
  today's `concord`), a `Derived` post with the same fact is a no-op that
  still records its edge.
- Every post ticks the shared clock exactly once, so cross-page order is
  recorded, as today.

### 6.3 Audit

Two generic findings over every registered page: `unsourced-fact` (a
resolved cell with neither an inbound edge nor a mint edge; while the latch
is OPEN the auto-tagged raw writes are listed here by page and stage, which
is the migration worklist) and `unsourced-identity` (a `MINTED` id in any
function with no mint edge). Existing per-page findings stay.

### 6.4 Steps 1-3, concretely

Declared vocabulary for steps 2-3 lives in `src/compiler/concordance_declarations.py`
(all pages, stages, transforms, reasons and fact types as objects; plans 60 and 70
are the specs).

Step 1 (`src/compiler/identity_concordance.py`, `src/compiler/monotonic_ids.py`):
the types above, the registry, `Concordance.post`, the latch (OPEN),
auto-tagging of raw `IdentityPage.set/revise/concord/bind_alias/PageMapping`
writes as `Unsourced(RAW_PRIMITIVE)`, `record_shape_transformation`
re-expressed through `post` (its three pages become the generic edge /
dependents / state pages), the two audit findings. Zero behaviour change.

Step 2 (ingestion roots, `graph_express2.py`, reducer canonical relabel):
`ssa_identity_tokens` (ingestion id -> canonical id) posted as DERIVED
`canonical_value` rows, one per relabelled node, from the `ingestion_value`
cell (corrected by plan 60: canonical ids are dense enumerate positions
consumed as dense, so they cannot be MINTED mints; the one NOVEL root per
source construct is the `source_span` row); `identity_table`,
`class_definitions`, `function_parameter_annotations`, `value_kind`,
`map_ir` object rows posted DERIVED from their AST-span rows (the span row
itself is the one NOVEL root per source construct). These pages are the
roots every later edge must reach.

Step 3 (reducer field state): `attribute_value_nodes` /
`attribute_effect_nodes` -> a field-state page keyed (receiver, field) whose
writes are DERIVED(receiver row, field row, value row) and whose conditional
merges are DERIVED(both arm states, test row); `return_slot_values` /
`return_record_field_states` become DERIVED rows from it; `_set_operands`
appends post their cause. The dicts become read views of the pages.

Proof for the movement: `probe_annotated_scalar_parameter` and
`probe_struct_intake` still pass; the audit tool's six cases report the same
findings plus the new `unsourced-fact` worklist; census 50's field has a
rooted DERIVED chain from its AST span to its return-site state rows.

## 7. Decided for steps 4-5 (user, 2026-09-30, plan 80's two questions)

1. **Every callee copy is its own variant.** Specialization records (the
   literal/default fold of `_propagate_callsite_planner_specializations`) are
   per copy, in the copy's own scope, always. A copy never carries a literal
   proven for another caller. How much variant inlining to do is a metric or
   contract choice made later; it costs almost nothing, and code that would be
   worse for the compiler ends up compiled as a shared function anyway. The
   fold itself is honest: `dt_system_over` omitted `rollback`, so under the
   contract it WAS the default; the retry lane became live when the source
   passed a state field (bc34adba), not because a fold was wrong.
2. **Falling up the scope ladder is correct; losing a write is the bug.** An
   arm that does not assign leaves the value at the enclosing scope's
   version, and the control builder taking the entered value there is right.
   The `hard_failure` defect was identity confusion (the write published
   under the SetAttr id, the arm looked up by the RHS id), fixed by keying
   arms on cells (steps 2-3). Rule for name-carried arms, identical to lane
   C's rule for fields: take the entered version when the book records no
   write in that arm (the arm cell is the entered cell); refuse with
   `Unresolved(ARM_VERSION_MISSING)` only when the book records a version for
   the arm and the builder cannot find it. Never refuse the scope ladder.
