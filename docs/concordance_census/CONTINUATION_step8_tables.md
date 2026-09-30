# Continuation: step 8 (book-backed tables), declaration phase

Lane G, 2026-09-30.  Plan `90_plan_steps6_8_records_linker_tables.md` section
4; inventory `75_raw_only_pages_inventory.md` sections 5 and 10 (step 8
DRAFT).  Declaration and plumbing only: no writer is routed, no caller
passes sources yet.  Nothing committed.

## What changed

### `src/compiler/concordance_declarations.py` (Steps 6-8 section only)

A "Step 8" sub-block after the `# (steps 6-7 declarations go here)` marker
(the marker is kept for the step 6/7 lanes).

Stages: `TABLE_REGISTRATION`, `SCOPE_MINT`, `DECLARATION`
(`_SSALayoutTable.register`'s default string today), `CTYPES_INTERCEPTION`
(`ctypes_layout.STAGE` today).
Transform: `TABLE_OWNER_SCOPE` (arity 1).
Reasons: `NO_FUNCTION_SCOPE`, `SEQUENCE_CLAIM_WITHOUT_PROPOSER` (plan 90),
`SEQUENCE_CLAIM_UNCHANGED` (new, see "Decisions").
Facts: `TableKind` (Enum: RECORD, SEQUENCE, STRUCT, UNION, CALL),
`RecordMergeFact(incumbent, incoming, merged, widened_fields, adopted_pool)`.

Pages, with the DRAFT block adopted as written unless noted:

| page | row | fact | note |
|---|---|---|---|
| `record_descriptor` | (owner SCOPE, record VALUE_ID) | object | draft |
| `record_member` | (owner SCOPE, member VALUE_ID) | tuple | draft |
| `sequence_descriptor` | (owner SCOPE, sequence VALUE_ID) | object | draft |
| `sequence_member` | (owner SCOPE, member VALUE_ID) | tuple | draft |
| `sequence_column_claims` | (owner SCOPE, sequence VALUE_ID, key LABEL) | tuple | draft |
| `call_record` | (owner SCOPE, caller NAME) | object | draft |
| `struct_descriptor` / `union_descriptor` | (owner SCOPE, id VALUE_ID) | object | draft |
| `struct_member` / `union_member` | (owner SCOPE, member VALUE_ID) | tuple | draft |
| `layout_state` | (owner SCOPE, kind NAME, row_id VALUE_ID) | tuple | draft |
| `layout_supersession` | (owner, kind NAME, target VALUE_ID, source VALUE_ID, stage NAME) | tuple | draft |
| `record_descriptor_merge` | (owner SCOPE, record VALUE_ID, revision INDEX) | `RecordMergeFact` | plan 90 1.2 (step-6 page, step-8 writer); declared here because item 3 posts it |
| `table_owner` | (owner SCOPE, function NAME) | `TableKind` | AMENDED from plan 90 4.2 (`(owner,)` -> `TableOwnerFact(kind, function)`): the lead's brief spells the function symbol as a row element and the kind as the fact |
| `sequence_row_layout_concordance` | (function SCOPE, sequence VALUE_ID) | tuple | draft ("nearest-8") |
| `sequence_row_dtype_concordance` | (scope SCOPE, sequence VALUE_ID) | tuple | draft ("nearest-8") |

Not declared from the draft: `sequence_contract_concordance` -- the step-5
lane already declares it (same shape) in its section; a second declaration
would be redundant.

No declaration had to be corrected after the audit: every observed writer's
row fits (owners are `(label, serial)` tuples, ids are ints, `call_record`
callers are `str(caller)`, `layout_state` kinds are `"struct"`/`"union"`,
the supersession stage is `str(stage)`).  Row validation only runs on
`post`, so declaration alone cannot refuse a raw writer; the shapes will be
exercised when the routing phase makes the writers post.

### `src/transmogrifier/ssa.py`

- `_table_post(book, page_name, row, fact, *, sources, stage, mode=None)`:
  the one sourced write path (`book.post(registry.page(name), row, fact,
  stage=TABLE_REGISTRATION if None, provenance=Derived(sources),
  mode=REVISE if None)`).
- `_BookRows.assign(key, value, *, sources=(), stage=None) -> Ref | None`,
  `_BookRows.remove(key, *, sources=(), stage=None)`; `__setitem__` /
  `__delitem__` delegate with no sources (raw `revise`, as before).
  `on_change(old, new, cell, stage)`: `cell` is the posted descriptor Ref or
  None.  The three table constructors' lambdas take the two extra arguments.
- `_revise_member_claims(page, owner, old, new, cell=None, stage=None)`:
  with a cell, each member row posts DERIVED(cell); else raw `revise`.
- `SSARecordTable.register(descriptor, *, sources=(), stage=None)`: no
  sources -> unchanged.  With sources: identical re-registration writes
  nothing; a plain registration posts the descriptor DERIVED(sources); a
  complementary-view merge posts `record_descriptor_merge`
  `(owner, id, revision)` CONCORD DERIVED(incumbent descriptor cell,
  *sources) with `widened_fields` / `adopted_pool` computed from the two
  views, then the merged descriptor DERIVED(merge cell).
- `SSASequenceTable.register(descriptor, *, sources=(), stage=None)`: no
  sources -> unchanged.  With sources: the column claim goes through
  `identity_concordance._post_or_unsourced(..., SEQUENCE_CLAIM_UNCHANGED)`
  (every attempt is still recorded), the descriptor posts DERIVED(sources)
  on first registration only.
- `_BookCallList._commit(records, *, sources=(), stage=None)`;
  `append(value, *, sources, stage)`, `replace(index, value, *, ...)`,
  `remove_at(index, *, ...)`; `__setitem__` / `__delitem__` delegate raw.
- `SSACallTable.assign(caller, records, *, sources=(), stage=None)` and
  `remove(caller, *, sources=(), stage=None)`; `__setitem__` /
  `__delitem__` delegate raw.

Not touched: `_mint_table_owner` (lane D owns scope minting), every caller
(no site passes sources), `_SSALayoutTable.register` / `_publish` /
`_record_supersession` (E8.2's layout edges are routing).

## Decisions taken here (the lead should confirm)

1. `SEQUENCE_CLAIM_UNCHANGED` reason: a sourced `register` that re-offers
   the same column typing from the same, unchanged cells is refused by the
   api as an uncaused REVISE; the table's contract is "every attempt is
   recorded", so it falls back to `Unsourced(SEQUENCE_CLAIM_UNCHANGED)`
   through the existing `_post_or_unsourced` helper.  Plan 90 names only
   `SEQUENCE_CLAIM_WITHOUT_PROPOSER` (for the no-sources path, which stays
   raw in this phase).
2. Sourced `register` of an identical descriptor writes no revision (today
   the raw path appends an identical revision).  Applies only when sources
   are given, so no current caller sees it.
3. `table_owner` row shape follows the lead's brief, not plan 90 4.2 (see
   table above).

## Verified

- `python -m py_compile` both files; `python -c` imports of
  `concordance_declarations` and `ssa` on the shared tree (with the other
  lanes' Step 4/5 edits present) and on the HEAD worktree.
- Line endings: both files CRLF only (2870 / 1028 CRLF, 0 lone LF); ASCII.
- Smoke (`scratchpad/smoke_step8_sources.py`, fresh book, no lowering):
  raw paths unchanged; sourced record register has one inbound edge from
  the source cell, member rows derive from the descriptor cell, identical
  re-registration adds no revision, the merge posts one
  `record_descriptor_merge` row (`widened_fields=("a",)`,
  `adopted_pool=False`) with edges from the incumbent cell and the source,
  and the merged descriptor derives from it (column +1); sequence claim and
  descriptor derive from the source, the repeated claim is tagged
  `SEQUENCE_CLAIM_UNCHANGED`; call-table `assign` / `append` / `replace` /
  `remove_at` / `remove` carry edges, a second edit from the same unchanged
  cell is refused by the api's REVISE rule, raw `append` still works;
  `new_layout_tables()` shares one owner.
- Probes on the shared tree: `probe_annotated_scalar_parameter` ok,
  `probe_struct_intake` ok (`failures: []`), `probe_record_in_tuple_return`
  ok (three ok lines), `probe_branch_written_field` ok on the second run
  (its first run failed on the step-5 lane's in-progress `fresh_value`
  signature, not on these edits).
- Audit, seven cases.  The shared tree could not be audited: every case
  fails in the step-5 lane's WIP (`fresh_value() missing 'transform'`;
  `identity_transition` fact type) before reaching any table page.  So the
  before/after was taken in a detached HEAD worktree at
  `C:\Users\alber\AppData\Local\Temp\wt8` (c67e9ed1) with ONLY this lane's
  two files applied (ssa.py copied; the step-8 block substituted for the
  placeholder).  First lines identical before and after:
  view 493/24/0, toplevel 322/31/1, energy 490/22/0, controller 528/27/1,
  controller_untyped 531/27/5, mapping 24/2/0, oscillator 331/22/0;
  `[unsourced-fact]` / `[unsourced-identity]` group counts identical
  (67/75, 53/132, 59/105, 57/141, 57/140, 25/15, 40/118).
  `unsourced:` fact totals before -> after: view 2909 -> 2928, toplevel
  5621 -> 5628, energy 3718 -> 3721, controller 5517 -> 5533,
  controller_untyped 5151 -> 5167, mapping 155 -> 155, oscillator
  3380 -> 3388; identities unchanged.  The rise is the worklist's unit
  change for a newly declared page (one "cell" per revision instead of one
  "raw row" per tagged row); no writer changed.
- Lead's measurement (`measure_completeness.py mapping controller`), in the
  worktree via a path-substituted copy (the script hardcodes the main
  tree).  BEFORE: mapping 425 cells, DERIVED 99 (23.3%), mint 36,
  unsourced 290, pages 100 declared 20 undeclared 80; controller 17834
  cells, DERIVED 9525 (53.4%), mint 428, unsourced 9755, pages 125
  declared 24 undeclared 101.  AFTER: not obtained -- the run was stopped
  externally twice (exit 137) before printing; expected unchanged DERIVED
  percentage and `undeclared` lower by the number of step-8 pages each case
  writes.  Re-run: `cd C:\Users\alber\AppData\Local\Temp\wt8 && python -u
  <scratchpad>\measure_completeness_wt8.py mapping controller`.

## Not done

- Routing (E8.1-E8.6): no caller passes sources; `_mint_table_owner` does
  not post `table_owner`; `_SSALayoutTable` first-declaration edge, `Stage`
  objects for its `stage` argument, `__deepcopy__` / `__setstate__` owner
  rows, the `fortran_c_shell` call-record edits, the audit's
  `record-merge-widened` finding, `--require-sourced`, `DEFAULT_LATCH`.
- The after-measurement (above).
- The shared-tree audit cannot run until the step-5 lane's tree compiles
  again; re-run all seven there once it does.

## Exact next edit (routing phase, E8.1 first)

`_mint_table_owner(book, label, *, function_scope: Ref | None = None)` in
`src/transmogrifier/ssa.py`: after `book.mint_scope(...)`, post
`TABLE_OWNER (scope, str(label)) -> TableKind.<kind>` with
`Derived((function_scope,))` under `SCOPE_MINT` when given, else
`Unresolved(NO_FUNCTION_SCOPE)`; the kind comes from the constructing table
(add a `kind` parameter; `new_layout_tables` / `IRModule._layout_owner`
pass STRUCT for the shared module owner).  Then E8.4's first caller:
`fortran_c_shell` `SSARecordTable(owner=symbol)` sites pass
`function_scope=` the function's `FUNCTION_SCOPE` cell once step 6 posts it.

## Working-tree state

Shared tree `C:\dev\Powershell\turing`: this lane modified only
`src/compiler/concordance_declarations.py` (Steps 6-8 section) and
`src/transmogrifier/ssa.py`; other lanes' modifications are present in
`topological_reducer.py`, `control_source.py`, `glsl_deployment_strategy.py`,
`hierarchical_plan.py`, `identity_concordance.py`, `loop_composer.py`,
`precompile_to_ssa.py`, `process_graph_function_linking.py`,
`view_identity_concordance.py`, the Step 4/5 sections of the declarations
file, and untracked `shots/`, `CONTINUATION_viewer_core_pin.md`,
`probe_planner_specialization_chain.py`.  Nothing committed.
Worktree `C:\Users\alber\AppData\Local\Temp\wt8` (detached at c67e9ed1)
holds HEAD plus this lane's two files; remove with
`git worktree remove --force C:/Users/alber/AppData/Local/Temp/wt8` when the
after-measurement has been taken.
