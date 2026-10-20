# Provenance sweep continuation — 2026-10-19 (second report)

Follows `20_provenance_bootstrap_continuation_2026_10_19.md`. Canonical
repository `C:/dev/Powershell/turing`, branch `main`, base `2b94890e`. This
report accompanies an UNCOMMITTED working tree of 16 files. No bootstrap and
no native build were run. Every claim below is labelled observed (seen in a
probe or a test) or inferred.

## Method

Five read-only investigators traced one blocker each from the checkpoint
report; a conference mapped their findings to shared causes; focused
correctors then edited disjoint files, each with a seconds-long repro, each
verified by its repro and one targeted test file. A second round traced the
links that the first round exposed. A clean pre-checkpoint worktree sits at
`C:/Users/alber/AppData/Local/Temp/wtb3` (7c560b04) for baselining; it was
used read-only.

## Shared cause found by the conference

Three of the five checkpoint blockers had one root: the reducer wrote
`function_output_shortfall` when return sites disagreed in arity and declared
no `function_outputs`. Nothing read that shortfall. `IdentityPage._resolved`
(arities `[1, 2]`), `IdentityBook._virtual_edges_into` (arities `[0, 1]`,
empty return paths) and the four unexplained inputs of unit 034 (projections
of union-returning calls with no result bindings) all descend from it.

## Corrections in this tree (observed unless noted)

1. **Units 041/065 — adaptation loop.** `ssa_call_input_adapters.py`
   `_concord_exact_region_feed_dtypes`: the restore branch put the feed-id
   resident back over an operand that was already its recorded conversion
   Cast (uncounted), and the adapter loop re-minted the Cast (counted) every
   round. The signature was right; two writers alternated. The restore now
   skips an operand that the `CALL_INPUT_CONVERSION` row and the Cast's own
   `call_input_conversion` accounting name as the conversion of that resident
   AND that already carries the exact dtype; a stale Cast is still restored.
   Unit 065 reruns in ~8 s under a hook with no refusal (the frozen plan's
   toolchain pin had to be stubbed in scratch because the edit changes the
   pinned hash). A tiny probe (`prog2.py`) that hit the same gate now passes
   it and stops at a separate, later fault (`undefined_operands=1` on a Call
   in `loop_exit`), not traced. Unit 041 was not run (needs linked units 3
   and 40).

2. **Finding 2 — region purity.** `precompile_to_ssa.py`: the scheduler's
   `pure_region_indices` ignored `table_region_operations` /
   `table_region_post_operations` while the builder's `_pure_region_index`
   treated them as barriers. One predicate now (`_region_table_operations`),
   and `sequence_accesses` yields the arena reads/writes of a region's
   recorded operations by their own `sequence_id`, so the existing
   RAW/WAR/WAW edges order them. Probe: region 0 carrying six operations is
   no longer pure.

3. **Finding 3 — trailing in-place pack.** The branch with no following
   numerical region did `continue` and recorded nothing. It now records the
   `pack_bits` into `table_epilogue_operations` with `callee_reference` and
   `callsite_id`. UNKNOWN: a trailing pack inside a loop body with no later
   region anywhere lands in the function epilogue, not at the loop boundary;
   the linker match was not watched (the injected probe stops at an
   injection artifact).

4. **Finding 4.** The unit receipt's `unexplained-public-arguments` shortfall
   now carries `"details"` from the existing
   `_unexplained_root_argument_details`.

5. **Fast-fail default.** `bootstrap_provenance_core` now defaults
   `stop_after_failure=True`, matching `ensure_native_provenance_core` and
   the CLI. `tests/test_whole_program_compiler_bootstrap.py`: 12 passed.

6. **Nine audit-contract failures classified and worked.** All nine fail
   identically at 7c560b04: five raw `name_binding.set` fixtures from
   c0127eeb, four declaration refusals from de5c10e4 (both 2026-10-17, both
   in the baseline). The five fixtures now post through `IdentityBook.post`
   (`Novel(INGEST_SOURCE, ())` roots, `Derived` capture) and pass. The four
   declaration refusals were the first link of a chain of real compiler
   faults that the audit tool's raw-string fixtures had masked. The fixtures
   now bind `Metrics`/`Targets`/`STController` via `python_bindings` (the
   real class-ingestion route). Links fixed, in order:
   - `frame_identity_book._record_field_declaration_cell` compared the bare
     `row[1]` to a qualified contract identity; explicit contract records
     override inferred schemas, so no contract record carries
     `source_record_view_schema`. The row's `(module, class)` pair is now
     joined to `module.Class`.
   - `identity_concordance.publish_program_abi_graph_identities`: the
     ProgramABI span-field `getattr` shape edge posted `Derived(())`. It now
     derives from the field's `class_field_declaration` cell and the owner's
     identity cell.
   - `item_operand` was an undeclared page written raw by
     `glsl_deployment_strategy._concord_item_operands`. Declared
     (`ITEM_OPERAND`) and posted DERIVED like its siblings.
   - `fortran_c_shell._class_surface_ssa_program` field-read loop replaced a
     value's contract and dropped the field's declared `float | None` arms;
     the carried `python_types` are kept.
   - One class had two keys: the reducer concords a parameter's class under
     the annotation spelling (`Metrics`) while the ProgramABI record states
     `module.Class`; the cross-module join failed because the class is
     declared in its defining module. `_annotated_parameter_record_identity`
     now joins parameter binding → record identity → source-record-class row
     → span → class declaration (spelling equality is a consistency check
     only). A second hunk substitutes the join result at the
     SOURCE_VALUE_CLASS admission; whether it is still needed after the join
     is UNKNOWN.
   - `assignment_projection_identity_concordance` was written raw in
     `_class_surface_ssa_program`: ANY `a, b = f(n)` unpack of a call result
     was refused, on the baseline too. Declared
     (`ASSIGNMENT_PROJECTION_IDENTITY`) and posted DERIVED from the
     projection-source, call-result-projection, call-node and projection-node
     cells. `plain_unpack.py` lowers.

7. **Return arms (the sweeping change), stages 1–4.** Gated on a record that
   exists only when return arities differ; single-arity functions take the
   unchanged path.
   - Stage 1: `_ReturnSiteView` yields `()` for a `return ()` site;
     `arm_return_control` and `loop_composer` take the site hint from the
     arm's own `return_site_cell`, so a zero-length arm owns its return edge.
   - Stage 2: `concordance_declarations.py` declares `ReturnArmsKind`,
     `ReturnArm`, `ReturnOutputArms`, page `RETURN_OUTPUT_ARMS`
     `(reduction_scope, output)`, reason `RETURN_ARMS_UNCLASSIFIED`;
     registered with the return-site pages and in
     `_relabel_field_state_pages`. The reducer (`post_return_output_arms`)
     classifies arms from the AST (`Constant(None)` = absent), sets
     `function_outputs=("result_0",)`, mints one output Phi and one member
     Phi per position, posts SOURCE_AGGREGATE_LAYOUT and the arms row DERIVED
     from arm container, slot and output cells. The shortfall writer is gone.
   - Stage 3: precompile lowers member Phi k AS the return-merge Phi for slot
     k (`SSAValue(member_cells[k])`, the join through the record, mirroring
     the conditional-join precedent); per-edge absence from the record;
     `ssa_optional_values` interleaves presence after each payload leaf in
     the output's layout and records per-position receipts in
     `optional_return_contracts`. VARIADIC posts the record and stops on one
     named shortfall `return_arms_variadic_unlowered` (the sequence lane,
     `SSASequenceDescriptor` + `ControlSequenceMutation`, is the real
     lowering; not built).
   - Stage 4: `_post_call_result_projections` runs in all lanes;
     `_accepted_semantic_result_bindings` joins CALL_RESULT_PROJECTION rows
     to layout position k (old equality still accepted); the caller side of
     `ssa_optional_values` builds N pairs for record-covered callees. Deviation:
     the accepted lane matches payload ids against the callee's per-position
     receipts, not SOURCE_AGGREGATE_LAYOUT, because the declaration carries
     no usable layout scope.
   - Observed on the repro (`while True` returning `None` or `(n, n+1)`):
     `semantic_output_ids=(16,)`, `aggregate_return_layouts=((16,(17,q,18,q)),)`,
     Ret width 4 with reciprocal `ssa_optional_*` accounting, 0 concordance
     findings; a caller doing `r = f(n); if r is None: ...; r[0] + r[1]` binds
     both projections and tests `LNot(presence0)`, 0 findings. New test
     `test_native_record_covered_optional_aggregate_binds_member_projections`.
   - NOT observed: native codegen or runtime parity for any of this; the
     real units 033/034/035 (needs a bootstrap).

## Validation (observed)

Six-file pre-commit gate: **217 passed, 4 failed** in 39.6 s (was 211/9).

```powershell
python -m pytest -q tests/test_precompile_to_ssa.py tests/test_value_union_abi.py tests/test_ssa_call_input_adapters.py tests/test_ssa_definition_dominance.py tests/test_hierarchical_plan.py tests/test_concordance_audit_contracts.py -p no:cacheprovider
```

Second batch plus `tests/test_project_compilation_product.py`: **126 passed,
2 skipped, 1 failed** in 182 s. The one failure,
`test_finished_ssa_artifact_releases_unpickleable_frontend_graphs`
(`function_address: no source cell for 'package.leaf'`), fails identically
at 7c560b04 and is not from this tree.

## Remaining frontier (the four audit-contract failures)

These are the next links, each observed with its message; none was chased:

- `[True]` controller: `precompile_to_ssa.py` ~11355, union Phi
  `floor = dt_min if dt_min is not None else 1e-30`: arm 13 (the optional
  field read) now has `VALUE_TYPE_ABI (NoneType, float)` but no tag SSA
  value. The field's `program_abi_optional_presence` Input IS its tag; no
  producer derives it yet.
- `[False]` controller: `fortran_c_shell.py` ~20588, `pi_update` parameter
  `dt_pen` has no authored native reference ABI. Under the bound-class route
  an undeclared isinstance subject on a pursued method is refused by design;
  the no-root-types variant needs `pi_update` value rows in the fixture
  contract or is no longer a distinct assertion. Decision for the user.
- `[energy]` and `[toplevel]`: `fortran_c_shell.py` ~46851, 3 incompatible
  physical call inputs (values 53, 47, 50 float64 in the specialized
  `_energy_time_limit` vs bool in `planned_region_7/10`).

## Raw page writers still in the compiler (inventory, read-only)

All are refused unconditionally by `_admit_raw_write` when reached.

| Site | Page | Declared |
|---|---|---|
| `fortran_c_shell.py` ~50807 `.concord` | `phi_edge_projection_placement_concordance` | yes |
| `fortran_c_shell.py` ~52476, ~52689 `.concord` | `static_record_piece_mapping/sequence_concordance` | no |
| `fortran_c_shell.py` ~25059 `.set` (inside try/except pass) | `member_formals` | yes |
| `frame_storage_roles.py:269` `.concord` (used as predicate) | `entry_record_handle_concordance` | yes |
| `glsl_deployment_strategy.py` ~1314 `.concord` | `operand_position_orphan` | no |
| `glsl_deployment_strategy.py` ~1348/1427/1493 fallbacks | consumer/item/call_argument_operand | yes |
| `loop_composer.py` ~459, ~1979 fallbacks | loop_carried_binding, loop_result_port_binding | yes |
| `planner_concordance_posts.py:316` `.set` fallback | `source_callsite_activation_concordance` | yes |
| `precompile_to_ssa.py` ~16316 `.set` | `tensor_shape_concordance` | yes |
| `src/transmogrifier/ssa.py:1630` `.set` | `layout_supersession` | yes |

## Other open items

- Design items not built: variadic return lane (stage 5); the independent
  conditional-count record (`source_conditional_count` is still written from
  the lowered count at fortran_c_shell ~25249, so the `==` accounting cannot
  catch a dropped If); audit checks for the arms record; a per-parameter
  `contract_demand` → `source_record_class_concordance` edge at
  `graph_express2.py` ~5992 (the join above walks the fact payload).
- Unit 034 shortfall 1 (`Registry.pages` keyed `.get`): `table_sequence`
  consults `field_aggregate_kinds` by the owner's own field name; the nested
  record's keyed descriptor is not declared/bound to value 23. Not fixed.
- `prog2.py` `undefined_operands=1` after the adaptation gate. Not traced.
- The frozen plan under fingerprint `9a2048…` pins toolchain hashes that this
  tree changes; a rerun needs a fresh plan (the normal bootstrap command).

## Prompt history

> I want you to use agents, audits, focused corrections of each blocker, and occasional conference between these dispersed goals to attempt to build the most sweeping corrective measures possible while respecting the architecture and timing as working in many cases, so worth considering before changing

## Addendum: first bootstrap on 64bed83a (user run, fingerprint 0e06287c...)

Observed: admitted=10/330, compiled-unverified=8, failed=2 (stop-after-failure).

- **Unit 032 `IdentityPage._reindex`** failed with a control-effect cycle:
  the ConditionalBlock sits at its `If` statement's authored position and
  now carries its body's `column_positions[column] = ...` store (arena 53),
  while the membership query that is its predicate sits at the later
  position of the `in` expression; the arena RAW edge opposed the value
  edge. Fixed in `_schedule_loop_callsites`: an arena edge is oriented by
  the exact value/hierarchy dependencies when they already prove the
  earlier-listed block executes later. Unit 032 compiles again
  (compiled-unverified). Preferable follow-up: attribute a nested region's
  accesses at the region's own authored position, as mutation blocks are.
- **Unit 042 `identity_spill._detached_ordinals`** (`if not detached:
  return []` then `return sorted(...)`): OBSERVED on the old compiler the
  guard If was silently dropped (no CondBr, `return []` gone), so
  "compiled-unverified" was wrong. The return-arms change makes it loud.
  Two findings: (a) the tail return edge (function ending in `return`)
  carried no `return_site_cell`, so `finish` dropped the arms record for
  every mixed-arity function ending in a return; fixed by a declared
  `ReturnArm.tail` the reducer posts from `statement.body[-1]` and the
  fallthrough terminator now carries that site cell. (b) the reducer
  flattens `ast.List` like `ast.Tuple` into slots (topological_reducer
  ~12131, ~5322), so `return []` is arity 0 and the function is VARIADIC;
  existing programs (vehicle tick `return [t1, t2, ...]`) rely on the list
  aggregate reading. When a list display at a return site is a SEQUENCE
  VALUE (one slot) versus an aggregate is an undeclared rule: a decision
  for the user (the `-> list` return annotation is the declared candidate).
  Until then unit 042 stops on `return_arms_variadic_unlowered`.

## Addendum 2: census, the declared-sequence return rule (commit after 1719fd10)

Second user run on 1719fd10 (fingerprint c0c303d9...): admitted=10,
compiled-unverified=9, failed=1 (unit 042 on `return_arms_variadic_unlowered`).
The bootstrap pruned the older fingerprints, so no unit-by-unit comparison
with the 28-unit run exists any more.

Census over the 348 functions of the bootstrap program (AST replica of the
reducer's return rule; scripts t2-t5 in the session scratchpad):
single arity 233 (untouched by the arms lane); mixed 39 = 3 OPTIONAL_AGGREGATE
(`_resolved`, `proven_shape_contract_of`, `shape_transformation_state`) +
36 VARIADIC. Of the 36: 27 annotated `list[...]` returning `[]` on one arm
(25 are `CorrelationTable._*_findings`), 7 annotated `tuple[T, ...]`
returning `()` on one arm (incl. `_virtual_edges_into`), 2 undeclared
(`shape_scope_of`: `Any`, spill `reduce`: none). 36 of 39 end in a tail
return. Effects inside control flow: 109 functions, 45 of which also read
the same container (the unit 032 shape).

Decided and implemented (user): when the DECLARED return annotation is a
sequence type (list / list[...] / List / Sequence / top-level
`tuple[T, ...]`, outer `| None` tolerated), a Tuple/List display at a
return site is ONE sequence value (arity 1), not flattened into slots.
Measured blast radius: 0 of the 233 single-arity functions change; 34 of
36 VARIADIC become single-arity. Implementation:
`topological_reducer._declared_sequence_return` /
`_return_display_flattens`, consulted at the four sites that flattened
displays (return_expression_lists, arm shapes, Return reduction, container
kind). `shape_transformation_state` now declares its fixed width.

Two faults found and fixed on the way:
- `_EllipsisExpander` (node_special_cases.py) rewrote `...` inside type
  annotations as tensor indexing, so `tuple[T, ...]` never reached the
  reducer intact. Annotation positions are now skipped.
- `loop_composer.materialize_retained_loop_ports`: when a loop's
  materializer (e.g. `sorted(<genexpr>)`) is superseded by its collection
  port, the function's `result_0` binding history kept the superseded id,
  so precompile saw two returned values for a one-slot function
  (`identity_spill._detached_ordinals`, unit 042). The history now moves
  to the port as the carried-binding path already did.

Observed: `_findings`-shaped and `return []` probes lower with the guard If
retained and the empty sequence value; fixed-tuple optional probes
unchanged. Known gap: a NON-EMPTY tuple display under a `tuple[T, ...]`
annotation (`return ((x, x + 1),)`, the `_virtual_edges_into` shape) still
becomes a structural aggregate with leaves, so the `()` arm cannot supply
them; affects 2 functions; needs the tuple-display-as-sequence-literal
lowering. Unit 042's growth pickle carries the old arms rows, so only the
next bootstrap (a fresh plan) shows the rule on real units.

Pre-existing failures classified against clean worktrees (fail at the
checkpoint 2b94890e, pass at 7c560b04; not from this session):
`test_abstract_tensor_topological_reducer.py::test_classmethod_receiver_uses_concorded_static_class_identity`
and four `test_ellipsis_store_ingestion.py` variants ([3],[5] x 2).
Gate on this tree: six files + reducer + ast ingestion = 313 passed,
5 failed (the four audit fixtures and the classmethod test).

## Addendum 3: sequence return merge; the dropped sort (commit after 11fb3fab)

User run on 11fb3fab (fingerprint a609d2a3...): admitted=14,
compiled-unverified=13, partial=1, failed=0. Unit 042 passed SSA lowering
and was partial at finalization with one shortfall,
`unresolved-boundary-types` on its single published output.

Traced: the output is the return-merge Phi of two sequence arenas (`[]`
and the `sorted(...)` result). Finalization accepts a value only if it owns
a sequence descriptor (a key of `module.sequence_tables`); the
`conditional_result` path (single-exit folding) already gives its merge
one, the `return_merge` path minted a bare Phi. Fixed in precompile
(`_return_merge_sequence_destination`): when every arm of a return merge
is a sequence arena of one policy/width, the merge is minted as a
sequence descriptor through the existing `_sequence_descriptor` helper and
claimed as the Phi's data arena exactly as the conditional merge does; the
arms reach it through the existing `output_identity_aliases`. Observed:
`p1.py e8` predicate False, Ret width 1, both CondBr present; unit 042
compiles from the growth pickle with no shortfall; the optional-tuple
repros unchanged; gate 313 passed / 5 failed, same ids.

Not done: element dtype. The destination's column dtype stays 'unknown',
as it does for the single-exit precedent. The declared return annotation
is not on the book (ingestion posts only PARAMETER_ANNOTATION), and the
append lowering sets no element dtype from its argument; a return
annotation row would be the record to add, and the audit must learn it.

FOUND, PRE-EXISTING, SILENT: `sorted(<generator expression>)` loses its
sort everywhere (old compiler too, observed on 7c560b04). The reducer
materializes the generator into the `sorted` call's own result arena and
stamps the loop's append with the call node as effect id
(loop_composer ~5609 via `materializer_node_id`, reducer ~11404/11470);
`post_source_sort_contract` posts the sort with the same effect id;
`_install_lexical_sequence_mutations` (fortran_c_shell ~12580) dedups
mutations by effect id and discards the sort as already placed. One
identity, two effects. `sorted(x)` over a plain sequence keeps its sort.
Unit 042's compiled form therefore returns UNSORTED ordinals; its
`compiled-unverified` status is wrong code until this is fixed. Owner: the
writer that gives the loop's append the call node's identity (the
materialization is the loop's effect; the sort is the call's).

Pre-existing failures introduced by the checkpoint commit 2b94890e (pass
at 7c560b04): three `test_ir_sequence_tables.py` tests
(`..._keeps_tuple_slots_and_object_references`,
`test_bare_tuple_constructor_field_is_an_opaque_native_reference`,
`..._keeps_literal_vocabulary_and_optional_scalar_tag`), in addition to
the classmethod and ellipsis-store tests already listed.

## Addendum 4: sorted(<generator>) keeps its sort (commit after 380c80d7)

Two writers, one identity each:
- loop_composer ~5609: the materializing loop's append now carries the
  LOOP's own node as its effect identity (`loop.node_id`), not the sorted
  call's; precompile's native-collection check compares against the
  receipt's `loop_node_id`. The placement filter in
  `_install_lexical_sequence_mutations` therefore no longer discards the
  sort as "already placed".
- loop_composer, after `rewire_continuation(materializer -> collection)`:
  the SOURCE_SORT_CONTRACT row posted by the reducer for the sorted call
  is REVISED so destination and source are the LoopResult (the collected
  arena), Derived from the old contract cell and the port row. The sort
  mutation's `argument_value_ids` then name the collection, which the
  existing `dependency_signature` ordering treats as a value feed produced
  by the loop block, so the sort is placed after the loop by dependency,
  not by authored position. `sorted(real_sequence)` is unchanged.

Observed: `gen_sort` and `e8` lower with append in `loop_body`, then
`replace`+`sort` on the SAME arena in `loop_exit`; unit 042 from the
growth pickle holds `..._20_append`, `..._20_replace_20`, `..._20_sort`,
no shortfalls, `repository_ssa_complete: True`. Nine-file gate 359
passed / 8 failed, the known ids. Not observed: native runtime of the
in-place `replace` on identical storage (documented as valid by
`lower_sequence_replace`); needs the user's bootstrap.

Pre-existing (fail on clean 380c80d7): three
`test_symbolic_law_batch_materialization.py` tests (shape concordance
double/float64 refusal). `test_loop_composer.py` has no baseline here
(one failure seen at ~290 s, `formal_shape ('tick','state')` Derived names
no source cell; unclassified).

## Addendum 5: record-covered optional output at finalization (after 7d7b2c63)

User run on 7d7b2c63 (fingerprint 735f91dc...): cumulative 28/330,
compiled-unverified=27, partial=1, failed=0. Units 041 and 065 (the
original adaptation-loop blockers) and 042 are admitted. Blocker: unit 033
`IdentityPage._resolved`, `native-result-layout-mismatch`: authored paths
`((0,0),56), ((0,1),57)` vs published `56, q, 57, q`.

Cause: `_accepted_physical_output_paths` (project_compilation_product.py
~6574) admitted optional layouts only as two-element entries keyed by
payload id (the old per-Phi receipts); the record-covered output publishes
ONE entry keyed by the semantic output, `(p0, q0, p1, q1)`. Added the
derived view: when the function has one semantic output whose layout is
twice the authored leaf count and whose even positions are exactly the
authored leaves in order, each leaf's physical ABI is its (payload,
presence) pair at `(*path, component)`. Observed: unit 033 compiled from
the live growth with no shortfalls, `repository_ssa_complete: True`.
Gate (ten files): 457 passed / 9 failed, all nine pre-existing.

## Addendum 6: linking a record-covered optional callee (after 0118f18e)

User run on 0118f18e (fingerprint 61aaaf83...): cumulative 31/330, failed=2:
units 034 `_virtual_edges_into` and 035 `_virtual_fact`, both callers of
`_resolved`, at the linker: "linked callee lost an authored output from its
return surface" (3 bindings vs a four-wide Ret; semantic output 55 unmapped).

Cause: a separately compiled callee arrives with presence already
interleaved into its Ret by `emit_outputs` from `aggregate_return_layouts`;
the linker's ordered-window match then has no window, while an in-unit
callee is linked on its payload-only Ret and `ssa_optional_values`
delivers presence afterwards. Fix (fortran_c_shell ~41473): when the
callee carries `optional_return_contracts` whose interleaving equals
`aggregate_return_layouts[return_output_id]` and every leaf is on the Ret,
the linker links against the payload-only view (drops exactly those
presence ids from `callee_outputs`); the existing pass then mints the
caller's presence values as in-unit. Observed on unit 034: call
`output_ids=(35, q, 37, q)`, `is None` -> `LNot(presence0)`, projections
35/37 bound. Gate 457 passed / 9 failed, the known ids.

Remaining, by observation:
- Unit 034 is `partial` on its pre-sweep shortfalls: unresolved source
  value 43 (`Tuple` of `(origin, *ref.row[1:])`, the starred row) and
  unexplained public inputs 13 and 19 (`arg:1`, `arg:0` of callsite 21;
  the preceding `_fork_of_scope` call binds only `output_ids=(11,)`, so its
  projections 12/13 have no producer). These predate the sweep (report 20).
- Unit 035 uses the optional result WHOLE: `found is None` then
  `self.cells[found]` with the tuple as dict key; no leaf projection exists,
  one binding against two leaves, the refusal is correct. A whole-value
  consumer of an optional aggregate needs the layout leaves bound to caller
  values and a tuple-key lookup lowering: a design decision.
- UNKNOWN: scalar-optional callees (the `contracts[...]` lane) linked from
  another unit may need the same payload-only view.

## Addendum 7: whole-value optional aggregate, stages 1 and 3 (after da4187de)

Design (read-only, records only): a caller that consumes an optional
tuple result whole (unit 035 `_virtual_fact`: `found is None`, then
`self.cells[found]`) has no `Indexed` projections, so no
CALL_RESULT_PROJECTION rows and one binding against two leaves; the
linker's refusal is correct. The precedent for caller-side leaf values is
`_publish_callsite_return_members` (CALLSITE_RETURN_MEMBER ledger,
`aggregate_leaf_value_ids`), not the record lane.

Found on the way, both silent and independent of optionals:
- a tuple VALUE as a dict key (`key = (row, column); cells[key]`, the
  shape of `IdentityPage.cell`) lowered as a one-column lookup while the
  display form lowers two columns; units then disagree on `key_columns`.
  FIXED (stage 1, fortran_c_shell ~9731 `tuple_value_leaves`,
  `authored_index_values`, the `.get` collector): the index parent's
  `aggregate_leaf_value_ids` are the queries, as the membership lane
  already reads them. Observed: `keyc.py value` == `display`.
- a mixed consumer (`r[0]` projected, `cells[r]` whole) compiled with the
  raw call aggregate as a single query. FIXED (stage 3,
  ssa_optional_values ~78-114): the two silent early returns of
  `_lower_record_covered_call` are loud refusals naming the call, callee,
  receipt positions and the missing projection.
Gate 457 passed / 9 failed (known ids). Four keyed-lookup test files
(`test_sequence_contract_concordance`, `test_source_object_context`,
`test_ssa_object_context`, `test_ssa_owned_heap`) carry 10 failures that
are identical with stage 1 reverted: pre-existing, unclassified against
an older baseline.

Stage 2 (mint members for an optional-aggregate accepted header) is
BLOCKED on a cross-unit cell: the member rows must derive from the
callee's `source_aggregate_layout` cell `((lexical_reads:(...,
'IdentityPage._resolved'), 0), 55)`, posted in unit 033's book; the
linked header carries only its Ref (`aggregate_return_layout_sources`),
and the consuming book refuses a Ref whose cell it does not hold. Also:
the member-publication loop (glsl ~17422) skips callees without a source
graph, so a linked accepted header never reaches the gate; and for an
in-unit callee the gate's `call_result_descriptor` is empty, so the
member descriptors must come from the arms record's layout leaves. Open
decision: how a callee's layout/arms facts cross the unit boundary (the
plan carries a `source_concordance` archive; whether it should carry
these rows, and who imports them, was not traced).

## Addendum 8: whole-value optional aggregate, stage 2 (after 4f26ab89)

Transport question answered by reading: a linked callee's concordance
facts are READ from its accepted book
(`function.metadata["accepted_ssa_component"]["identity_book"]`, the
callee unit's own book, reached via `host_ssa_module`), and every
caller-side post derives from the caller's own cells (call node identity
cell, `call_binding` cell, else the caller root), as the record-return
lane's `_frame_mint` does. No cross-book derivation exists and none was
added; the plan's `source_concordance` archive has no reader.

Implemented (glsl_deployment_strategy.py): an OPTIONAL_AGGREGATE callee's
one semantic output has member leaves declared by RETURN_OUTPUT_ARMS +
SOURCE_AGGREGATE_LAYOUT; the caller gets one `Indexed(call, k)` member
per leaf through the existing `_publish_callsite_return_members`, all N
positions, presence not a member. Linked header (`callee is None`
branch): layout/arms read from the accepted book at the row
`aggregate_return_layout_sources` names, descriptors from
`optional_return_contracts`, rows derived from the caller's call cells.
In-unit callee: members from `_return_output_arms_members`, replacing the
single-kind proxy and the empty `call_result_descriptor` for that case.
Minted member/index nodes get node-identity rows via
`post_derived_node_identity` from the call cells.

Two further faults found and fixed on the way:
- the fixed-point digest omitted `aggregate_leaf_value_ids`, and
  `_tensor_descriptor_rule` rewrote the pair-nested physical ledger every
  round against the flat member ledger ("repeated a completed state");
  both writers now state the one record for an OPTIONAL_AGGREGATE header.
- `_graph_control_expression` (fortran_c_shell ~10942/11028) read an
  aggregate operand of `is None` / `is not None` as TRUTHINESS
  (`Eq(Gt(len(r), 0), None)`), so `ssa_optional_values` never found the
  aggregate to rewrite into `LNot(presence)`; an identity comparison now
  takes the aggregate as a value. This changes every aggregate compared
  with `is None`, not only optional calls.

Observed: unit 035 `compiled-unverified`, no shortfalls; `whole.py whole`
and `partial` equal the projection oracle (Call output_ids (p0,q,p1,q),
`LNot(q)` optional_presence_test, two-query lookup on key_columns (0,1));
0 concordance findings, latch CLOSED; unit 034 unchanged (its pre-sweep
shortfalls); gate 457 passed / 9 failed, known ids. Not observed: the
linked case's `found is None` lowering to `LNot(presence)` inside unit
035 (only the in-unit probes show it); the region ordinals shift
(`planned_region_6` -> `_4`), not traced. Stage 4 (optional aggregate
passed/returned/compared whole: `shape_transformation_state`, 4 sites)
remains a separate decision.

## Addendum 9: three consumer-side repairs redone at their generators (2026-10-20)

Rule (user, absolute): never fix generated content after the fact; fix the
generator. Three committed edits violated it and were reverted and redone:

1. Linker payload-only view (da4187de) REVERTED. The callee's Ret stays as
   generated (presence interleaved by `emit_outputs`); the linker's result
   window now binds payload positions by reading the callee's
   `optional_return_contracts` receipts (`callee_link_positions`), not by
   editing `callee_outputs`. In-unit callees (no receipts yet at link time)
   are unchanged. Chosen over moving the optional lowering before linking,
   which runs once over the whole module after all links.
2. Arena-edge orientation (1719fd10) REVERTED. The generated run order was
   wrong: a ConditionalBlock sorted at its `if` keyword, before the hoisted
   items of its own test. `_conditional_block_position` (fortran_c_shell
   ~11961) places a conditional where its TEST ENDS, at all five
   `block_position` sites; one lexical rule, test before body. Unit 032
   compiles with no orientation rule.
3. Sort-contract REVISE in the composer (7d7b2c63) REVERTED. The reducer
   posted SOURCE_SORT_CONTRACT before the composer generated the final
   source identity. Now: the reducer's `sorted(generator)` branch
   materializes the call's ARGUMENT (one identity before composition);
   `post_source_sort_contract` runs after loop stamps and DEFERS when the
   sort's source value is a loop materialization target (the
   `loop_iteration_outputs` record, not AST shape), leaving
   `source_sort_contract_pending` on the call; the composer posts the
   contract ONCE when it mints the collection port (dest=call,
   source=collection, Derived from the port row and the call cell,
   CONCORD). `sorted(d)` on a real sequence posts at the reducer as before.
   Observed on gen_sort, lc_sort, nb_sort (name-bound comprehension), e8:
   one post with final ids, append in loop_body, replace+sort in loop_exit.

Also committed before this: a0072423 declares fixed-arity tuple mapping
values as N value columns at the origin (reducer element schema,
`ResolvedSequenceSchema.value_column_count`); the keyed-field ABI's single
`values` part, the N-value lookup, and reducer-minted leaf identities for a
lookup result are still to be generated from it; until then
`dict[K, tuple[...]]` stops loudly at the lookup helper (unit 027).

Gate on this tree: 457 passed / 9 failed, the known ids. Units 032, 034
(partial on its pre-sweep shortfalls), 035, 042 verified from the live
growth. Not run: a bootstrap.
