# Continuation: porting proposal 02 (return-site identity) onto 854e145c (2026-10-03)

Source: `02-return-site-identity.patch` (author "dot", written against
3559a8fd, 29 commits behind). Applied with `git am --3way` in a detached
worktree at 854e145c; conflicts resolved by hand, the am'd commit amended.

## The key fact

The proposal fixes the same defect our head already fixed in two lanes
(`CONTINUATION_return_site_identity.md`, f7603483: return sites keyed by
site; `CONTINUATION_written_field_formal.md`, dbc59f0a: a written field's
formal is its incoming value). The proposal was written before both. Every
conflict is the same mechanism implemented twice. Our side keeps the
mechanism; the proposal's extra capabilities are re-expressed on it.

Measured first (baseline worktree at 854e145c + only the proposal's new test
file): 9 passed / 6 failed. Failing on our head: pickle republication,
the two audit-finding tests (the finding did not exist), the `for_loop`
variant (C compile error: `impl_..._step` called with 2 args, declared 1),
`static_branch`, and the selection-representation test.

## Conflict resolutions

- `control_source.py` (LoopControlBlock fields; post_control_program
  describe): OURS. Our describe lives in `_describe_control_block` (shared by
  writer and the read-only `control_block_cell`) and already owns a return
  by `return_site_cell`. Dropped the proposal's `return_source_span`,
  `return_field_state_cells` fields and its `authored_return_*` helpers:
  the helpers read the site from a private AST attribute
  (`_turing_return_site_cell`), which is a private key, not a book row. Our
  site comes from the reducer's `return_site_slot` / `return_site_field_state`
  rows through `_ReturnSiteView`'s cell->span join (`return_site_cell_for`).
  The proposal's `control_dependency_value_ids` extension (return field-state
  values) only feeds its OBSERVED-projection chain (below); dropped with it.
- `loop_composer.py` (descriptor shape, retarget, describe, body_items):
  OURS (4-tuple `(hint, chain, slots, (site cell, span))`, construct-position
  placement). KEPT the proposal's `evaporate_unrolled_loops` rule: a loop
  with `return_controls` is blocked from evaporation like one with
  break/continue sites. This is what fixes the `for_loop` variant.
- `glsl_deployment_strategy.py` (`arm_return_control`, overlay dedup): OURS
  (`return_site_cell_for`, `return_site_key`). Dropped the proposal's
  `_fold_callsite_structural_values` keep-set for return field-state
  GetAttrs: those GetAttrs exist only under its entry projection.
- `precompile_to_ssa.py` (edge stamps): OURS (stamp only when a site
  exists). Dropped its OBSERVED-version publication at the return edge
  (needs the entry projection). KEPT the `field_state_cell` attribute on the
  scalar field Store: a declared book cell naming the write's own state.
- `ssa_record_return_state.py`: OURS for the span->cell join and the lookup
  (site-keyed book path with `entered_version`, value-matched path only for
  edges that carry no site). KEPT the proposal's module-book wrapper:
  `publish_scalar_record_return_fields` runs under
  `begin_identity_book(identity_book(module))` and restores the ambient book
  (uses the `begin_identity_book(book)` parameter the proposal adds in
  identity_concordance). Fixes pickle republication. Dropped its rewrite of
  receipts to site keys (ours joins span<->cell on the book). Its selection
  improvements are RE-EXPRESSED on our site branch (next section).
- `topological_reducer.py` (auto-merged, REVERTED to ours): the AST private
  key; `initial_field_state` projection of every declared scalar parameter
  field at entry (overlaps WF's NOVEL formal for no-read fields); the relabel
  refresh of `field_state_cell` attributes (WF's `version_cells` follows the
  `canonical_relabel` edge instead).
- `fortran_c_shell.py` (auto-merged, REVERTED to ours): position by
  `return_source_span` (field not ported) and SSA_FIELD_VERSION following a
  record storage alias (only reached by the entry projection's GetAttrs).
- `identity_concordance.py`: KEPT `begin_identity_book(book=None)`. KEPT
  the `return-site-edge-disagreement` finding, RE-EXPRESSED: it reads the
  `return_site_slot` rows at the function's `record_return_state_scope`
  from the module's book (not the span-keyed metadata receipts).
- Tests: `test_pruned_loop_return.py`: ours + an assertion that the
  surviving control is the live site (slots and span); the proposal's
  version planted the private AST key. `test_ssa_record_return_state.py`:
  ours (the proposal's edits assume site-keyed metadata receipts and
  string sites). `test_record_return_site_identity.py`: kept; the
  selection test accepts our entered-version cell (the formal's
  `ssa_value` cell, checked to be a formal of the function) beside
  `ssa_field_version`; `static_branch` is a strict xfail, HELD (below).

## Selection re-expressed (ssa_record_return_state, site branch)

Measured on 854e145c (probe source, selection rows by predecessor): of six
live rows, `value@if_true.1` was Unresolved(version_not_const_or_carried_phi)
and `value@if_merge.1` Unresolved(version_not_uniquely_defined); native was
right only because the fallback (the in-place storage formal) held the
written value. After the port all six are sourced: four `ssa_field_version`
cells, two entered-version formal cells (sites with no state row).

1. OBSERVED site state with no published version: `observed_formal` finds
   the formal whose `ssa_value` cell derives, along posted book edges, from
   the state's value cell (the authored read). It is posted on
   `ssa_field_version` at the field-state cell, DERIVED(field-state cell,
   formal cell), mode CONCORD. The proposal matched the source-graph id
   against formal ids; that id join is replaced by the edge walk.
2. A published version with no defining instruction is a formal of the
   function (same SSA id space): defined at entry.
3. A version the book publishes for the exact state is admitted whatever
   instruction produced it; Const / carried-Phi admission only guards
   versions recovered without a published cell (the proposal's rule).
4. The intervening-write scan starts after the Store stamped with the
   version row's field-state cell (the proposal's rule; join by declared
   cell, not by `source_effect_node_id`).

## Held

- `static_branch`: `if False: m.hard_failure = True` is pruned, so
  `hard_failure` is neither read nor written and is not a record output --
  identical to a field never mentioned (measured: both lower to args
  `[rejected, value]`, outputs `[value]`). Making it an output is the
  proposal's entry-projection ABI change for every declared scalar
  parameter field. Not ported; needs a decision.
- `test_child_record_conditional_write_reaches_return` fails on 854e145c
  and after the port alike: the native child unpickles the module and
  `ArgumentBindingFact` (tuple subclass, `__new__(cls, kind, source)`, no
  `__getnewargs__`) raises TypeError. Pre-existing, not touched here.
- `while_exit` selection rows stay `Unresolved(predecessor_not_a_return_edge)`
  (open since lane RS).

## Results (2026-10-03, worktree, Windows, Python 3.11.7)

- Proposal batch: 58 passed, 1 xfailed (static_branch), 1 failed (the
  pre-existing unpickle above), 140.8 s. Same batch's new file at 854e145c
  before the port: 9 passed / 6 failed.
- `probe_record_return_merge.py --native 0 2.0`:
  `NATIVE {"hard_failure": false, "value": 0.5}` (CPython: False, 0.5).
- Selection rows (probe source): 6 live rows sourced (were 4 sourced, 2
  Unresolved at 854e145c); `while_exit` x2 Unresolved as before.
- Gates: collocation Jacobian `--compile` max rel 4.138e-16 (forward
  2.39e-15), 3 min 60 s; graph-reverse VJP 1 passed; native scalar loss
  adjoint 2 passed; process-graph linear motion 1 passed; orbital transfer
  compile 4 passed / 7 xfailed. (The Jacobian probe needs
  `PYTHONPATH=C:\dev\Powershell\engine_toy` from a worktree: it finds
  engine_toy as the repo's sibling.)
- Audit, all seven cases, 854e145c vs port: identical. view 493 rows / 24
  fns / 0; toplevel 322 / 31 / 1 (operand-never-written); energy 490 / 22 /
  0; controller 530 / 27 / 1 (operand-never-written); controller_untyped
  533 / 27 / 5 (operand-never-written); mapping 24 / 2 / 0; oscillator 331
  / 22 / 0. Unsourced facts / identities identical (4083/0, 2831/10,
  3800/10, 4546/1, 4431/1, 197/0, 2952/0). The new
  `return-site-edge-disagreement` finding fires nowhere.

## Bearing on the held name-arm alias fix (main tree, IndexedStore)

None directly. The proposal does not touch the indexed-assignment path,
`_branch_compartments` membership or IndexedStore. What it shares is the
defect class: a construct whose graph node is gone or never located must
still be identified and ordered. Both this head's return-site lane and the
proposal settle identity on a reducer cell rooted at the construct's
`source_span` row and use the span only to order (the proposal's
`return_source_span`, ours `_ReturnSiteView`'s cell->span join). That is a
precedent for the open "expr_obj vs source_span + effect_span_nodes"
decision on the side of a book-rooted span, not an answer to it. The one
precompile_to_ssa.py hunk here (the field Store's `field_state_cell`) is
far from `_carried_name_arm`; a cherry-pick should not conflict with the
held fix.
