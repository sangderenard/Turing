# Continuation: a return site is its construct, not its value (lane RS, 2026-10-01)

Repro: `python -u tools/compiler_probes/probe_record_return_merge.py`
(its docstring holds the two fault chains this lane worked from).

## Status

- Fault 1 (lost fall-through return, native spin): FIXED.  The finished
  function has three return edges for three authored returns; no case spins.
- Fault 2 (predecessor matched to sites by returned value): FIXED at the
  lookup.  Every selection row now reads its OWN site's cells.
- Probe still FAILS two native cases: `hard_failure` returns True where
  Python returns False (`rejected=False`, value 2.0 and 0.5).  Cause is a
  third link, outside this lane (below).
- Gate otherwise green: struct_intake, branch_written_field,
  scalar_write_only_arm, record_in_tuple_return, row_handle_record_parameter,
  planner_specialization_chain, control_binding_chain pass; native
  correctness 0 failures; audit first lines unchanged (view 0, toplevel 1,
  energy 0, controller 1, controller_untyped 5, mapping 0, oscillator 0).

## What changed

The site's identity is the reducer's return-site cell (the key of the
`return_site_slot` / `return_site_field_state` rows).  Once the construct's
nodes leave the graph, the only join between that cell and the span-keyed
receipts is the reducer's `_ReturnSiteView` span map.

1. `ssa_record_return_state.py`: `return_site_span`, `return_site_cells`,
   `return_site_cell_for` read that join (node attribute as a fallback).
2. `loop_composer.py`: `LoopDescriptor.return_controls` entries are
   `(position hint, chain, slots, (site cell, span))`, one per authored
   return.  `expanded_loop_body_nodes` records each Return construct's
   lexical position during the source-order walk and places the return
   there; the value node is only a hint (and `site_node_id` only when it is
   a body node).  The block carries `return_site_cell`.
3. `control_source.py`: `LoopControlBlock.return_site_cell`; a return
   block's control-program owner is that cell.
4. `precompile_to_ssa.py`: the return edge gets `return_site_cell` beside
   `return_source_value_ids` (both emission sites).
5. `glsl_deployment_strategy.py`: `arm_return_control` stamps the same
   cell; the overlay's duplicate-return removal keys by site cell (it keyed
   by `site_node_id`, i.e. the value, and would have removed the
   fall-through return as a copy of the arm returns).
6. `scalar_return_field_versions.lookup`: an edge carrying a site cell
   reads `(scope, site, receiver, field)` on `return_site_field_state`; the
   version is that state's value cell.  No row = the site wrote nothing:
   the entered version, selected only when the descriptor value is a
   formal, posted DERIVED(site slot cell, formal cell).  Edges without a
   cell keep the old value-matched path.

Out-of-list files touched (needed to thread the cell): control_source.py
(one field, one owner line), precompile_to_ssa.py (two stamps), glsl
overlay dedup (`return_site_key`).

## The open link (linker, not this lane)

Observed: when the selection runs, `hard_failure`'s formal is id 4, the
same id as the `True` literal written at site A; `_recover_late_source_literals`
(fortran_c_shell.py, `function.args = retained_args`) then turns that
formal into `Const True`.  Inferred (read, not hooked): in
`materialize_parameter_record_abi`, a scalar field with no getter has its
write sources outside `scalar_write_sources`, so the write value becomes
the field's candidate storage id.  The entered `hard_failure` therefore
never exists as its own formal, and every site without a write returns
the Const.  Fix belongs where the field's formal is minted: a written,
never-read scalar field needs a formal distinct from its write values.

Also open: the merge Phi still lists `while_exit` (post-loop fall-through,
unreachable) when the selection runs; its row is
`Unresolved(predecessor_not_a_return_edge)`.
