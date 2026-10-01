# Continuation: the function subgraph's own operand-position scope (lane SC)

Base: HEAD 8462a2a9 (uncommitted).  File edited:
`src/common/tensors/topological_reducer.py` only.

## Finding (from CONTINUATION_step9_graphs.md)

The mapping case's source ProcessGraph had 25 operand edges but 23 live
positions on the book.  The 2 were `Assign.type_comment` positions retired
by `function_subgraph_filter`: the extracted function subgraph
(`copy.copy(graph)` + `G.subgraph(included).copy()`) inherited the source
graph's metadata, so `_operand_position_scope` resolved to the build scope
(`ingestion_value_scope`) and the subgraph's `_set_operands` posted Retire
rows into the SOURCE graph's scope.  One operand table, two writers.

## Edit

- `fork_operand_position_scope(graph, members, cause)` (beside
  `fork_read_scope`): mints `"<source label>|operands"` through
  `book.mint_scope` (stage FUNCTION_SUBGRAPH); copies every
  `identity_transition`, `lexical_read_binding`, `consumer_operand` row of
  the source scope whose consumer is a member, each posted DERIVED from the
  cell it copies (raw primitive only for a fact its declared page would
  refuse, as `fork_read_scope` does); posts the fork's origin on the
  declared `scope_origin` page, `ScopeFork(source, cause)` DERIVED from the
  source scope's `scope_registry` cell; sets the subgraph's
  `operand_position_scope` to the fork.
- Called right after the subgraph is cut, before the `PARAMETER_INPUT` and
  `FUNCTION_SUBGRAPH_FILTER` rewrites, so both post into the fork.
- `_normalize_lexical_values` records the scope the graph arrived with
  (`entry_operand_scope`) before switching to the ingestion read scope; the
  canonical relabel continues THAT scope's `identity_transition` rows (the
  fork), falling back to the build scope when the graph arrived without one.

Readers checked: `node_identity_cell` (fork has no `ingestion_value` rows;
falls through to `ingestion_value_scope`), `_operand_position_scope`, the
relabel tail, `_relabel_field_state_pages` (keyed on the ingestion read
scope, untouched).

## Verified

Scratch `sc_subgraph_scope.py`, mapping case:

- source graph, build scope: edges 25, live positions 23 -> 25; graph-only
  2 -> 0, book-only 0.
- subgraph `root` at normalization entry, fork scope: edges 12, live 12
  (before: 23 live in the shared scope, 11 book-only).
- subgraph canonical, read scope: edges 5, live 5, one graph-only
  `(n, 'operand', 0)` and one book-only `elts` Move -- unchanged from
  before; a separate drift (remove_node Move vs a later operand), not this
  lane.

Probes struct_intake, branch_written_field, scalar_write_only_arm,
record_in_tuple_return, row_handle_record_parameter,
planner_specialization_chain, control_binding_chain: all ok.  Audit first
lines and unsourced counts identical before/after (view 0 / 4105,
toplevel 1 / 2829, energy 0 / 3800, controller 1 / 4536,
controller_untyped 5 / 4427, mapping 0 / 197, oscillator 0 / 2951).

## Next

The canonical subgraph's 1/1 drift above; `operand-view-drift` in
`identity_concordance.py` can now be written against graphs whose scopes
are their own.
