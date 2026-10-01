# Continuation: step 5, the control SSA builder (plan 80 part B)

Lane E, 2026-09-30.  Everything below names functions, never line numbers or
compiler ids.  Earlier edits of this lane were committed to main by another
session ("a lot of active work", "Return merge mints with its declared
transforms (lane E's two unwired sites)"); this note describes the tree after
the uncommitted follow-up in the working tree.

## What changed (files / functions)

`src/compiler/concordance_declarations.py`, Step 5 section only: stages
`CONTROL_SSA_ENTRY/CONDITIONAL/LOOP/REGION/FINISH`, `REDUCER_READ`; the
nineteen `fresh_value` transforms of plan 80 B1 (all arity 1 -- a mint made
from several cells first posts one `cell_set` row and names it; the lead's
N4 answer), plus `CONTROL_FUNCTION_ROOT` (arity 0, the one root row a
lowering posts for itself) and `FORK_READ_SCOPE`; the eight reasons of B1
plus `GRAPH_ID_WITHOUT_CANONICAL_CELL`; facts `SSAValueFact`,
`ControlBinding`/`BindingKind`, `AliasFact`/`AliasKind`,
`ParameterDeclaration`, `RegionSignature`, `OperandTransition` family,
`ScopeFork`; pages: the nine new ones of B1.1 plus `cell_set`;
`lexical_read_binding`, `identity_transition`, `scope_origin`,
`scalar_item_merge` (B1.3); the fifteen re-declared builder pages (B1.2);
the census-75 drafts for `ssa_call_input_adapters` / `ir_identities`
(declared only, writers still raw).

`src/compiler/precompile_to_ssa.py`:

- `_ControlSSABuilder`: `_scope`, `_function_root`, `_cell_set`,
  `_canonical_cell`, `_declared_cell`, `_cells`, `_value_cell`,
  `fresh_value(*, transform, operands)` (NOVEL on `ssa_value`; the id is
  the book's; `GLOBAL_MONOTONIC_IDS.mint()` is gone from the builder),
  `_revise_value`, `_binding_cell`, `_binding_fact`, `_binding`, `_bind`,
  `_bind_unresolved`, `_declare_parameter`, `_rebind_view`, `_withdraw`,
  `_restore_view`, `_post_carried_snapshots`, `_recorded_name_write`,
  `_carried_name_arm`, `_post_concord`, `_post_alias`,
  `_post_scalar_item_merge`, `_post_tensor_shape_contract`,
  `_sequence_contract_cell`, `_producing_region_cell`,
  `_bind_sequence_query_result`, `_read_binding_cells`, `_finish_pages`.
- Every `fresh_value` site names its transform and operands; every
  `external_values` write goes through `_bind` / `_bind_unresolved` /
  `_withdraw` / `_restore_view` / `_rebind_view` (the dict is the read view).
- `__init__` (uniforms, parameter seeds + `declared_parameter`, PLANNING
  alias rows, the book-numbered control scope when none is handed in),
  `external_value`, `produced_value(..., cause=)`, `constant_value(literal,
  *operands)`, `indexed_load`, `_note_callsite_arguments` (posts
  `callsite_argument`, try/except removed), `emit_plan_callsite`,
  `emit_region_call`, `_enter_loop_state` (posts `loop_carried_entry` /
  `loop_entry_state`; `CARRIED_ENTRY_NOT_ATTRIBUTED`), `_region_feed`
  (`scalar_item_merge`; `REGION_FEED_NO_OPERAND_ROW`),
  `lower_control_expression`, `_lower` (dispatch / external / field write
  binds), the sequence and table helpers, `lower_conditional`
  (`carried_snapshot` rows, the name-carried arm rule, PHI_CONDITIONAL,
  CONDITIONAL_MERGE / RESTORED binds), `_publish_loop_result_ports`
  (`carried_port_value` rows, LOOP_RESULT_PORT binds),
  `_bind_loop_result_ports_inside_body` / `_restore_loop_result_port_aliases`
  (`control_value_alias` LOOP_BODY_SPELLING / RESTORED), `lower_loop`,
  `lower_while` (`while_carried_test` posted DERIVED or
  `Unresolved(WHILE_TEST_NO_READ_EXPRESSION)`), `finish` (`function_parameter`
  / `function_output` posted; `parameter_names`, `named_outputs`,
  `carried_port_values`, `value_aliases` materialized from the pages with a
  byte-identity guard that keeps the computed value when a page cannot
  reproduce it).
- Module level: `_post_derived_or_raw`, `_post_region_signature`,
  `_function_root_cell`, `_mint_ssa_id`; `lower_control_sections_to_ssa`
  mints the control scope through `mint_scope` (`<name>@control:<serial>`,
  no process id), posts `control_uniform_dtype`, `region_value_dtype`,
  `region_signature`, `tensor_shape_concordance` (two sites) and the
  `control_value_concordance` alias site DERIVED; `_concord_region_feed_consumers`
  and `_split_region_captures_by_binding` post DERIVED, the split formal is a
  NOVEL `REGION_FORMAL_SPLIT`; `_inject_field_slot_access.fresh` and
  `lower_class_navigation_to_ssa.Builder` mint through the book; the fused
  lowering mints its scope and posts its region signatures;
  `_canonicalize_non_dominating_loop_result_uses._note` posts
  `loop_result_reconciliation` DERIVED when a port cell exists.

`src/common/tensors/topological_reducer.py`, `_set_operands` body only:
move / retire / fork post `identity_transition` NOVEL(OPERAND_MOVE / RETIRE /
FORK, (position cell,)) with `OperandMove` / `OperandRetire` / `OperandFork`
facts; the arriving `lexical_read_binding` follow-ups post DERIVED from the
transition cell; vacating `None` revisions stay raw (the declared page holds
`str`).

`tools/compiler_probes/probe_control_binding_chain.py`: new (plan 80 B8).

## The name-carried arm rule (decision 7.2), as implemented

`_carried_name_arm`: arm id equal to the initial id -> the snapshot (the
entered version), source = the `carried_snapshot` cell; arm id with a
binding in this lowering -> that binding; arm id with NO binding and NO
authored `name_binding` row naming it (`_recorded_name_write`) -> the arm
did not write, the snapshot, never refused; arm id with an authored
`name_binding` row and no binding -> `Unresolved(NAME_ARM_VERSION_MISSING)`
at the arm id plus the `carried-name-arm-missing` shortfall, the snapshot
standing in for the emitted Phi.  The earlier regression treated every
differing arm id as a recorded write; the record the rule now reads is the
reducer's `name_binding` page under the graph's read scope.

## Verified

- `probe_control_binding_chain`, `probe_struct_intake`,
  `probe_branch_written_field`, `probe_scalar_write_only_arm`,
  `probe_record_in_tuple_return`, `probe_planner_specialization_chain`: all
  checks ok.  `probe_scalar_native_correctness`: 14/14 equal to CPython.
- Completeness (lead's `measure_completeness.py`, before -> after):
  mapping DERIVED 23.3% -> 45.5%, tagged unsourced 68.2% -> 35.9%, MINTED
  ids with a mint record 0/15 -> 14/15; controller DERIVED 53.4% -> 71.4%,
  tagged unsourced 54.7% -> 32.8%, MINTED ids with a mint record 0/141 ->
  79/141.  (The after numbers include lanes D and F's committed work.)
- Audit (`tools/audit_identity_concordance.py`, seven cases) after this
  follow-up: first lines identical to the baseline (view 0, toplevel 1,
  energy 0, controller 1, controller_untyped 5, mapping 0, oscillator 0
  findings; same row and function counts).  `unsourced-identity` counts
  before -> after: view 75 -> 48, toplevel 132 -> 35, energy 105 -> 55,
  controller 141 -> 62, controller_untyped 140 -> 62, mapping 15 -> 1,
  oscillator 118 -> 2.  Unsourced fact counts moved both ways (view 2909 ->
  4101, toplevel 5621 -> 2775, controller 5517 -> 4530, mapping 155 -> 195,
  oscillator 3380 -> 2866): rows on newly declared pages that other lanes
  still write raw now count as cells, and lanes D and F's committed work is
  included.

## Not done from plan 80 part B (do not substitute; these are the gaps)

- `external_values`, `field_version_values`, `_carried_port_values`,
  `_carried_port_groups`, `control_identity_receipts`,
  `declared_parameter_only_ids` still exist as read views (F14 deletion
  waits for every reader to move to `_binding` / the pages).
- `value_names` is still computed from the dict (plan B2.6 calls it a join
  of two pages; no page was added).  `function_output` derives from the slot
  value's cells, not the `return_site_slot` cells (the return construct cell
  is not reachable from `function_return_edges`).
- `commit_sequence_contract`, `declare_loop_scope`, `rebind_loop_scope_inner`
  live in `identity_concordance.py` (not this lane's file) and still write
  raw; the builder keeps calling them (the loop-scope calls still wrapped in
  `try/except`).  `_set_operands`' vacating `None` revisions and the
  `consumer_operand` follow-ups stay raw; `fork_read_scope` (lane D) still
  writes `(forked, "scope")` rows raw; `_concord_lexical_reads` (not this
  lane's) still writes `lexical_read_binding` raw -- the largest remaining
  group in the probe's unsourced listing.
- The `value_concordance` (`planning_value_concordance`) `bind_alias` sites
  stay raw: that page is undeclared and may be a detached page.
- `ssa_call_input_adapters.py` / `ir_identities.py` pages are declared but
  their writers still write raw (plan 80 B9 leaves them to steps 6-7).
- Exit Phis for ports under the graph's own id are adopted rows (DERIVED
  from the incoming values), not `PHI_LOOP_EXIT` mints; `PHI_LOOP_EXIT` is
  declared and unused.  Return-merge Phis are `PHI_RETURN_MERGE` mints.
- The builder's pages are keyed by the per-lowering control scope
  (`tensor_shape_concordance_scope`), not `lexical_read_scope` as B1.1 says:
  one ControlProgram is lowered more than once under one read scope and
  CONCORD rows holding physical cells cannot be shared across lowerings.
  `carried_port_value` is REVISE (equivalence-class ports are rebound by an
  enclosing loop), not CONCORD as the declaration comment says.
- A mint whose operand cell the builder cannot name (a bare literal, a
  function the reducer never saw) names the function root
  (`CONTROL_FUNCTION_ROOT`); an adopted graph id with no `canonical_value`
  cell is posted `Unsourced(GRAPH_ID_WITHOUT_CANONICAL_CELL)`.

## Exact next edit

Route `_concord_lexical_reads` (reducer, stage `REDUCER_READ`): the
occurrence row DERIVED from the Name node's identity cell, each consumer
position row DERIVED from the occurrence row and the consumer's identity
cell -- it is the largest raw group the new probe lists.  Then delete
`field_version_values` (`_carried_field_arm` can read the object from
`ssa_value_objects` by the row's fact).

## Working-tree state

Uncommitted: `src/compiler/precompile_to_ssa.py` (`_recorded_name_write`,
the `_carried_name_arm` ladder), the new probe, this note.  Line endings
are CRLF in `precompile_to_ssa.py` (kept).  Baseline and after outputs of
the audit and the measurement are in this session's scratchpad only.
