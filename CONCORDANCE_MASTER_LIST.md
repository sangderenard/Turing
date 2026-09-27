# Concordance master list

The concordance (identity book, `src/compiler/identity_concordance.py`) is the
central authority, durable from source reduction to emission.  A decision
"uses the concordance" only when all three hold:

- **Source** — the facts it decides from are read from book pages.
- **Record** — the decision is written to a page (`concord` for a fact that
  never changes, `revise` for one that legitimately evolves; both keep
  history), so a later disagreement is visible or raises.
- **Consumed** — every later reader reads the page, not a private copy.

Working rules (2026-09-26):

- Any id confusion is a failure to use the concordance.  Join identities
  through book rows, never by matching value ids or pairs of ids.
- One base fact per identity; other pages record structure (which operand,
  which entry), never a restatement of the fact.
- SSA structures (tables, call records) keep their interfaces; their
  storage is the book.
- Scopes are minted by the compile's book (`IdentityBook.mint_scope`,
  page `scope_registry`), never by process counters.
- A missing row raises; no value-id fallback where the book is expected.
- Every new record is taught to the audit
  (`CorrelationTable.findings`, `tools/audit_identity_concordance.py`).

Status used below: **BOOK** (all three hold), **RECORD-ONLY**, **PRIVATE**.

---

## Part A — state of the work

### A1. SSA structures stored on the book (turing, uncommitted)

| Structure | Pages (row -> fact) | Status |
|---|---|---|
| `SSARecordTable` (`ssa.py`) | `record_descriptor` `(owner, record id)` -> descriptor; `record_member` `(owner, value id)` -> claims `(record id, field, storage identity, role)` | BOOK |
| `SSASequenceTable` (`ssa.py`) | `sequence_descriptor` `(owner, sequence id)`; `sequence_member` `(owner, value id)` -> `(sequence id, role)`; `sequence_column_claims` `(owner, sequence id, "column_dtypes")` (replaces the private `attempts`) | BOOK |
| `SSACallTable` (`ssa.py`; linker `call_records`, `final_call_records`, `IRModule.call_table`) | `call_record` `(owner, caller)` -> records; in-place list edits are revisions | BOOK |
| `TransformationLedger` (`transformation_priority.py`) | `transformation_decision` `(scope, identity)` -> `(rule, proof, target)`; `transformation_event` `(scope, n)` | BOOK |
| Table owners / scopes | `scope_registry` `(label, serial)` | BOOK |

The audit cross-checks `record_member` / `sequence_member` against the
descriptors they index (`table-member-disagreement`).

### A2. Linker working state moved onto the book (`fortran_c_shell.py`)

| Former private structure | Page | Status |
|---|---|---|
| `record_field_demands` | `record_field_demand` (via `PageMapping`) | BOOK |
| `record_field_writes` | `record_field_write` | BOOK |
| `record_parameter_by_value` | `record_parameter_value` (concord) | BOOK |
| `record_forwarding_edges` | `record_forwarding_edge` | BOOK |
| `record_parameter_specs` | removed; read from the graph's `parameter_record_abi` | BOOK |
| `record_storage_aliases` | `record_storage_alias` | BOOK |
| `scheduled_call_sources` (keyed by caller value id) | `scheduled_call_argument` `(caller, callsite, callee formal)` | BOOK |
| `callee_callers` snapshot | removed (dead) | — |
| `_linked_caller_member` scan of descriptors | reads `record_member` / `sequence_member` | BOOK source; decision still unrecorded |

### A3. Reads, loops and bindings (the carried-value miscompiles)

One base fact decides every read:

| Page | Row -> fact | Writer | Kind |
|---|---|---|---|
| `lexical_read_binding` | `(read scope, consumer, role, ordinal)` -> binding; `(read scope, "return", "root", position)` for a bare `return name` | reducer, at each Name load, AugAssign target, return root | base fact |
| `loop_carried_binding` | `(read scope, loop, updated, initial)` -> bindings | `loop_composer` | entry identity |
| `loop_result_port_binding` | `(read scope, port)` -> binding | `loop_composer.add_port` | port identity |
| `consumer_operand` | `(read scope, node, value)` -> `((role, ordinal), ...)` | planner, region nodes | structure |
| `call_argument_operand` | `(read scope, callsite, position)` -> `(role, ordinal)` | planner, call sites | structure |
| `region_feed_consumer` | `(control scope, region, feed)` -> consumer nodes | `lower_control_sections_to_ssa` | structure |
| `loop_entry_state` | `(control scope, loop, initial)` -> `(carried bindings, pre-loop value)` | lowering, loop entry | state |

The lowering keeps only its position (`enclosing_loop_states`) and resolves
every loop read through one method, `_resolve_read`: a loop carrying the
read binding gives its header (or latch update); a loop carrying none of the
read bindings passes through to its pre-loop value; a mix, or an
unattributed read, raises.  Readers routed through it: region feeds, call
arguments, break-bound result ports, and the `while` latch condition.
Exit ports join port -> binding -> carried entry.  Carried Phis and updates
are keyed by carried-entry position, not by updated id.  `loop_composer`
rewires a continuation only for reads of its own binding.

Miscompiles fixed (each a guard in
`tests/test_loop_carried_producers.py::test_one_value_held_by_two_bindings_computes_the_authored_answer`,
all failing at HEAD): region read of a parameter that seeds a carried
binding; call argument; one value passed as two call operands; while
predicate through a region (never terminated); direct while predicate;
`for` exit of two bindings sharing an update; shared-update tuple return.

### A3a. Operand positions are identities (2026-09-26, woodshop)

The position `(consumer, role, ordinal)` that keys `lexical_read_binding`
lived only in the graph node's private `parents` list, rewritten directly
by ~57 sites in 15 files.  A rewrite changed what a committed row's key
named without the book knowing.

- Found by: woodshop `lower_newton_dt_system` (LLVM, O0, piece link)
  refused with `region 9 reads carried initial 195 through an operand no
  lexical_read_binding row attributes`.  `AbstractTensor.minimum(dt_cap,
  remainder)` (`dt_controller.py:698`, in `run_superstep`'s while) had its
  read committed at `(call, 'args', 0)`; the reducer's Call branch then
  rebuilt the operands through `_replace_inputs` as `arg:0`/`arg:1`, and
  the planner's `consumer_operand` row `('arg:0', 0)` found no binding.
- Fix: `topological_reducer._set_operands` is the one writer of an
  operand list.  It pairs old and new positions by operand identity (same
  role first, then in order; `same` maps a swapped id), revises page
  **`identity_transition`** `(scope, consumer, role, ordinal) -> ("move",
  consumer, new role, new ordinal, cause)` (`"retire"` when the operand
  left; see A3c for `"fork"` and `"merge"`), moves every
  position-keyed page (`lexical_read_binding` rows, `consumer_operand`
  facts) and revises a vacated position to None.  The graph names its
  scope (`operand_position_scope`: the ingestion read scope, then the
  canonical read scope).  `_append_operand` appends through it.
- Migrated: every reducer writer (`_replace_inputs`, `_remove_node`,
  `_redirect_value`, the canonical relabel, Expr/Return dissolve, the
  function-subgraph filter, parameter inputs, four appends).
- Audit: the planner (`_concord_consumer_operands`) records any live read
  row at a position its consumer no longer has on page
  **`operand_position_orphan`**; `CorrelationTable` reports it as
  `operand-position-orphan`.  With the writer bypassed the repro records
  `cap` at `(12, 'args', 0)`; with it, none.
- Guard: `tests/test_loop_binding_reads_native.py[call-argument-carried]`
  (fails at `f9432d80`, passes now).  Gate: the 14 recorded gate failures
  are identical at `f9432d80` on Windows; scorecard 18/19.
- Open: the ~43 raw writes outside the reducer (`glsl_deployment_strategy`
  10, `loop_composer` 8, `process_graph_fusion` 6, autograd 3, 11 others)
  still bypass the writer; `call_argument_operand` facts are not yet moved.
  Specialized graph copies share one read scope, so a post-canonical move
  in one copy revises rows another copy reads: needs a per-copy scope
  before those files migrate.
- Next behind it (observed, not fixed): the woodshop loop 354 carries
  `dt_cap` (update 305) and `last_dt_next` (update 297) from initial 195
  (`last_dt_next = dt_cap`), but `loop_entry_state` owns only
  `last_dt_next` -- the control loop's carried list lacks `dt_cap`'s entry,
  so a body read of `dt_cap` would take the pre-loop value.

### A3b. A rejection is a keyed fact (2026-09-26, woodshop)

- Found by: the woodshop lowering, past A3a, sat in
  `settle_call_result_types` round 1 (py-spy: two samples a minute apart
  on one `propose` for `woodshop_newton_momentum`, value 11; 980 s CPU).
- Cause: `TransformationLedger.propose` decided "already rejected?" by
  scanning `events`; `_BookEventLog` had no `__iter__`, so each item
  rebuilt the whole log from the book (O(E^2) per rejected proposal), and
  `__len__` materialized every row to compute each event serial.  200
  identities: 87 s.  A regression from moving the ledger onto the book
  (`56255f4`).
- Fix: page **`transformation_rejection`** `(scope, identity, rule, proof,
  retained rule, retained proof) -> event serial`, read by one `latest`,
  written by `concord`; `_BookEventLog.__iter__` builds once;
  `IdentityPage.scope_row_count` counts a scope's rows without
  materializing them (also used by `mint_scope`).  20000 identities:
  0.72 s, same events and rejections.
- Guard: `tests/test_transformation_priority.py::test_repeated_rejection_is_found_by_its_book_key_at_scale`.
  Gate unchanged (14 recorded failures, scorecard 18/19).

### A3c. Identity operators: fork and merge (2026-09-26, woodshop 195)

The woodshop's `loop-scope-outer-read` on value 195 reproduced in seconds
as `total_value = float(total.item())` inside a while that carries
`total` (LLVM, 65.9 for 35.9).  It was not a missing fork of `dt_cap`: it
was a merge applied without the loop-generation constraint.

- Chain (observed): no region computes the rank-0 `item` (value 22), so
  region 5 captured 22; the loop carries 7 (`total`), not 22, so the
  control lowering passed 22 through; `fortran_c_shell` structural recovery
  resolved 22 as `item(7)` by graph id and receipted `(22, 7,
  scalar_item_identity)`; `_rebind_recorded_scalar_identities` rewrote the
  call to id 7's only definition -- the pre-loop one.
- **fork** (`_set_operands(fork_from=...)`): the call rebuild gives
  `total.item()` a new `operand` position fed by the Name occurrence the
  attribute load already read at `(attr, 'value', 0)`.  That second read
  position now carries the binding; `identity_transition` records
  `("fork", attr node, "value", 0, cause)`.  Also cleared the refusal
  `control expression leaf ... no lexical_read_binding row` for
  `if cap.item() > 100.0: break`.
- **merge**: the planner (`_concord_item_operands`) records page
  **`item_operand`** `(read scope, item) -> (operand, role, ordinal)`,
  structure only.  The control lowering (`_region_feed`) reads the operand
  by the binding at that position in the site's loop generation; a rank-0
  value read is the item, recorded as `identity_transition` `(control
  scope, item) -> ("merge", resolved id, "scalar_item")`.  The call then
  never holds 22, so the id-keyed rebind has nothing to rewrite in loops.
- Proof: with `_concord_item_operands` disabled the program computes 65.9;
  with it, the authored 35.9.  Guard: `tests/test_loop_identity_operators_native.py`
  (2 programs, LLVM; both fail at `f9432d80`).  Gate unchanged.
- Found on the way, not fixed:
  - LLVM `Const` with a tuple payload writes `i32 int(item)` into a buffer
    sized by the result's declared shape: a float static iterable
    `(0.7, 1.9)` (dtype `ssa.aggregate`, shape `()`) loads as 0.0.  The
    Const declares neither its extent nor its element dtype.  The woodshop's
    `for boundary in boundary_values` has this shape.
  - C lane only: `while` + call + `break` refuses `%tN is unavailable`
    although the SSA dominates; LLVM computes the authored answer.
  - A dead `Cast` of `step` before an inner loop still reads the outer
    generation (`loop-scope-outer-read`, no effect on results).
  - `_rebind_recorded_scalar_identities` still rewrites by id
    function-wide; with the merge resolved at the read site it no longer
    fires inside loops, but it has no generation check of its own.

### A3d. Heuristics replaced by graph facts (2026-09-26, woodshop)

Found while driving the woodshop dt lowering to zero findings; each was a
silent wrong answer or a refusal, reproduced in seconds, native LLVM (and C
where noted) against the authored Python.

- **Loop ledger reads are graph edges.**  A loop reads its carried updates
  (backedge) and break continuations (exit) through ledgers on the loop
  node, not edges, so values read only by the loop had out-degree zero.
  Evaporation's dataflow collection (`evaporate_unrolled_loops`) and the
  fold's dead-call pruning (`_fold_callsite_structural_values`) each
  deleted a carried update (woodshop `dt_cap`, Phi 305; `dt =
  minimum(dt * 1.5, 1.0)` after a `continue`); the composer then dropped
  the pair and the body read the pre-loop value.  The reducer now makes
  the loop node consume each update (`carried_update`) and continuation
  (`break_value`).  Updates are never descendants of their loop (its only
  children are its ports), so no cycle.  No pruner needed a change.
- **Continue sites are recorded like break sites.**  The reducer claimed
  only break sites (`loop_break_sites`, site-node inputs); a continue arm's
  value had no consumer but a synthesized merge.  Continue sites now go
  through the same claim (no break bindings), and `control_dependency_
  value_ids` counts a `LoopControlBlock`'s `site_values`.
- **No history-synthesized conditional merges.**  `_ordinary_conditional_
  control_programs` synthesized merges from a name's identity history ("the
  first version after the arms", and a nearest-dataflow-ancestor rule).
  For a continue arm that version was the fall-through update, bound as
  the if-merge.  Both paths are deleted; merges are the reducer's Phis,
  which it creates for every runtime conditional whose arms differ.
- **No name-history call-argument override.**  `_build_shell_hierarchy_
  plan` replaced a call's argument edge with the source-latest definition
  of the argument's name; it chose a continue arm's `dt * 0.5` over the
  edge to the pre-branch `dt`.  Deleted; the edge is the identity.
- **Non-falling-through arms restore bindings.**  The conditional lowering
  snapshotted only merged identities; an arm that branches away (continue
  in the arm) now restores every binding it published.
- **Keyword and receiver reads fork from the occurrence.**  A keyword
  wrapper (and its Name) can be absent from the function subgraph; the read
  is committed at `(scope, "occurrence", id)` and a call's `kw:<name>` /
  `operand` position forks from it (`ctrl.pi_update(dt_prev=dt_tensor)` in
  `step_with_dt_control_used`, refused).
- **LLVM tuple constants** are typed and sized from the payload, as in C
  (float static iterables loaded as zeros).
- Guards: `tests/test_loop_identity_operators_native.py` (7 programs).
  `test_control_branch_compartments.py`: the synthesized-merge test now
  asserts no merge is invented; a suppression-receipt assertion for the
  deleted path was dropped.
- Evidence: gate + `test_process_graph_function_linking.py` 23 failed /
  283 passed, all 23 known at `f9432d80`; `test_fortran_c_shell.py` 20
  failed identically on both trees (override deletion).
- Open: `_set_operands` still pairs old/new positions by operand identity
  and order; writers should declare their moves.  LLVM loop Phis are `phi
  ptr` over the backedge slot (lost copy: `grow(cap, factor=f)` read `f`
  after `f = f + 1`; C is correct).  C lane refuses `while` + call +
  `break` (`%tN is unavailable`).  The kernel-emitter tuple Const
  (`@const.vec`, i32) has the same shape as the fixed one.

### A4. Other fixes on the way

- `ir_indexing._propagate_scalar_dtypes`: a declared scalar `Const` dtype
  is kept; the Python literal only decides when nothing is declared
  (`Metrics.advanced_dt` absent payload retyped int64).
- `precompile_to_ssa._materialize_control_constants`: an undeclared literal
  is typed by its own Python type (was a float64 default for `True`).
- `ssa_python_materializer`: aggregate-convention callees return their
  aggregate tuple (callee side, checked against the callers' projection);
  loop-exit phis are bound; contracts read `callee_output_ids` under the
  aggregate convention only.
- `tools/translation_scorecard.py` passes the program contract (was 0/19;
  now 18/19, matching the manifest).
- Tests fixed on sight: 43 contract-less compiler calls in 19 files;
  stale expectations (region ordinal, descriptor `rank`, specialized callee
  names, materializer Ret convention, zero-trip expectation that encoded the
  old miscompile).

### A5. Numeral work committed in `2ab44759`

Unchanged from the 2026-09-25 list: 5 BOOK (deferred numeral literals,
`type(self)`, reducer class/numeric lookups, renumbering carry, T4 resident
at its write), numeral result limbs read by string key
(`storage_identity.rpartition(".")`), about 17 RECORD-ONLY.

### A6. Open (next, in order)

1. Direct predicate expressions: `lower_control_expression` resolves a
   `value` leaf by id; needs operand rows from `structured_control_expression`
   and a repro first.
2. The lowering environment itself: `external_values` is still a private
   value-keyed dict with loop reads carved out of it.  The line: binding
   state at each causal point on the book, `external_values` a view of it.
3. Remaining private linker maps: `frame_ledgers` (caller -> ledger),
   `result_storage_bindings_by_call`, `constructor_instance_pools`,
   `call_anchor_value_ids`, `function.metadata["value_aliases"]` snapshot.
4. Name matchers: `_graph_sequence_record_abi` annotation fallback and the
   demand check (`identity.rsplit(".", 1)[-1] == name`); the positional
   result window; numeral result limbs by string key.
5. Emitted parameter order differs from authored declaration (scorecard
   level 12's recorded defect).
6. Linker decisions that read the book but record nothing (resolver,
   discovery, reconcile, ownership, demand growth, pruning).
7. The 19 remaining failures in the focused regression set (baseline 42).

---

## Part B — pipeline areas that decide identity outside the concordance

(Static scan of 2026-09-25; line numbers predate the A1–A4 work.)

Method: every function (nested functions separately) in the 12 pipeline modules
was parsed; a function counts as **identity-reading** if it reads an ABI label
string (`program_abi_*`, `binding_name`, `parameter_names`, `identity_table`,
`value_aliases`, `record_field`, `callsite_id`, `propagated_formal_id`,
`linked_call_frame_storage`) or a private identity structure (record/sequence
tables, call records and their bindings, `frame_ledger`, `storage_identity`,
`record_return_layouts`). It is **no-book** if it never touches
`current_identity_book` / `identity_book` / `.page` / `.concord` / `.latest`.
This is a static scan: it finds where identity is read, not whether each read is
a lookup or a label write, so each row still needs reading.

Totals: 1316 functions; 173 identity-reading; **138 no-book**; 35 mixed.

| Module | identity-reading | no-book |
|---|---|---|
| fortran_c_shell.py | 76 | 59 |
| glsl_deployment_strategy.py | 54 | 48 |
| topological_reducer.py | 11 | 5 |
| precompile_to_ssa.py | 11 | 8 |
| ssa_c_backend.py | 5 | 5 |
| ssa_record_return_state.py | 4 | 4 |
| tensor_ssa_lowering.py | 3 | 2 |
| ssa_call_input_adapters.py | 3 | 2 |
| ir_identities.py | 2 | 1 |
| ssa_fortran_backend.py | 2 | 2 |
| ssa_llvm_backend.py | 1 | 1 |
| graph_express2.py | 1 | 1 |

Known name/position matchers still live (read directly, not from the scan):

- `materialize_record_phis`: fields merged by `field.name`, intersection of names.
- Record-result binding: `positional_result_window`, `callee_signature` /
  `caller_signature` tuples of `(name, storage_identity, count)`.
- `bound_record_pairs` child pairing by `storage_identity` string (A1 #11).
- `record_field_demands` keyed by `(symbol, parameter name)`; forwarding edges
  only between parameters.
- Returned-receiver field matching by `field.name` in `allocate_result_storage`'s caller.
- `_linked_authored_parameter_aliases` (ids converted to names) still used at 35176.
- `select_return_arguments` / record-return scalar Phis: `field.name == record_field`.
- Numeral result limbs: book rows scanned by `storage_identity.rpartition(".")` (A1 #8).

### B1. No-book functions (138)

| Module | Function | Lines | Identity read from |
|---|---|---|---|
| fortran_c_shell.py | `_drop_unused_private_sequences` | 605–728 | `parameter_names`, `program_abi_parameter` |
| fortran_c_shell.py | `_drop_unused_root_private_formals` | 731–781 | `parameter_names`, `program_abi_parameter` |
| fortran_c_shell.py | `_recover_late_source_literals` | 807–899 | `parameter_names` |
| fortran_c_shell.py | `_recover_late_source_slice_offsets` | 902–999 | `parameter_names` |
| fortran_c_shell.py | `_rebind_recorded_scalar_identities` | 1601–1726 | `parameter_names`, `binding_name` |
| fortran_c_shell.py | `_record_module_closure_formals` | 1742–1807 | `parameter_names`, `binding_name` |
| fortran_c_shell.py | `_intern_writable_region_outputs` | 1978–2040 | `program_abi_storage` |
| fortran_c_shell.py | `_linked_frame_physical_shape` | 2114–2126 | `program_abi_storage` |
| fortran_c_shell.py | `_same_declared_span_storage` | 2129–2150 | `storage_identity` |
| fortran_c_shell.py | `_linked_frame_storage_role` | 2153–2173 | `program_abi_record`, `program_abi_field`, `program_abi_storage` |
| fortran_c_shell.py | `_preferred_linked_field_candidates` | 2176–2203 | `program_abi_storage` |
| fortran_c_shell.py | `_preferred_linked_field_candidates.priority` | 2195–2200 | `callsite_id` |
| fortran_c_shell.py | `_restore_linked_sequence_member` | 2542–2571 | `linked_call_frame_storage`, `callsite_id`, `propagated_formal_id` |
| fortran_c_shell.py | `_linked_frame_storage_owner` | 2604–2640 | `program_abi_field`, `storage_identity`, `program_abi_record`, `program_abi_storage` |
| fortran_c_shell.py | `_reconcile_post_aggregate_record_results` | 2643–2838 | `frame_bindings`, `storage_identity`, `record_tables`, `callsite_id` |
| fortran_c_shell.py | `_prune_unused_callee_formals` | 3104–3120 | `call_records`, `sequence_tables` |
| fortran_c_shell.py | `_is_dead_conceptual_record_argument` | 3337–3367 | `program_abi_field`, `program_abi_storage`, `linked_call_frame_storage` |
| fortran_c_shell.py | `_lower_planned_region_record_projection_captures` | 3370–3653 | `record_field`, `record_tables` |
| fortran_c_shell.py | `_lower_optional_record_presence_graph` | 3855–4033 | `program_abi_parameter`, `program_abi_field`, `identity_table` |
| fortran_c_shell.py | `_sequence_row_record_slots` | 4524–4559 | `identity_table` |
| fortran_c_shell.py | `_authored_text_parameter_transforms` | 6844–6872 | `identity_table` |
| fortran_c_shell.py | `_field_slot_ops.table_sequence` | 7449–7527 | `binding_name` |
| fortran_c_shell.py | `_sequence_augassign_ops` | 8098–8175 | `identity_table` |
| fortran_c_shell.py | `_source_mapping_mutations` | 8795–8839 | `binding_name` |
| fortran_c_shell.py | `_identity_return_aliases` | 8842–8882 | `binding_name` |
| fortran_c_shell.py | `_linked_sequence_propagation_kind` | 9013–9037 | `program_abi_parameter`, `program_abi_record`, `program_abi_field`, `program_abi_keyed_owner` |
| fortran_c_shell.py | `_joined_byte_sequence_ids` | 9163–9262 | `identity_table`, `binding_name` |
| fortran_c_shell.py | `_plan_callsite_projection_ids` | 9879–9926 | `result_bindings` |
| fortran_c_shell.py | `_control_block_consumes_values` | 11559–11628 | `argument_bindings` |
| fortran_c_shell.py | `_utf8_encode_aliases` | 12115–12158 | `binding_name` |
| fortran_c_shell.py | `_bytes_join_source_transforms` | 12161–12221 | `binding_name` |
| fortran_c_shell.py | `_scalar_source_transforms` | 12224–12304 | `binding_name` |
| fortran_c_shell.py | `_sequence_prepend_concat_ops` | 12591–12673 | `identity_table` |
| fortran_c_shell.py | `_sequence_prepend_packed_call_ops` | 12676–12727 | `identity_table` |
| fortran_c_shell.py | `_sequence_inplace_bit_pack_call_ops` | 12730–12816 | `identity_table` |
| fortran_c_shell.py | `_nested_row_projection_ops` | 12819–12862 | `identity_table` |
| fortran_c_shell.py | `_record_sequence_projection_bindings` | 12896–13210 | `identity_table` |
| fortran_c_shell.py | `_sequence_row_operations` | 13213–13344 | `identity_table` |
| fortran_c_shell.py | `_sequence_column_dtype_contracts` | 13401–13754 | `identity_table` |
| fortran_c_shell.py | `_sequence_record_identity_contracts` | 13757–13787 | `identity_table` |
| fortran_c_shell.py | `_authored_source_sequence_ids` | 13790–13835 | `identity_table` |
| fortran_c_shell.py | `_linked_authored_parameter_aliases` | 13838–13910 | `argument_bindings` |
| fortran_c_shell.py | `_linked_authored_parameter_aliases.identities` | 13856–13894 | `parameter_names`, `identity_table` |
| fortran_c_shell.py | `_call_argument_identity` | 14014–14050 | `program_abi_field`, `propagated_formal_id` |
| fortran_c_shell.py | `_propagate_record_field_demand` | 14053–14246 | `program_abi_field`, `program_abi_record`, `call_records`, `all_record_tables` |
| fortran_c_shell.py | `_propagate_record_field_demand.linked_member` | 14098–14173 | `all_record_tables`, `all_sequence_tables`, `call_records` |
| fortran_c_shell.py | `_propagate_record_field_demand.linked_member.grow` | 14116–14162 | `program_abi_parameter`, `program_abi_record`, `all_record_tables` |
| fortran_c_shell.py | `_harmonize_call_argument_shapes.parameter_names_of` | 14295–14306 | `parameter_names` |
| fortran_c_shell.py | `_prune_dead_local_sequences` | 14796–14934 | `program_abi_parameter`, `linked_call_frame_storage`, `all_sequence_tables` |
| fortran_c_shell.py | `_class_surface_ssa_program.recover_structural_source_outputs` | 18841–20734 | `program_abi_storage`, `linked_call_frame_storage`, `identity_table`, `parameter_names` |
| fortran_c_shell.py | `_class_surface_ssa_program.recover_structural_source_outputs.structural_boolop_value` | 18867–18974 | `program_abi_storage` |
| fortran_c_shell.py | `_class_surface_ssa_program.recover_structural_source_outputs.structural_boolop_value.destroys_value` | 18904–18916 | `program_abi_storage` |
| fortran_c_shell.py | `_class_surface_ssa_program.recover_structural_source_outputs.structural_membership_value` | 18976–19284 | `program_abi_field`, `program_abi_storage`, `program_abi_keyed_owner`, `all_sequence_tables` |
| fortran_c_shell.py | `_class_surface_ssa_program.materialize_parameter_record_abi` | 20977–22850 | `program_abi_field`, `program_abi_parameter`, `program_abi_record`, `program_abi_storage` |
| fortran_c_shell.py | `_class_surface_ssa_program.resolve_keyed_mapping_iterables` | 23973–24270 | `program_abi_storage`, `program_abi_keyed_owner`, `program_abi_field` |
| fortran_c_shell.py | `_class_surface_ssa_program.loop_constructor_requires_instance_pool` | 24483–24526 | `identity_table` |
| fortran_c_shell.py | `_class_surface_ssa_program.frame_fixed_point_digest` | 27305–27317 | `all_record_tables`, `all_sequence_tables`, `call_records` |
| fortran_c_shell.py | `_report_unmaterialised_record_parameters` | 37950–38014 | `parameter_names` |
| fortran_c_shell.py | `compile_ast_fortran_c_shell` | 40050–40669 | `identity_table` |
| glsl_deployment_strategy.py | `_observe_process_graph_node` | 473–710 | `value_aliases` |
| glsl_deployment_strategy.py | `_identity_parameter_projection` | 1340–1393 | `identity_table` |
| glsl_deployment_strategy.py | `_synthetic_device_scalar_shell` | 1396–1434 | `identity_table` |
| glsl_deployment_strategy.py | `_declared_output_terminals` | 1884–1960 | `identity_table` |
| glsl_deployment_strategy.py | `_build_shell_hierarchy_plan` | 2209–3501 | `argument_bindings`, `result_bindings`, `identity_table`, `binding_name` |
| glsl_deployment_strategy.py | `_refresh_hierarchy_control_captures` | 3730–3781 | `argument_bindings`, `result_bindings` |
| glsl_deployment_strategy.py | `_build_hierarchical_glsl_artifact` | 3785–6530 | `binding_name`, `identity_table`, `result_bindings` |
| glsl_deployment_strategy.py | `_build_hierarchical_glsl_artifact.find_identities` | 3804–3832 | `argument_bindings` |
| glsl_deployment_strategy.py | `_build_hierarchical_glsl_artifact.gather` | 3865–3903 | `argument_bindings` |
| glsl_deployment_strategy.py | `_build_hierarchical_glsl_artifact.value_meta` | 4204–4288 | `identity_table` |
| glsl_deployment_strategy.py | `_build_hierarchical_glsl_artifact.static_endpoint_value` | 4310–4736 | `binding_name` |
| glsl_deployment_strategy.py | `_build_hierarchical_glsl_artifact.static_predicate` | 4738–4912 | `binding_name`, `identity_table` |
| glsl_deployment_strategy.py | `_build_hierarchical_glsl_artifact.leaves` | 4914–5360 | `binding_name`, `identity_table` |
| glsl_deployment_strategy.py | `_structural_region_program_from_subgraph` | 7173–7318 | `record_field` |
| glsl_deployment_strategy.py | `_retained_control_value_id` | 7661–7695 | `identity_table` |
| glsl_deployment_strategy.py | `_optional_presence_control_expression` | 7698–7784 | `identity_table`, `binding_name`, `program_abi_parameter`, `program_abi_field` |
| glsl_deployment_strategy.py | `_ordinary_conditional_control_programs` | 7947–8814 | `binding_name`, `identity_table` |
| glsl_deployment_strategy.py | `_repair_missing_phi_initial_identities` | 8817–8929 | `binding_name`, `identity_table` |
| glsl_deployment_strategy.py | `_closure_routing_dependencies` | 9472–9546 | `identity_table` |
| glsl_deployment_strategy.py | `_closure_routing_dependencies.free_binding_names` | 9498–9514 | `binding_name` |
| glsl_deployment_strategy.py | `_coordinate_scheduled_capture_impl` | 10541–15575 | `binding_name`, `identity_table` |
| glsl_deployment_strategy.py | `_coordinate_scheduled_capture_impl.evaluate_region.run_region` | 11274–11788 | `value_aliases` |
| glsl_deployment_strategy.py | `_coordinate_scheduled_capture_impl._resolve_reference_node` | 11927–11967 | `identity_table` |
| glsl_deployment_strategy.py | `_coordinate_scheduled_capture_impl.evaluate_node` | 11969–14224 | `binding_name`, `identity_table` |
| glsl_deployment_strategy.py | `_coordinate_scheduled_capture_impl.evaluate_reduced_control_expression` | 14262–14677 | `identity_table` |
| glsl_deployment_strategy.py | `_coordinate_scheduled_capture_impl.execute_generator_statements` | 14679–15359 | `identity_table` |
| glsl_deployment_strategy.py | `_source_static_value` | 15953–16001 | `binding_name` |
| glsl_deployment_strategy.py | `_source_static_literal` | 16004–16073 | `binding_name` |
| glsl_deployment_strategy.py | `_tensor_descriptor_rule` | 17775–18673 | `binding_name` |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values.exact_return_formal_actual` | 18729–18785 | `binding_name` |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values._record_field_descriptor` | 18826–18877 | `binding_name` |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values._declared_record_span_descriptor` | 18879–18915 | `binding_name` |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values.descriptor_attribute` | 18917–19006 | `binding_name` |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values.evaluate` | 19131–19740 | `binding_name`, `identity_table`, `program_abi_storage` |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values.remove_node` | 19831–19870 | `identity_table` |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values.replace_alias` | 19872–19918 | `identity_table` |
| glsl_deployment_strategy.py | `_apply_callsite_tensor_descriptors` | 21237–21283 | `identity_table`, `binding_name` |
| glsl_deployment_strategy.py | `_alias_projection_to_member` | 21351–21420 | `identity_table` |
| glsl_deployment_strategy.py | `_apply_callsite_aggregate_descriptors` | 21423–21562 | `identity_table`, `binding_name` |
| glsl_deployment_strategy.py | `_expand_specialized_unbroadcast_identity` | 21565–21653 | `identity_table`, `binding_name` |
| glsl_deployment_strategy.py | `_callsite_specialized_shell_type.caller_record_for_value` | 21699–21717 | `binding_name` |
| glsl_deployment_strategy.py | `_callsite_specialized_shell_type.publish_call_result_shape` | 21900–21958 | `identity_table` |
| glsl_deployment_strategy.py | `_resolve_grounded_method_references` | 22009–22312 | `binding_name`, `identity_table` |
| glsl_deployment_strategy.py | `_resolve_grounded_method_references.receiver_class` | 22114–22156 | `binding_name` |
| glsl_deployment_strategy.py | `_resolve_grounded_tensor_operations` | 22315–22400 | `binding_name` |
| glsl_deployment_strategy.py | `ProcessGraphGLSLDeployment.capture_fused_programs` | 23512–24309 | `binding_name` |
| glsl_deployment_strategy.py | `ProcessGraphGLSLDeployment.compile_discovery_program` | 24336–25393 | `binding_name`, `parameter_names` |
| glsl_deployment_strategy.py | `ProcessGraphGLSLDeployment.install_composed_control` | 25395–25587 | `identity_table` |
| graph_express2.py | `_expand_unresolved_ast_parents` | 1564–2600 | `argument_bindings` |
| ir_identities.py | `_refresh_precision_call_records` | 3080–3160 | `call_table`, `argument_bindings`, `frame_bindings` |
| precompile_to_ssa.py | `lower_fused_program_to_ssa` | 204–514 | `binding_name`, `parameter_names` |
| precompile_to_ssa.py | `lower_fused_integral_to_repository_ssa` | 517–680 | `sequence_tables` |
| precompile_to_ssa.py | `_ControlSSABuilder.__init__` | 881–1722 | `storage_identity` |
| precompile_to_ssa.py | `_ControlSSABuilder._emit_table_delete` | 2078–2174 | `storage_identity` |
| precompile_to_ssa.py | `_ControlSSABuilder.finish` | 7925–8352 | `value_aliases`, `parameter_names` |
| precompile_to_ssa.py | `_schedule_loop_callsites` | 9408–10263 | `argument_bindings`, `result_bindings` |
| precompile_to_ssa.py | `lower_control_sections_to_ssa.collect_resolved_plan_dependencies` | 11431–11453 | `argument_bindings` |
| precompile_to_ssa.py | `lower_precompile_and_control_to_ssa` | 13381–13620 | `sequence_tables` |
| ssa_c_backend.py | `emit_ssa_module_to_c` | 1350–4867 | `program_abi_storage`, `parameter_names`, `callsite_id` |
| ssa_c_backend.py | `emit_ssa_module_to_c.authored_array_contract` | 1410–1430 | `program_abi_storage` |
| ssa_c_backend.py | `emit_ssa_module_to_c._root_public_ids` | 1558–1585 | `program_abi_storage` |
| ssa_c_backend.py | `emit_ssa_module_to_c._trusted_declared_shape` | 1596–1614 | `program_abi_storage` |
| ssa_c_backend.py | `emit_ssa_module_to_c.is_structural_abi_value` | 4580–4588 | `program_abi_storage` |
| ssa_call_input_adapters.py | `physical_call_input_conflicts` | 288–321 | `program_abi_storage` |
| ssa_call_input_adapters.py | `_adaptation_state_signature.descriptor` | 726–734 | `program_abi_storage`, `storage_identity` |
| ssa_fortran_backend.py | `_FunctionEmitter.__init__` | 769–1057 | `program_abi_storage` |
| ssa_fortran_backend.py | `emit_module` | 4344–6195 | `parameter_names`, `program_abi_storage`, `program_abi_parameter`, `linked_call_frame_storage` |
| ssa_llvm_backend.py | `_emit_repository_call_module` | 1192–4221 | `linked_call_frame_storage` |
| ssa_record_return_state.py | `normalize_declared_scalar_record_shapes` | 8–64 | `record_tables`, `storage_identity` |
| ssa_record_return_state.py | `publish_inout_scalar_return_snapshots` | 67–253 | `record_tables` |
| ssa_record_return_state.py | `reconcile_forwarded_record_results` | 256–417 | `call_records`, `callsite_id`, `call_table`, `record_tables` |
| ssa_record_return_state.py | `publish_scalar_record_return_fields` | 1263–1346 | `record_field`, `record_tables` |
| tensor_ssa_lowering.py | `propagate_repository_ssa_call_metadata.enrich` | 913–1030 | `program_abi_storage` |
| tensor_ssa_lowering.py | `propagate_repository_ssa_call_metadata.fixed_point_state` | 1137–1191 | `program_abi_storage` |
| topological_reducer.py | `specialize_python_precision_widths.call_return_identity` | 547–611 | `identity_table` |
| topological_reducer.py | `_normalize_lexical_values.input_value` | 2577–2635 | `binding_name` |
| topological_reducer.py | `_normalize_lexical_values.static_constant` | 2759–2798 | `binding_name` |
| topological_reducer.py | `_normalize_lexical_values.bind_loop_target` | 2911–2933 | `binding_name` |
| topological_reducer.py | `reduce_abstract_tensor_topology` | 6759–10792 | `binding_name` |

### B2. Mixed functions (35) — touch the book but also decide from private identity

| Module | Function | Lines | Identity read from |
|---|---|---|---|
| fortran_c_shell.py | `_concordant_function_aliases` | 88–144 | `value_aliases` |
| fortran_c_shell.py | `_concord_record_return_phi_inputs` | 189–250 | `record_field` |
| fortran_c_shell.py | `_publish_concordant_function_aliases` | 385–443 | `value_aliases` |
| fortran_c_shell.py | `_prune_dead_entry_field_aliases` | 2206–2359 | `program_abi_parameter`, `program_abi_field`, `program_abi_storage`, `linked_call_frame_storage` |
| fortran_c_shell.py | `_linked_caller_member` | 2391–2539 | `storage_identity`, `argument_bindings` |
| fortran_c_shell.py | `_complete_propagated_frame_tails` | 2859–3059 | `linked_call_frame_storage`, `callsite_id`, `propagated_formal_id` |
| fortran_c_shell.py | `_prune_unused_callee_formals_once` | 3123–3334 | `parameter_names`, `program_abi_parameter`, `linked_call_frame_storage`, `call_records` |
| fortran_c_shell.py | `_field_slot_ops` | 6875–8095 | `storage_identity`, `binding_name`, `identity_table` |
| fortran_c_shell.py | `_class_surface_ssa_program` | 14939–36866 | `storage_identity`, `result_bindings`, `all_sequence_tables`, `all_record_tables` |
| fortran_c_shell.py | `_class_surface_ssa_program.recover_structural_source_outputs.ensure_structural_value` | 19286–19817 | `binding_name` |
| fortran_c_shell.py | `_class_surface_ssa_program.materialize_parameter_record_abi.materialize_nested_record` | 21273–22061 | `program_abi_field`, `program_abi_record`, `program_abi_parameter`, `program_abi_storage` |
| fortran_c_shell.py | `_class_surface_ssa_program.materialize_program_abi_record_literals` | 22852–23307 | `program_abi_record`, `program_abi_field`, `program_abi_storage`, `parameter_names` |
| fortran_c_shell.py | `_class_surface_ssa_program.materialize_record_phis` | 23309–23662 | `record_field`, `storage_identity`, `all_record_tables` |
| fortran_c_shell.py | `_class_surface_ssa_program.materialize_loop_record_phis` | 23664–23971 | `storage_identity`, `all_record_tables`, `call_records`, `result_bindings` |
| fortran_c_shell.py | `_class_surface_ssa_program.complete_linked_literals` | 25397–25484 | `all_record_tables` |
| fortran_c_shell.py | `_lower_ast_source_to_ssa_impl` | 38017–39904 | `call_table`, `sequence_tables`, `callsite_id` |
| fortran_c_shell.py | `_lower_ast_source_to_ssa_impl.numeric_parameter_record_views` | 38975–39033 | `binding_name` |
| glsl_deployment_strategy.py | `_propagate_callsite_tensor_specializations` | 16611–17132 | `identity_table`, `binding_name` |
| glsl_deployment_strategy.py | `_propagate_callsite_tensor_specializations.call_result_descriptor` | 16647–16789 | `identity_table` |
| glsl_deployment_strategy.py | `_tensor_descriptor` | 17651–17772 | `binding_name` |
| glsl_deployment_strategy.py | `_fold_callsite_structural_values` | 18676–21234 | `identity_table`, `program_abi_storage`, `binding_name` |
| glsl_deployment_strategy.py | `_callsite_specialized_shell_type` | 21656–22006 | `identity_table`, `binding_name` |
| glsl_deployment_strategy.py | `ProcessGraphGLSLDeployment.__init__.plan_callsites` | 22561–22728 | `callsite_id` |
| ir_identities.py | `lower_precision_operations` | 1879–2998 | `value_aliases`, `parameter_names` |
| precompile_to_ssa.py | `_ControlSSABuilder._emit_table_lookup` | 1734–1938 | `program_abi_storage` |
| precompile_to_ssa.py | `_ControlSSABuilder.lower_loop` | 6692–7524 | `program_abi_storage` |
| precompile_to_ssa.py | `lower_control_sections_to_ssa` | 10593–13284 | `sequence_tables`, `storage_identity`, `program_abi_storage`, `record_tables` |
| ssa_call_input_adapters.py | `_adapt_physical_call_inputs_round` | 392–686 | `program_abi_storage` |
| tensor_ssa_lowering.py | `lower_tensor_calls_to_repository_ssa` | 1652–4602 | `program_abi_storage` |
| topological_reducer.py | `specialize_python_precision_widths` | 361–1045 | `binding_name` |
| topological_reducer.py | `_normalize_lexical_values` | 2217–6701 | `identity_table` |
| topological_reducer.py | `_normalize_lexical_values.resolve_expression` | 2948–4531 | `record_field`, `binding_name` |
| topological_reducer.py | `_normalize_lexical_values.reduce_statement` | 4884–6216 | `record_field`, `binding_name` |
| topological_reducer.py | `reduce_abstract_tensor_topology.specialize_concorded_same_type_numeric_operator` | 8172–8286 | `binding_name` |
| topological_reducer.py | `reduce_abstract_tensor_topology.propagate_call_formal_numeric_types` | 10492–10734 | `binding_name`, `identity_table` |
