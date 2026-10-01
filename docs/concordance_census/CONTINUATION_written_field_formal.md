# Continuation: a written field's formal is its incoming value (lane WF, 2026-10-01)

Repro: `python -u tools/compiler_probes/probe_record_return_merge.py`.
Follows `CONTINUATION_return_site_identity.md` ("The open link").

## Status

- Probe PASSES: native == Python for all three cases, no spin.
- Gate green: struct_intake, branch_written_field, scalar_write_only_arm,
  record_in_tuple_return, row_handle_record_parameter,
  planner_specialization_chain, control_binding_chain, emission_chain;
  native correctness 0 failures (emission_chain and native correctness were
  run in a clean worktree at HEAD with only this lane's two files, because
  another lane's in-progress `control_source.py` in the shared tree raises
  NameError `_describe_control_block`).
- Audit findings unchanged (view 0, toplevel 1, energy 0, controller 1,
  controller_untyped 5, mapping 0, oscillator 0).  Rows: controller
  528 -> 530, controller_untyped 531 -> 533 (update_dt_max now has an
  incoming `dt_max` formal; it writes the field and never reads it).

## Confirmed (read-only hooks, before the edit)

`materialize_parameter_record_abi`, the per-field `candidate_ids`: for
`hard_failure` (written, never read) there is no read, and its write value
is not in `scalar_write_sources` (that set is filled only for fields with a
getter), so `candidate_ids` was the write value itself -- the `True` literal.
It became the field's formal; `_recover_late_source_literals` then matched
that formal's id to the graph Constant with the same value id and replaced
it with `Const True`.

A second link: site A's selection read the field state's VALUE cell (a
source-graph id) as the version.  Control SSA posts the write's version on
`ssa_field_version` at the ingestion field-state cell; the return site names
the canonical re-post of that cell (DERIVED from it at `canonical_relabel`).
The lookup never found the version.

## What changed

1. `fortran_c_shell.py`, `materialize_parameter_record_abi`: a scalar field
   with no authored read takes no write value as a candidate.  Its incoming
   formal is minted NOVEL through `_RecordAbiMinter` (NESTED_RECORD_PART,
   the declared record's `contract_demand` cell).  Fields with a read are
   unchanged: the read is the incoming formal, and writes stay storage
   aliases (pi_update's `acc` needs that).
2. `_recover_late_source_literals`: a formal claimed on `record_member` by a
   table owned by this function is retained.  The fold matched by id
   equality (formal id == Constant value id); that is the same confusion,
   and the guard now reads the book row instead.
3. `ssa_record_return_state.py`, `version_cells`: when there is no version
   row at the canonical field-state cell, follow its `canonical_relabel`
   edge to the source `reducer_field_state` cell.  In the site branch,
   `version_id` is the posted `ssa_field_version` fact when one exists.

## Open

- A no-read scalar field whose write is region-produced and has no Store
  now relies on the selection, which admits only Const or carried-Phi
  versions.  No gate program has this shape.
- `while_exit` is still `Unresolved(predecessor_not_a_return_edge)`.
