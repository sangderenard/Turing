# CONTINUATION: native dt-system lowering to full-native

Target: `examples/llvm_dt_system.lowered_system` over air + pool (b1, C
backend) passes the full-native contract.  Full run: scratchpad
`lower_two_pieces_spans.py` (see CONTINUATION_dt_compile_stall.md).
Seconds-long repros: scratchpad `fv/repro_field_version_loop.py`
(`FV_VARIANT=a|b|c`, real Metrics/Targets ABI + full-native execution file,
reuses `tools/repro_record_row_effects._contract`) and `fv/repro_d.py`
(`FV_VARIANT=d|e`, pure scalars).

## 2026-10-03 -- wall 1: ConcordanceRefusal on ssa_field_version (Metrics.unresolved_report)

Repro (variant b, 2 s): the shape of step_with_dt_control_used --

    while True:
        ok, m = advance(metrics, x); m = coerce_metrics(m)
        rejected = m.mass_err > targets.mass_max
        if rejected and allow_unresolved:
            lines = [...]; lines.append(...)
            m.unresolved_report = list(lines)
            rejected = False
        if rejected:
            x = x * 0.5; continue
        return m, x

Same refusal as the full run, same row shape (column 0 of the field-state
row = the WRITTEN cell).  Two writers on that row (hooked
`IdentityBook.post`):

1. `_class_surface_ssa_program` posts `Unresolved(FIELD_WRITE_DTYPE_UNPROVEN)`
   at the write cell -- correct: `list(lines)` is not a scalar write.
2. `_carried_field_arm` then posts `Unresolved(ARM_VERSION_MISSING)` from
   the SAME cell -> REVISE without a changed source.

Why (2) ran at all: the merge of `m.unresolved_report` is a sequence merge
(initial = the seeded `GetAttr`, aggregate_kind list from the Metrics
contract; true arm = `builtins.list`), and
`fortran_c_shell._promote_conditional_sequence_aliases` moved it from
`carried_aliases` to `carried_sequence_aliases` WITHOUT removing its entry
from the index-aligned `carried_field_cells`.  Every later alias shifted onto
its predecessor's cells: the `rejected` name alias was lowered as the field
merge of `unresolved_report` (its snapshot, the field's cells).  Variant a
(no `coerce_metrics`, receiver class unknown, so no promotion) fails the same
way for a different reason; the real program has `coerce_metrics -> Metrics`.

Fix (identity): the promotion filters `carried_field_cells` in lockstep with
`carried_aliases`, as the retained-values projection in control_source
already does.

## 2026-10-03 -- wall 2 (repro): literal name arm `rejected = False`

Next in the repro: `carried-name-arm-missing arm=<Constant False>`.  An
authored literal is published by no region; the control function owns it
(`_materialize_control_constants`).  `_carried_name_arm` only looked in
`external_values`.  Fix: a literal arm id (`constant_value_ids`) resolves
through `external_value` (the provisional formal later materialized as the
function's own Const), as the break-edge site values and owned-literal
outputs already do.

## 2026-10-03 -- wall 3 (repro, silent): region-less guard nested into a later sibling's arm

With wall 2 fixed the repro lowered with a COMPLETE gate but miscompiled:
`if rejected and allow: rejected = False` (no region) was anchored before the
first region after it -- `x = x * 0.5`, which lives in the arm of the next
`if rejected: ...; continue`.  `_insert_before_marker` descends into
conditional arms, so the guard was placed INSIDE that arm; its merge Phi was
never emitted and the outer test read the merged id as an invented `bool`
formal of `step` (pure-scalar variant d shows it; the full-native gate does
not catch fabricated formals).
Fix (control_source.py + the class-surface overlay call):
- `overlay_scheduled_control(..., lexical_encloses=)`; `_insert_before_marker`
  places an anchored conditional BEFORE a conditional that holds the anchor
  marker but does not lexically enclose it (answered from the AST by the
  caller, same signature-walk rule as `parent_by_child`).
- `nested_root` inserts anchored region-less children before embedding the
  region-owning children (the gds pre-overlay path).
Variants b, d, e now lower with the merge Phi before the second test and no
invented formal.

Still visible in the repro (not addressed): step's signature carries the
`unresolved_report` arena of the call-result record (`27`, capacity, status)
and one `unknown`-dtype formal passed to `advance` -- the open
"table-storage field of a call-result record has no ABI columns" item.
