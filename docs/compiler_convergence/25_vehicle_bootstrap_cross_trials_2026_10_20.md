# Vehicle/bootstrap cross-trials — 2026-10-20

This follows `24_vehicle_validator_creep_2026_10_20.md`.  No compiler or
vehicle source was changed.  Two isolated real-source units were rerun to
separate deterministic compiler defects from project-creep scheduling and the
deliberately interrupted constructor attempt.

## Cross-reference

| Vehicle observation | Bootstrap observation | Shared problem |
|---|---|---|
| `_outer_rate_hz` refuses a raw `call_record` rewrite | Reports 21–22 retain a raw-page-writer worklist and require every fact to be written through `IdentityBook.post` | A provenance-bearing table consumer mutates a record without publishing the cells from which the replacement was derived |
| `visual_snapshot` raises `KeyError(20)` while lowering `source.items()` | Report 23 units 081 and 115 stop around compiler-object/mapping traversal, generator consumers, unavailable call results, and missing physical representations | Mapping/iterator consumers assume a sequence payload representation that no producer published |
| Constructors are partial with `missing_used_parameters`; setter regions have `no-published-outputs` and unresolved address bases | Reports 22–23 describe consumers/finalization using operands without reading or publishing the authoritative producer/effect facts | Effect and receiver-field consumers are not closing the exact input/output accounting chain |
| Strict whole-entry planning reports duplicate class identities | Reports 21–23 repeatedly identify one source class acquiring more than one spelling/key | Class navigation still contains a path that admits aliases instead of joining to the canonical class identity |

The first two rows are observed at exact stacks below.  The latter two are
cross-family classifications only; their writers have not yet been traced.

## Trial 1: isolated call-record refusal

Command:

```powershell
python -u -m tools.compile_project_catalogue --source tools/run_vehicle_native_assembly.py --entry _outer_rate_hz --output build/vehicle_cross_trial_outer_rate_20261020 --jobs 1
```

Result: deterministic failure in 12.8 seconds.  The one-unit manifest reports
zero linkable, zero partial, and one failed unit.

Exact chain:

```text
_prune_unused_callee_formals_once
  caller_records[record_index] = replace(...)
_BookCallList.__setitem__
_BookCallList.replace(..., sources=())
_BookCallList._commit
IdentityPage.revise
IdentityBook._admit_raw_write
ConcordanceRefusal: raw write refused: page 'call_record'
```

`SSACallRecord.cell` already carries the `call_record` cell from which the
record was read, and `_BookCallList.replace` already accepts `sources` and a
stage.  The failing consumer uses list-assignment syntax, which cannot pass
either.  This site was absent from report 21's ten-writer inventory because
that inventory found direct page writes; this write is hidden by the mutable
sequence abstraction.

Receipt:
`build/vehicle_cross_trial_outer_rate_20261020/units/_outer_rate_hz/failure.json`

Manifest:
`build/vehicle_cross_trial_outer_rate_20261020/manifest.json`

## Trial 2: isolated mapping-iteration failure

Command:

```powershell
python -u -m tools.compile_project_catalogue --source tools/run_vehicle_native_assembly.py --entry _PythonVehicleMaterial.visual_snapshot --output build/vehicle_cross_trial_visual_snapshot_20261020 --jobs 1
```

Result: deterministic failure in 7.8 seconds.  The one-unit manifest reports
zero linkable, zero partial, and one failed unit.

The saved resolved ProcessGraph identifies value 20 as the exact call
`source.items()` at authored line 330.  Its edge is the iterator consumed by
the dictionary comprehension.  Control lowering reads a union fact for that
iterator and enters `_native_type_tag_sequence_column`, which then performs:

```text
self.sequence_storage_values[20][column_projection]
```

No sequence-storage entry for value 20 exists, so it raises `KeyError(20)`.
The product frontier labels the failure `repair-stale-planned-node-reference`,
but the observed evidence establishes only an absent sequence-storage
publication.  It does not establish that node 20 is stale.

This is the same broad surface described by report 23 unit 115: iteration over
a mapping/object model reaches generator lowering without a complete physical
representation.  The vehicle failure is earlier and less disciplined because
the consumer raises a bare key error rather than a named concordance refusal.

Receipt:
`build/vehicle_cross_trial_visual_snapshot_20261020/units/_PythonVehicleMaterial.visual_snapshot/failure.json`

Manifest:
`build/vehicle_cross_trial_visual_snapshot_20261020/manifest.json`

## Focused coverage check

The existing pruning and mapping-union unit tests were run together:

```powershell
python -m pytest -q tests/test_fortran_c_shell.py::test_dead_callee_formal_and_matching_region_operand_are_pruned_together tests/test_value_union_abi.py::test_mapping_value_column_uses_writer_union_abi
```

Result: one passed, one failed in 3.71 seconds.

- The mapping-union writer test passes.  It does not exercise an `.items()`
  iterator consumer whose sequence storage is absent.
- The pruning test fails earlier at
  `pruned_callee_formal_concordance: no source cell`, before exercising a
  book-backed `SSACallTable`.  Its origin was not baselined in this pass, so it
  is not classified as a regression.  It does show that the present focused
  test does not cover the real call-record mutation path.

## Narrow next trials

1. **Call-record provenance trial.** Exercise
   `_prune_unused_callee_formals` with a real book-backed `SSACallTable`, capture
   the incumbent `SSACallRecord.cell`, the formal-removal decision cell, and
   the replacement record at the failing write.  The acceptance criterion is
   one DERIVED `CALL_RECORD` revision from those exact cells; no raw-write
   bypass and no function-name special case.
2. **Mapping-iterator representation trial.** Replay the saved
   `visual_snapshot` graph while observing publication of value 20's iterator
   descriptor, union fact, and `sequence_storage_values` entry.  The consumer
   must either read the published payload column or refuse by name at the
   missing producer.  A default dictionary lookup or scalar stand-in is not an
   acceptable result.

Only after those trials name their producer/writer chains should another
project creep be useful.  Rerunning the 12-unit creep now would reproduce the
same two deterministic failures and the same constructor closure expansion.
