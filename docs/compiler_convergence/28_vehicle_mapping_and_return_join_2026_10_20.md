# Vehicle mapping and return-frame repairs — 2026-10-20

This continues reports 23–27 in the canonical repository
`C:/dev/Powershell/turing`. No vehicle or bootstrap source was changed.

## Outcome

The isolated real `_PythonVehicleMaterial.visual_snapshot` unit is now fully
linkable. Three compiler ownership gaps were repaired.

1. Source partitioning previously discarded aggregate field facts produced by
   methods outside the selected slice. The project product now harvests exact
   class-field aggregate producers from the complete authored source, including
   local aliases such as `snapshot = {...}; self.current = snapshot`. `None`
   initializers contribute no storage claim and conflicting aggregate kinds
   erase the claim.
2. `mapping.items()`, `.keys()`, and `.values()` iterator views were treated as
   storage owners. Loop composition now projects their lexical targets from the
   exact mapping operand's resident keyed columns. The type tag and payload now
   share the producer-owned descriptor.
3. A lexical tail returning a multi-column sequence frame was omitted when the
   early-return join computed its physical width. The join now includes the open
   tail's accepted width and expands only an exact authored `None` literal over
   that absent arm. The literal IDs come from the compiler's existing constant
   table; no dtype guess or synthetic payload is used.

## Real-source receipts

Before these repairs, the visual unit reported
`missing-union-payload-column` and then an unresolved
`return_slot[1]`. After the repairs:

```powershell
python -u -m tools.compile_project_catalogue --source tools/run_vehicle_native_assembly.py --entry _PythonVehicleMaterial.visual_snapshot --output build/vehicle_fix_trial_visual_snapshot_return_join_20261020 --jobs 1
```

Result: **1 linkable, 0 partial, 0 failed, 0 blocked**. The compiler completed
all eleven stages. Receipt:
`build/vehicle_fix_trial_visual_snapshot_return_join_20261020/manifest.json`.

Two cross-trials establish the next frontiers:

- `_outer_rate_hz` is partial only because its declared native boundaries
  `math.ceil` and `math.log2` remain unmaterialized. The earlier
  `args.rate_hz` projection issue is absent. Receipt:
  `build/vehicle_fix_trial_outer_rate_hz_after_mapping_20261020/units/_outer_rate_hz/unit.json`.
- Bootstrap `_result_layout` first reported
  `unresolved-address-base-contracts` for value 4, the loop binding `values`
  consumed as the base of two nested structural reads. The control builder
  already selected the exact nested child-row address, but its rank was lost
  at the independently compiled region formal. Nested row bases and their
  exact array-feed formals now publish `ssa_call_rank=1`. The follow-up receipt
  has **no incomplete source-region integrals**. Its remaining root coordinator
  frontier is six unexplained structural formals (`function.blocks`,
  `b.instrs`, `i.op`, `i.args`, the generator `any(...)`, and the layout
  tuple), a missing represented `function` parameter, no published return, and
  unresolved final call value 58 (`next(iter(layouts))`). There are no
  unmaterialized extraction boundaries or unresolved Program ABI references.
  Receipts:
  `build/bootstrap_result_layout_after_mapping_return_20261020/units/_result_layout/unit.json`
  and
  `build/bootstrap_result_layout_nested_row_rank_20261020/units/_result_layout/unit.json`.

## Tests

The four focused regressions pass. `tests/test_precompile_to_ssa.py` passes
132/132. The relevant broader gates reached their previously documented
source-less fixture faults without a new failure: loop-composer reached 142
passing before `formal_shape` refused a `Derived` post with no source cell;
project-product reached 99 passing before a `FunctionTable.declare` fixture
named no source span.

## Next repair

The highest-leverage isolated bootstrap target remains `_result_layout`, but
its numerical regions are now complete. Continue at the root structural
projection chain `Function.blocks -> BasicBlock.instrs -> Instr.args`: these
values must be supplied by the accepted compiler record/sequence descriptors,
not left as public formals. Then settle the existing `next(iter(layouts))`
intrinsic result and rerun this one unit before another full bootstrap.
