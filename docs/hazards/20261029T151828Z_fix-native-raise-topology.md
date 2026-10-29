# 2026-10-29 Raised class authority on source exception transfers

The source Raise transfer now carries the exact `STATIC_SYMBOL_DEFINITION`
cell reached from its constructor's static-use edge. The control overlay's
cached class spelling is checked against that cell. Native SSA reads the
transfer's class cell for handler matching and status selection, and refuses
a missing or disagreeing class. The exception audit checks that the class
cell is a static-definition cause of the transfer. This removes an
independent class key at the SSA consumer; the native status values are still
the handwritten table in `native_faults.py` and remain unresolved work.

Measured native C corpus: `c1166_try_raise_handler_loop_state` and
`c1167_try_raise_superclass_handler` passed (2 passed, 651 deselected).
The focused codec exception-transfer audit returned zero findings. The
public codec compile still refuses at `region value 4 (input 'value' in
function 'tensor_octets_to_bytes') has no settled dtype`.

The exact graph path is input 3 (`octets`, declared int64 rank-1 span in
the public compile contract) to slice 76, `tolist()` call 78, `enumerate`
call 79, and loop input 4 (`value`). Node 78 has only the slice as its
operand and no native sequence-view or element-type record. The current
`_planned_scalar_dtypes` propagates list element types through collection
slices and `enumerate`, but has no rule for an AbstractTensor `tolist()`
view. This is an observed missing producer, not evidence that an int64
default should be assigned to value 4. The short public compile reproduced
the same refusal in 5.8 seconds after the class-cell change.

`py_compile` passed for all five changed compiler modules. The broad `view`
audit remains red at `selected return has no exact source call reachability
authority`, matching the prior receipt. Native conversion failure edges,
handler Phi, finalizers, and the codec artifact are still unfinished.
