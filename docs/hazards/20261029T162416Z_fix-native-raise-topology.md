# 2026-10-29 Codec's Slice formal is a mislowered tensor view

The bounded public full-native compile of `tensor_octets_to_bytes` still
refuses at formal parity. Its caller function has an unnamed formal 75,
accounted as source `ast.Slice` `:limit`. Source graph node 75 has upper
bound node 70 and feeds node 76 as its `index`; node 76 is the authored
`octets[:limit]` Subscript over ABI span node 3. The reducer records that
Subscript as generic `Indexed`. Planned region 5 has formals `(3, 75)`
and emits `GetElementPtr(3, 75)` followed by `Load` to value 76. Its caller
passes `(3, 75)` into region 5; there is no value producer or ABI name for
Slice 75. The full-native formal parity check correctly refuses it.

This is not a missing name for a legitimate argument. The producer of the
wrong value semantics is the generic Indexed-to-element lowering of a
Subscript whose index is a Slice. The authored operation is a tensor view
bounded by `limit`; it cannot be repaired by treating Slice 75 as a scalar
formal or by defaulting its index to zero. The exact next work is to follow
the existing tensor span/view contract through region planning and emit the
slice view with its dynamic extent, then materialize `tolist()` natively.

The independent focused `finally` probe also remains red. A protected
`int(x)` call with a finalbody reached exception topology with a handler
entry naming source Call 15 after that node had left the current function
subgraph; its transfer was absent. A direct-Raise variant refuses earlier
in loop composition with `try-exception-entry-state-not-composed` because
the composer explicitly excludes finalbody. Neither fault was changed in
this receipt. The previously green native raise gates remain the last
verified execution evidence.
