# Weighted runtime broadcast and reduction geometry (2026-11-04)

Measured in the validation checkout with `%TEMP%\probe_weighted_native.py`,
using `lower_ast_source_to_ssa`, the explicit C repository reference, and a
rank-one int64 `AbstractTensor` ProgramABI input. The source reshapes the input
to `(-1, 8)`, converts it to float64, multiplies by the fixed `(1, 8)` weight
view, then reduces axis one and returns a private scalar.

The mixed Mul's booked descriptor is dynamic despite its display shape
`(1, 8)`. Its result23 has dynamic shape/rank/count IDs; the downstream
`reduce_dim_double` had read the display tuple and passed constant shape
`(1, 8)`, producing a false static one-row result27. The `shape_unknown`
reader now checks the booked descriptor before the display tuple. This exposed
two incumbent callers of `_frame_physical_fields(...).get("rank")` that did
not handle the reader's documented `KEEP_PHYSICAL` sentinel; both now use a
posted mapping only when present.

After those reader changes, the weighted source passes the mixed broadcast
validity audit. Its remaining exact final findings are three
`extent(result27)` reads at planned region5 entry instructions 2–4 before
result27 is defined at entry instruction 6. The dynamic reduction result
geometry writer is being repaired separately. No C or native parity for this
weighted source has yet been claimed.

Focused compatibility command:
`python -u -m pytest -q tests/test_native_runtime_reshape_scalar_first.py tests/test_native_private_mixed_broadcast.py --tb=short`
passed 4 tests in 29.56 seconds. Selected concordance command:
`python -u tools/audit_identity_concordance.py mapping` reported 28 rows,
0 findings, 0 unsourced facts, latch CLOSED in 7.43 seconds.

After the separate axis-reduction producer landed, a fresh bounded lower
reached one different final finding: region5 result27 has an `axis-runtime`
receipt derived from source23 and live source/output geometry, but the
`reduce_dim_double` Call output was later adapted to float64 temporary4122.
The reduction audit found no live Call matching result27. The three earlier
self-extent findings were gone. This later observation is still before native
execution and requires the exact output-carrier conversion relation.
