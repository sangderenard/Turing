2026-10-28 LANE D, D8: AN EXTERNAL CALLEE NAME IS A STATIC SYMBOL (MEASURED, cloud):
`_decide_value_shape` returns (None, ()) for an external Input read only as
the callee of tensor-namespace calls (`np.asarray`), as it already did for a
StaticReference and a receiver-only external name. Before, any `np.asarray`
in root refused with "region value 1 (input 'tensor') has no settled dtype".
New corpus c810 and c811 (block c810-c819) PASS on C and LLVM and refuse on
main 0fc16947. Typed corpus on both trees: the same 6 non-PASS (c363, c460,
c463, c467, c646, c671), all on main. Fuzz seed 37, FUZZ_EXT=3: 30 PASS.
Gate at b7b7c708 vs main 0fc16947: failure names identical in all four parts
(aa 134, ab 27, ac 58, rec 48). Part ab is OOM-killed when the four parts run
in parallel on this 15 GB container, on main too, so ab was run alone for both
trees. Not green, still open: `np.asarray([n, 2], dtype=int)` stops at a
named C emission shortfall ("array cast requires a matching static shape").
