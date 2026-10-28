2026-10-24 LANE C, c428 REGRESSION FROM #879 (MEASURED, cloud): #879 placed a
plan callsite where its call ends, but a region member still completed at its
own start. In `Q(b=x, a=y).get() + Q(x, y).a` (corpus c428) the `+` region
sorted ahead of both calls, pulled `get()` ahead of itself, and left the
`__init__` callsite behind. C returned 5 where Python gives 310 (silent). A
member reading a call operand now completes after that call ends, which is
the rule display operands already follow.
Hazard: placement rules for calls, queries, displays and regions must agree on
"completes where it ends". A new placement rule that uses a node's start
re-opens this.
Corpus 381/381 PASS at 8dbb0b7b; fuzz seed 11 FUZZ_EXT=1: 119/120 (p11_41 loud,
also on main). Gate vs main 8fb85e7b (llvmlite installed): failure names
identical in all four parts (aa 86 / ab 8 / ac 26 / rec 20).
