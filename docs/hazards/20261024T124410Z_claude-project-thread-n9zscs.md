2026-10-24 UNANNOTATED RETURNED LIST ELEMENT (MEASURED, cloud): a helper
with no return annotation that builds a list in a loop and returns it
(c546) refused with a heap context conflict: the caller's ABI edge had no
element dtype, so the frame defaulted the member to float64 while the
callee's heap owner was int64. ABI settlement now takes the element dtype
from the caller graph's own planned scalar dtypes when the linked list
value has none, and states it as a rank 1 mutable list span contract. The
heap conflict error now names both functions and the conflicting key.
C corpus 380/381 at 8fb85e7 + this change; the one failure is
c428_class_keyword_construct WRONG, identical on main 8fb85e7. Fuzz
FUZZ_EXT=3 seed 71 (40): 40 PASS. Gate names identical to main 8fb85e7 in
all four parts (both trees list the same names beyond the baselines).

Same PR, second change (MEASURED, cloud): a list returned inside a tuple
(`return out, flag`, then `t, big = compact(n)`) now compiles. Three
records were wrong: the list member's placeholder projection was retired
with no producer, so `t[0]` read a dangling cell (its consumers now read the
caller arena the linked call passes); an equal-priority type proposal that
records LESS than the incumbent (a callee list handle with no element type
vs the caller arena's int64) raised a conflict (now an ordinary tie, the
converse of the existing refinement rule, also in the C call-result
conflict list); and the call result of a multi-output function took the one
member's class that had a fact (list), so an annotated
`-> tuple[list[int], bool]` became a list/tuple union (a call result takes a
member's class only when the callee returns one value). Corpus c547-c549.
C corpus 383/384 (c428 WRONG on main too). Fuzz FUZZ_EXT=3 seed 73 (40):
40 PASS. Gate names identical to main 8fb85e7 in all four parts.
