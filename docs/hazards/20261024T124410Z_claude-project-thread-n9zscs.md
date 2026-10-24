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
