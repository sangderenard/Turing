2026-10-24 LANE D, D2: SEQUENCE COPIES, COLLECTION BRANCH JOINS AND SLICES (MEASURED, cloud):
`_planned_scalar_dtypes_from` gains two element-class rules. A copy of a
sequence (`list(xs)`, `tuple(xs)`, `set(xs)`, `frozenset(xs)`) holds its
argument's rows; a dict argument proves nothing, since iterating it yields
keys. A branch join (Phi) that is itself a collection joins its arms' rows
with its own writers. Emitted-plan float64 fallbacks (last hierarchy plan per
function, 347 programs at main cc62e7e8): 914 -> 611 with #875 and this
change; c430 52 -> 0, c441 28 -> 0, c442 39 -> 6. Gate at 086c4b49 vs main
8fb85e7b (llvmlite and ziglang installed): failure names identical in all
four parts (aa 86, ab 8, ac 26, rec 20). Typed corpus 380/381 on both trees:
c428_class_keyword_construct is WRONG on main too (keyword construction of a
crossed-field class), not this change. Fuzz seed 30, FUZZ_EXT=3: 30 PASS.
Not green, found on the way, WRONG on main as well: a list rebound on a
branch and then appended (`rep = [1]; if k > 2: rep = [k, 2];
rep.append(3); sum(rep)`) loses the appended row in `sum` (k=0: 12, want
42); sent to the silent-wrong lane.
Re-measured with the slice rule (a slice is a collection of its base's
rows, not one row of it) at a45ea3d1 vs main 3aaa36d1: gate names identical
in all four parts (aa 86, ab 8, ac 26, rec 20); typed corpus 380/381 on both
trees (the same c428); fuzz seed 31, FUZZ_EXT=3: 30 PASS. Emitted-plan
fallbacks over 381 programs: 732 -> 676 (c386 33 -> 7, c294 16 -> 6, c291
and c292 14 -> 4). Also WRONG on main: a float store into an int slice
(`ys = xs[:2]; ys[0] = 0.5`) is truncated, not refused; sent to the
silent-wrong lane.

2026-10-24 re-gate after #904, #913, #920 and #922 merged into main
`d84d0107`: merged main into this branch without rewriting history. Focused
typed corpus c291, c292, c294, c386, c441 and c442 passed in C, LLVM and
Fortran. `tools/audit_identity_concordance.py mapping` found 0 discrepancies
across 26 rows and 2 functions. The Linux gate completed on fresh main and
this branch with matching failing names in all four parts (aa 77, ab 7,
ac 25, rec 20); every part log ended in a pytest summary. Fuzz seed 31 with
`FUZZ_EXT=3` and LLVM/Fortran backends passed all 30 programs. The full
C/LLVM/Fortran typed corpus finished: 376/387 programs passed every backend;
11 had a nonpass. Those names are c22, c367, c398, c532, c536, c537, c538,
c547, c548, c549 and c550. Fresh main reproduced the same failure family
for c22, c367, c398, c532, c536, c537, c538 and c550 in the typed checker;
direct fresh-main native Fortran children for c547, c548 and c549 each
exited -11. The fresh-main comparison was targeted at these 11 names, not
a second full 387-program run. No new failing name appeared on this branch.
The previous c428 wrong answer is resolved on current main and this branch.
