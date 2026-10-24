2026-10-24 LANE D, D2: SEQUENCE COPIES AND COLLECTION BRANCH JOINS (MEASURED, cloud):
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
