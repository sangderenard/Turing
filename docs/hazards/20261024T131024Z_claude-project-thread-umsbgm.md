2026-10-24 FORTRAN RECORD PROGRAMS (MEASURED, cloud): Lane E's parity run
found C-passing record programs refused by Fortran (doc 40 F6). The refusals
were loud, and no answer was wrong. (1) "record 'S' field 'n' names absent
SSA values" (c397, c399): `emit_module` now checks and exports a record
descriptor only when the record or one of its field values is resident in
the function. A resident record with an absent field still refuses, and so
does any record with a field that names storage (span, sequence, nested
record, reference). (2) "missing Fortran feed N" (c422-c429): the
constructor frame link's cloned formals kept `program_abi_parameter` and
`program_abi_field`. They are now `linked_call_frame_storage` of the
constructor. Hazard: a looser first version of (1) silenced
`test_record_table_rejects_storage_references_absent_from_function`. The
gate caught it before the PR. Typed corpus at main 830aa1d9: C 380/381 (main
the same), Fortran 373/381 (main 363). The one C failure is c428, wrong at
main since #879, and Lane C's #896 fixes it. Gate at 8fb85e7b (llvmlite
installed): aa and ac match main. Main's own ab and rec runs were killed for
memory, so those parts were read against the baselines. The new names there
were the record-table test (fixed) and
test_project_unit_recognizes_exact_lowered_closure_formals, which fails on
main too. Fuzz seed 101 at FUZZ_EXT=4: 55 PASS, 5 LOUD, 0 WRONG.

2026-10-24 RE-GATE ON MAIN c8bf22db (MEASURED, Ubuntu24 WSL): Merged
main into this PR without rewriting history. The full typed C/LLVM/Fortran
corpus completed all 387 programs. Eleven nonpass program names remain,
all established on fresh main: c22, c367, c398, c532, c536-c538,
c547-c550. Eleven formerly nonpass names now pass: c397, c399,
c422-c429, and c552. c550 still refuses by name; c547-c549 still crash
the isolated Fortran child with exit -11, also reproduced directly on main.
The focused record storage-refusal and constructor frame-ownership tests
passed 2/2. Identity concordance audit: 26 rows, 2 functions, 0 findings,
0 unsourced (CLOSED).

FUZZ_EXT=4 seed 101 completed 60 programs: 57 PASS, 3 LOUD, 0 WRONG.
The Linux gate ran on the merged PR and a fresh main worktree with
requirements, llvmlite, ziglang, setuptools, and gfortran installed.
Each of its four part logs ended in a pytest summary. Part aa/ab/ac/rec
had 77/7/25/20 failing names on both trees, with exact name equality
and no new failures. The PR is mergeable on GitHub at this receipt.
