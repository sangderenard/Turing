2026-10-24 PRECISION LIMBS ACROSS REGIONS (MEASURED, Ubuntu24 WSL):
Observed on fresh main ef82644e: c621's authored Precision[2] loop lowers
through lower_ast_source_to_ssa, but the C artifact publishes neither x nor
y as a buffer; its buffer_order contains the count and a limb-count formal.
Running it would pass unallocated pointers, so the test refuses before native
execution. The same source on this branch publishes x and y and returns
[1.0, 2^-53, 4.0, 2^-51] for input limbs [1.0, 2^-54, 2.0, 2^-53].
Two planned regions record appended result-limb fields. This is the small
program in tests/precision_corpus/c621_precision_span_square.py.

Inferred from the writer and records: the old typed-parameter receipt joined
the numeric Precision descriptor to the projected class record, treating
the y span as the class's limb-count field. The precision pass also inferred
array striding from local use only, so a planned-region call hid the authored
array declaration. The branch uses the declared descriptor at that seam and
returns each result limb through the existing aggregate call contract.

Focused Linux check: c621 passes on C and LLVM. tests/test_precision_pipeline.py has
34 passed, 2 failed, 1 skipped; the two Exp/Log source-class concordance
refusals reproduce on untouched main with the same records. The identity
concordance mapping audit before the corpus addition found 26 rows across
2 functions, 0 findings and 0 unsourced. GPU execution is unverified in
this WSL environment: PyOpenGL resolves no glDispatchCompute entry point,
even when llvmpipe reports GL 4.5. Windows Python 3.11 also lacks math.fma,
which the pre-existing test interpreter uses.

The authoritative Linux gate was run after installing requirements.txt,
llvmlite, and ziglang. At final branch head 05535301, each of its four part
logs ends in a pytest summary. Against a fresh ef82644e main worktree, the
sorted failing names match exactly: aa 77/310, ab 7/255, ac 25/476,
rec 20/255 (failed/passed). FUZZ_EXT=3 seed 74, 30 programs on LLVM and
Fortran, completed 30 PASS and 0 nonpass. The full C/LLVM/Fortran typed corpus
finished all 387 programs with 11 nonpass names: c22, c367, c398, c532,
c536, c537, c538, c547, c548, c549, and c550. Fresh main reproduces the
eight noncrash names and the directly terminated c547-c549 Fortran children;
the latter report exit code -11 on this branch. This corpus ran on eb5c48cc;
the only subsequent change before final head 05535301 was extending c621's
focused assertion to LLVM, without changing compiler source.
