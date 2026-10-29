2026-10-29 LIST REBOUND IN ONE ARM, THEN APPENDED (MEASURED, cloud): a silent
wrong answer at main 09e36ef7, rebuilt from #902. `rep = [1]; if k > 2: rep
= [k, 2]; rep.append(3)` gave 1112 for 4312 at k=0 (c485; float row c486
18.5 for 21.0). Each arm now leaves the pre-branch arena with the extent of
the value it selects, and when one column has several descriptors the
argument's own descriptor supplies the length operand. Focused: list,
append and branch corpus programs show no branch-only failures against main
(branch 318 pass, main 316); 7 list/capacity test files fail the same 16
names on branch and main. Seen, not fixed: the sequence replace helper's
capacity-exhausted path returns status 2 without a validation error.
