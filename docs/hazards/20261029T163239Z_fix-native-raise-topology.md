# 2026-10-29 Native Try else takes only the protected success edge

The exact source program is `root(x: float, n: int)`: a loop Try assigns
`y = int(x)`, the conversion handler raises TypeError, and `else` assigns
`y = y + 1`. The loop composer refused this program with
`try-exception-entry-state-not-composed` solely because `orelse` existed.
It already assigned each scheduled region to its exact source Try arm, and
the SSA builder already lowered the else only after the protected body.
The composer gate no longer excludes `orelse`; it still excludes `finalbody`
and exceptional value joins it cannot compose.

Native C and LLVM both compiled and executed the real program. With x=2.4,
n=0 yielded 0, n=1 yielded 3, and n=2 yielded 3. With x=NaN and n=1,
both reported NativeTypeError from the handler. The focused concordance
audit returned zero findings. All eight native exception control cases
passed. Syntax compilation and `git diff --check` passed. The broad `view`
concordance audit still refuses at its prior selected-return source-call
reachability authority error.

This does not implement `finally`: the direct-Raise loop probe still refuses
at `try-exception-entry-state-not-composed`, and SSA has no exceptional
finalizer edge or status carry. The real codec still has an unaccounted
dynamic Slice formal 75 before native artifact emission.
