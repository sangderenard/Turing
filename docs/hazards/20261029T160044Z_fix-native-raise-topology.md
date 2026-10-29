# 2026-10-29 Standalone Raise transfer in a specialized callee

Program: `fail_if_zero` raises ValueError under an `if`, and `root` calls it
inside a loop Try. The deployment strategy's topology trigger examined
retained Try records only. The callee has no Try, so it posted no
`SOURCE_EXCEPTION_TRANSFER` for its Raise. SSA then refused the callee's
authored Raise at source node 6 for missing class definition, although the
static-use edge identified `builtins.ValueError`.

The trigger now also notices live Raise graph nodes. In a bounded trace, the
specialized callee's transfer row for Raise 6 contained the exact
`STATIC_SYMBOL_DEFINITION` cell for `builtins.ValueError`. Standalone native
C and LLVM compilation and execution of this helper passed: x=1 returned,
and x=0 raised NativeValueError after the native run. The targeted test was
2 passed, 2 deselected.

The protected caller program advanced beyond the missing callee transfer,
then failed final concordance: `root value 3: alias of 0 which is never
defined`. This is still red. The protected helper call has not been shown
to execute natively, and no alias repair was attempted here.
