2026-10-29 NATIVE STATUS TAKE HANDOFF (MEASURED, Windows worktree)

The existing native validation runtime records the first nonzero status
in a thread-local cell and `turing_validation_error_take` returns and clears
it. The first focused SSA report/take test found that C emitted a local
status value but did not give it an address; Ret's output publication left
the caller's slot at 0. The C call emitter now records the scalar address.
The same single-block SSA reached LLVM's legacy leaf path, which looked up
the validation runtime symbol in the handwritten tensor kernel table and
raised KeyError. Such calls now use the existing repository module emitter.

Native C and LLVM both returned code 97 and cleared the runtime status for
literal and SSA-supplied report codes. The public-source float-to-int tests
also passed in both lanes. The combined native-status, source exception and
existing validation-runtime regression gate passed 15/15 in 6.50 s.
`git diff --check` passed. The broad concordance audit remains red at the
preexisting selected-return reachability refusal. No authored Try has yet
consumed a taken status; source admission remains closed.
