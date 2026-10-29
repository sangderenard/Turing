# 2026-10-29 Direct raise handler loop state (measured, Windows)

`tests/native_semantics_corpus/c1166_try_raise_handler_loop_state.py` is a
native C corpus case. `root(1)` must return 2: the protected body updates `x`
and raises, and the handler reads that updated `x` and adds 2. Before the
reducer correction, the handler read pre-Try `x`; lowering refused because
its loop-carried update was the Try node itself, with no producer. After the
reducer correction, lowering accepted but compiled C returned 0 for `root(1)`.
An SSA dump showed the constant-false direct Raise emitted a two-way branch;
the impossible success block reached the Try exit, making the handler result
fail loop-latch dominance. The latch then chose its incumbent Phi. The
validation lowerer now emits one edge for a literal boolean predicate.
The loop composer now refuses a source Try without a direct terminal Raise
or with protected calls, `else`, `finally`, or additional handler control:
those shapes still need explicit exceptional entry states. The codec graph
now reports `try-exception-entry-state-not-composed` and `Raise` blockers.

Targeted native corpus run: `python -m pytest -q -p no:cacheprovider -rf
tests/test_native_semantics_corpus.py -k c1166_try_raise_handler_loop_state`
passed 1, 642 deselected, after the correction. This compiles and executes C;
it does not establish generic protected-call failure edges, exceptional
`finally`, or the real octet codec. The new loop Try composer remains a draft
outside this direct terminal Raise case. The Windows battery script created
`build/battery` logs and an untracked script; its checkout and fast-forward
pull failed, the branch HEAD and tracked changes were preserved, and it left
no running Python battery process.
`python -u tools/audit_identity_concordance.py view` remains red with the
previously observed `selected return has no exact source call reachability
authority` error. `git diff --check` and syntax compilation passed after
the explicit source-Try refusal was added.
