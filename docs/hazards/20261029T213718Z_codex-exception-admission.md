2026-10-29 HANDLER PHI ARM VALUES (MEASURED, Windows worktree)

The bounded `y=1; a=int(x); y=2; b=int(x)` Try has two failing Call
sites and one handler Phi for y. The source Phi cited the two exact
NAME_BINDING cells, but the planned control tree did not carry the Phi
or its forked arm values. The Try control fact now carries each handler
entry Phi. SOURCE_EXCEPTION_HANDLER_ARM_VALUE links each source transfer,
binding and value to the forked canonical Phi, handler and value. The
concordance audit checks arm order, canonical reachability, provenance,
and completeness against a retained Try's carried Phi.

The focused exception, control ownership and native status suites passed
25/25 in 6.02 s. The broad concordance audit remains red at the
preexisting selected-return reachability refusal. These rows define
handler inputs; native SSA still needs executable status dispatch and
handler Phi instructions. Authored Try admission remains closed.
