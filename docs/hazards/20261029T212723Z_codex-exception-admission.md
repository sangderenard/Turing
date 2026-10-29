2026-10-29 EXACT SOURCE TRY JOIN ARM VALUES (MEASURED, Windows worktree)

The bounded authored Try `try: y = int(x); except ValueError: y = 2`
already had a source continuation Phi with two exact NAME_BINDING cells.
The success binding's source value reaches forked canonical value 6; the
handler binding's source value reaches forked canonical value 1. Reading
the final name `y` would erase that distinction. The planner now posts
SOURCE_EXCEPTION_CONTINUATION_ARM_VALUE with source Phi, control, binding,
value and their forked canonical identities for each arm. The concordance
audit checks the source Phi tuple, binding's value identity, canonical
reachability and receipt provenance.

The focused source exception admission suite passed 9/9 in 3.03 s. The
broad concordance audit remains red at the preexisting selected-return
reachability refusal. Native SSA does not yet consume these arm facts;
authored Try admission remains closed.
