2026-11-12 SCENE RECORD INDEX (MEASURED CPU FIXTURES, WINDOWS):

Isolated base 0485ded4cc251f8d8175459a5877d2e1426230fb plus exact copied
canonical decoder baseline identified in STUDIO_SCENE_RECORD_INDEX_2026-11-12.md.
Canonical shared files, Pluck, native transport and shaders are unchanged.

Existing decoder checks pass. New bounded record-index fixture: 17 tests pass,
including 150 deterministic finite segments independently checked against each
cell's slab intersection. Opt-in, budget rejection, exact payload bits, family
and context separation, unknown/zero distinction and boundary ownership pass.
Node syntax and index-only whitespace checks pass. Full suite not run.
The copied preexisting decoder test has one trailing blank line at EOF; the
baseline snapshot preserves it unchanged. It is not part of the index diff.

Independent read-only review found inherited object property names were accepted
as filter fields. The whitelist now uses own-property checks; negative tests
cover constructor, toString and __proto__. No other blocking finding reported.

Initial baseline test command had the wrong working directory and could not
locate the test module. It was corrected to the isolated worktree before edits;
both original check groups passed. This was not a source failure.

The index is endpoint/finite-segment record references only. It does not provide
new rays, deposited-energy values, unavailable producer records, complete
non-opaque-volume coverage, shader blending or application readiness. No GPU
context, compiler, link, dispatch, render, restart, deployment or push ran.
