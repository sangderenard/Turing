2026-11-13 RETAIN EXACT CLEAR EFFECT AND LENGTH AT RESET CONSTANT MINT (MEASURED):

The resident sequence clear producer now mints its zero with the exact resident length cell and the existing validated mutation effect/current control sources. This uses existing CONTROL_CONST/CELL_SET and _sequence_mutation_call_sources, without a new operation table or a change to runtime reset behavior. Missing/ambiguous source evidence remains unaugmented and does not globally reject source-less lowering.

This is necessary evidence, not sufficient permission to accept arbitrary length-zero stores. List births themselves produce branch-local clear operations, in addition to entry initialization. The separate closed writer reader must validate unique source effects, native occurrence/placement, entry initialization recipes, and refuse injected late resets. Ownership/native transfer is not granted by this change.

At HEAD 4c647158 plus coordinated working-tree prototypes: candidate self-contained clear receipt tests (three list births, plus repeated explicit clears after append), incumbent mutation-call and helper-identity receipt tests: 5 passed in 23.78s. With only the five-line clear producer change removed: 3 passed and 2 new receipt tests failed in 23.81s. Failures are absent new provenance, not a historical native behavioral regression claim. Exact-byte restoration verified; source-less/missing/ambiguous mutation resolver behavior remains covered by incumbent tests.

Mapping audit: 28 rows across 2 functions; 0 findings, 0 unsourced facts/identities; latch CLOSED. No selected-owned-record native execution, actual string-return correctness, or complete reset closed-use proof claimed.
