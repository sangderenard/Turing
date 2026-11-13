# Runtime annotation arms exclude container dependencies

## Evidence and change
Source `nodes: list[Node]` with a nested dataclass field acquired a false runtime union of Node and builtins.list. Annotation ingestion correctly records all class dependencies; the semantic ABI consumer incorrectly promoted every dependency to a runtime alternative. Select only class names at the annotation value boundary, preserving the existing class-declaration provenance join. Union/Optional arms remain eligible; container element dependencies do not.

## Verification
- New runtime boundary tests plus existing authored Registry|None test: 8 passed.
- Immediate prechange HEAD comparison for source regression and Registry union: 1 failed (false Node/list union), 1 passed; repaired both pass.
- Full existing union ABI suite: 45 passed, same 6 existing failures as preceding commit. Initial combined run additionally caught a test assertion wrongly requiring a singleton fact that is legitimately omitted; corrected and reran focused tests.
- Actual class-equipped GraphSolver compile_nodes remains refused with 49 findings: 45 descriptor-member-unknown, 1 elementwise geometry, 1 unaccounted formal, 1 operand-never-written, 1 source parameter default.

## Limits
Nested record sequence materialization is not fixed here. Source regression intentionally preserves and inspects that refusal; it does not suppress final concordance or claim native execution. No real GraphSolver LLVM product. No deployment or shared checkout edits.
