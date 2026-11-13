# Direct nested record sequences reuse the declared row ABI

## Root cause and change
Direct list[Record] row declarations used a nonrecursive physical-column enumerator. A nested record field became one unknown scalar column; unused-frame pruning then removed columns that the sequence descriptor still named. The existing explicit table row_record path already defines recursive dotted leaf columns, optional presence, and cycle refusal.

Expose that same existing row normalizer over record receipts and reuse it from explicit ProgramABI table construction and direct authored nested record sequence views. Preserve the record's semantic fields; retain the normalized row-column receipt alongside them. Direct loop field projections now walk exact receiver identities and select the same dotted leaf column order, including fields after nested records. No opaque handles or independent layout were introduced; concordance remains enforced.

## Verification
- New nested LLVM execution tests cover ordinary and optional-present nested records over 0, 1, and 4 rows, projecting nested x/y and a following outer field. All outputs match Python arithmetic.
- Shared normalizer regression checks exact optional presence order and explicit recursive cycle refusal; existing annotation boundary source test now lowers rather than inspecting a refusal.
- Repaired existing extraction-contract, annotation-alias, IR sequence suites: 75 passed, 16 pre-existing failures.
- Immediate HEAD baseline, same suites plus the two new native source tests: 75 passed, 18 failed. Exact failure-set difference is those two newly passing nested tests. No new existing-suite failures.
- Mapping identity audit: 28 rows, 2 functions, 0 findings, 0 unsourced, latch CLOSED.
- Actual retained-class compile_nodes through source-divider LLVM attempt: 49 findings reduced to 4. All 45 descriptor-member-unknown findings resolved.

## Remaining limitations
Real compile_nodes still does not emit LLVM. Remaining findings: string comparison retains a stale span receipt, one unaccounted formal, missing producer for isinstance(..., nn.Module), and optional parameter default provenance. Guarded optional nested record absence (`if row.inner is not None`) remains a separate missing presence producer and is refused; native tests here exercise valid present records, not that unsupported guard. No merge, push, deployment, or shared checkout edits.
