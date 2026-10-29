# 2026-10-29 Exact region placement for protected calls

The retained `NativeTryBlock` now carries each protected source Call with
the exact scheduled region marker that contains it. The composer derives
the pair from the region's source node membership and refuses a call with
no unique region. The control block's concordance fact carries both the
source call cells and deployment region cells. The exception audit checks
each pair against `DEPLOYMENT_REGION_MEMBER`, `SOURCE_EXCEPTION_TRANSFER`,
and the Try owner cell. SSA refuses a mismatched mapping before lowering.

Measured on the real codec graph: Try 56 owns `(region 1, call 65)` and
`(region 8, call 80)`. The bounded public compile still refuses with the
two explicit `protected call has no lowered native failure edge` shortfalls;
the focused exception/control audit returned zero findings. Native C corpus
cases c1166 and c1167 passed (2 passed, 651 deselected). The broad `view`
audit still refuses at its earlier `selected return has no exact source call
reachability authority`. The next change must emit a native failure branch
at each of these region calls before any failed result is read.
