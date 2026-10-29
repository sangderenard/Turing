2026-10-29 EXCEPTION SITE REGION IDENTITY (MEASURED, Windows worktree)

The bounded `try: y = int(x)` source Call's ingestion cell maps through
CANONICAL_VALUE to forked node 6. The planner's region-member row records
node 6 in region 0 under planning scope ('plan', 1). The source Try builder
now posts SOURCE_EXCEPTION_REGION_SITE from the exact transfer, canonical
value, deployment region and deployment member cells. A new audit verifies
the source-to-canonical edge, the current planner cells and provenance.
The bounded source test observes exactly one such row and the audit returns
no region-site findings. The nested authored raises have canonical nodes
but no deployment member at this stage; no region fact is invented for them.

The source exception and control ownership suites passed 17/17 in 3.76 s;
the source exception suite passed 8/8 after the audit refinement. The broad
concordance audit remains red at its preexisting selected-return source-call
reachability refusal. This is a planning-to-source identity correlation,
not an SSA exceptional successor. Admission remains closed.
