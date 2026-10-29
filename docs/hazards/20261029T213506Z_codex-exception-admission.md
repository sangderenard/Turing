2026-10-29 DEPLOYED EXCEPTION SITE STATE (MEASURED, Windows worktree)

The bounded `try: y = int(x); except ValueError: y = z` program retains
the binding of z at the failing Call. The planner now posts
SOURCE_EXCEPTION_REGION_STATE for each required live name at each exact
deployed site, linking the region site, source transfer, NAME_BINDING,
its source value and the forked canonical value. The concordance audit
checks those source and canonical edges and requires a state row for each
transfer binding.

An initial lookup through the Call's ingestion scope failed on a second
Try whose live y was a Phi in the reducer scope. `post_name_binding`
already records the value's exact INGESTION_VALUE cell as a provenance
edge. Both this new site state mapping and the existing continuation arm
mapping now follow that edge. The bounded source exception suite passed
10/10 in 3.17 s, and the combined exception, control ownership and native
status gate passed 25/25 in 5.95 s. The broad concordance audit remains
red at the preexisting selected-return reachability refusal. The source
Try gate remains closed because SSA exception execution is incomplete.
