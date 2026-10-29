2026-10-29 EXCEPTION REGION CHECKPOINT BOUNDARY (MEASURED, Windows worktree)

The bounded source `try: a = int(x) + int(y)` planned three regions:
one executable cast node in each of regions 0 and 1, then the addition
in region 2. A region checkpoint can attribute a native status to its
source transfer only when that protected site is the sole executable
node. DEPLOYMENT_REGION_MEMBER classifies the cast output as OUTPUT;
INPUT members are feeds, while NODE and OUTPUT members execute. An
initial audit check that counted NODE only failed on the one-cast case;
the corrected check reads both executable roles.

The planner now refuses a protected region with any other executable
member, and the SOURCE_EXCEPTION_REGION_SITE audit checks the same
condition. The source exception, control ownership, and native status
gate passed 25/25 in 5.80 s. The broad concordance audit remains red
at the preexisting selected-return reachability refusal. No native
Try status checkpoint has been emitted yet; admission remains closed.
