2026-10-29 TRY OVERLAY AND FLOAT-INT FAULT STATUS (MEASURED, Windows worktree)

The source Try control program was built on the forked function shell, but
its caller omitted it from the existing heterogeneous control overlay. The
function shell now passes the Try program into the existing nesting helper
and overlay. A bounded hook observed the overlay's function-shell input
contain one Try program and its output root contain a NativeTryBlock. The
admission refusal moved to the already existing end-of-preparation gate,
which still fires before SSA emission. The first attempted move placed the
refusal after every shell; the first, nonfunction shell then refused with
zero Try control facts. The final-gate placement restored all eight source
exception tests.

The public source `return int(x)` lowers its intrinsic Cast inside a planned
region. Native C and LLVM now check NaN, infinity, and int64 range before
the cast. Native execution returned 1 and -1 for finite 1.9 and -1.9 and
raised NativeValueError for NaN and NativeOverflowError for infinity in
both lanes. The combined source exception, native cast, and control ownership
tests passed 19/19 in 5.11 s. The concordance audit remained red at the
preexisting selected-return reachability refusal. The two backends also have
unexercised status-take and dynamic-status call support for future SSA
exception dispatch; this run does not prove those paths.
