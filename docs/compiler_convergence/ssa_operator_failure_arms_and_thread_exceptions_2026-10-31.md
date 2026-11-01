# SSA operator failure arms and threaded exception control

Current implementation (2026-11-01): admitted try syntax converts during
source ingestion into explicit exception-record operations and conditional/
loop control, with exact concordance provenance. The withdrawn synthetic
position overrides remain removed. The combined native exception/thread/
synchronization/sequence gate passes 102 cases, including generated standalone
C. See `docs/hazards/20261101T013030Z_work.md` for measured identities and limits.
TryStar, custom resource scopes and ordinary protected calls without a
failure-return contract remain unsupported. Standard Condition/Lock/RLock
resource scopes now have native C coverage, including waits, nested return/break
cleanup, try around a scope, caught cleanup failure and recursive restoration
after a park failure. See
[Condition composition](condition_ssa_composition_2026-11-01.md) and the newest
hazard receipt for current measurements. This is not complete Python exception
semantics.

## Contract

Failure conditions belong to fundamental SSA operators. Ingestion supplies
source exception regions and their handler ownership; it does not define what
division means. The compiler option `operator_safety_arms` enables an operator's
failure arms when its instruction is constructed. It is a compile-policy cell,
retained by that instruction. This is neither a pre-filter nor a post-filter
pass over a completed program.

A safety-enabled operator has a normal result and defined failure outcomes.
On failure the normal result is unavailable for ordinary publication. A
backend must test before a potentially undefined operation, or use a target
fault mechanism with a demonstrated recovery contract. A harmless temporary
value used by C on its failure path is not a successful source result.
Proof that a condition cannot fail can remove that condition's physical check;
it does not remove the operator's semantic failure definition.

`operator_safety_arms=False` disables the newly configured arms. It does not
remove established checks required by Python integer/native-width contracts.
It is not a promise that unchecked operations become safe. Target fault
recovery is not selected by this boolean.

## Implemented and measured slice

`transmogrifier/operator_safety.py` defines division failure outcomes;
`OperatorDef.failure_arms` exposes them and `Instr.failure_arms_for(book)` selects them
using its retained compiler option. `Div`, `FloorDiv`, and `Mod` share the
zero-divisor outcome. The signed native quotient-width outcome applies to an
integral quotient that cannot be represented, including minimum / -1; modulo
minimum % -1 is zero and does not take that width-error arm. Floating division
does not acquire a signed integer-width failure. Existing exact integer
quotient/conversion limitations remain named native faults.

The C backend materializes a zero-divisor arm before evaluating scalar division.
Safety-enabled shaped division refuses until its per-lane failure contract is
materialized, rather than silently dropping the configured arms.
Existing integer helpers retain their safe width, rounding, and overflow
behavior. The selected operator, exact SSA inputs/result, definition, and
compile-policy identity are concorded in `C_OPERATOR_SAFETY`. The audit checks
that record against the actual instruction. No signal handler is installed.

Authored try regions currently normalize into existing conditional control
and Phi construction. Native exception-record operations create, capture,
match, and propagate failures. Arithmetic itself owns the division check;
there is no source-level zero-divisor guard. The runtime record is thread-local
and contains a code, exact supported exception type, optional static message,
and worker origin. Numeric faults without a declared type do not match an
invented Python exception type. The evaluator-based Try-value alias has been
removed; an unnormalized ambiguous merge refuses explicitly.

This is an intermediate exception transport: an operator reports to the
active native error slot, and explicit `ExceptionTake` consumes that outcome
at the protected activity boundary. It is not yet a universal SSA exceptional
successor ABI. In particular, arbitrary source callees need an admission contract that stops their own
effects after reporting a fault. Do not
claim protection across such calls from the local arithmetic tests.

Thread-table C lowering copies scalar arguments, retains exact captured
storage views, starts native tasks, and waits for completion at join. A worker
may catch its own error. An uncaught worker record is retained by its task;
join reports it through the parent's separate reported-error channel and
returns normally. A join does not execute a parent's except arm just because
a worker failed. The standalone shell reports an unhandled record at entry
completion. The current channels retain the first error, not an arbitrary
queue of failures.

## Remaining design, not implemented guarantees

The final SSA failure ABI should expose normal and exceptional successors
explicitly, including a typed failure-record operand and the exact handler or
caller-propagation destination. Region and ordinary-call signatures must
carry that outcome so failure exits the callee before subsequent effects.
Unhandled raise already branches to the existing SSA fault-exit Trap, with
resource cleanup, rather than relying on a C-only jump. General exceptional
call returns still need their explicit ABI. All identities and rewrites must be concorded at their
writers. Later transformations must retain the outcome and control dependency
when folding, scheduling, inlining, or changing storage representation.

An operator definition needs the following policy fields:

| Field | Meaning |
| --- | --- |
| Applicability | Exact operation and operand/result representations |
| Condition | Failure predicate, or a proof that it cannot occur |
| Failure | Exception identity, code and owned payload schema |
| Publication | Which effects/results exist before failure, if any |
| Destination | Nearest active handler, or exceptional caller return |
| Materialization | Explicit check, proven-elided check, or verified target trap |
| Provenance | Authored try/raise identity and exact compiler policy |

Initial extension table:

| Family | Conditions | Boundary |
| --- | --- | --- |
| Division/modulo | Zero divisor; integer quotient width; exact conversion limitations | Division arms exist; full exceptional call ABI remains |
| Integer arithmetic | Result/narrowing outside declared representation; negative shift | Existing native checks; consolidate into fundamental failure definitions |
| Sequence access | Index type, negative-index normalization, bounds, live storage extent | Local protected normalization exists; move definitions into access operators |
| Pointer/span access | Null where forbidden, extent/alignment/lifetime validity proved by storage contract | No general recovery from arbitrary invalid C pointers promised |
| Mapping access | Missing required key | Preserve existing storage contract and explicit failure edge |
| Allocation/queue | Capacity/allocation failure; empty/full operation policy | Blocking, nonblocking, and exception behavior must be explicit |
| Mutex/synchronization | Invalid token/state; broken barrier; timeout if authored | Cleanup uses exact acquired token; no automatic synchronization |
| Math/conversion | Domain, range, invalid conversion | Respect actual dtype and Python/native representation contract |

Handler matching must follow the declared exception hierarchy. Dynamic owned
messages, `except ... as value`, general custom exception classes, exception
chaining, and richer payload lifetimes remain outside this slice. Only the
explicit supported builtin hierarchy is admitted today.

Cleanup runs on success, failure, return, break and continue as appropriate.
A finally failure replaces a pending failure; a finally return suppresses it;
a finally body otherwise preserves the pending outcome. Locks/resources are
released only by their actual cleanup obligations. Loop iteration state and
pending outcomes are distinct SSA carries. A try inside a loop is supported
by the tested slice; loops/break/continue inside a protected try still refuse
until their cleanup continuations are represented.

Backend policy may select hardware faults only after specifying trap coverage,
compiler optimization assumptions, instruction attribution, worker-local
recovery, cleanup, and nested/reentrant handler behavior. Portable C integer
undefined behavior and default IEEE infinity/NaN are not that contract.
Licensed Python external boundaries remain permitted by explicit extraction
contracts; basic types, exception control, and concurrency mechanics cannot
use Python as their implementation.

## Verification ladder

Use authored Python through the public ingestion entry, inspect the actual
SSA and concordance, emit C, compile a standalone executable and run it in a
separate process. Compare supported successful/caught outcomes with the same
Python source. Check unhandled native failures through the documented shell
record ABI. Never substitute handwritten C or an in-process Python executor.

Current combinations are in `tests/test_source_c_exceptions.py`: division
success/zero/negative operands; except/else/finally in a loop; worker-local
handling; a worker containing that loop; nested authored raise/rethrow and
finally; and both compile-option settings on the same authored division.
The former strict return/finally failure now passes. Structural folding had
identified distinct normalized ExceptionNone calls by identical copied source
coordinates and syntax, deleting the try-entry record along with a rejected
cleanup arm. The shared source signature reader now uses the existing stamped
source occurrence row for transformed nodes, preserving identity across copies
without equating different occurrences. Both strict xfails were removed only
after standalone execution passed. Worker-return cases observe the finally
store through captured storage after join. While loops with try/except/else/
finally also run in the parent and inside an authored worker.

Ordinary source calls in protected expressions refuse until they declare a
failure-return contract. Next establish failure-exit loop carries, then
add exceptional call propagation and side-effect stopping and protected loops
with all cleanup exits, before general locks/queues under failure. No check or
try elimination is part of this milestone.
