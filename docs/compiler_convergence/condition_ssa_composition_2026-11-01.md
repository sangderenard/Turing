# Condition composition in SSA

For the complete per-operation porting table, including operands, outcomes,
ordering, lifetime and C symbols, see the
[backend threading operation reference](backend_threading_operation_reference_2026-11-01.md).

## Implemented and verified

Authored `threading.Condition`, `Lock` and `RLock` calls enter through
`lower_ast_source_to_ssa`. Their exact callable identities become
`SOURCE_PROCESS_OPERATION` facts. Condition's constructor, acquire, release,
wait, notify and notify_all become primitive SSA and actual control-flow
subgraphs. No opaque Condition instruction remains in the tested modules.

Standalone generated C applications have been compared with the same authored
Python under ASAP and ALAP. The cases include recursive waits, timeouts,
notification counts, captured worker handles, shared locks, queue isolation,
loops, returns, breaks and exception handling. This is verified C execution,
not a claim about LLVM, other backends or every Python threading convenience.

## Construction and exact identity

Default construction projects to `MutexCreate(OBJECT, reentrant=True)`. Its
existing lock-box row includes empty notification head/tail links, separate
from mutex-acquisition waiters. Every execution allocates a fresh instance;
a source factory site is not a runtime singleton.

`Condition(lock)` and `Condition(lock=lock)` project to
`WaitQueueCreate(exact_mutex, OBJECT)`. Each queue is distinct, while
`WaitQueueMutex` retrieves the exact authored guard. Two Conditions sharing a
mutex do not share notifications. Lock-box binding follows the incumbent
source operand and closure-capture edges; names and allocation spelling do
not choose identities. Supplied locks use the ingested Lock/RLock protocol;
arbitrary Python lock implementations are not admitted as native mutexes.

Captured handle dtypes enter SSA from `SOURCE_MUTEX_BINDING` at the argument
writer, with that binding as provenance. C does not reconstruct a missing
formal type from a callee name. Native source-worker ownership tokens derive
from the existing thread-management instance ID, installed at dispatch.
Legacy native callers retain a separate, aligned caller-identity encoding.
Ownership therefore does not depend on a source worker retaining a particular
hardware lane or a recycled native thread address.

## Fundamental contracts

All omitted `after` operands are the ordinary explicit SSA activity tokens.
They order effects, not unrelated workers.

| Primitive | Meaning |
| --- | --- |
| `WaitQueueCreate(mutex, execution_scope)` | Allocate an empty notification FIFO referring to that mutex, without allocating a replacement lock. |
| `WaitQueueMutex(queue)` | Retrieve its guard identity. |
| `WaiterRegister(queue, worker, token, latch)` | Validate ownership and append one registration while the guard is held. Return a status/activity token. |
| `WaiterSelect(queue, worker, token)` | Select the first unsignaled registered latch, or the null handle. Do not notify a future registration. |
| `WaiterRetire(queue, worker, latch)` | Remove that exact registration under the guard. |
| `MutexSuspend(mutex, worker, token, memory_scope, RELEASE)` | Save full recursion depth and release ownership completely. |
| `MutexRestore(mutex, worker, depth, memory_scope, ACQUIRE)` | Reacquire without a timeout and restore the full saved depth. Return a current ownership token. |
| `LatchCreate(participants, generation, execution_scope)` | Allocate generation-specific retained control state. |
| `LatchWaitTimeout(latch, generation, seconds, finite)` | Return released/timeout/error status. A missing/None timeout is explicit non-finite mode; an authored negative timeout polls. |
| `LatchSignal(latch, generation)` | Retain release for this generation, including a signal arriving before parking. No user-memory fence is implied. |
| `LatchDestroy(latch)` | Dispose exclusively after retirement. |

Acquire/release reuse `WorkerToken`, `MutexAcquire`, `MutexAcquired`,
`MutexToken` and `MutexRelease`. All memory scopes/orders remain explicit.
Host lowering accepts its declared scopes and ordering masks; it does not
reinterpret device memory classes or broaden an object wait to a global one.

## Actual procedure topology

`wait` first validates ownership, allocates its one-shot latch and registers it
before suspending the mutex. It then parks, restores the same guard/depth,
retires its registration and destroys its latch before returning the outcome.
Guarded failure paths skip parking when setup fails. A timed-out or failed park
still passes through restoration and retirement. Real status and ownership
Phis join these paths; effect Phis connect restoration to retirement and
retirement to destruction. This is not one ordinary release of an RLock.

`notify(n)` is an SSA loop over `WaiterSelect` and `LatchSignal`, with a real
count Phi/decrement and CFG backedge. `notify_all` traverses until no pending
waiter remains. Both validate ownership, retain the guard and return None.
No guessed participant count or all-worker rendezvous is inserted.

Registration precedes unlock. The latch retains an early notification. The
worker returns only after reacquiring its guard. These are the edges that
prevent lost wakeups. Mutex release/acquire supplies user-memory ordering;
parking and waking are separate control operations. A join remains a task
completion dependency, not a Condition notification or participant barrier.

## Reused runtime and concordance

The existing native Condition already held a recursive owner/depth, acquisition
waiters and a FIFO notification list. That state implementation was factored
into the primitives above in `turing_pool.c`. Its historical Condition/Event
ABI delegates to those same primitives. There is no second Condition state
engine, Python interpreter or general deque substitute. The native FIFO already
provided the required wait-registration mechanism; the general sequence/object
context's unsupported concurrent mutation was not treated as permission to
substitute it.

The existing `turing_thread_scope` now also owns synchronization allocations.
It joins captured workers before releasing parent-owned objects. Per-wait
registrations/latches are retired during the procedure, including loop
iterations. Escaping synchronization handles require an explicit lifetime
transfer ABI and are refused rather than returned after their storage is freed.

`SOURCE_SYNCHRONIZATION`, `SOURCE_MUTEX_BINDING`, `LOCK_BOX_ENTRY` and the
incumbent SSA projection emitter remain authoritative. The new
`SSA_SYNCHRONIZATION_PROCEDURE` joins each source operation to its actual
entry/exit blocks, primitive receipts, typed outcome identity and CFG edges.
Pure unused outcome materializations may be reduced away; their typed identity
remains in the book. The audit checks live primitive/CFG coverage and C
scope/order selection. C decisions use `C_SYNCHRONIZATION_POLICY`.

Worker contexts borrow the existing invocation extent ABI under the same
joined-lifetime proof as captured storage (`C_THREAD_EXTENTS`). No second
shape/extent protocol was introduced. Source error records and the existing
shell channel carry primitive failures. A protected discarded call needs no
invented scalar temporary. Resource entry/body/cleanup remains the existing
`ResourceScopeBlock` mechanism, now also accepting its declared Lock/RLock
recipes. Try may surround a resource scope; body and cleanup failures reach its
handler, including cleanup after a protected return.

ASAP/ALAP use the existing activity configuration. Neither adds comparative
edges between worker bodies. Blocking work still needs native concurrent
progress or a separate resumable-context lowering; simply running a blocked
worker body to completion serially is not an implementation of Condition.

## Remaining boundary

This is the basic Condition protocol, not every Python context manager or
threading convenience. Callable-predicate `wait_for`, arbitrary/custom lock
protocols, native storage for rebound closure cells and escaping ownership
transfers remain separate work. The C runtime's direct generation/timeout
checks do not establish general barrier reset/arrival lowering. The outstanding
run-to-join dual-IR/barrier projection failures are recorded in the hazard
receipt. No new LLVM or other-backend execution support is claimed.
