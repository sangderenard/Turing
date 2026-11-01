# Source synchronization and the SSA lock box

Current implementation/porting reference:
[backend threading operation reference](backend_threading_operation_reference_2026-11-01.md).
The C Condition/mutex/latch slice now executes; the historical remaining-boundary
section below describes the earlier ingestion-only stage. Current limits are
recorded in the linked reference and newest hazard receipt.

## Observed implementation

Source calls are ingested through the existing Python identity, control placement,
and DispatchBlock path. `src.compiler.synchronization` declares compile-only
intrinsics; executing these declarations in Python raises an error. Tests compile
actual source with `lower_ast_source_to_ssa`; they do not substitute constructed SSA.

| Source operation | SSA projection | Explicit operands/policy |
| --- | --- | --- |
| memory_barrier / memory_fence | MemoryBarrier | memory scope, semantics mask, storage |
| compiler_fence | MemoryBarrier | invocation scope, semantics mask, storage; compiler ordering domain |
| control_barrier | ControlBarrier | latch, execution scope, generation, participants |
| control_memory_barrier | release MemoryBarrier, ControlBarrier, acquire MemoryBarrier as requested | separate execution/memory scopes, mask, storage, generation, participants |
| latch_create | LatchCreate | participants, generation, execution scope |
| latch_arrive / wait / signal | LatchArrive / LatchWait / LatchSignal | latch, generation; arrival count where applicable |
| latch_reset | LatchReset | latch, previous generation, new generation, participants |
| completion_wait | CompletionWait, with requested surrounding memory phases | handle, memory scope, semantics, storage |
| mutex_create / threading.Lock / threading.RLock | MutexCreate | execution scope, reentrant policy |
| mutex_acquire / recognized acquire | WorkerToken, MutexAcquire, MutexAcquired | handle, logical worker, blocking, timeout, acquire semantics, object scope |
| mutex_release / recognized release | WorkerToken, MutexToken, MutexRelease | handle, logical worker, held token, release semantics, object scope |
| mutex_locked / recognized locked | MutexLocked | handle |

Each projection carries the same exact SOURCE_SYNCHRONIZATION identity and its
own SSA_SYNCHRONIZATION receipt. Intermediate tokens are concorded. An explicit
`after` operand orders synchronization activity within a basic block; existing CFG
placement governs branches and loops. These operations are effect barriers to SSA
fusion. The resource-scope entry phase places generated acquisition before its body;
early-return cleanup retains release after acquisition.

Scopes include cross-device, device, workgroup, subgroup, invocation, queue-family,
shader-call, thread-region and object. Memory order is acquire, release,
acquire-release or sequentially consistent, with explicit memory-class,
availability, visibility and volatile bits. Invalid scope/order/mask combinations
are rejected. Split projections retain release flags before rendezvous and acquire
flags after it. Memory-only operations create no rendezvous. Completion wait is
not an all-participant barrier. Scopes are never promoted to global scope.

## Lock-box preparation

`IRModule.lock_box` is an SSALockBox view over the module's identity book.
LOCK_BOX_ENTRY records binding/layout policy; SOURCE_MUTEX_BINDING records exact
source receiver bindings. Captured mutex inputs follow SOURCE_CLOSURE_CAPTURE to
the parent's entry. Aliases do not create another lock. Unrelated same-spelled
methods are ordinary calls. Explicit unknown mutex parameters retain unknown
reentrancy rather than inventing a concrete allocation.

A descriptor is not a runtime allocation. Runtime lookup is keyed by the mutex
handle: repeated executions of a factory site require distinct rows, while aliased
handles require the same row. State columns are object_token, held_token,
owner_token, depth, generation, waiters, policy_flags and execution_scope.
WorkerToken denotes logical activity identity, not a hardware lane or OS thread.
Acquire produces an opaque held token and a separate Boolean success projection;
release explicitly retrieves the held token for that mutex/owner. Blocking calls
require communicating-task progress; nonblocking attempts do not require waiting.

## Remaining policy and lowering boundary

The next implementation must connect source worker bodies and region worker tokens
to outlined SSA activity, then lower these effectful operators to target primitives.
Standard `threading.Barrier.wait` still has source control projections with unresolved
memory policy; it is not automatically converted to the explicit intrinsic protocol.
The original worker-body/shared-cell SSA regression remains a strict expected failure.

The remaining backend policy table must map each (operation, execution scope,
memory scope, order, memory class, progress capability) to supported native mechanics
or an explicit refusal. It must define generation-safe latch transitions, release
publication and acquire observation, sequential-consistency domain, timeout/failure
results, mutex ownership and reentrancy, waiter wakeup, and destruction/lifetime.
Mutex table allocation and atomic transitions are not implemented here. Plain Python
Lock release ownership and RLock owner enforcement need distinct native policies.
No unsupported target scope may silently broaden. A serial lowering is eligible
only when its progress/dependency contract proves it safe; waiting for another
activity cannot become an ordinary blocking serial call.

Existing ASAP/ALAP scheduling remains configurable on the source region and adds
no comparative ordering between workers. Hardware lane choice, backend emission
and native execution are outside this slice. No Python executor or automatic lock
for shared storage was introduced.
