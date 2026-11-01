# Unified source-built threading runtime

## Direction and status

Superseded requirement: the user subsequently clarified that source-built Python
procedures are optional. Compiler-owned SSA and native C runtime procedures are
authorized and preferred for the initial C slice. The ownership, identity and
primitive contracts below remain useful; source construction is not a gate.

The intended system is a source-built threading core, analogous to compiling
Token Orchard methods, not a growing set of backend-specific implementations
of Python threading. Ordinary Python runtime procedures must become ordinary
ProcessGraph bodies, ControlProgram operations and SSA procedures. The target
provides a small set of primitives and may specialize a larger procedure when
the capability contract permits it.

This document coordinates the intended wider core. Source worker bodies and
their captured contexts now reach standalone compiled C. The basic Condition
protocol is composed from mutex, waiter-queue and latch SSA operations; see
[Condition composition](condition_ssa_composition_2026-11-01.md) for the measured
implementation and remaining limits. `synchronization.py` source declarations
remain compilation intrinsics, not a Python executor. SSALockBox is the
concorded layout/binding view; native activation scopes own allocated state.
General source atomics, deques, barriers and other backends are not established
by this C slice.

## Existing source-build route to reuse

`lower_ast_source_to_ssa` is the sanctioned entry. ProcessGraph source pursuit
resolves exact Python callables using the ExtractionContract and FunctionTable.
Token Orchard tests use `ExtractionContract.with_sources` to admit authored
implementations, preserving methods, internal calls, mutable fields and returned
storage. The dynamic glyph formatter route in fortran_c_shell.py also admits
ordinary Python helper definitions before pursuit, with exact module/qualname,
_python_bindings and private exports. Neither route requires hand-built SSA.

Python identity programs and lower_python_threading already select recognized
thread/synchronization operations. Existing source and SSA synchronization pages
retain exact contracts and projection receipts. Extend this route to call the
source runtime procedures: preserve source call identity, bind a distinct exact
implementation callable, pursue its body under the same compilation book, and
record the substitution. Do not replace the authored identity with an unrelated
callee name. Do not compile the current RuntimeError marker bodies as a runtime.

The thread-runtime source module must be admitted as an implementation source,
not copied into user files or executed by a new Python interpreter. Source
procedures use existing record/sequence/storage machinery, including caller-owned
runtime state passed through explicit SSA arguments. Compiling a procedure only
when demanded is compatible with having the full runtime contract available.

## One application runtime root, related tables

“One thread table” means one runtime ownership/lifetime root. It need not mean one
wide physical row for every unrelated object. Child tables have different instance
cardinality and retain their exact relationships; target layout may change them
without changing identities. No backend-specific Python object becomes authority.

| Logical child | State owned by the application runtime | Authoritative compile-time relationships already present |
| --- | --- | --- |
| Contexts | logical context token, activation generation, entry/frame, continuation, run/wait/completed state, result/error | SOURCE_THREAD_REGION and MEMBER; exact create/start/join/target and captures |
| Wait registrations | waiting context, condition/object token, requested generation, resume continuation, result/timeout | SOURCE_SYNCHRONIZATION and source control/dependency edges |
| Lock box | runtime mutex handle/generation, held token, logical owner, depth, waiters, policy, execution scope | LOCK_BOX_ENTRY and SOURCE_MUTEX_BINDING |
| Latches/events | object token/generation, required/remaining participants, released generation, event predicate, waiters | SOURCE_SYNC_OPERATOR, CONTROL_RENDEZVOUS and SOURCE_SYNCHRONIZATION |
| Atomic storage | exact backing storage identity/address and lifetime, width/alignment, scope, ordering domain, selected fallback lock | Existing atomic result/memory projections; source bindings still to implement |
| Deques | object token/generation, element storage and lifetime, head/tail/count/capacity, policy and protecting synchronization | Existing sequence/storage identities; deque source runtime still to implement |

The runtime root owns allocation and destruction dependencies. A source factory
site is not a runtime instance. Repeated creates instantiate fresh objects;
aliased handles and captured references address the same state. Deferred work
keeps captures, table rows and synchronization objects live until its final use.
Parent activity has a logical context too; its identity is not a hardware lane.

Do not duplicate concordance facts into an independently mutable registry.
Compile-time views consume existing cells; additional source-procedure binding,
layout, demand and policy decisions require declared concordance facts when
implemented. Application runtime rows are actual data, built by compiled source.

## Source procedure inventory and primitive boundary

Names below specify roles, not newly declared compiler opcodes.

| Source-built procedure family | Ordinary source work | Primitive/target obligation |
| --- | --- | --- |
| Create/start/complete/join | Allocate and initialize context/frame; lifecycle state; completion/result publication; register/resume waiters | Activation/suspension/progress service, ordered publication |
| Mutex/RLock | Handle lookup, ownership/depth policy, try/failure behavior, wait registration, release/wakeup | Atomic state transition or target mutex, scope-correct acquire/release |
| Condition | Ownership validation, full recursion-depth release/restore, notification and waiter policy | Atomic unlock-and-register/wait protocol; no lost wakeup |
| Event | Set/clear predicate, broadcast and future-wait policy | Atomic predicate/wait registration and publication |
| Latch/barrier | Count/generation update, arrival, readiness, release, reset and wakeup | Atomic transition; progress; requested memory phases remain separate |
| Atomic cell | Type/order validation, lock-backed algorithms, observed/success/value results | Compatible shared-storage lock or native atomic operation; coherent SC domain |
| Deque | Both-end indexing, capacity/empty/eviction rules, element lifetime, protected operations | Existing storage operations and declared mutex/atomic protocol |

Source algorithms must expose an indivisible transaction boundary for shared
state. Compiling a Python read/modify/write sequence does not make it atomic.
Similarly checking a predicate then registering a waiter is not sound if a
notification can occur in between. The primitive boundary must cover that race;
it is not enough to add a fence afterward. Backend swaps may implement the whole
transaction using a native primitive, retaining the same exact source contract.

Blocking methods compile to readiness/continuation activity or native task waits
with sufficient progress. They are not polling Python functions run synchronously
on the compiler host. A serial target schedules another ready logical context;
a parallel target may block an OS task if progress is guaranteed. Unsupported
scope, external participation or progress requirements must be explicit refusals.

## Ingestion and implementation binding

| Authored operation | Required implementation selection |
| --- | --- |
| threading.Thread constructor/start/join | Context allocation/activation/completion procedures, exact target/capture frame |
| threading.Lock and RLock methods/context scopes | Shared lock-box procedures with distinct owner/reentrancy policies |
| threading.Condition methods/context scopes | Condition procedures sharing the exact underlying lock, not an implicit different lock |
| threading.Event methods | Event predicate/waiter procedures |
| threading.Barrier constructor/wait | Generation-aware barrier procedure; explicit Python memory policy projected with the control operation |
| Explicit synchronization intrinsics | Same runtime procedures or primitive leaves according to declared contract |
| Source atomic API | Existing atomic identities extended with exact storage, scope and legal success/failure orders |
| collections.deque and selected methods | Exact deque API procedures; preserve maxlen eviction and empty exceptions |
| Unrelated same-spelled methods | Ordinary authored calls, unchanged by threading recognition |

Known identity from pursuit/capture is mandatory. Constructor spelling alone,
a variable called mutex, or a method called wait/pop does not select a runtime
procedure. Existing explicit intrinsic and standard-library entry paths must
converge on the same implementation and state, rather than two runtimes.

## Backend compromise contract

The SSA contract is complete before backend selection. Lowerings may choose
native operations, lock-backed algorithms, cooperative serialization, copied
independent deployments, or dispatch continuations only where they preserve the
requested semantics. Copies cannot replace shared communicating storage.
A physical-thread identity cannot substitute for logical RLock ownership when
multiple contexts share that physical worker. Sequential consistency cannot be
reduced to uncoordinated per-object locks. A GPU workgroup barrier cannot become
a device-wide rendezvous by renaming its scope. Lock-backed atomicity is a
supported compromise only when lock-free progress was not required.

## Concrete next implementation boundary

Build one source runtime module with the application root and context/lock/latch
state records, admitted through normal source pursuit. First close worker-body
SSA retention and give context/effect/held tokens their target representation.
Compile and execute the create/start/complete/join and plain mutex vertical slice
using existing C runtime leaves. Confirm exact Python operation-to-procedure
bindings and shared table storage with source-to-native tests and concordance.
Then implement source atomics against existing atomic projections, condition/
event/barrier transitions, and deques. Retain the full schema and callable contract
throughout; do not install empty procedures that appear to implement later parts.

Use common_threading_contract_and_test_matrix_2026-10-31.md as the coordinated
battery. Add checks that source methods are present as actual SSA procedures,
that calls reach them, and that aliases/captures share allocated runtime state.
An intrinsic marker or a table descriptor is not sufficient evidence. Backend
specialization tests compare the same source runtime scenario and preserve the
same source/implementation/policy concordance trail.
