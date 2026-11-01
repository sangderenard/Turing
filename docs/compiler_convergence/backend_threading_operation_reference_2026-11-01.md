# Backend threading operation reference

This is the porting reference for the threading/synchronization set exercised
by the C Condition, latch, barrier and completion work. The tables describe the SSA contract; the C column
records the current implementation. Another backend may use native instructions,
runtime calls, or a sound composition of existing operations. It must preserve
the same state transitions, dependencies, scope, outcomes and lifetime.

There are **27 mapped synchronization primitives**, plus task lifecycle, fences,
exception records and ordinary CFG operations. The declarations and concordance
remain authoritative; this document does not introduce a second registry.

## Reading the tables

| Term | Contract |
| --- | --- |
| `worker` | Opaque logical activity/context identity, stable across suspension or hardware-worker migration. |
| `mutex`, `queue`, `latch` | Opaque runtime object identities. Repeated factory execution creates fresh instances. Aliases preserve the same instance. |
| `token` | `ssa.lock_token`: ownership generation for that mutex/context, not a Boolean acquisition result or a hardware lane. |
| `effect` | `ssa.sync_token` or the source contract's completion/None result. A sequencing carrier is not a substitute for the actual effect. |
| `scope` | Explicit execution or memory scope. Current host C mapping accepts invocation=4, thread-region=256 and object=257, subject to the operation's contract. |
| `order` | Explicit memory semantics. C mutex acquire/restore accepts acquire=2; release/suspend accepts release=4. Other orders require another mapping or refusal. |
| `timeout` | Numeric seconds. Mutex acquire uses -1 for indefinite acquisition; nonblocking acquisition requires the supported default timeout. Latch timed wait has a separate `finite` operand, preserving None/indefinite versus authored negative/poll. |
| `after` | Optional additional SSA operand carrying preceding activity. Omitted from signatures below. Keep its dependency, including effect Phis at merges. |

Use each instruction's declared result dtype and role list. Some straight-line
source operations expose a Python None/completion result where a compound
procedure uses an internal effect/status carrier. C's signed 64-bit handle and
token representations are an implementation choice, not a portable pointer ABI.

## A. Context identity and mutex state — 9 primitives

| SSA operation and operands | Logical result | Required transition / progress | Failure / normal absence | Current C selection |
| --- | --- | --- | --- | --- |
| `WorkerToken()` | `opaque_ref` context token | Read current logical context; no wait, rendezvous or user-memory fence. | Dispatch must establish an appropriate identity. | `turing_sync_worker_token` |
| `MutexCreate(execution_scope, reentrant)` | Mutex handle | Fresh state: unheld owner/token/depth, declared policy/scope. Default Condition also has empty notification links. | Allocation failure is an error, not a usable null mutex. | `turing_source_mutex_create` |
| `MutexAcquire(mutex, worker, blocking, timeout, memory_scope, memory_semantics)` | Ownership token; zero on unsuccessful acquisition | Enforce exclusion and reentrancy policy; acquire user-memory ordering on success. Blocking mode needs progress. | Try/timeout is normal unsuccessful acquisition; invalid policy/timeout or platform failure is an error. | `turing_sync_mutex_acquire` |
| `MutexAcquired(acquisition)` | `bool` | Decode acquisition success separately from ownership token. No wait or extra fence. | Does not turn a recorded failure into success. | Inline positive-token test (`token_status`) |
| `MutexToken(mutex, worker)` | Current `ssa.lock_token` | Validate/read that context's current ownership generation. No acquisition. | Missing ownership is an error for the procedures that require it. | `turing_sync_mutex_token` |
| `MutexRelease(mutex, token, worker, memory_scope, memory_semantics)` | Completion/effect | Validate token/owner; release one acquisition. On final release clear ownership and permit acquirers to proceed. Release memory ordering. | Invalid ownership/token is an error. | `turing_sync_mutex_release` (C argument order is mutex, worker, token) |
| `MutexLocked(mutex)` | `bool` | Read held state without acquiring or rendezvousing. | Unlocked is normal false; invalid handle is an error. | `turing_sync_mutex_locked` |
| `MutexSuspend(mutex, worker, token, memory_scope, memory_semantics)` | Saved full depth, `int64` | Validate ownership, save depth, release all recursive acquisitions; release ordering. | Invalid ownership/state is an error. One ordinary release is not equivalent. | `turing_sync_mutex_suspend` |
| `MutexRestore(mutex, worker, depth, memory_scope, memory_semantics)` | Current `ssa.lock_token` | Reacquire without a timeout, then restore saved depth; acquire ordering. Needs progress if another context holds the mutex. | Invalid restoration/platform failure is an error. A prior wait timeout does not skip restoration. | `turing_sync_mutex_restore` |

The source Lock/RLock lowering uses Turing's existing token-ownership policy.
A backend must not silently substitute a different ownership/reentrancy policy.
Broader Python plain-Lock cross-owner-release behavior is a separate policy
question, not established by these Condition tests.

## B. Guarded waiter queues — 5 primitives

These queues carry wait registrations/latch identities. They are not the
general byte FIFO/deque API.

| SSA operation and operands | Logical result | Required transition / synchronization | Failure / normal absence | Current C selection |
| --- | --- | --- | --- | --- |
| `WaitQueueCreate(mutex, execution_scope)` | Fresh queue handle | Empty FIFO referring to the exact supplied guard. Does not allocate a replacement mutex. | Allocation/invalid guard is an error. | `turing_source_wait_queue_create` |
| `WaitQueueMutex(wait_queue)` | Exact guard handle | Resolve association only; does not acquire. Default Condition's combined row may resolve to itself. | Invalid handle is an error. | `turing_sync_wait_queue_mutex` |
| `WaiterRegister(wait_queue, worker, token, latch)` | Status/effect carrier; positive success | Validate guard ownership and append one registration while the guard is held. Publish before suspend/unlock. | Invalid ownership or already-registered latch is an error. Failed registration must not lead to parking. | `turing_sync_waiter_register` |
| `WaiterSelect(wait_queue, worker, token)` | First pending latch handle, or null | Under the guard, choose the first unsignaled registration in FIFO order. Selection itself does not signal or release the guard. | Empty pending set is normal null. Ownership validation precedes selection in the composed procedure. | `turing_sync_waiter_select` |
| `WaiterRetire(wait_queue, worker, latch)` | Completion/effect | Under the same guard, unlink this exact registration, preserving other rows and head/tail state. | Wrong registration/ownership is an error. | `turing_sync_waiter_retire` |

Two queues sharing a mutex share exclusion, not notifications. A signal cannot
be deposited into an empty queue for an unregistered future waiter.

## C. Retained latch state — 7 primitives

| SSA operation and operands | Logical result | Required transition / progress | Failure / normal absence | Current C selection |
| --- | --- | --- | --- | --- |
| `LatchCreate(participants, generation, execution_scope)` | Latch handle | Fresh generation/count/release state; zero participants is initially ready. | Invalid count/generation or allocation failure is an error. | `turing_source_latch_create` |
| `LatchSignal(latch, generation)` | Completion/effect | Retain release for exactly this generation and wake its current waiters. Independent of arrival count. No requested user-memory fence. | Wrong generation/handle is an error. | `turing_sync_latch_signal` |
| `LatchWait(latch, generation)` | Completion/effect | Wait indefinitely for this generation's release; an earlier signal remains observable. Needs progress unless already ready. | Wrong generation/handle or platform failure is an error. | `turing_sync_latch_wait` with indefinite mode |
| `LatchWaitTimeout(latch, generation, timeout, finite)` | `int64` outcome: released, timeout, error | Retained wait with explicit finite mode. C logical status is 1/0/negative. No requested user-memory fence. | Timeout is normal, not an exception. Invalid/overflowing timeout or platform failure is an error. | `turing_sync_latch_wait` with supplied timeout/mode |
| `LatchDestroy(latch)` | Completion/effect | Exclusive destruction after registration retirement and all uses. | An in-use/registered latch cannot be freed. | `turing_source_latch_destroy` |
| `LatchArrive(latch, generation, count)` | Completion/effect | Atomically subtract a positive count, reject over-arrival/stale generation, release and wake at zero. No implicit requested user-memory fence. | Invalid count/state is an error; no underflow or wrapping count. | `turing_sync_latch_arrive` |
| `LatchReset(latch, previous_generation, generation, participants)` | Completion/effect | Require the exact completed previous generation and a strictly newer generation. Rearm count/release state; retain old-generation completion for waiting and later old-generation waits. Zero participants is ready. | Incomplete/mismatched/reused generation, registration or allocation failure is an error. | `turing_sync_latch_reset` |


Explicit latch generations retain their completion records until scope disposal;
a backend cannot silently recycle them while an old-generation wait is legal.
The cyclic source barrier below instead retires an epoch after its last waiter
leaves. Both use the same native count/epoch state implementation.

### Cyclic source barrier — 4 primitives

| SSA operation and operands | Logical result | Required transition / progress | Failure | Current C selection |
| --- | --- | --- | --- | --- |
| `BarrierCreate(participants, timeout, finite, execution_scope)` | Opaque barrier handle | Allocate positive participant count, initial generation and default timeout. Scope is explicit host object scope in this source policy. Owned by the joined activation lifetime. | Invalid count/allocation failure is reported through the existing error channel. | `turing_source_barrier_create`, shared latch allocation/state |
| `BarrierWait(barrier, timeout, finite)` | Arrival index, integer in `[0, participants)` | Count one arrival, park until the cohort completes, advance generation once, preserve release for old waiters, and permit repeated phases. An explicit timeout overrides the default; None/no override uses the constructor default. Requires concurrent progress. | Timeout/abort/reset faults the affected waiting epoch with `threading.BrokenBarrierError`; invalid numeric timeout/overflow/allocation/platform errors use checked failure arms. | `turing_sync_barrier_wait` |
| `BarrierAbort(barrier)` | Source None/effect | Mark broken and wake current and still-parked cyclic participants. Future waits fail until reset. | Invalid handle/state is an error. | `turing_sync_barrier_abort` |
| `BarrierReset(barrier)` | Source None/effect | Break an incomplete old generation, preserve its waiters' failure, and admit a fresh generation. Completed epochs remain released unless aborted. | Generation overflow/allocation/invalid state is an error. | `turing_sync_barrier_reset` |

Python `threading.Barrier` construction and these methods become typed process
operations at ingestion, including captured and aliased barriers. No Python
callee survives for their implementation. `wait` composes release fence ->
`BarrierWait` -> acquire fence under one source contract, preserving its arrival
index separately from the final effect token. A non-None action callback still
requires a compiled callback ABI and is explicitly refused; it is not called
through Python. The supported source methods here are wait/reset/abort.

## D. Task lifecycle and dispatch integration

| SSA operation / service | Result | Backend obligation | Current C implementation |
| --- | --- | --- | --- |
| `ThreadCreate` with exact target/context operands | Task handle (`opaque_ref`) | Allocate an instance/context from its body-table identity; preserve actuals, captures, storage and failure packet. Creation does not start activity. | `turing_source_thread_create`, generated context/trampoline, `C_THREAD_ACTIVATION` |
| `ThreadStart(handle)` | Source None | Start that instance with concurrent-progress semantics. No ordinary blocking serial-call substitute. | `turing_source_thread_start` |
| `ThreadJoin(handle)` | Source None | Wait for that task's completion under the supported lifecycle policy; preserve repeated-join behavior and worker failure reporting. Not an all-participant barrier. | `turing_source_thread_join` |
| `ThreadIsAlive(handle)` | `bool` | Observe task lifecycle without joining. | `turing_source_thread_is_alive` |
| Install/clear current context — runtime service, not a new SSA opcode | Context state | Bind logical worker identity around dispatched activity; retain context across suspension/migration. | Generated trampoline calls `turing_sync_set_worker_context`; reuses management-table instance IDs |
| Capture/extent ABI — runtime service | Borrowed/copied fields | Honor exact argument/capture roles, types and joined lifetime. Do not derive identity from spelling or copy mutable storage as a scalar. | Existing context bindings; `C_THREAD_EXTENTS` borrows invocation extent ABI within current capture limits |
| Allocation-scope teardown — runtime service, not `MutexDestroy` SSA | Completion/failure | Finish outstanding captured activity before freeing reachable state. Retire waiters before latch disposal. Reject missing ownership-transfer contracts. | Extended `turing_thread_scope`, `turing_thread_scope_clear` and registered native resource disposal |

The four task operators and twenty-seven synchronization primitives are distinct
from compiler-generated numerical `Deploy`/`Join` work. Their progress and
ownership contracts must not be exchanged merely because a target has a pool.

## E. Memory ordering and rendezvous

| SSA operation / source form | Operands and semantics | Current C status |
| --- | --- | --- |
| `MemoryBarrier` in memory-ordering domain | `memory_scope, memory_semantics, storage`; order selected memory effects without making workers rendezvous. | `atomic_thread_fence`; host-local scopes and plain acquire/release/acq_rel/seq_cst orders only |
| `MemoryBarrier` in compiler-ordering domain (`compiler_fence`) | Invocation scope, semantics, storage; compiler reordering constraint. | `atomic_signal_fence`; supported plain orders only |
| `ControlBarrier` | `latch, execution_scope, generation, participants`; rendezvous constraint, not an implicit memory fence. | `turing_sync_control_barrier`; shared counted latch state and native monitor wait |
| Combined control/memory barrier | Requested release phase -> control rendezvous -> requested acquire phase, one exact source concordance identity. Separate execution/memory scopes and storage operands. | Declared release fence -> `turing_sync_control_barrier` -> acquire fence, preserving explicit requested scopes/orders |
| `CompletionWait` | Wait for one handle; source composition supplies explicitly requested surrounding memory phases. | `turing_source_thread_join` for an exact source thread handle, with the requested fence phases. Other handle families need their own target mapping. |
| `LatchArrive(latch, generation, count)` | Reduce remaining participant count; release at readiness. | `turing_sync_latch_arrive` |
| `LatchReset(latch, previous_generation, generation, participants)` | Rearm a completed generation while retaining old waiters' release semantics. | `turing_sync_latch_reset` |

Backend policy must select from **operation, execution scope, memory scope,
order, memory class, progress and lifetime**. Unsupported combinations require
an explicit refusal. A local primitive cannot stand in for a device/global
barrier, and detecting shared storage does not add synchronization.

## F. Failure records and ordinary SSA support

| SSA operation / support | Required behavior | Current C route |
| --- | --- | --- |
| `ExceptionNone` | Absence of an active exception record. | Native null record identity |
| `ExceptionMake([message])` | Construct declared code/type/message payload with provenance. A nonliteral message is an explicit dependency operand. | Static messages: `turing_error_packet_init` and native packet storage. Dynamic messages: refused pending an owned native message ABI. |
| `ExceptionTake` | Transfer pending current-context failure into a record; clear the active channel appropriately. | `turing_validation_error_take_active` |
| `ExceptionActive(record)` | Test presence/active code. | Native presence/code test |
| `ExceptionMatch(record)` | Test the declared supported exception class/hierarchy. | `turing_error_packet_matches` |
| `ExceptionRaise(record)` | Propagate the record along the selected exceptional/fault continuation. | `turing_validation_error_raise_packet`, existing SSA fault exit |
| Worker-error delivery — runtime service | Keep worker identity/function and error payload distinct from parent active failure. Deliver by the supported join/shell policy. | Existing worker packet and shell reporting channel |
| `ExceptionIndexType`, `ExceptionLength` | Basic protected-index type/length checks when the source procedure needs them. | Existing basic-type and bounded-sequence descriptor lowering |
| `Const`, `Ne`, `Gt`, `Sub`, `Phi`, `Br`, `CondBr`, `NoneValue` | Actual procedure data/control graph. Preserve guards, loop counts, predecessor-specific values and cleanup continuations. | Existing ordinary SSA/C lowering |

Raw C primitive status conventions are not the portable exception ABI. C's
acquire/wait helpers use -1 for runtime failure, -2 for timeout overflow and -3
for invalid timeout; ownership probes/suspension also have zero failure cases.
The emitter records named failures in the existing channel and exposes the
declared SSA result. Preserve **success, ordinary false/timeout/empty, and error**
as separate outcomes on every backend. A target's native status codes may
differ, but lowering must normalize them to the SSA sign/zero/Boolean contracts
before feeding the shared procedure's comparisons and Phis.

## G. Compositions every equivalent backend must preserve

| Source operation | Shared SSA composition / required edges |
| --- | --- |
| Default Condition | Fresh recursive `MutexCreate` with empty notification state. |
| Condition with supplied lock | Fresh `WaitQueueCreate` referencing that lock; separate queues remain separate. |
| Acquire / release | Context identity, exact guard resolution, existing ownership-token protocol and explicit acquire/release ordering. |
| Wait | Validate -> allocate latch -> register -> suspend full depth -> wait -> restore -> retire -> destroy -> expose outcome. Ownership/status/effect Phis retain the actual failure/cleanup edges. |
| Notify(n) | Validate -> count Phi -> select pending waiter -> signal -> decrement/backedge; stop at n or empty queue. |
| Notify-all | Same select/signal loop until empty pending set. Keep the guard held. |
| Resource scope | Declared entry/body/cleanup topology. Cleanup on normal exit, break and return; supported exceptional paths capture body/cleanup errors. |

A backend may lower this shared graph to its own facilities. It must not
reconstruct these procedures from Python AST, keep an opaque Python callee,
or independently invent another Condition policy. Sound specialization or
graph rewriting must itself be concorded.

## H. Port record and acceptance evidence

For **each operation in A-F**, record these columns in the backend's mapping
table. This is the common comparison format for subsequent ports.

| Required column | What to record |
| --- | --- |
| SSA operation and contract revision | Exact registered operator/roles and source/projection identity. |
| Target mapping | Instruction, runtime ABI symbol, or shared SSA expansion; no assumed host fallback. |
| Supported scopes/orders/classes | Exact accepted set and explicit refusal for the rest. |
| Representation/state ownership | Handle/token/depth/generation representation, table association and allocation lifetime. |
| Progress mechanism | Native concurrent task, target subgroup/workgroup participation, or implemented resumable-context scheduling. |
| Failure/timeout mapping | Normal unsuccessful outcomes, recorded errors and exceptional continuation. |
| Concordance writer/reader | Source, SSA, policy and emitted-unit receipts; audit coverage for any rewrite. |
| Evidence/status | Source test, compiled artifact, target/platform and result. Distinguish declared, mapped, compiled and executed. |

| Backend group | Reference status for this operation set |
| --- | --- |
| C | Concrete reference mapping above; standalone source execution and concordance audited on Linux/GCC. Windows code paths exist but were not executed in this work. |
| LLVM | Equivalent port/evidence still required. A compatible native runtime call is a possible mapping; merely representing an opcode is not execution support. |
| Fortran | Equivalent port/evidence still required; shared native ABI calls are an option if its contracts apply. |
| Wasm / JavaScript | Equivalent port/evidence still required, including the distinction between a blocking worker context and a host context that cannot park. |
| WebGPU / GLSL / native_glsl / SPIR-V / WebGL | Equivalent supported subset and refusals still required. Local shader cooperation does not establish arbitrary host-Condition or global-rendezvous support. |

These rows are a porting worksheet, not a newly measured inventory of other
backends. Consult the [backend concurrency inventory](backend_concurrency_inventory_2026-10-31.md)
for the wider existing machinery. A serial backend may elide/prove-ready work
or use implemented resumable contexts; blocking a serial context whose peer
must run is not a valid fallback.

Acceptance uses the **same authored source through the public compiler API**:
default/shared locks, recursive restoration, early notification, finite/None/
negative timeouts, notify-one bound, queue isolation, ownership failures,
try/except, loops and resource cleanup. Run ASAP and ALAP, verify required edges
without adding comparative worker order, compile/run target artifacts and audit
before/after. Report unsupported cases. General atomics, events, byte deques,
barrier arrival/reset and broader lifecycle operations are not established by
having the Condition subset.

## Implementation links

- [SSA vocabulary](../../src/transmogrifier/ssa_registry.py) and
  [arity declarations](../../src/transmogrifier/operator_defs.py).
- [Source contracts/projection writer](../../src/compiler/synchronization.py),
  [shared Condition procedures](../../src/compiler/condition_ssa.py),
  [lock-box binding](../../src/compiler/lock_box.py).
- [C policy mapping](../../src/compiler/c_synchronization.py),
  [C emitter](../../src/compiler/ssa_c_backend.py),
  [native primitive ABI](../../src/common/tensors/accelerator_backends/c_backend/turing_pool.h),
  [native task/scope ABI](../../src/common/tensors/accelerator_backends/c_backend/turing_thread_table.h).
- [Concordance declarations](../../src/compiler/concordance_declarations.py) and
  [audit](../../src/compiler/identity_concordance.py).
- [Standalone source tests](../../tests/test_source_c_conditions.py),
  [Condition composition design](condition_ssa_composition_2026-11-01.md),
  [measured implementation receipt](../hazards/20261101T140315Z_work.md).

The [structured dual-IR region contract](thread_structured_dual_ir_regions_2026-11-01.md)
defines the ingestion-level host `threading.Barrier.wait` participant/effect
policy and phase projections. The native C mappings in sections C/E now
implement that host policy for the supported source methods.

## Hardware-specific operator requirements

These operators are registered in `ssa_registry.Handler` and `operator_defs`;
the latter marks substrate signatures `hardware_specific` and `effectful`.
They are not numerical lane-selection instructions. Each backend must provide
a faithful native mapping, a concorded SSA composition, or an explicit refusal.

| Operation family | Required target facility | C mapping | Porting constraint |
| --- | --- | --- | --- |
| Context/mutex/waiter queues | Stable logical context, exclusion, checked state and a wake mechanism | Existing native monitor/condition state | An implementation must preserve token ownership and recursive suspend/restore. |
| Latch arrival/signal/reset | Atomic count/generation state plus retained completion | Same native latch monitor and epoch records | Do not recycle a generation still observable by a valid waiter. |
| Control/cyclic barrier | Concurrent progress, counted rendezvous, wake all and phase isolation | Same count/epoch engine, native parking | A plain serial call cannot wait for a peer it prevents from running. No implicit GPU/global scope widening. |
| Memory/compiler fence | Target memory/compiler ordering instruction or faithful runtime boundary | C11 `atomic_thread_fence` / `atomic_signal_fence` | Accept only mapped scopes/orders/classes. Monitor internals can be stronger than the requested order but do not enroll extra graph participants. |
| Completion wait | Completion of a proven handle kind and its resource/error lifetime | Existing source-thread join and worker packet delivery | Completion is not an all-participant barrier. A raw opaque value is not proof of a thread handle. |

Current C execution evidence is Linux standalone compilation/execution. Windows
uses the existing platform monitor abstraction but was not executed here;
other backends remain unverified ports. See the latest native barrier hazard
receipt for the actual test results and explicit source API limits.

C cyclic barrier counts must be positive and fit `INT64_MAX`. An invocation-scope
control barrier requires exactly one participant; object/host scopes retain
their explicit cohort. Unsupported workgroup/device scopes are refused.
The [native implementation receipt](../hazards/20261101T165117Z_work.md)
records the source identities, graph measurements and final regression results.

## Whole control surface and deployment types — inspection 2026-11-01

Every backend must account for these families by native emission, concorded
composition, or a precise refusal. This is a porting obligation, not a claim
that every backend currently implements every family.

| Family | Registered operations / existing structure | Backend obligation |
| --- | --- | --- |
| Basic control | Phi, Br, CondBr, IndirectBr, Call, Ret, Trap | Exact carried values, branch targets, predecessor copies, call/return ABI and nonreturning traps. |
| Structured control | Sequence, conditional, counted/while loops, break/continue, state-machine cases, recursion regions | Consume ingested ControlProgram and its lowered CFG; these are not additional SSA opcodes. |
| Exceptions and cleanup | ExceptionNone/Make/Take/Active/Match/Raise/IndexType/Length; NativeTryBlock, ResourceScopeBlock, ValidationBlock | Checked operator failure arms, ordered matching, propagation, else/finally and cleanup continuations; preserve records rather than replacing handled errors with termination. |
| Authored tasks | Dispatch, ThreadCreate/Start/Join/IsAlive; management-table bodies/captures | Separate logical contexts, exact storage/capture identities, progress, completion and error lifetime. |
| Deployment permission | Deploy, Join | Validated frame, iteration space, live-ins/live-outs, join mode, serial or supported parallel strategy. |
| Exclusion and wait state | The 27 mapped primitives in A–E | Ownership, wait registration, retained generations, cyclic phases and completion; no extra source ordering. |
| Memory ordering | MemoryBarrier (compiler or memory domain) | Explicit scope, order and storage classes, distinct from rendezvous. |
| High-level coordination | ConditionCreate/Acquire/Release/Wait/Notify/NotifyAll; EventCreate/Set/Clear/Wait/IsSet | Condition already has shared SSA composition; registration alone is not evidence of Event backend support. |
| Atomic data effects | AtomicCompareExchangeObserved/Success/Memory; AtomicExchangeAddObserved/Memory | Existing registered projections; each backend needs a faithful indivisible operation or sound mutex composition, with exact storage and memory effects. Registration does not establish lowering support. |
| Queue transport | ByteQueueCreate and queue access procedures | Preserve payload bytes, element/length metadata, endpoint state and synchronization. A general deque surface is not established by one factory opcode. |
| Shell-facing control effects | StreamPublishBlock, external-reference calls, sequence mutations/queries, scalar field writes | Declared ABI, effects, bounds, lifetime and error delivery; these require the applicable data operators as well. |

Pause/resume, cancellation/stop, a complete atomic load/store/RMW API and the
full deque API are remaining design/coverage work, not registered lifecycle
features inferred from the above list. For each port, inventory its actual
operations and compositions rather than assuming a language-level equivalent.

Deployment already distinguishes three independent dimensions:

| Dimension | Existing alternatives | Meaning |
| --- | --- | --- |
| Join mode (`deployment_frame.py`) | barrier, indexed, product, reduce | Completion only; indexed results; product-shaped results; declared reducer. These are contracts, not four proven parallel implementations on every backend. |
| Execution class (`deployment_classification.py`) | graphics-output, shader-compute, thread-workers, host-linear | Placement eligibility, with evidence/reasons; thread-worker classification requires barrier-lane membership and pointwise extent effects. |
| Lowering strategy (`deployment_lowering.py`) | serial, pool, dispatch, simd | Actual target execution mechanism. Profiles gate available strategies and parallel join modes. |

Framing is not a new numerical type system. Binding gives Deploy an
`ssa.deploy_token` and Join reads that token plus lane live-outs. Values inside
the frame retain their ordinary SSA dtypes. Fortran consumes shared
`dtype_layout`/`fortran_kind_table`; its numerical ABI is distinct from
opaque worker, mutex and error-record ABIs. Floating reductions retain the
explicit `allow_reassociation` policy; recognizing an operator name does not
permit arbitrary floating-point reassociation.

Observed Fortran path: the declared profile is serial-only; the SSA emitter
keeps the numerical body and prints Deploy/Join as boundary comments, with
Phi copies on predecessor edges and ordinary conditional/unconditional branches.
It also has a separate structured control renderer. It can therefore form
useful serial deployments without native thread support. The profile's note
about future C-shell pool integration is not evidence that this launch route
already exists. ISO_C_BINDING/native wrapper integration is a sensible next
port boundary, subject to exact capture/buffer ABI, reentrancy, error and lifetime
contracts; no Fortran pool implementation was added by this inspection.

Serial deployment is not permission to erase authored communicating threads.
A worker parked at a rendezvous must not prevent its peer from progressing.
A resumable cooperative lowering would need explicit continuation/state support;
otherwise such authored work needs native workers or refusal.

Proof limit observed: `deployment_ssa_binding.analyze_deployment_dataflow`
checks sibling SSA value uses. That check alone is not an alias/effect proof
for shared stores or opaque calls. Outlining has separate shared-store/sequence
and loop-shape checks. Preserve those gates and the upstream eligibility
contracts; do not generalize the binding check into a shared-storage-thread
serialization proof. Join aggregate result/destructuring and some lane live-outs
remain explicitly documented future work in binding/outlining.

Inspection validation: deployment lowering/classification/SSA binding and
Fortran control-target suites: 35 passed, 1 skipped (no Fortran compiler),
1 optional nodus warning. This verifies those contracts/renderers; it does not
claim a newly compiled or parallel Fortran deployment.

## Optional C orchestration for LLVM and Fortran AOT products

`c_shell_control=CShellControl(...)` now approves the existing C control/runtime
around explicitly selected scalar numerical SSA functions emitted as LLVM or
Fortran. The product is a CModuleArtifact and standalone executable; this is
wrapper support, not new native LLVM/Fortran thread/exception opcode coverage.
See [the section ABI and exact limits](c_shell_control_native_sections_2026-11-01.md).
Native coarray/multi-lane/loop-dispatch mappings and finer subgraph division
remain future capability refinements of these same operations.

General C SSA emission now honors `deployment='auto'` for a narrow discovered
source-SSA case: a bounded unit-stride loop with direct per-index writes into
one rank-one span, scalar numerical callees and no shared reads/arbitrary
calls. It reuses the existing iteration outliner and native pool. Discovery,
subgraph outlining and the C worker-budget selection are separate concordance
facts. The `deployment_cores` C-emitter option is the TOTAL execution-slot
budget; two means one background pool worker plus the participating caller.
This is not inferred synchronization for authored threads. Full-source Linux
standalone tests assert actual OS thread counts for explicit workers, mixed
LLVM/Fortran sections and passive deployment; serial and shared-read controls
remain one-thread executions. Other discovery shapes are not claimed.
