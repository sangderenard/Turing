# Common source threading contract and coordinated verification

This is a proposed contract and implementation sequence, grounded in the current
backend inventory. It adds no compiler operators or claims of runtime support.
See backend_concurrency_inventory_2026-10-31.md for current-source evidence.

## Separate two permissions

Authored threads require exact captures/shared storage, logical contexts, start,
completion/join and synchronization activity. Dispatch chooses where eligible
activity executes: serial, host pool, SIMD, compute grid or browser workers.
Dispatch does not make arbitrary communicating threads independent. A backend
being able to launch parallel work does not prove it can service a blocking wait.

There is no currently implemented common end-to-end source-threading subset across
all parallel targets: the worker SSA and synchronization lowering gaps remain.
The useful common *semantic* core below is a target contract, with explicit
capability refusal when a target cannot realize the requested scope/progress.

## Small common core

| Ability | Minimal semantics | Optional stronger facility |
| --- | --- | --- |
| Logical context | Separate private state and continuation; exact captured/shared storage | Physical concurrent execution |
| Activation/completion | Start eligibility, completion token, join dependency | Native task handle and asynchronous completion |
| Memory access ordering | Explicit storage, memory scope, order and effect dependencies | Target fence/visibility instructions |
| Atomic cell | Fixed-width integer load/store, exchange, strong compare-exchange; indivisible at declared scope | Native lock-free instructions, wider types |
| Atomic arithmetic | Fetch-add derived under same atomic-cell protocol | Native fetch-add and other RMWs |
| Waiting | Readiness/suspension without blocking all required progress | OS wait/notify, cooperative suspension, target dispatch continuation |
| Mutex | Object-keyed exclusion; try-acquire/release, explicit logical owner policy | Blocking acquire, timeout, recursion |
| Rendezvous | Explicit participant set/count, scope and generation | Native barrier/latch |

Use 32-bit integer cells as the first cross-target atomic width. Width, alignment,
storage class, scope and operation support are capabilities, not inferred from
thread count. No promise that device-wide mutexes or rendezvous work on every GPU.
GPU workgroup cooperation, dispatch completion and native communicating threads
remain distinct realizations. Copied browser tile storage is not shared memory.

One physical worker may service all logical contexts. Preserve their identity
where mutex ownership, joins and participation observe it. Static scheduling may
resolve waits where readiness is proved; otherwise retain cooperative progress or
refuse. Do not execute a blocking wait on the only worker that could release it.

## Atomics using mutexes

A lock-backed implementation is valid when every participating access to a cell
uses the same exact storage-identity/alias-resolved protocol. Under that lock,
load/store/exchange/compare-exchange/fetch-add have a single observable
linearization point. Compare-exchange returns observed value AND success;
fetch-add returns the old value. Fixed-width arithmetic must retain its declared
wrapping behavior. Strong compare-exchange avoids inventing spurious failure.

Mutex acquire/release must provide the requested publication/observation ordering
at the declared scope. Relaxed/acquire/release/acq_rel/seq_cst stay explicit in SSA;
a stronger lock-backed realization must be a concorded policy decision. Sequential
consistency needs a coherent order across the declared domain, not merely unrelated
per-object locks. A domain-wide service lock is a possible first policy, subject to
progress and scope legality. Mixed native and lock-backed operations require one
compatible protocol; never let native access bypass a fallback lock for the same
cell. Address reuse/lifetime and aliases must use canonical storage identity and
runtime handle/generation, not a source spelling or factory-site number.

Lock-backed atomics do not promise lock freedom, wait freedom, interrupt/signal
safety, or unrestricted GPU forward progress. They cannot implement a request
whose contract requires those guarantees. A spin loop is not automatically a
valid fallback. All memory participants must fit the supported scope; unsupported
scope is refused instead of broadened.

Existing AtomicCompareExchangeObserved/Success/Memory and
AtomicExchangeAddObserved/Memory in ssa_registry.py and machine_code_lifting.py
already preserve observed results and memory transitions for lifted locked x86
operations. Reconcile source atomics with those authoritative identities and
result/effect projections before extending the vocabulary. Do not lower a shared
RMW as an independently schedulable load followed by store. Source compare-exchange
will also need success/failure orders; failure cannot request release or acq_rel,
and the allowed pair must be validated under the selected memory model.

## Deques after atomics

Start with a correctness-first object-keyed mutex implementation of both-end
push/pop/try-pop, exact item/storage identity, explicit empty/full results and
fixed-capacity behavior. Add the authored Python deque API only with its exact
semantics: deque(maxlen) evicts at the opposite end on append; it is not a queue
that refuses full. pop/popleft on empty raises IndexError. A concurrent try API
must distinguish empty from a legitimate None payload. Compound checks such as
`if deque: deque.popleft()` are not made atomic by atomic individual methods.

Do not confuse an authored data deque with a dispatcher work-stealing deque.
Work stealing adds owner/thief rules, last-item races, publication and reclamation;
those can follow after the shared-container contract. Existing DeploymentFIFO
is a bounded Python single-reader helper (publish fails when full, consume returns
availability/value), not a concurrent deque or proof of SSA lowering. Deque3D is
a three-double ctypes structure, not a double-ended concurrent queue.

## Coordinated battery

One authored scenario should pass through source ingestion, dual IR, SSA, target
lowering and execution. Assertions belong at each boundary; a graph-only pass is
not a runtime pass. Reuse existing tests rather than constructing replacement IR.

| Layer | Existing coverage to retain | Next source-driven assertions |
| --- | --- | --- |
| Thread ingestion | test_thread_dual_ir_sync_contract, test_python_two_threads_shared_location | Worker bodies retained in SSA; independent activation state, exact captured alias, join live-outs |
| Dispatch policy | test_deployment_lowering/classification/stage, test_repository_ssa_dispatch, test_deployment_outlining/ssa_binding/native_emission | Same source semantics under each legal strategy; communicating work never receives an invalid independent-lane serial fallback |
| Synchronization SSA | test_source_ssa_synchronization, test_source_ssa_lock_box | Tokens receive target representation; source identity shared across projections and rewrites |
| Native runtime | test_turing_pool_runtime, test_deployment_host_pool | Source-to-runtime linking after worker SSA; serial cooperative and concurrent contexts preserve logical ownership |
| Atomic semantics | Existing binary lifting tests located by AtomicCompareExchange*/AtomicExchangeAdd* | Load/store order validation; CAS observed/success including failure; exchange/fetch-add old value; wrapping; aliases and distinct cells |
| Atomic concurrency | New authored producer/consumer and contended counter scenarios | No lost RMWs; release/acquire publication; SC multi-cell constraints; matched native and lock-backed realizations |
| Deque semantics | DeploymentFIFO only as its own helper contract | Both ends, wraparound, empty/full/eviction policy, None payload, aliases, lifetime |
| Deque concurrency | New authored producer/consumer scenarios | No lost/duplicate items; per-producer order; contention on final item; publication and reuse |
| Negative capability | Existing explicit refusal checks | Unsupported width/alignment/scope/progress, illegal memory orders, unavailable shared storage, unrelated same-spelled methods |

Run graph/SSA checks under both ASAP and ALAP on the same source. Neither may add
an authored comparative worker edge. Run common supported scenarios with one
physical worker and several when such realizations exist. Validate histories
against legal linearizations rather than asserting a particular race winner.
Use bounded native probes with diagnostic timeout handling; stress and litmus tests
supplement structural proofs and never establish a memory model by frequency.
A test skip names its missing toolchain/device/capability; aggregate separately:
source pass, SSA pass, emission pass, execution pass, capability refusal and skip.
Do not use expected failures or synthetic graphs as evidence of backend support.

## Implementation order

1. Close existing worker-body SSA and synchronization token-layout gaps; attach
   logical context and progress to existing dispatch seams.
2. Specify atomic operand/result/effect identities against the existing lifted
   projections; source-driven single-context SSA tests and invalid-order refusals.
3. Implement one native correctness path using the existing mutex/task ABI,
   with explicit storage and ordering domains; compile and execute authored tests.
4. Add supported GPU/Wasm/native atomic mappings with truthful scope/progress
   capabilities and equivalent tests. Dispatch-only targets retain their limits.
5. Compile mutex-backed deques, then specialize atomics/work-stealing only where
   measured need and target guarantees justify it.

No new executor, automatic locking of arbitrary unsynchronized accesses, or
backend mutation exception is part of this proposal. Every selected realization
and graph rewrite must be concorded to the source operation and capability policy.
