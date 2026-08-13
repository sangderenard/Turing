# Automated system-boundary binding and event-capture prospective

Date: 2026-08-13
Context: `compile()` host-library extraction and the remaining external/indirect
machine-state surface

## Direct answer

Yes: the compiler can accurately notice each irreducible system interaction and
route that exact occurrence either to a real operating-system capability or to
a durable event stream. The correct design is not “call unknown Python” and not
“pretend the system call succeeded.” It is an explicit, typed suspension and
completion protocol carried by the compiled program.

The automated decision at each reached boundary should be:

1. **Internalize it** when readable code and machine semantics are available.
2. **Execute a deterministic virtual capability** when the system behavior is
   intentionally represented by the program's virtual filesystem, registry,
   environment, devices, threads, clock, or loader state.
3. **Replay an exact prior completion** when its descriptor, input read-set, and
   environmental preconditions match.
4. **Invoke an approved real host capability** through a generated ABI bridge or
   isolated native broker when the behavior is genuinely irreducible and live
   authority is granted.
5. **Record and suspend** when no sound completion source exists. The event can
   be exported for an operator or another executor, but execution does not
   continue using an invented result.

Those are deployment policies over one boundary ABI. They must not become five
different translations of the program.

## What the repository already has

The present code is materially close to this architecture:

- `SSAMachineIndirectLink` records each indirect transfer's source function,
  machine address, edge kind, operand/slot, and strongest known target identity:
  internal function, PE import, unresolved slot, or dynamic state.
- `MachineExecutor.step()` resolves an indirect target. If it names a registered
  external address, it constructs `MachineExternalCallRequest`, retains it in
  `MachineExecutionState.external_requests`, and returns `WAITING_EXTERNAL`.
- The request records the external library/symbol, callsite, return address,
  RCX/RDX/R8/R9, stack pointer, and up to eight stack words.
- `CapabilityGatedExternalPort` admits only explicitly registered
  library/symbol handlers. Unsupported calls remain pending.
- `MachineExternalCallCompletion` already has a broad effect vocabulary:
  result, register and memory writes, virtual memory, filesystem, registry,
  environment, textual state, device output, control transfer, deployments,
  guest callbacks, thread spawns, symbol resolution, termination, and exit code.
- `complete_external_call_state()` validates the pending request and reversible
  call stack, applies typed effects to guest state, and returns to guest control.
- `MachineSystemTape` is append-only, checkpointed, resumable, and records causal
  request-to-completion edges.
- `TapeForwardingExternalPort` can replay a previous narrow completion and
  records which request IDs it forwarded.

This means the fundamental execution shape already exists:

```text
machine/SSA callsite
    -> exact external request retained in program state
    -> approved completion source
    -> typed effects validated and applied
    -> event committed to reversible tape
    -> execution resumes at the recorded return address
```

The live system port today is a deterministic virtual capability library, not a
general real-Windows-call binder. The isolated real-host bridge described below
does not yet exist. `ctypes.WinDLL` use in host-code identity discovery locates a
loaded module base; it is not an external-call implementation.

## Important current gaps

### The request is not yet an adequate native ABI record

`MachineExternalCallRequest` captures four integer registers and a fixed maximum
of eight stack words. That is sufficient for existing hand-authored handlers,
but not for automatic native binding. It lacks:

- an authoritative function signature and calling convention;
- XMM/vector argument registers and vector return policy;
- parameter direction (`in`, `out`, `inout`) and pointee layout;
- the relationship between pointer parameters and length parameters;
- nested pointer/record layouts, strings, arrays, callbacks, and variadics;
- handle type/domain and ownership transfer;
- thread-local `LastError`/errno behavior;
- the input guest-memory read-set used to determine the result;
- environmental preconditions such as filesystem, registry, clock, locale, and
  process/thread generations.

A generic bridge cannot safely pass guest pointer integers to a host DLL. Guest
addresses name the executor's virtual memory, not valid host buffers.

### The tape stores state transitions, not the typed completion itself

The tape durably stores before/after machine states and causal edges. The
current forwarding code reconstructs register and memory effects by diffing
those states. It deliberately refuses filesystem, registry, virtual-memory,
environment, device, thread, deployment, termination, and control-transfer
effects because reconstructing their cause from a state diff would be unsound.

The present replay identity is also only
`(library, symbol, raw scalar/stack arguments)`. Two calls can use the same guest
pointer value after the pointed-to bytes have changed. Replaying the first
completion for the second call would be incorrect.

### Policy is attached to a symbol handler, not a proved occurrence contract

The current port chooses a handler by normalized library/symbol. The final
system needs a stable descriptor tied to:

- the exact callsite and pursued/forwarded identity chain;
- the ABI shape;
- declared read/write/effect domains;
- allowed execution policies and authority;
- implementation version/digest.

The same symbol may legally have different call shapes or policy at different
sites. Conversely, API-set aliases and forwarders should converge onto one
resolved capability identity.

## Exactly where a system boundary is declared

Classification must be based on proof, not DLL spelling.

An import such as `KernelBase!WriteFile` is not automatically irreducible. The
compiler should continue ordinary PE pursuit through API-set resolution,
forwarders, KernelBase, and `ntdll` as long as readable machine code exists.
That source remains part of the compiled ecosystem.

A boundary becomes a capability only when pursuit reaches an operation whose
result depends on authority or state outside the closed program, for example:

- an OS kernel transition (`syscall`, interrupt, or approved exported native
  service chosen as the deployment boundary);
- device, console, network, process, thread, clock, entropy, locale, or host
  configuration input;
- host filesystem/registry/environment state not included in the compiled
  initial state;
- an externally supplied callback, plugin, or dynamically loaded image whose
  bytes are not part of the program;
- a privileged or nondeterministic instruction.

The compiler emits one `SystemBoundaryDescriptor` at that exact occurrence. If
source later becomes available, the descriptor can be replaced by an internal
function without changing callers or inventing a new translation pipeline.

## One shared descriptor table

Add a backend-neutral descriptor table beside `machine_indirect_table`. A
descriptor should contain at least:

```text
capability_id
source module digest + function + machine address
requested identity + API-set/forwarder chain + resolved identity
boundary kind
target platform and calling convention
result and parameter ABI records
pointer/record/string/array layout and length expressions
parameter direction, alias, lifetime, and handle domain
scalar and vector register assignments + stack layout
declared state read domains and write/effect domains
blocking, determinism, idempotence, and callback properties
LastError/errno/exception/trap policy
allowed policies: virtual, replay, native-in-process, native-broker, suspend
implementation/schema digest and authority identifier
```

This is compile-time table data, not an OOP runtime object. Repository SSA can
represent the invocation as an ordinary declared external function slot whose
arguments and results include explicit state/effect bundles. Machine-originated
calls additionally retain their full machine frame correlation. PE-specific
details stay in the descriptor and machine tables rather than changing the
meaning of repository SSA.

The stable `capability_id` should hash the resolved identity, ABI descriptor,
effect contract, and source occurrence. It must not be a sequential ID that
changes when an unrelated function is added.

## Runtime request and completion records

### Request

Extend the existing request with:

- `capability_id` and exact occurrence ID;
- integer, vector, and stack arguments selected by the ABI descriptor;
- normalized logical handles rather than raw host handles;
- each declared input/inout guest span, its bytes or content-addressed blob, and
  digest;
- pointer graph/object correlation so relocated guest allocations can still
  match a recording;
- versions/digests for every declared environmental read domain;
- thread/core identity and causal tape sequence;
- requested policy and granted authority.

The executor must read only descriptor-declared guest ranges while constructing
the request. An unmapped or undersized range is a guest fault, not permission to
truncate an argument silently.

### Completion

Keep `MachineExternalCallCompletion` as the semantic core, and add provenance:

- exact request fingerprint;
- completion source: virtual model, replay record, native bridge, native broker,
  or operator/external executor;
- handler/broker binary and schema digest;
- return value, vector result, `LastError`/errno, exception/trap status;
- complete typed effect list using the existing effect domains;
- a domain coverage map (`complete`, `partial`, `not-observed`, `not-applicable`);
- preconditions under which replay is valid;
- content hashes for large buffers/blobs;
- causal ordering and any blocking/wakeup relationship.

Serialize this typed completion directly into `external_completion` tape
metadata. The after-state checkpoint remains valuable, but replay should apply
the recorded cause, not reverse-engineer it from the consequence.

## Deployment options

### Option 1 — internal compiled implementation

Use when readable source or machine code is available and all necessary state
is represented.

- The call is linked to the pursued SSA function.
- No external event is emitted merely because the original code crossed a DLL.
- Normal machine/repository SSA execution supplies the result.
- This is the preferred route for API-set hosts, CRT code, `ntdll` code before
  the actual kernel boundary, and any dynamically discovered image whose bytes
  are admitted into the compiled library.

This preserves the user's requirement that external code be ingested rather
than waved away as a runtime language.

### Option 2 — deterministic virtual capability

Use when the program intentionally owns the environment model.

Examples are the existing virtual filesystem, registry, environment, device
buffers, virtual memory, loader, handles, synchronization objects, and threads.

- A schema-generated or reviewed handler consumes a typed request.
- It returns only declared `MachineExternalCallCompletion` effects.
- Effects update reversible guest/virtual system state.
- The exact request and typed completion are journaled.
- Replay is deterministic when the request and environmental preconditions
  match.

This is not a fake host call. It is the system implementation selected for that
compiled deployment.

### Option 3 — in-process native binding

Use only for narrow, well-described leaf capabilities where latency dominates
and process isolation is unnecessary.

- Generate a C ABI thunk from the descriptor; do not use an untyped generic
  `ctypes` call.
- Marshal guest input buffers into host-owned bounce buffers.
- Replace guest pointers with host pointers to those buffers.
- Call the resolved export using its declared calling convention.
- Capture the result, `LastError`, and declared output/inout buffers.
- Convert those outputs into typed completion effects and validate destination
  guest ranges before applying them.

This is suitable for pure queries and bounded transformations. It is unsuitable
for arbitrary pointer graphs, host callbacks, process mutation, or a capability
whose side effects cannot be completely observed. A crash or memory error here
would affect the executor process, which is why this should not be the default.

### Option 4 — isolated native capability broker (recommended live boundary)

Use for genuine Windows/system behavior.

- The compiled executor sends a descriptor ID, normalized scalars/handles, and
  content-addressed input buffers over a versioned binary protocol.
- A separate same-architecture broker validates the capability grant, loads the
  approved DLL/export, constructs host buffers and ABI arguments, and invokes
  it.
- Broker-specific adapters observe declared filesystem, registry, environment,
  process, device, timing, callback, and handle effects.
- Host handles never enter guest state directly. The broker returns durable
  logical handle IDs associated with live broker resources; the mapping is an
  explicit session resource and journal event.
- Guest callbacks become nested capability messages. They re-enter known guest
  functions through the existing `guest_calls`/dispatch-target machinery.
- The broker returns a typed completion plus its coverage certificate.

The broker is executing an approved host capability, never forwarding arbitrary
guest instructions. Unsupported guest AVX/AVX2 remains decoded/emulated/lowered
inside the compiler. Native OS DLL code runs only as ordinary host-approved
broker code on the actual supported processor.

For direct NT syscalls, bind to a named, hashed `ntdll` export when possible.
Raw syscall numbers are OS-build-specific; if used, the descriptor must include
the OS build and exact provider-image digest and cannot be considered portable.

### Option 5 — exact recorded replay

Use only when all replay preconditions match:

- same capability/ABI/effect descriptor digest;
- equivalent scalar arguments and logical handles;
- matching contents for every declared input read span;
- matching versions/digests for every declared system-state read domain;
- compatible destination mappings for every recorded write;
- compatible causal/thread ordering and no unrecorded callback dependency.

Raw guest pointer equality is neither necessary nor sufficient. Pointer values
must be normalized to logical guest objects/spans, with writes relocated to the
current corresponding objects.

Full typed completion serialization removes the current registers+memory-only
restriction. Filesystem, registry, virtual-memory, environment, device, thread,
deployment, callback, control-transfer, and termination effects can be replayed
because their original typed causes are available.

### Option 6 — record and suspend

Use when the event is recognized but cannot be soundly serviced.

- Journal the exact occurrence, descriptor, ABI arguments, readable input
  snapshots, current environmental digests, and the reason no completion source
  was admitted.
- Return a suspended status and export the request to an operator, broker, or
  another executor.
- Accept a later completion only if its request ID, descriptor digest, effect
  contract, and authority validate.

If observation is partial, mark each uncovered effect domain explicitly. A
partial recording may be useful evidence, but it is not replay-complete and does
not make `repository_ssa_complete` true. Recording without a result cannot let
the guest continue.

## Automated policy selection

Policy should be a compiled manifest, not a hard-coded live-first fallback:

| Mode | Selection order |
|---|---|
| Hermetic/reproducible | internal -> virtual -> exact replay -> suspend |
| Capture live system | internal -> virtual where mandated -> broker -> journal -> suspend on incomplete coverage |
| Production native | internal -> approved in-process leaf -> approved broker -> suspend |
| Audit/observe only | internal -> emit request -> suspend; an external authority may complete it |
| Differential validation | execute virtual/replay and broker in isolated branches, compare typed completions, commit the approved branch |

Each descriptor restricts its permitted policies. A global mode cannot authorize
an undeclared network, filesystem, process, or device capability.

## End-to-end compile wiring

1. **Pursue code normally.** Resolve API sets, PE forwarders, exports, interior
   targets, and finite indirect tables. Compile every readable dependency.
2. **Detect the first external-state operation.** At a syscall, device/nondeterminism
   operation, unresolved approved plugin boundary, or chosen OS export, create a
   boundary occurrence. Do not classify by spelling alone.
3. **Resolve ABI evidence.** Combine PE/debug metadata where available, known OS
   declarations, callsite register/stack use, and verified adapter schemas. If
   evidence conflicts or leaves pointer direction/extent unknown, retain an ABI
   shortfall; do not guess.
4. **Infer and verify effect contract.** Derive candidate read/write domains from
   the implementation and declarations, then require an explicit verified
   contract for native execution/replay.
5. **Emit the shared descriptor.** Add its stable ID to the module table and link
   the exact machine/control occurrence to it.
6. **Lower a suspension-capable call.** Backends emit a call to a small stable
   capability ABI, for example
   `turing_capability_invoke(descriptor_id, frame, guest_memory, journal)`.
   The ABI returns completed, waiting, trapped, or denied plus typed effects.
7. **Package implementations and authority.** The final executable carries its
   virtual models, replay references, broker protocol version, and capability
   grants. It does not carry Python handlers.
8. **Execute and journal.** On reach, build the typed request, append its tape
   node, select an allowed completion source, validate the completion, apply it
   through the shared state transition, append the typed completion, and resume.
9. **Certify coverage.** Report every descriptor occurrence, selected policy,
   invocation count, completion provenance, effect coverage, and outstanding
   suspended request. Completeness fails if an executed boundary lacks a fully
   covered completion.

For Fortran output, `turing_capability_invoke` should be an `ISO_C_BINDING`
interface to the same small C runtime/broker client used by other native
backends. The compiled graph remains the program; the runtime implements only
the declared system boundary ABI.

## Accuracy and safety invariants

- Never pass a guest pointer directly to a native host function.
- Never expose a raw live host handle as durable guest state.
- Never infer complete effects from an after-state diff when the typed cause is
  unavailable.
- Never replay by library/symbol/raw pointer arguments alone.
- Never call a native export without a verified ABI and bounded pointer graph.
- Never continue a record-only request with a fabricated return value.
- Never hide an unobserved effect domain; mark the completion partial and keep
  the program incomplete or suspended.
- Never treat an external DLL boundary as irreducible while readable code can be
  recursively pursued.
- Never forward arbitrary guest machine code to the host. Native binding is only
  for capability implementations authorized by the descriptor.
- Every request and completion is append-only, causally linked, versioned, and
  content-addressed where it carries bytes.

## Concrete implementation order

1. Add descriptor, parameter ABI, effect-contract, coverage, and stable-ID data
   structures without changing execution policy.
2. Extend `MachineExternalCallRequest` with descriptor ID, vector arguments,
   input span snapshots/digests, logical object/handle IDs, and state
   preconditions.
3. Add complete serialization for every `MachineExternalCallCompletion` field
   and store the typed completion in tape metadata.
4. Replace narrow state-diff replay with typed replay and strict precondition
   validation. Keep legacy narrow recordings readable but label them narrow.
5. Add a policy router that composes internal, virtual, replay, broker, and
   suspend sources according to descriptor authority rather than hard-coded
   live-first order.
6. Generate ABI marshaling thunks for a small scalar/bounded-buffer subset and
   prove request -> native result -> typed completion on isolated test APIs.
7. Move live binding into an out-of-process broker and add logical handles,
   callbacks, cancellation, timeouts, crashes, and broker provenance.
8. Connect the descriptor/call ABI to repository SSA and the Fortran/C section
   renderer. Do not use the `FusedProgram` path.
9. Feed the completed compile blocker ledger through classification: pursue API
   sets and `ntdll`; resolve indirect targets; create descriptors only for true
   environmental boundaries reached afterward.
10. Add a coverage report proving there are no silently dropped or implicitly
    successful system events.

## Recommended result

The best architecture is a hybrid, but not an ambiguous one:

- readable program and library code stays compiled SSA;
- deterministic OS personality lives in typed virtual capability handlers;
- irreducible live behavior crosses an isolated, schema-generated native broker;
- every live or virtual completion is recorded as its original typed event;
- replay is exact and preconditioned;
- anything not fully understood becomes a durable suspended request.

That wiring accurately notices that a system interaction occurred, preserves
the exact callsite and ABI evidence, allows a real system effect when authorized,
and retains the greatest honest recording possible without confusing partial
observation with a complete executable program.
