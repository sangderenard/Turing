# Backend concurrency inventory — 2026-10-31

The [backend threading operation reference](backend_threading_operation_reference_2026-11-01.md)
defines the shared operation set and the mapping/evidence columns for each port.

Current-source inventory at base 3eeab1bf plus the uncommitted dual-IR/SSA slice.
This distinguishes existing executor machinery, target-language possibilities,
and actual lowering of the new SSA protocol. Runtime tests below were rerun;
GPU/browser execution and native thread-source end-to-end compilation were not.

2026-11-01 update: authored threads and composed Condition procedures now have
standalone C execution coverage under ASAP/ALAP, including shared Lock/RLock,
recursive release/restore, notify counts, loops and exception cleanup. The C row
below reflects that update. Other backend rows remain the earlier inventory,
not newly remeasured execution claims. See
[Condition composition](condition_ssa_composition_2026-11-01.md).

The subsequent C barrier slice also implements counted arrival/reset,
control/combined barriers, source-thread completion waits and Python Barrier
wait/reset/abort. These hardware-specific SSA operators and target obligations
are in the [operator reference](backend_threading_operation_reference_2026-11-01.md).
Other backend rows are still unverified inventory, not equivalent ports.

## Counts and scope

The authoritative deployment registry (`src/compiler/deployment_lowering.py`)
contains **8 names**, **7 with nonserial strategies**. GLSL/native_glsl are two
names for the same desktop-compute family: **7 families, 6 parallel**. Four
profile names use pools (Python, LLVM, C, Wasm); three use compute dispatch
(WebGPU, GLSL, native_glsl); Fortran is serial-only in this registry.

The table extends that set with JavaScript, WebGL and SPIR-V emitters:
**10 distinct compiler/executor families**. Tensor adapters (NumPy, Torch, JAX,
Rust, nodus) are not counted again as independent source-SSA threading backends.
Their libraries may use threads internally; that is not authored-thread lowering.

**Zero of these families currently has lowering for the complete new SSA
MemoryBarrier/ControlBarrier/Latch/Mutex protocol.** There are existing native
runtime and shader-specific synchronization implementations to reuse.

## Existing repository implementation

“Internal” means machinery used by that executor/kernel, not a general source
synchronization API. “Not found” means no implementation was identified in the
inspected emitter/executor; it is not a claim about the target language.

| Family / names | Parallel work currently implemented | Synchronization / completion | Memory fences | In-work rendezvous barriers | Mutexes / ownership | Atomics | New sync SSA lowering |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C / c | Native persistent pool; authored worker bodies/contexts and separate OS task lifecycle | Source task joins; Condition composed in SSA; historical native Event ABI | C11 host-local fences; explicit mutex release/acquire phases | Counted latch arrival/reset, control/combined barriers and cyclic Python Barrier wait/reset/abort lower through the shared native epoch state | Source Lock/RLock token protocol, shared guards, recursive suspend/restore and waiter queues use the existing native state engine | Internal native runtime atomics; no general source SSA atomic emitter established | 27 native synchronization primitives selected; standalone C coverage for fences, Condition, latch generations, barriers and completion; host scopes/lifetimes explicit |
| LLVM / llvm | HostDeploymentPool over native ctypes artifact calls; SIMD option | Host pool barrier/indexed/reduce joins | Host lock synchronization; no LLVM fence emission found | Host end-of-frame join | Host pool condition/lock internals; no native lock-box lowering | No general atomicrmw/cmpxchg SSA emission found | Native lowering missing; explicit C-control AOT wrapper available around selected scalar LLVM sections |
| Fortran / fortran | Registry serial natively; explicit C-control AOT wrapper available around selected scalar sections | Serial deployment boundaries | No explicit fence lowering found | No authored barrier lowering found | No native lock-box lowering found | No general atomic SSA emission found | Native lowering missing; C-wrapper control uses the C row's admitted operations |
| Python / python | Existing persistent daemon-thread deployment pool | Condition-backed barrier/indexed/reduce joins | Host synchronization internally | End-of-frame completion | Internal threading.Condition lock | Lock-protected claims; no general atomic SSA API | Missing; no new Python executor added |
| Wasm / wasm | Browser Worker tile batches, copy-in/copy-out | Message/batch completion and result merge | No shared-memory Wasm fence path found | Batch completion only | No shared token-lock box found | No shared-memory Wasm atomic instruction path found | Missing |
| JavaScript emitter | Structured host control/device-call orchestration | Device results awaited at orchestration boundaries | No authored fence lowering found | No authored rendezvous lowering found | No lock-box lowering found | No Atomics-based SSA path found | Missing; no declared deployment profile |
| WebGPU / webgpu | Compute grid; tiled GEMM kernel | Dispatch completion; internal GEMM cooperation | workgroupBarrier provides workgroup-memory synchronization in GEMM | Existing GEMM workgroupBarrier calls | No mutex lowering found | No general atomic SSA emitter found | Missing |
| Desktop GLSL / glsl, native_glsl | Compute dispatch, composed kernels, stream/control machinery | Dispatch boundaries and internal shader cooperation | Existing memoryBarrierBuffer and host glMemoryBarrier | Existing barrier() in composed/device-resident kernels | No general lock-box lowering found | Existing uint atomicAdd for debug counters; not general source atomics | Missing |
| WebGL emitter | Fragment-invocation data parallelism | Raster/render-pass completion | No general source fence lowering found | No compute workgroup rendezvous path found | No mutex lowering found | No general atomic SSA path found | Missing; no declared deployment profile |
| SPIR-V emitter | Deployable compute entry/local size and shader manifests | Deployment interface exists; no declared deployment profile | No OpMemoryBarrier lowering found | No OpControlBarrier lowering found | No mutex lowering found | No OpAtomic lowering found | Missing |

Evidence anchors: `deployment_host_pool.py`, `repository_ssa_dispatch.py`,
`deployment_native_emission.py`, `deployment_ssa_binding.py`,
`deployment_outlining.py`, `wasm_html_shell.py:ensureTileWorkers/executeDeploy`,
`ssa_webgpu_backend.py` tiled GEMM, `ssa_glsl_compute_backend.py` host barrier,
`common/tensors/accelerator_backends/glsl_backend.py:compose_shader` and stream
publication/debug instrumentation, and the corresponding `ssa_*_backend.py`
operator dispatches. Native ABI is in
`common/tensors/accelerator_backends/c_backend/turing_pool.h` and `.c`.

Binary ingestion also preserves locked x86 compare-exchange/exchange-add as
AtomicCompareExchange*/AtomicExchangeAdd* SSA projections. This is ingestion
support, not evidence that the target emitters generate native atomic instructions.

## Target facilities available for subsequent lowering (proposed direction)

| Target | Facilities to map explicitly | Scope/progress limits |
| --- | --- | --- |
| C / LLVM native | Existing task/condition/event runtime; C atomics or LLVM fence/atomicrmw/cmpxchg; token table service | Reuse native ABI; distinguish plain Lock ownership from RLock; do not serialize communicating tasks as blocking calls |
| Fortran | Existing C interoperability seam; native atomic/lock facilities where supported | Current serial deployment profile is not permission to erase communicating waits |
| WebGPU / WGSL | workgroupBarrier, storageBarrier, supported integer atomics | Workgroup rendezvous; no general cross-workgroup barrier within a dispatch; no global-progress promise for spinning locks |
| Desktop GLSL | barrier, memoryBarrier variants, integer atomics, dispatch/API barriers | Workgroup rendezvous versus memory scope must stay distinct; cross-workgroup progress is not guaranteed |
| SPIR-V | OpControlBarrier, OpMemoryBarrier, OpAtomic* with explicit scopes/semantics | Capability/environment validation and structured/uniform control constraints |
| Wasm / JavaScript | Shared memory, atomic operations and wait/notify where the host permits them | Existing copied tiles do not provide shared storage identity; browser main-thread blocking restrictions |
| WebGL | Host-mediated passes and continuation boundaries | Fragment path cannot be treated as compute shared-memory threading |
| Python executor | Existing host concurrency primitives | Existing deployment executor only; no new Python thread emulation proposed |

First direction: retain worker SSA bodies, specify token representation/layout,
and connect source logical context to dispatch. Then reuse the native C ABI and
add explicit target policy mappings. GPU policies require scope-limited mappings
and continuation/dispatch boundaries for wider synchronization. No silent scope
promotion or spinning-lock progress assumption.

## Serial SSA execution and logical state

Yes: physical worker assignment can be absent, and different logical contexts
can execute on one physical worker. SSA dependencies can express a legal
interleaving of their activity. Preserve each context's private state, closure
binding, logical owner token and progress position, with exact shared storage.

Do not erase logical identity while ownership, recursive locks, joins or latch
participation depend on it. A fixed schedule can make identities implicit in
its state slots; removing them requires a concorded proof that observable
semantics do not change. Two activities using one OS thread must not appear to
be the same recursive-lock owner.

At a barrier, schedule all required arrivals before their continuations. At a
contended acquisition or wait, suspend that context and execute another runnable
context. Running worker A to completion while it waits for B cannot work. A
static topological schedule suffices when readiness is statically established;
dynamic waits, timeouts, polling and external participants require retained
progress machinery or a target refusal. General DFS/ASAP/ALAP alone does not
supply those runtime state transitions.

Existing `legalize_deployments_serial` drops deployment-frame Deploy/Join markers
around independent lane instructions already in valid serial order. LLVM and
Fortran likewise treat those boundaries as scheduling receipts. This contract
does not authorize dropping the new ControlBarrier, MemoryBarrier, Mutex or
CompletionWait operations. A serial realization chooses one permitted execution;
it does not establish an authored comparative ordering guarantee.

## Measurements

Environment: CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types
PYTHONPATH=/workspace/turing-deps.

- Native runtime plus deployment-policy tests: **25 passed**, 0.43 s. Includes
  task waits/nested numeric deployment, recursive condition ownership,
  notify-one/all, event broadcast/clear, exactly-once claims and zero-worker drain.
- Existing host deployment pool tests: **9 passed**, 0.31 s.
- Before the C fence slice, actual source `compiler_fence(2, 1)` ingested to SSA, then C module emission
  refuses `KeyError: no dtype layout declared for 'ssa.sync_token'` at the
  result declaration. This is an earlier blocker than opcode selection; no
  native fence was emitted or executed.
- Other rows are current-source inspection, not freshly executed platform tests.
  This documentation changes no compiler decisions or concordance pages.

The subsequent C fence slice resolves that token-layout blocker for fences; see
`c_synchronization_fence_slice_2026-10-31.md` for current measured support.

The optional mixed AOT route is documented in
[c_shell_control_native_sections_2026-11-01.md](c_shell_control_native_sections_2026-11-01.md).
It returns a C artifact and does not promote wrapper support into native LLVM
or Fortran opcode support. Linux standalone tests verify both mixed numerical
backends with real native worker counts; coarray/native lane dispatch remains
future porting work. General C passive deployment now discovers the documented
bounded disjoint-store case and respects a configured total core budget.
