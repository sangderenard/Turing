# C synchronization: first native fence slice

C is the first backend implementation target. Runtime algorithms may be native
C or compiler-owned SSA; authored Python runtime procedures are optional. The
existing task/condition/event/pool ABI remains the next connection point.

## Implemented and executed

Actual Python source synchronization intrinsics travel through normal ingestion,
ControlProgram placement and repository SSA to C module emission and compilation.
MemoryBarrier maps to C11 atomic_thread_fence for memory ordering and
atomic_signal_fence for the compiler-fence domain. The source projection selects
acquire, release, acq_rel or seq_cst; no backend spelling is inferred from a name.

The C host policy currently accepts invocation (4), thread-region (256) and object
(257) scopes, and order-only masks 2/4/8/16. Device/workgroup and other unsupported
scopes, memory-class flags and MakeVisible/MakeAvailable/Volatile masks are explicit
shortfalls. It does not silently discard flags or promote scopes. This deliberately
small policy can be extended only with a demonstrated host-memory interpretation.

ssa.sync_token now has a declared inert 64-bit layout in the existing dtype
layout authority. C emits zero as its carrier; it is a dependency value, not a
native lock handle or memory-state representation. The fence itself provides the
native ordering. Held-lock tokens still need their own representation/implementation.

Each native selection is posted to C_SYNCHRONIZATION_POLICY, keyed by the exact
SSA receipt and derived from that receipt and its SOURCE_SYNCHRONIZATION cell.
The existing C emission recorder relates actual text units to SSA instructions.
Policy records retain intrinsic, order, scope, mask and domain. Source and SSA
receipts are checked before selection. No backend-specific source graph is built.

## Verification and limits

`tests/test_source_c_synchronization.py` compiles authored source, checks all four
orders and all accepted scopes, emits C, compiles through CModuleArtifact.compile,
loads and executes it, and verifies its ordinary return. Negative tests assert
explicit refusals and no successful fence policy/emission for unsupported requests.
There are two native execution scenarios and four refusal scenarios, all passing.
This verifies the C11 mapping and integration; it is not a multithreaded publication
litmus or a proof of arbitrary communicating-thread execution.

The standard compiler path required ziglang, which was absent and was installed
into /workspace/turing-deps. No compiler fallback or mocked compilation was added.
Focused combined source/SSA/native-runtime tests: 51 passed, 1 xfailed. The xfail
is the existing missing worker-body SSA assertion. Broader aggregate/dtype results
and clean-worktree comparison are recorded in the matching hazards entry.

Remaining boundary: retain/outline actual worker bodies, connect logical context
tokens and captures to the existing C task ABI, lower lock-box operations and
latch/barrier state with ownership/progress guarantees, then add source atomics
using existing atomic identities and native C atomic/mutex mechanics. A fence
alone provides neither mutual exclusion nor rendezvous nor a task launch.
