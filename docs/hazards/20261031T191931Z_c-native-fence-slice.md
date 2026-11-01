# 2026-10-31 C11 fence SSA-to-native slice (MEASURED, cloud Linux)

Canonical clone /workspace/Turing, base 3eeab1bf with earlier uncommitted work.
User clarified native/compiler hardcoding is authorized; source-built Python runtime
procedures are optional. Started C backend integration through its existing module
emitter, emission concordance and compile/load path. No native executor substituted.

Added ssa.sync_token layout, C11 atomic_thread_fence/atomic_signal_fence selection,
and authoritative C_SYNCHRONIZATION_POLICY facts. Native policy keeps host-local
scope and exact order-only masks; unsupported scope/class/visibility is refused.
Details: docs/compiler_convergence/c_synchronization_fence_slice_2026-10-31.md.

Before editing, source synchronization + native runtime tests: 27 passed, 9.62 s.
New authored-source C tests: 6 passed, 7.97 s, including two compiled/loaded/executed
scenarios across all four orders and accepted scopes, plus four explicit refusals.
After: source C + source SSA synchronization + lock box + dual IR + shared location
+ native pool runtime tests: 51 passed, 1 strict xfail, 18.19 s. No xfail removed.

Broader run including test_ssa_c_aggregate_constants: 79 passed, 14 failed,
1 xfailed, 35.88 s. Native dtype/storage checks: 38 passed, 2 failed, 2.30 s.
Untouched /workspace/scratch/turing-thread-baseline at 3eeab1bf ran the same
aggregate/dtype/storage modules: 65 passed, 16 failed, 3.07 s. Same failure names:
14 aggregate failures (old text/ABI assertions, missing llvmlite and the existing
unsourced dead_structural_retirement refusal), plus int128/uint128 public ABI
layout assertions. No unrelated fixes or test relaxation. Sync-token adds one
passing dtype-alias case relative to baseline.

Mapping concordance audit still refuses unsettled Add 7/root dtype, as before.
Source-to-C tests assert zero concordance findings before emission, after emission
and after native compile/execute. Full measured probe receipt is appended below.

Environment: CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types
PYTHONPATH=/workspace/turing-deps. Initial native compile refused absent ziglang;
installed ziglang into the existing dependency directory, then normal artifact
compilation/execution passed. No runtime host fallback. py_compile and
 git diff --check pass.

Next boundary: worker-body SSA and task ABI/capture binding; then C lock-box and
latch/barrier operations and atomics. Do not claim native authored threads,
rendezvous or mutex operations now work because fences compile.

Measured release-memory/acquire-compiler source probe: before and after C emission,
9 rows across 1 function, legacy=8/minted=1, 0 findings, 0 unsourced facts/identities.
Policy keys are SSA receipts ('root__root@control:0', 'entry', 0) and site 2;
values respectively (atomic_thread_fence, memory_order_release, 256, 4, memory)
and (atomic_signal_fence, memory_order_acquire, 4, 2, compiler).
