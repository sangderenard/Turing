# 2026-10-31 Backend concurrency inventory (MEASURED and SOURCE-INSPECTED)

Current `/workspace/Turing` worktree, base 3eeab1bf plus uncommitted dual-IR slice.
Read current emitter switches, deployment profiles/executors, native pool ABI,
GLSL composition/stream/debug shader code, WebGPU GEMM and browser tile workers.
Extended inventory: `docs/compiler_convergence/backend_concurrency_inventory_2026-10-31.md`.

Registry: 8 named profiles, 7 nonserial; GLSL/native_glsl share a family. The
extended inventory has 10 families including JS, WebGL and SPIR-V emitters.
No complete lowering of the new SSA synchronization protocol was found.
Existing C tasks/Condition/Event, host pools, GPU kernel-local barriers/fences
and GLSL debug atomicAdd are real implementations and are recorded separately.
Binary locked-atomic ingestion is distinct from target atomic emission.

Reran with CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types
PYTHONPATH=/workspace/turing-deps:
`python -m pytest tests/test_turing_pool_runtime.py tests/test_deployment_lowering.py -q --tb=line --show-capture=no`: 25 passed, 0.43 s.
`python -m pytest tests/test_deployment_host_pool.py -q --tb=line --show-capture=no`: 9 passed, 0.31 s.
Actual authored source compiler_fence(2, 1) reaches SSA; C emission refuses
missing dtype layout ssa.sync_token before opcode selection. No native fence
or barrier execution was claimed. Other platform rows are inspection only.

Serial scheduling may remove hardware assignment, but logical owner/context
identity and wait readiness remain until proven removable. Independent deployment
serial fallback does not cover blocking communicating tasks. Next boundary:
worker-body SSA retention, token representation/layout and target policy mapping,
then reuse existing runtime ABI. Documentation only; no compiler rewrite/audit
change in this inventory follow-up.
