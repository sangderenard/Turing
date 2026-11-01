# 2026-10-31 Direct worker tokens on both region graph halves (MEASURED, Linux cloud)

Follow-up to `20261031T145608Z_thread-dual-ir-ingestion.md` in the same
`/workspace/Turing` worktree. The authoritative membership facts already
contained each worker activation's exact thread-create cell. Both graph
halves now expose that cell directly as `attributes['worker_token']`.
Create/start/join operations remain parent-owned and expose the same target
activation through `dispatch_worker_token`. No worker number or hardware
assignment was invented.

`ThreadRegion.worker_tokens` enumerates those source-scoped tokens.
`worker_nodes(token)` selects the exact membership from concordance, and
`worker_boundary_edges(token, half)` retains the capture, lifecycle and
synchronization crossings needed when separating a native worker. Shared
storage, parent context and rendezvous completion retain `worker_token=None`.
The audit compares these views against the existing source membership and
lifecycle facts; there are no new pages or graph edges.

The actual two-thread SOURCE measures tokens from creates **8 and 9**, each
owning **7** body nodes despite sharing worker function reference **1** and
parent allocation **4**. BARRIER_SOURCE measures create token **10** owning
**3** body nodes; its barrier-arrival edge remains a crossing to shared
completion node **16**. The source-driven checks assert distinct tokens on
both halves, exact worker selection, parent lifecycle association, capture
crossings and barrier crossings. Existing ASAP/ALAP independence checks
remain in the same source test.

With the previous entry's GCC/PYTHONPATH environment prefix,
`python -m pytest tests/test_thread_dual_ir_sync_contract.py
tests/test_python_two_threads_shared_location.py -q --tb=line --show-capture=no`
measured **12 passed, 1 xfailed**. Both current source probes report
**0 concordance findings and 0 unsourced facts**, as before the token views.
The mapping audit was rerun and still refuses its unsettled **Add 7 dtype**;
it is not a clean mapping audit. `py_compile` and `git diff --check` pass.

Native lowering can bind these existing tokens directly to its worker
objects. Native allocation/emission/execution was not added. The prior SSA
consumer and memory-policy boundaries, including the remaining strict SSA
xfail, remain unchanged.
