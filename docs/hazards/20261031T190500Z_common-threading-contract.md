# 2026-10-31 Common threading/atomic/deque contract (SOURCE-INSPECTED, PROPOSED)

Inspected current SSA atomic handlers and machine_code_lifting projections,
DeploymentFIFO and memory-graph Deque3D. Existing x86 compare-exchange retains
observed/success/memory projections; exchange-add retains old value/memory.
DeploymentFIFO is a bounded Python single-reader helper, not concurrent SSA.
Deque3D is a three-double ctypes structure, not a double-ended queue.

Recorded minimal common semantics, mutex-backed atomic legality/limits, exact
Python deque versus try-queue semantics, and coordinated source-to-runtime tests
in docs/compiler_convergence/common_threading_contract_and_test_matrix_2026-10-31.md.
This is a design/test inventory, not newly implemented target support. No compiler
code changed and no new tests were fabricated or executed for this documentation.
Previous entry's 34 runtime/policy test passes were not rerun here.

Next implementation boundary remains worker-body SSA and token layout, followed
by source atomics reconciled with existing atomic projections, one native
lock-backed correctness path, then deques. Lock-free progress, cross-device
visibility and general GPU mutex progress are not implied by mutex equivalence.
