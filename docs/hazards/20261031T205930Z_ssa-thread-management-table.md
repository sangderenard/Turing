# SSA thread-management table and worker-body inclusion (MEASURED)

Worktree /workspace/Turing, base 3eeab1bf plus the ongoing uncommitted changes.
Canonical source is C:/dev/Powershell/turing; this run used its cloud clone.
Read root/compiler AGENTS and the newest hazard before this slice. Current
source and failing frame records supplied the diagnosis, not old logs.

## Before and observed missing connection

Targeted thread ingestion/shared-storage tests before this slice: 6 passed,
1 strict xfailed, 3.21 seconds. Worker graph1 existed with seven source nodes,
but planned-shell traversal admitted ordinary calls and omitted thread targets.
The previous mapping audit refusal was unsettled root Add7 dtype.

Added IRModule.thread_table / SSAThreadTable and book-backed source body,
activation, capture-use, capture-contract/column, SSA body/activation/context
pages. Source body identity is the exact FunctionTable address cell. Worker
identity is each exact create occurrence; one loop create remains a factory
site. All source operand roles, direct start/join cells and authored control
scopes are retained. Bodies reference the existing FunctionTable ProcessGraph
and actual module SSA Functions; they are not rewritten Python executors.

Planning now follows table targets through thread_worker_shells, separately
from ordinary call edges/public entrypoints. It lowers the existing worker body
through the normal region/control/sequence/call machinery. The Thread target
address stays in the table and no longer becomes a callable host input.
Ordinary callsite-specialized copies are excluded from worker-entry selection.

## Exact failures exposed and repairs

Admitting the actual worker first exposed Indexed3 without a dtype. The exact
SOURCE_CLOSURE_CAPTURE linked worker input0 to parent storage1. The parent
initializer's element class had a source receipt, but no input row contract was
carried into worker planning. Capture contracts now retain that exact class;
the existing writer-checked type fixed point handles a concorded closure input
as it handles a formal. Its physical column proof is posted as well, so the
sequence builder and region planner read the same type rather than one getting
unknown. Numeric scalar captures retain their exact source class too.

The worker body then existed in SSA, but the parent post-join read still folded
to initializer0. The existing escape/mutation classifier could not see a write
through a thread's captured storage. SOURCE_THREAD_CAPTURE_USE records the
exact parent/input identities and whether the body mutates. Retained storage
is an escape through that same classifier; the existing structural fold now
preserves its identity. Explicit capture operands are added using _set_operands
under registered thread_context_binding. No comparative worker order, lock or
synchronization is introduced by storage aliasing.

The unrelated Other.wait case exposed an empty receiver capture without a
physical SSA value. SSA context formation now consults the actual lowered body
signature and existing sequence/record member claims. An unused physical
context is explicitly recorded and removed from the physical operand list;
its source capture remains in the table. Live captures retain storage operands
and all descriptor member claims. The ordinary dominance/concordance refusal
was not bypassed. The context audit checks both live and unused status against
the real worker signature.

## Measurements and verification

Original two-thread SOURCE:
- source worker reference1, seven source nodes, one closure capture;
- one source body, two source activations, one SSA body, two SSA activations,
  two live SSA context rows;
- worker function root__worker, captured parent storage1 -> worker input0;
- each context has four member claims on each side from the existing sequence
  descriptors (column/handle, length, capacity, status);
- worker load, arithmetic region call, and store exist in SSA;
- parent's post-join read is not folded to a constant;
- concordance59 rows across4 functions, legacy16/minted43, zero findings,
  zero unsourced facts/identities, latch CLOSED.
IDs above are measurements of this source, not a numbering policy.

The original strict expected failure for missing worker body/shared-cell access
passed for these assertions and was removed. tests/test_ssa_thread_table.py
adds eight Python-source compilation cases/checks: shared body/distinct handles,
exact capture/member bindings, argument+branch body, unrelated wait receiver,
loop factory/control scope, read-only list capture, scalar capture, and exclusion
of an ordinary callsite specialization from worker entry selection.

Final regression: 65 passed, 35.94 seconds, one environment warning about absent
nodus_tensor_core. Command:
`CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types PYTHONPATH=/workspace/turing-deps python -m pytest tests/test_ssa_thread_table.py tests/test_thread_dual_ir_sync_contract.py tests/test_python_two_threads_shared_location.py tests/test_thread_ingestion_semantic_nodes.py tests/test_source_ssa_synchronization.py tests/test_source_ssa_lock_box.py tests/test_source_c_sequence_fidelity.py tests/test_source_c_standalone_validation.py tests/test_source_c_synchronization.py -q --tb=short --show-capture=no`

This includes actual source dual-IR ASAP/ALAP edge checks, authored/captured
barriers and unrelated wait classification, plus the existing source-to-native-C
sequence and standalone-validation/fence cases. It does NOT execute native
threads. Mapping audit remeasured during and after this slice still refuses
Add7/root without settled dtype after0.7 seconds; it never produces a clean
report. No claim is made that the repository-wide audit passes. py_compile and
git diff --check pass.

## Next exact boundary

The table's per-handle state columns declare the runtime state contract; no
native state-table allocation or task launch is implemented by this slice.
Constructor args/kwargs remain exact source/SSA operands. Next is the native
argument/context ABI and C task lifecycle servicing of those table rows,
including storage lifetime through join. A capture into a different enclosing
activation currently refuses explicitly; general runtime handle-container flow
still needs lifecycle correlation beyond direct edges. Python Barrier scope /
memory policy and backend synchronization remain separate work. The authored
threaded queue benchmark has not been compiled/executed by this change.

Design/API details: docs/compiler_convergence/thread_management_table_2026-10-31.md.
