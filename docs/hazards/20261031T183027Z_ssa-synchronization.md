# 2026-10-31 Source synchronization SSA and mutex lock-box preparation (MEASURED, Linux cloud)

Worktree `/workspace/Turing`, branch `work`, base `3eeab1bf`; follows the source
region and worker-token entries from this date. This records observed compiler
behavior, not native barrier/lock execution.

Added exact-identity compile-only synchronization intrinsics, separate control and
memory SSA operators, generation-scoped latch operators, completion waits, and
concorded token-lock projections. Plain/reentrant mutex factories and exact methods
are recognized. SSALockBox is a book-backed binding/layout registry; runtime handle
lookup and atomic servicing remain a lowering contract. Captured worker mutexes
share their parent's exact entry. All compiler facts/projections use concordance.

Actual source probes exposed two placement issues: terminal special-case calls
lost computed argument expressions, and generated with-entry acquisition inherited
the whole resource span and was placed after cleanup. Keeping normal Call argument
roles and using the generated resource entry phase fixed them. No mutation exemption
was added. An authored post-join memory fence preserves a real shared-cell Load.

Measured full source concordance: synchronization example **77 rows, 3 functions,
0 findings, 0 unsourced facts/identities**; RLock example **16 rows, 1 function,
0 findings, 0 unsourced facts/identities**. Before and after this work the mapping
audit refuses unsettled dtype for **Add 7 in root**. The latest command accidentally
included `--case`, which the positional CLI rejected as a case name; it subsequently
ran `mapping` and reproduced that same dtype refusal. This is not an audit pass.

Verification command (environment prefix below):
`python -m pytest tests/test_source_ssa_lock_box.py tests/test_source_ssa_synchronization.py tests/test_thread_dual_ir_sync_contract.py tests/test_python_two_threads_shared_location.py tests/test_control_source.py tests/test_string_source_callee_identity.py tests/test_ssa_shared_memory_threads.py -q --tb=line --show-capture=no`
Result: **65 passed, 1 skipped, 1 xfailed**, 15.90 seconds. These compile authored
source, including all declared scopes/orders, combined masks, CFG placement,
completion, aliases/captures, unrelated methods, and early-return cleanup. Original
ASAP/ALAP and exact Barrier identity tests remain included. No new expected failures.
Earlier pre-mutex synchronization regression run: 58 passed, 1 skipped, 1 xfailed.
The original missing worker-body SSA assertion stays strict xfail.

Environment: `CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types PYTHONPATH=/workspace/turing-deps`.
Incumbent threading-special-case failures remain the previously clean-worktree
verified unsettled Add 9/root n/counter n failures; broad regression observed 3
failures alongside 48 passed, 1 skipped, 1 xfailed before mutex additions.

Exact next boundary: outline/retain worker-body SSA with source worker tokens;
resolve standard Python Barrier memory policy explicitly; implement target policy
selection, runtime mutex table allocation, atomic token/latch transitions and progress
servicing. Neither native execution nor backend synchronization emission was verified.
See `docs/compiler_convergence/synchronization_ssa_contract_2026-10-31.md` for the
implemented operand table and remaining policy requirements.
