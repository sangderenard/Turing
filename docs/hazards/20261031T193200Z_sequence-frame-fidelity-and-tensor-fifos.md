# 2026-10-31 Sequence call-frame fidelity and tensor FIFO wrap (MEASURED)

Worktree /workspace/Turing, base 3eeab1bf plus ongoing uncommitted work. Read
current sources, traced actual source calls and compared untouched baseline;
no diagnosis inferred from older logs. Details/inventory/proposal:
docs/compiler_convergence/tensor_fifo_deque_reuse_2026-10-31.md.

Returned-list failure: factory result edge (callee5, caller1) was present in the
call record and both sequence tables. Frame ownership consulted argument edges
only; factory length/capacity/status/owner became private leases. The following
mutator's proven argument members collided with those leases and were split.
Standalone trace: root sequence1 had heap owner2305843010213693955; split mutator
owner2305843010213694372 contradicted it. IDs are process-local measurements.
Traced the ownership selector before mutation and the refusing heap projection.
Untouched /workspace/scratch/turing-thread-baseline reproduces the same refusal.

Repair: unique result bindings participate in exact sequence-member ownership;
retention is posted by the existing frame transformation ledger under the new
exact_sequence_member_binding rule with call/member provenance. No mutation
exemption and no weakening of the heap-owner refusal. Returned-alias probe now
compiles, emits and executes C with output34 for n=0,1,8; concordance160 rows,
8 functions, legacy24/minted136, 0 findings, 0 unsourced facts/identities.

Added tests/test_source_c_sequence_fidelity.py: seven authored programs, each
compiled natively and compared against Python over eight inputs including repeated
invocations. Covers loops/helper branches, ring wrapping, alias mutation,
append/pop/negative indexing, factory results, distinct factories and growth.
First run7 passed,10.52 s. Broader source/frame/returned-view/C-fence run:
33 passed,1 failed(no Fortran compiler),49.72 s. The exact Fortran failure is
reproduced on clean baseline. Follow-up excludes only unavailable Fortran;
final receipt appended below.

Tensor FIFO investigation found a real PremixRing bug: capacity3 with pushes1..4
had idx1 and no spectral window. Baseline repeats it. New actual RingHarness /
gather_recent_windows tests before fix: 2 failed(capacities3,5),1 passed(capacity1).
Keep logical idx monotonic; only physical slot wraps. Related tensor tests after:
7 passed,2 failed; those same two existing tests(premix histogram/output psi) fail
on untouched baseline with array-to-scalar/setting-array-element errors.

Resident GPU FIFO vertical composition/fanout tests: default PyOpenGL platform
fails glProgramParameteri lookup on both worktrees. With PYOPENGL_PLATFORM=egl,
2 passed,1.08 s. The existing graph-built owns_named_and_fifo_execution test
refuses no ingestion scope on both worktrees. No fallback, fake GPU slots or
stand-in graph was introduced. Tensor gradient/version checks included4 passes.

Mapping concordance audit before and after still refuses unsettled Add7/root dtype.
An authored Ring class independently refuses Indexed18 without source class/dtype
on both worktrees. Remaining boundaries: those source type facts and deque
endpoint/lease integration with the existing typed sequence storage model; C
concurrency/runtime work from prior entries remains unfinished.

Environment: CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types
PYTHONPATH=/workspace/turing-deps; GPU positive run additionally
PYOPENGL_PLATFORM=egl. py_compile and git diff --check pass.

Final source/frame/returned-view/C-fence receipt with unavailable Fortran excluded:
33 passed,1 deselected,46.63 s. Command:
`python -m pytest tests/test_source_c_sequence_fidelity.py tests/test_transformation_priority.py tests/test_source_call_input_conversion_frames.py tests/test_string_core_returned_slot_views.py tests/test_string_core_returned_merge.py tests/test_source_c_synchronization.py -q -k 'not fortran' --tb=line --show-capture=no`.
The scalar/list ring sources in this compiler battery are fidelity tests; actual
tensor-ring investigation and repair use fs_harness.PremixRing and the real
spectral_readout.gather_recent_windows consumer. No tensor deque was implemented.
