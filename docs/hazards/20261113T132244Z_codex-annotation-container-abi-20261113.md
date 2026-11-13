# Opt-in direct child-pool selection, not public escape

The selection and local-lifetime readers now accept keyword-only
`include_child_pool=False`. Existing callers still refuse child pools. Opted-in
source-built direct leaves retain the complete distinct child storage vector
before the shared owner, preserving incumbent outer status ordering. Lifetime
admission separately joins `validated_sequence_child_storage` and retains its
proof references; structure alone is not an ownership grant.

Child address-Phi selections remain refused. This does not change the producer,
public projection, ownership transfer, LLVM/C emission, or GraphSolver branch
contract mismatch. No nested-string public native result is claimed.

Verification (Windows Python 3.11, service Nodus tensor-core DLL):

- New `tests/test_direct_child_selection.py` before compiler change: 1 passed,
  6 expected failures (unsupported keyword), 9.63s. After change: 7 passed in 6.10s.
  Covers exact complete vector/proof, primitive non-null parity, default refusal,
  missing/swapped/aliased roles, foreign owner, and child-Phi refusal.
- Existing child storage tests: 2 passed. Existing lineage test fails before
  selection at `owned_record_return_sources` empty unpack. Focused combined run
  including the new 7: 9 passed, 1 failed in 16.29s.
- Primitive transfer plan/projection/suffix/native suites: 12 passed in 89.71s.
  Includes native LLVM and C retained copies, repeated release and copy-failure
  cleanup. These are primitive regressions, not native child transfer proof.
- Full-source detached incumbent acf1a37a57df00b1b8a9e55b6b7cbd6fb25a8a63 ran
  the exact same 15 existing tests: 14 passed, the same 1 lineage fixture failure,
  109.84s. No incumbent fixture was edited or suppressed. Detached worktree was
  removed without force after clean tracked/untracked status verification;
  test timing/last-failure summaries retained in local developer scratch.
- `git diff --check` passed. No full repository suite or sanitizer run here.

The pre-existing lineage fixture failure remains visible and separate; it is
not evidence of child-Phi support. Actual nested-child producer and escape
contracts remain follow-on work requiring exact predecessor and lifetime proof.
