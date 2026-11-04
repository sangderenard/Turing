2026-11-04 Stable tensor metadata Call binding audit (MEASURED, isolated compressor-results worktree):

In the arange→cast→sum bounded source, two TENSOR_METADATA_CALL_BINDING rows
originally used instruction positions 47 and 60. Later aggregate output
projection insertions moved both Calls; those positions named GEPs. A later
tensor lowering invocation moved the Calls again. The exact Call result SSA
identity is stable across those insertions. Union changed the page row's
fourth member to that result ID, and both writers cite the result SSA_VALUE
as a direct cause. The C reader now resolves one live Call by that booked
result ID in the recorded block and checks paired actual/formal positions.

The new concordance audit checks the same unique Call identity, callee,
positional actual/formal IDs, and direct SSA_VALUE plus shape/result-binding
causes. It does not accept a coincident signature or a historical instruction
index. `tests/test_tensor_metadata_call_position.py` lowers the bounded
source, confirms two bindings, inserts a GEP before one Call without changing
the audit result, then forges the Call result ID and observes refusal:
1 passed in 5.79 seconds. Mapping audit: 28 rows, zero findings/unsourced,
CLOSED. No full compressor rerun or native parity claim is made here.
