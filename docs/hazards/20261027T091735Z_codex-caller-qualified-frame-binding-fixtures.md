2026-10-27 Caller-qualified native call input receipt fixtures (MEASURED, `tests/test_native_call_input_receipts.py`):

The two caller-storage frame-tail fixtures previously posted
`ARGUMENT_BINDING` with `(callee, formal, callsite)`, which does not match the
current four-field page key `(callee, caller, formal, callsite)`. The fixture
now publishes an `SSACallRecord` for the exact caller and callsite, derived
from the formal, actual, and storage `SSA_VALUE` cells. Its caller-qualified
`ARGUMENT_BINDING` row derives from that call record and the same exact value
cells. Production schema and rejection behavior are unchanged.

Before the fixture update, the two tests
`test_exact_binding_reuses_concorded_late_sequence_storage` and
`test_exact_binding_restores_removed_caller_storage_identity` failed at the
raw three-field row. After the update:

```text
python -m pytest -q tests/test_native_call_input_receipts.py --tb=short
8 passed in 1.30s
```
