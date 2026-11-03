2026-11-03 Public bytes input ports independent of returned slices (MEASURED)

Checkout: C:/Users/alber/.codex/worktrees/compressor-results/turing.

The actual `_read_header` public lower has a bytes packet Input with one
SEQUENCE_DESCRIPTOR. Its data, length, and capacity IDs are 2,
2305843010213693953, and 2305843010213693954 in the bounded run. The
descriptor cites one SEQUENCE_CONTRACT_CONCORDANCE; the contract reaches the
same INGESTION_VALUE as the authored SOURCE_VALUE_CLASS builtins.bytes.
`sequence_capacity_sources` is empty because this function returns no bytes
slice. The previous C wrapper initialized public bytes length and capacity
only while processing returned-slice relations, leaving this input's native
ports uninitialized.

`public_sequence_input_ports` now reads the exact descriptor, contract,
ingestion origin, and class independently of any returned output. It refuses
a conflicting class on that same source identity. It does not infer a value
from the parameter name or sample packet contents. C wrapper consumption and
native parity are separate integration work.

Focused check: `tests/test_public_sequence_input_ports.py` 1 passed in 9.34s.
Mapping audit: 28 rows, zero findings, zero unsourced, CLOSED. The bounded
scratch command was `$env:PYTHONPATH=(Get-Location).Path; python -u
$env:TEMP\probe_header_input_ports.py` in the isolated checkout.
