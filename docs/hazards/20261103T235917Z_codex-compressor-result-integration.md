2026-11-03 Full compressor run32 and byte-input integration (MEASURED)

Canonical commit 12060134, C:/dev/Powershell/turing. Exact original octet_codec
module, compress_octets entry, module bindings, program_extraction contract and
explicit C reference. Command: `$env:PYTHONPATH=(Get-Location).Path;
$env:PYTHONIOENCODING='utf-8'; python -u
C:/Users/alber/AppData/Local/Temp/compressor_native_32.py`.

Lowering stopped after 510.955s at the same reverse shape metadata row as
run31: root compress_octets|shape value31 from append_slice_30 value31.
The proposed row was (ptr -> int64); the prior row and unchanged sources
stated (ptr -> ptr). The earlier matching-counterpart cause correction did
not solve this full-program failure. No full SSA, C or DLL completion claim.

The atomic failure artifact exists at
C:/dev/Powershell/turing/build/compressor_native_32_failed_ssa_1.pkl,
123678713 bytes. It loads as IRModule with 7 functions and a 222-page
IdentityBook. Root formal31, its helper call, helper formal31, reserve call,
reserve formal and reserve memmove all use ptr. The append_slice_body
GetElementPtr alone reconstructs the same id31 as int64. Their SSA_VALUE
cells prove ptr with physical_dtype int64. The independent real-source
concat-tail probe reproduces this producer mismatch in seconds. The fix in
ir_sequence_tables reads the existing storage formal instead of creating an
SSAValue from the column element dtype. The producer receipt records its
focused actual-source test; the saved failing artifact remains untouched.
A fresh whole-program lowering is required to validate that correction.

The separate public bytes input reader now drives C wrapper length/capacity
initialization for all proved bytes inputs, even when no returned-slice bound
exists. Each emitted initializer derives from the exact descriptor, contract,
ingestion origin and bytes-class cell in addition to both SSA member cells.
No parameter-name matching or host evaluation of source control is used.

Measured integration in the isolated checkout:
- tests/test_dynamic_byte_tensor_view.py and
  tests/test_public_sequence_input_ports.py: 3 passed in 18.58s. Actual
  _octets DLL needs only data, including empty and high-byte inputs; metadata
  ports are initialized by native code. A byte-length DLL returns 0, 4 and
  256 from only its public data feed. Actual _read_header descriptor proof
  and conflicting-class refusal are also covered.
- tests/test_linked_accepted_c_emission_owner.py and
  tests/test_public_byte_slice_capacity.py: 2 passed in 17.77s after the
  append-slice producer fix. This includes actual linked _split DLL parity,
  direct slice DLL parity and a Python-free executable using the same
  capacity allocation, three calls and native-written logical-length cells.
- Canonical 12060134 tests/test_public_byte_slice_capacity.py plus
  tests/test_runtime_result_extent_ports.py: 24 passed in 11.92s before this
  subsequent input-initializer/producer integration. That older canonical
  result does not validate the later edits.
- Mapping audit for the input-reader and producer slices: 28 rows, zero
  findings, zero unsourced, CLOSED. git diff --check passed.
