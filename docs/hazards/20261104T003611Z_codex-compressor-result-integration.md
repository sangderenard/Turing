2026-11-04 Eager bytes return class relay (MEASURED, isolated compressor-results worktree):

The saved full compressor run33 book has SOURCE_FUNCTION_REACHABILITY for
compress_octets Call 21 and 27 to encode_octets, but no
SOURCE_CALL_RESULT_IDENTITY or SOURCE_VALUE_CLASS on either call. Its
encode_octets SEMANTIC_OUTPUT 0 names Add 62; operands 61 (`bytes(header)`)
and 57 (`stream.to_bytes()`) both have builtins.bytes class cells, while
Add 62 has no class in that historical saved book. This was read from
`C:/dev/Powershell/turing/build/compressor_native_33_failed_ssa_1.pkl`;
run33 was not replayed after the repair.

The eager bytes Add publisher first posts Add 62's class from its canonical
operand snapshot. The new source_call_return_class publisher then reuses
the exact reached call, canonical function address, sole SEMANTIC_OUTPUT,
returned canonical value, and returned SOURCE_VALUE_CLASS. It posts the
incumbent SOURCE_CALL_RESULT_IDENTITY and SOURCE_VALUE_CLASS for the caller
after callee class publication. It admits only a single proven bytes output;
it does not use an annotation or callee name as the result class. The new
audit checks those direct causes and rejects a forged returned class.

Bounded public source `encode(data: bytes)->bytes: return bytes(data)+bytes(data)`
called twice from `root` completes SSA and final concordance in 8.53 seconds.
Its eager flush posts five changed cells in round 1 (callee Add class, two
call result identities, two caller classes) and zero in round 2. Focused test
`tests/test_reached_bytes_return_class.py`: 1 passed in 6.84 seconds. Mapping
audit: 28 rows, zero findings/unsourced, CLOSED. No full compressor or native
DLL parity claim follows from this bounded source result.
