2026-11-03 TITLE (MEASURED, where): 
2026-11-03 Dynamic byte append-slice storage identity (MEASURED)

Checkout: C:/Users/alber/.codex/worktrees/compressor-results/turing.

Current sanctioned bounded source: `compress_tail(header: bytes,
encoded_length: bytes, coded_commands: bytes, coded_literals: bytes)` returns
their four-way concatenation. All four parameters have the actual compressor's
int64 rank-1 bytes span ABI and the explicit C repository SSA reference. Run:
`$env:PYTHONPATH=(Get-Location).Path; python -u
C:\Users\alber\AppData\Local\Temp\probe_compress_tail_capacity2.py`.
Before correction it raised shape_transformation_concordance REVISE without
changed source at the generated sequence-4 append-slice helper value 4 to its
reserve helper value 4. The reverse row proposed ptr to int64 after ptr to ptr.

The writer was `ir_sequence_tables.lower_sequence_append_slice`. Its
`_storage_values(destination)` gave the dynamic column value 4 a ptr formal
with `heap_element_dtype=int64`. The reserve Call passed that ptr. The body
recreated a separate `SSAValue(id=4, dtype='int64')` from the row dtype as the
GetElementPtr base. The same generated function therefore held two physical
types for one storage identity. Hooked `SSAValue` writes showed no ptr-to-int64
mutation of id 4; the int64 occurrence was created at the GetElementPtr writer.
The enrichment reader then transported both occurrences and tried to revise
the reverse shape edge without a new causal source.

The append-slice body now uses the exact `_storage_values` column formal for
each GetElementPtr base. A focused helper test checks the dynamic pointer
formal, and a real-source regression checks helper storage/formal agreement
after public SSA lower. The bounded `compress_tail` lower now completes in
9.87s. `tests/test_ir_sequence_tables.py::test_dynamic_sequence_append_slice_reserves_clipped_result_before_copy`
plus `tests/test_reverse_metadata_transport_provenance.py` passed 2 in 12.20s.
`tests/test_concat_tail_storage_identity.py` passed in 6.84s. The mapping
audit reports 28 rows, zero findings, zero unsourced, CLOSED.

This establishes the small concat-tail SSA path only. Full compressor run32
on earlier canonical 12060134 stopped at the same reverse row after 510.955s;
the corrected producer has not yet run through a new full compressor lowering.
The earlier matching-forward-edge metadata revision remains exercised by a
distinct linked byte-slice fixture and is retained.
