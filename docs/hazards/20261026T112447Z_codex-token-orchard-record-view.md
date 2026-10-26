2026-10-26 TOKEN ORCHARD DECLARED ARRAY RECEIVER VIEW (MEASURED, Windows):

Observed: The saved whole-core identity book at `string_core/build/turing_whole_core_audit/artifacts/identity_logs/token_orchard_whole_core.20261026T055549.failed.log.xz` states `Vocabulary.exact_begin` is an authored `array('Q')`. In specialized `_intern_validated_span`, sequence descriptor 54 already exists with column 54 and `uint64` dtype, yet record 5 still names that field as a raw SPAN of value 54. `_cover_span` requires its sequence column. The linker correctly refuses that mismatch at callsite 39.

Observed: `materialize_parameter_record_abi` built the declared sequence field but its `declared_surface` merge kept every incumbent field by name, discarding the stronger declaration for the same storage identity. The fix advances only a raw span whose exact storage identity, first column value ID, field slot, and dtype agree with the newly materialized sequence descriptor. The revised record stands on the incumbent record, declared ABI, and sequence descriptor cells. Any disagreement refuses by name.

Observed: Before the fix, the focused canonical `array('Q')` method relay raised a record/sequence descriptor conflict. After the fix, `tests/test_ir_sequence_tables.py::test_declared_array_receiver_refines_raw_span_before_method_linking` passes; its identity-book history shows the same record field transition from raw SPAN to SEQUENCE, retaining the `uint64` first column. Corpus `c624_array_receiver_relay.py` is the authored source.

Observed: The complete Token Orchard `Vocabulary.intern_batch` lowering audit was run once against this branch through the canonical entry. It passed the former `_intern_validated_span` -> `_cover_span` `exact_begin` refusal and reached a separate named frontier: `token_orchard_whole_core_entry` callsite 5 -> `intern_batch` requires `Vocabulary._candidates` sequence member `('column', 2)` but the bound caller record 3 names no such member.

Unknown: That new `_candidates` caller-record mismatch needs its own provenance trace. This repair does not claim native execution parity or clear later arena, heap, or ABI frontiers.
