2026-11-13 FIXED CHILD-POOL GLYPH KEY LOOKUP

This is a low-level, read-only extension of lower_table_lookup, not source dispatch, insertion, or new storage. An explicit query_sequence descriptor selects the existing linear row scan plus lower_sequence_equal over a selected child span. Each character is represented using the existing string_table.string_token contract. Equality compares complete token-array lengths and components; numeric token ordering is not character ordering and is not implemented or claimed.

The admitted contract is one outer key equal to the child handle column, one dense unshaped int64 child column, fixed caller-provided storage, and a readonly borrowed leaf query. Owned/dynamic tables, multi-key tables, shaped or live-flag child layouts, child status, query status/ownership, and incompatible schemas refuse before generation. An omitted explicit query cannot silently make a declared child-handle key compare scalar handles. Missing outer dtypes refuse explicitly; descriptor construction already checks handle/key column bounds.

The helper uses existing GEP/Load/Mul instructions to select the child span, then calls the incumbent full-content sequence equality function. The existing dependency bundle and unique-entry discovery register the composition. Typed metadata records the query sequence identity and child handle position. No new ABI, allocator, token encoding, or hash equality was introduced. The fixed caller ABI still requires valid initialized handle, length and stride buffers. Native fixtures explicitly establish these extents; this change does not claim hostile runtime handle-bound validation.

Storage union validates identities before deduplicating: cross-role/type/shape collisions refuse. Identical readonly glyph-data roles may share their arena; this alias is covered by native repeated reads. Scalar hash carriers are not admitted in this phase. Delete, compaction, child-slot reuse, copied insertion, set membership and source integration remain separate work. No existing scalar token or numeric lookup is rerouted to glyph equality.

The clean dd00b686 new-test baseline failed all three initial cases because query_sequence was not an accepted argument, before native execution. First candidate emitted both backends; its initial test observations incorrectly read the original host status array (rather than the normalized execution buffer) and indexed an LLVM scalar as a one-dimensional array. Correcting only those test observations produced 3 passes in 1.96s. Expanded fixture with readonly arena alias produced 5 passes in 3.07s. Native data uses existing per-character string_token values, empty table, empty key, Unicode, unequal prefix/length, 129-character key, permuted row handles, distinct handles holding equal contents, absent lookup status and repeated alias reads. There were no adversarial invalid-pointer native executions.

Mapping audit before: 28 rows / 2 functions / 0 findings / 0 unsourced, latch CLOSED (19.6s including initial bindings compile). Mapping audit after: same result in 5.1s. The direct helper native fixtures are deliberately low-level IR with no source identity book; they do not claim a source-to-native admission. Source dispatch remains protected by the resident-glyph-key refusal.

Final full candidate/baseline results will be appended before commit. Separate worktree codex/glyph-key-lookup-20261113; original annotation worktree unchanged. Existing Nodus DLL is read-only and is not rebuilt. Native binaries and test temporary data reside in user scratch.
Final frozen comparison at dd00b686: tests/test_fixed_glyph_key_lookup.py plus all of tests/test_ir_sequence_tables.py, 61 cases. Candidate: 46 passed, 15 failed in 159.68s. Exact HEAD ir_sequence_tables module loaded in an isolated baseline process: 41 passed, 20 failed in 146.64s. Exactly the five new tests improve; the same fifteen incumbent tests fail in both runs. The baseline uses the canonical module name and original __file__ resource path, without changing any shared source. Final mapping audit after the last validation changes: 28 rows, 2 functions, 0 findings, 0 unsourced, latch CLOSED in 4.7s.

Unchanged incumbent failure identities in test_ir_sequence_tables.py:
- test_authored_ordinary_class_infers_only_proven_receiver_state
- test_authored_constructor_schema_keeps_tuple_slots_and_object_references
- test_bare_tuple_constructor_field_is_an_opaque_native_reference
- test_keyed_record_value_derives_nested_enum_row_table_columns
- test_authored_schema_keeps_literal_vocabulary_and_optional_scalar_tag
- test_nested_table_pool_is_structural_and_validated_against_function_values
- test_compiled_retained_loop_mutates_caller_sequence_record
- test_empty_aggregate_const_becomes_explicit_sequence_arena_argument
- test_retained_field_sequence_populates_typed_record_from_lowered_storage
- test_declared_array_receiver_refines_raw_span_before_method_linking
- test_empty_dict_constructor_keeps_accepted_receiver_table_roles
- test_guarded_array_lengths_and_append_keep_one_resident_sequence
- test_readonly_forwarding_keeps_typed_mutable_sequence_owner
- test_array_u32_receiver_column_keeps_authored_dtype_before_append
- test_static_program_reference_is_typed_ssa_and_native_handle_storage

These include existing source record/schema, formal-occurrence and unknown-layout refusals, plus a Fortran expected-validation failure. They are not a green suite and are not repaired here. The earlier limited scalar regression subset was 27 passed, 2 failed, 27 deselected; the full paired comparison above supersedes it. Full native positive tests read returned status from the execution's normalized buffer, preserve raw buffers on readonly alias queries, and compare existing token-per-character arrays only. No token numeric-order or alternative encoding claim is made.
