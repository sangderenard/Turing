2026-10-27 Dynamic append-slice reserve path (MEASURED, repair worktree)

- Bounded canonical source: `values=[]; for i in range(n): values.append(i+3); selected=values[:end]; return selected[0] if len(selected) else 0`.
- Before this change, `lower_sequence_append_slice` rejected a valid DYNAMIC destination with an explicit owner, scalar unkeyed rows, and no live flags. The copied slice needed capacity beyond the destination's current extent.
- The helper now computes the Python-clipped slice count, asks the existing `lower_sequence_reserve` for `old_length + clipped_count`, copies only after a successful reserve, and publishes the new length after the copy. Fixed destinations retain their capacity check and overflow error.
- `python -m pytest -q tests/test_ir_sequence_tables.py -k 'dynamic_growth_is_an_explicit_typed_shortfall or append_slice' --tb=short`: 3 passed. `python -m pytest -q tests/test_native_semantics_corpus.py -k c296_dynamic_loop_slice --tb=short`: 1 passed (C native).
- Direct minimal all-backend execution: audit0, C3, LLVM3. Fortran emission complete, but the native compile fails on scalar initialization of assumed-size array formals in the generated root. This backend issue is outside the append-slice helper change.
- Fresh admitted two-call StringCore source: lowering71.30s, audit0, C native7, LLVM native7, Fortran emission complete but compilation fails on generated internal helper symbols `t39` lacking a declaration at lines2299 and13377. Snapshot: `build/real_two_batch_after_dynamic_slice.pkl`; generated source: `build/real_two_batch_after_dynamic_slice_fortran/turing_fortran_c_shell.f90`.
- The full `tests/test_ir_sequence_tables.py` gate returned 40 passed and 15 failures on the shared dirty repair tree. Failures include unrelated constructor-schema, concordance, unknown-dtype and Fortran API cases; no clean baseline was run for those failures.
- No full manager acceptance is claimed.

