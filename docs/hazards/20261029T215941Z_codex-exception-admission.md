2026-10-29 Native handler entry Phi from concorded failure routes (MEASURED, codex/exception-admission):

- The direct two-site control case now branches ten selected status matches into one `exception_handler` block. Its status Phi has ten source route operands. Its authored `y` Phi has ten incoming SSA values, each read through `SOURCE_EXCEPTION_SSA_HANDLER_ARM`; the first site's matches carry `y=1`, the second site's matches carry `y=2`.
- `SOURCE_EXCEPTION_SSA_HANDLER_ENTRY` and `SOURCE_EXCEPTION_SSA_HANDLER_PHI` identify the entry, source handler, incoming route set, source Phi, result and exact incoming arms. The focused audit checks these records and the selected branches reach the recorded entry block.
- The entry currently re-reports the merged status to the existing fault exit. It does not execute the handler body or finalizer, and the public admission gate and `native_try_dispatch` shortfall remain. This preserves refusal while executable outcome control is built.
- Targeted gate: `python -m pytest -q tests/test_source_exception_admission.py tests/test_metadata_copy_control_ownership.py tests/test_native_float_int_status.py` completed, 25 passed in 6.51s. `git diff --check` passed.
- Broad `python -u tools/audit_identity_concordance.py view` still stops after 1.8s at the pre-existing `selected return has no exact source call reachability authority` refusal.
