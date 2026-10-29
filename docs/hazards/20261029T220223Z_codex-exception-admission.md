2026-10-29 Complete live state at selected handler routes (MEASURED, codex/exception-admission):

- The bounded two-site Try has two different source bindings for `y` at its failure sites. Its direct SSA lower now posts `SOURCE_EXCEPTION_SSA_HANDLER_STATE` for each selected status route and each live name. The ten state rows agree with the ten handler Phi arm operands on this case.
- The new audit checks that every deployed live name for a selected route has a native state row, that it cites the route's exact site and match block, and that any handler Phi arm uses the same SSA value. The focused two-site audit reports no state finding.
- The state is recorded for future handler body reads; the current entry still reports the status to the fault exit. Handler body execution, post-Try continuation join, finalizer outcomes, nested propagation and public admission remain unfinished.
- Targeted gate: `python -m pytest -q tests/test_source_exception_admission.py tests/test_metadata_copy_control_ownership.py tests/test_native_float_int_status.py` completed, 25 passed in 7.44s. `git diff --check` passed.
- Broad `python -u tools/audit_identity_concordance.py view` still stops after 1.7s at the pre-existing `selected return has no exact source call reachability authority` refusal.
