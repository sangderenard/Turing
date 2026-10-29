2026-10-29 Source-owned native exception route selection (MEASURED, codex/exception-admission):

- The direct authored `try: y = int(x); except ValueError: y = 2` lowering now emits status comparisons for source-selected handlers. Each comparison records the checkpoint, source status route, exact handler identity, predicate, and successor blocks in `SOURCE_EXCEPTION_SSA_ROUTE`. The route audit reports no finding on this focused case.
- The handler selected by the source route is required to reach the handler's forked canonical cell before the SSA comparison is admitted.
- Targeted gate: `python -m pytest -q tests/test_source_exception_admission.py tests/test_metadata_copy_control_ownership.py tests/test_native_float_int_status.py` completed, 25 passed in 6.58s.
- `python -u tools/audit_identity_concordance.py view` still fails after 1.7s with `selected return has no exact source call reachability authority`; this is the same pre-existing failure observed before the route edit. It does not establish a clean broad audit.
- Authored handler bodies, handler joins, pending finalizer outcomes, and native parity are not yet executable through this route. The public admission gate remains closed and the SSA lowering reports `native_try_dispatch`.
