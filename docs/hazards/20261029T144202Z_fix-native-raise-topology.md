# 2026-10-29 Exception identity through graph and class concordance

The full codec compile repro first failed in validation discovery with
`source raise has no lexical concordance scope`. The aggregate ProcessGraph
had `ingestion_value_scope` but no function-local `lexical_read_scope`; the
raised constructor Call retained its exact `callee` graph edge. Its
`STATIC_REFERENCE_USE` row named that callee node and the authored function
occurrence. `validation_raise_identity` now selects by those exact graph and
concordance edges on the aggregate graph. The forked function graph still
uses its `SCOPE_ORIGIN` chain because relabeling changes the callee's graph
node ID. The full codec repro now reaches the loop's explicit
`try-exception-entry-state-not-composed` / `Raise` blockers.

The direct-raise handler matcher previously read raised and handler class
identities from source edges, then called the compiler host's `getattr` and
`issubclass` to choose a handler. `ensure_builtin_class_topology` now posts
builtin exception direct-base edges on `CLASS_SUBTYPE` from static definition
cells. SSA handler selection reads that page. Native C corpus
`c1166_try_raise_handler_loop_state` passes 1/644 deselected; new corpus
`c1167_try_raise_superclass_handler` passes 1/645 deselected and proves that
ValueError reaches an Exception handler in the compiled loop. Neither test
proves protected-call failures or the codec.
`python -u tools/audit_identity_concordance.py view` remains red with the
previously observed `selected return has no exact source call reachability
authority` error. `git diff --check` and compiler-file syntax compilation
passed.
