2026-10-29 TITLE (MEASURED, where): 
2026-10-29 EXCEPTION DECLARATION IDENTITY, DRAFT PR #1166 (Windows)

Observed: the reducer's ast.Try arm reduced each handler body but never resolved
the handler's exception declaration. The codec's
`except (TypeError, ValueError, OverflowError)` therefore reached later passes
as bare Name operands, with no static class identity for native dispatch.
The reducer now resolves each declaration in the handler's lexical scope and
rewrites the existing type operand edge to the static reference value. A
concordance `static_reference_use` row ties each authored occurrence to its
`static_symbol_definition`. A focused graph probe showed a tuple handler
resolving to `builtins.TypeError` and `builtins.ValueError`; a separate single
handler probe showed `builtins.ValueError` with the handler's `type` parent
pointing at the static reference value. These were graph/concordance probes,
not compiled native artifacts.

The source Try has not yet been composed into a native loop control block, and
protected call failure edges are not yet lowered. The original codec remains
blocked at its `Raise` in the handler. `python -u
tools/audit_identity_concordance.py view` remains red with `selected return has
no exact source call reachability authority` (exit 1); no audit green claim.
No broad suite or full codec lowering was run.
