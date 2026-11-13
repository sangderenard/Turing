2026-11-13 Public source entry is an actual export decision, not a name (MEASURED, source/SSA boundary):

Owned return projection must not silently change an internally called function's ABI. Added a typed PublicSourceEntry receipt at the existing non-private export writer, joining existing LOWERED_SOURCE_FUNCTION, FUNCTION_ADDRESS, SOURCE_DEFINITION, SOURCE_PURSUIT_ACTIVATION and PURSUIT_ROOT demand. Exact shared source spans join address/definition/activation; no symbol-name parsing or new export boolean substitutes for those facts.

Global audit validates authority independently of projection eligibility. A valid recursive or internally used public entry remains a valid entry, but the projection reader refuses it. Missing root demand, wrong address/definition, inactive activation, same qualname from another module, recursive/incoming calls and non-exported helpers are covered. Unsupported explicit-unit/library selection routes simply receive no new source-entry receipt.

Verification: seven combined tests pass; the same six incumbent tests pass with publisher/audit hooks temporarily removed and restored byte-for-byte. The new inspection test is not counted as a previously fixed runtime bug. Mapping remains 28 rows/2 functions/zero findings/zero unsourced identities, latch CLOSED. Actual GraphSolver still refuses only sequence100's missing public ownership escape, without native execution.

This receipt does not grant ownership transfer or prove primitive/string element identity. Native projection, exact source item binding and result lifetime verification remain separate work.
