2026-11-13 Optional span control tests consume their existing exact tag (MEASURED)

Minimal source: Optional[list[str]] parameter, `if items is not None: return 7; return 3`. Actual compile_nodes has this control edge too. _union_none_identity_subject excluded every nonscalar arm to avoid manufacturing pending tags for optional aggregate call returns. But the parameter already has an exact published native type tag, so the fallback emitted Ne(span,None), rejected as scalar/elementwise geometry mismatch.

Use the already registered non-pending tag via the existing native type predicate before the fixed-width fallback. If there is no such declared companion, retain the existing aggregate guard. No invented discriminant, source cell, tag number or payload meaning; no loosened geometry detector.

Verification:
- New source-built control regression tests:2passed; both is/is-not branches have zero concordance findings and their Eq reads the real tag named by the predicate receipt. Clean base554ac55e:2failed at the original span-geometry refusal.
- Existing native optional merge suite:7passed1failed (its optional-scalar expectation is already failing immediately preceding this change).
- Full value-union suite:45passed6failed. Four are long-standing failures; two direct-return Optional[List[str]]/Optional[list[str]] tests also fail at immediate pre-change HEAD5d022251. Combined immediate pre-change check confirms these same three selected failures; this change introduces none. Direct return and control branch follow different lowering paths and are not claimed equivalent fixes.
- Mapping audit unchanged:28rows,2functions,0findings,0unsourced,latch CLOSED.
- Actual class-equipped compile_nodes (lazy-import source experiment) changes from50 to49 final findings: exactly the Optional[list] control geometry issue disappears. Other45 unbound descriptor fields,1 string comparison geometry,1 unaccounted formal,1 undefined isinstance predicate,1 default source-provenance finding remain.

Attempted native execution, not claimed successful: LLVM emission of the minimal source reaches its independent `program_abi_field_slot` refusal, no source cell for items/PAYLOAD. Source lowering is verified; no payload-slot identity was fabricated to make emission pass. No shared-source edit, merge or deployment.
