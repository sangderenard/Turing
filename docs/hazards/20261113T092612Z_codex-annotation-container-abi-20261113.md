2026-11-13 Captured owned-record source lineage is not a current ABI grant (MEASURED, record source proof):

Public result projection rewrites current Ret/record fields. Re-running the ordinary source-return reader afterward cannot prove the original source because it correctly requires the original value to remain in Ret.

Added a separate read-only validator for the captured immutable source record/constructor/keyword/return-slot facts and the unchanged source sequence descriptor. It checks exact function/table identity, owner, field source, operand provenance and constructor ancestry. It deliberately does not claim that the current Ret returns the original value, nor authorize public projection, ownership transfer or string payload decoding.

Tests: two source-proof tests pass. The new case removes the current Ret and proves that historical source lineage remains valid while the ordinary live-return reader and public ownership reader both refuse. Wrong record, field, sequence, owner, source and constructor keyword evidence are rejected. Existing source-proof behavior remains covered. This adds a new inspection boundary; no existing compilation failure is claimed fixed by this helper alone.
