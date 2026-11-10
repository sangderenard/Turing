# Explicit fixture triangle policy

User-requested floor stands require more than the old hidden 5000-fixture-triangle cap. `marble_photo._config` now admits fixture triangles up to the selected `triangle_budget`; combined object/fixture preflight remains authoritative. No geometry is truncated.

The independent 2 MiB config bound and 48 MiB whole-upload bound remain. `decode_upload` now checks the whole-upload bound directly as well as at HTTP ingress. Thus very verbose fixture JSON can reach the byte bound before the triangle budget; no claim that every 1-million-triangle scene fits the upload format.

Verification: geometry-policy suite49 passed; photo suite23 passed.6500-triangle fixture request accepted intact for photo/live at budget6500, rejected at6499. Oversized config header rejected before JSON parsing. Synthetic6500-triangle JSON measured247244bytes and parse+validation peak2515173bytes; not worst-case coordinates.

Source only: running Python server has not been restarted for this change. Coordinate backend release with material and stand UI integration.
