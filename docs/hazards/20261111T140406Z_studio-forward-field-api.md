2026-11-11 STUDIO OPT-IN FORWARD FIELD API (Windows, measured):

Added receive_light_field as a strict top-level MRP1 boolean. Omitted and false
leave canonical config unchanged; true participates in the ordinary digest and
requires fixed:3, one camera, no tiled/oversampled or camera-pose request.
The existing owner-authorized photo GET route now streams /api/marble_photo/field.
It requires photo_read and a claimed job even for submitters, with no legacy
ownership fallback. Only the current status-advertised unique field name is
accepted. Schema, layout, color space, digest, live-control revision, bounded
integer dimensions, finite ordered bounds, and exact file size are checked.
Incomplete inactive, canceled, failed, wrong-name and stale fields fail closed.
Complete still fields remain readable to owners. No renderer or estimator changed.

Measured: python -m pytest -q tests/test_meowdoku_forward_field.py
 tests/test_meowdoku_marble_photo.py tests/test_meowdoku_guest_photo_read.py
 tests/test_meowdoku_access.py: 86 passed in 7.41 seconds, one existing cffi
DeprecationWarning. Initial focused run had 51 pass and one new HTTP test fail
because access.py's PHOTO_READ_APIS had not yet listed the field route. Adding
that route to the existing gate fixed it; no read/submit privilege was expanded.
No GPU producer, browser integration, deployment, restart or public HTTPS claim.
Concurrent server CRON and deployment edits were preserved and are outside this
receipt. Agent photo options use a separate camelCase allowlist in
web/marble_studio_agent.js; the browser owner is handling its packer integration.

Follow-up integration correction: Spectral's live worker reports state='live',
not 'exposing'. The field gate now admits live alongside exposing/complete,
while yielded remains unavailable. Added an actual HTTP live-state test with
revision 7, non-cubic [2,3,4] dimensions, signed float32 payload, exact response
length and full byte-for-byte body verification. The lifecycle test now uses
the real live state and explicitly rejects yielded. Same four suites: 87 passed
in 7.80 seconds. No live service was touched.
