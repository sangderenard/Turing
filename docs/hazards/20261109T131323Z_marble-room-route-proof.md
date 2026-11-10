# MARBLE STUDIO AUTHORED ROOM ROUTE PROOF

2026-11-09. CPU-only inspection and focused tests; no production source change.
The bridge's commented-out room is unused legacy procedural geometry. Current
Studio sceneForPhoto already includes checkerboardFaces and backdropRollFaces
with real material rows in fixtures. The worker selects scene_object=None and
imports that uploaded fixture mesh once at identity scale/center.

New tests/meowdoku_marble_photo_room_check.mjs executes the production sceneForPhoto
function with isolated scene inputs, actual backdropFaces and packMarblePhoto.
New test_meowdoku_marble_photo_room.py proves actual decode_upload/_config preserves
all faces/material rows exactly once: 1 passed in 0.31 seconds. The corresponding
Spectral worker test proves no procedural room and one exact float32 mesh import;
46 tests in test_marble_film_response.py passed in 3.42 seconds.

This proves CPU routing, not GPU shadow sharpness or image quality. No user
exposure was interrupted; no GPU render, restart, deployment or compiler run.
Historical display_white ~5.5e5 is not a measured gain factor; accumulated HDR
values and the old multi-bounce instability were not reproduced in this audit.
The old room/comment was intentionally left unchanged.
