# Expanded Studio room and publication receipt

2026-11-09, local Turing checkout; existing dirty work preserved.

Added shared pure geometry for all four walls, floor, and ceiling. Default room
scale is 3, range 1..6. Floor contact stays -1.02 Studio units; marble, fixture,
and authored mat dimensions are unchanged. Rear backdrop mounts move with the
rear wall. Exact fixed rear-window boundaries subdivide adjacent patches,
keeping paired edges and inward normals without T-junctions. Lighting profiles
save the scale; earlier profiles without the field use scale 1 to retain their
former backdrop mounting distance. Cameras outside the room see physical walls.

Focused geometry, profile, and current UI browser checks passed. The actual
sceneForPhoto / packMarblePhoto / decode_upload path preserves all 777 faces
(one fixture, 768 room triangles, eight swept-backdrop triangles), remapped PBR
rows, and fixed photo world scale. No procedural room is added by the worker.
The old commented-out config.room field is obsolete, not a missing room path.
Historical display_white growth alone does not establish physical energy gain.

Combined browser QA used read-only localhost port 8880 with source modules,
staged existing MaterialDatabase tables, and the separately validated Phong
shadow shader. Scale 3 -> 6 -> 3 left marble radius 10 mm and object triangle
count unchanged; two backdrop materials and sweep rendered. Existing unrelated
ad-filter-extension corruption errors were observed, no app shader error.
New material tables change only four PBR rows, keeping all indices/meshes.
Phong remains an approximation, and intersecting/nested-medium ray transport
is unresolved. No claim of calibrated or physically exact preview is made.

User-authorized immediate film publication was performed separately first:
specific exposure 7f5e184ddd5028c6e64d279ba0662f5826b29c29d3851cb5cf0078ad853c654c
was canceled, five tested film assets published, and server gracefully restarted
using recorded launch arguments. New PID 52488, boot 83ead214e5c7, restored one
session. Defaults now 30 s in HTML/client packer/server, preserving explicit
choices. HTTP 200 SHA verification passed for all five files. No unrelated
processes stopped. Larger room/Phong/defaults are not part of that publication.

A transient new room-route test failed at the old hard-coded ten-face count
after replacing its mocked floor with six real walls; expected count corrected
to 777 and rerun passed. This was a test expectation update, not dropped faces.
