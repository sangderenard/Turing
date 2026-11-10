2026-11-09 PHOTOGRAPHY TABLE AND BACKGROUND CAN (MEASURED, local Windows source):

New Studio sets include a closed neutral clear-glass slab, top Y=-1,
thickness .12, width5.2/depth3, with four closed supports to floor Y=-4.02.
Subject/camera target and hashed local marble geometry are unchanged. Table
materials are selectable from actual catalogue IDs and use the same live PBR
rows in preview/photo. New profile table state saves material/enabled; missing
state preserves the former table-off floor and exact authored fixture poses.
Interactive table toggles translate only the four floor practicals by3 units.
New practical defaults are grounded from their actual complete housing bounds.
Concentric layers and backdrop sweeps consume the actual lowered floor.

The background can now extends1.2 units forward, open on-axis, around its
unchanged emitter. Existing kicker shroud dimensions stay unchanged. CPU ray
checks show axial clearance and camera-side occlusion. Circle geometry now
carries its documented light group ID. Emission winding correction is owned
separately by the ten-light worker; no spectral renderer was edited here.

Critical contact finding: client nominal scale_m=.025 is not authoritative.
The real Python decoder uses the saved object's physical radius. Table-enabled
scene/camera conversion now uses a radius captured on successful save/verified
load; an unsaved DOM radius cannot move furniture away from that saved mesh.
Table-off retains the former .11/3.45 conversion and 18-degree photo FOV.
Table-enabled Current Studio/live cameras use the raster's actual 360/7-degree
vertical FOV, avoiding the cropped sphere caused by pairing radius-correct
units with the old close-up angle. Built-in framing is unchanged. Existing backend
focus bounds remain .02..10m, with clear client validation for too-close macro
views before photo upload or live camera updates. Zoom out or use built-in
framing; no silent focus clamp or altered optics.

Transparent furniture and marble use one cached world-space triangle sort,
with opaque furniture rendered in the depth-writing pass. This remains Phong
centroid-sorted transparency, not traced refraction or order-independent
intersection handling. Repartitioning follows shared material opacity changes.
Profile invalidation happens after all scene state/meshes are applied.

Passed focused checks: table closed/winding/contact geometry; both lowered and
legacy room bounds/window/watertightness; production sceneForPhoto and MRP1
packer; real decode_upload for10mm and25mm saved spheres with identical local
SHA and contact error below1e-9m (2 Python tests); table/default/legacy/material
profile roundtrips and atomic invalid-profile rejection; combined transparency
cache/sort; background can ray checks; axial saved geometry/defaults; existing
lighting profiles; light-strength component;31 photo lifecycle checks; JS
syntax and scoped tracked diff whitespace. No browser, actual GPU exposure,
full suite, export, server restart, commit or publication by this worker.

The table is photography furniture. Native particle-course collision and finite
table-edge rolling dynamics are unchanged, and the UI says so.
