2026-11-09 INDEPENDENT MARBLE SCALE (MEASURED, canonical Windows source)

The saved object's physical radius-mm remains the only subject-size source.
The size slider writes that existing field and emits its normal bubbling input
event, so autosave, physical mass and inertia follow it. Normalized geometry,
analytic/bubble descriptors and material records are not rescaled or rewritten.
The range is 0.1..200 mm; existing bubble admission still requires >=1 mm.

New lighting profiles own scale={version:1,mode:'fixed',unitMm:10}. A profile
contains no subject radius. Fixed unitM converts the complete room, furniture,
fixtures and orbit camera to meters. Preview subject radius is radiusM/unitM;
table-on centerY is -1+radiusM/unitM, preserving bottom contact at table Y=-1.
Photo/live object_center_m is OBJECT_CENTER plus that center times unitM;
the server still derives scale_m only from the authenticated saved radius.
Table-off mode retains origin placement; there is no new floor-motion rule.

Omitted scale in old profiles migrates in memory to explicit legacy mode,
without writing browser storage. Legacy table-on uses saved radius for the
rig unit; table-off retains exactly .11/3.45 meters and 18-degree photo FOV.
Legacy physical photo camera/geometry stay unchanged. Phong now shows that
actual physical ratio/FOV rather than preserving its former mismatch. The UI
labels this. Choosing Independent explicitly freezes the then-current rig unit.
Fixed mode uses Studio's 360/7-degree vertical FOV. Profile loading never edits
the object's radius, poses, materials or saved geometry.

Radius edits do not auto-move/reframe the orbit camera. Frame marble explicitly
targets its current center and fits its radius with the viewport aspect ratio.
The built-in close-up retains its existing distance/FOV and follows the subject
center, including table offset. Very large marbles can exceed the room or table;
advisories warn without relocating or clamping geometry. Top-fill height is now
relative to the current marble center, while its existing absolute pose remains
the sole saved value. Resizing updates the readout without moving the light.

Temporary blank, zero, nonfinite or out-of-range radius input holds the last
derived valid preview transform and displays validation. A corrected value
recovers immediately. Raw save/photo validation is not replaced by that cache.
The scale controls belong to lighting, not object autosave. The save-lifecycle
owner separately added photo.sceneChanged(), preserving the accepted object
digest for rig-only edits; this worker did not replace that lifecycle.

Tests: ten Node suites pass: marble_scale, marble_studio_lighting,
marble_table_lights, marble_studio_table, marble_photo_room, marble_gobo_scene,
marble_light_strengths, marble_photo_camera, marble_materials and
marble_photo_retouch. The scale suite executes production scale/camera/scene
and slider/frame handlers, authentic generated bubble geometry, immutable PBR
and mesh receipts, r^3 mass/r^5 inertia, fixed rig/camera, contact, malformed
input recovery, migration and equivalent live/still wire options. Profile tests
cover old/no-rewrite, new roundtrip, atomic invalid units and tiny-marble fill.

CPU Python gate: 28 passed in 5.10s across test_meowdoku_marble_scale.py,
test_meowdoku_marble_photo_room.py, test_meowdoku_table_lights.py,
test_meowdoku_marble_spheres.py and test_meowdoku_air_bubbles.py. Used installed
C:/Users/alber/AppData/Local/Programs/Python/Python311/python.exe and an external
45-second timeout. Actual MRP1 decoding and Spectral prepare_photo_geometry
confirm 1/10/25/200 mm radii, translated exact outer/bubble centers and radii,
unchanged PBR/owner shade bytes, contact and fail-closed unsupported metadata.
No Spectral changes were required. Initial new-test fixture/schema assumptions
were corrected before the final gate; they were not product regressions.

JS syntax and scoped whitespace checks pass. The new module is in web_export.py
and server.py's no-store list. Source-only: no browser QA, GPU, production API,
active-job interruption, server restart, export, deployment, commit or push.

Follow-up after coordinator browser QA: the preview transform guard recovered
from an empty radius, but the separate mass input listener still called
marbleMass with zero and raised before its size listener. Physical input
listeners now validate before either reportMass or bubble rebuild; reportMass
also guards entry from independent material/geometry refreshes. Invalid raw
data is retained for strict save/photo rejection, and the mass display states
that it is waiting for a valid radius. No error catch around actual valid-mass
calculation or fallback physical radius was introduced.

The scale regression now executes the actual listener registration, covering
empty/zero/nonfinite/out-of-range inputs with and without bubbles, density
edits while radius is incomplete, direct reportMass entry, and valid recovery.
All six rerun Node suites (scale, lighting, materials, camera, retouch, light
strengths) pass; the two actual-wire Python scale tests pass in 3.72s. This
worker did not perform browser QA or publish the correction.
