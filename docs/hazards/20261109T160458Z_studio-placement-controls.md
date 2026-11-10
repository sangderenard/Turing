# Studio table dimensions and fixture placement controls

Source-only on canonical Windows checkout. Table width X and length Z accept
2..20 Studio units; height accepts .30..10 and means floor-to-top. The default
is 5.2 x 3 x 3.02. The contact plane remains Y=-1; height changes the room floor,
leg length and the existing floor-mounted fixture translation. Legs retain
their .25 edge inset and .18-square footprint. Marble geometry, camera and
physical-radius conversion are unchanged. Old v1 tables without dimensions
use the original geometry; profile loads preserve explicit world poses exactly
and never rewrite stored profile bytes. A height edit that would exceed the
existing pose Y bounds is rejected before either table or poses change.

Independent stage-left/right arc and radius fields derive from the existing
pose X/Z, around subject origin. Arc 0 is +Z/front, -90 is -X/left, +90 is +X/right.
They preserve Y, saved manual yaw/pitch and aim-at-subject lock. Three-decimal
display formatting never rounds the saved pose or untouched polar coordinate.
Top-fill Y accepts 0..50, with 0 meaning subject-centre height. Advanced and saved
below-zero poses still load exactly. No second shortcut state enters profiles.
Fresh-only background and kicker X/Z defaults retreat 20%; the existing manual
pre-floor aim/mounting semantics remain unchanged.

Guaranteed invariant: the marble-bottom/tabletop contact remains unchanged.
Advisory checks separately warn about the whole stage-stand footprint, raised
or lowered stand bases, disabled-table support, table/room-wall overhang,
underlight/floor or tabletop clearance, and top-fill/unit-sphere overlap. They
do not claim comprehensive collision detection and never relocate fixtures.
Top-fill clearance currently uses a unit sphere at origin; any later independent
subject-scaling change must supply the changed centre/radius to these checks.

Passed: all eight Node suites for Studio lighting, Studio table, table lights,
room, photo room, lighting poses, lighting profiles and gobos. Added event,
dimension-validation, migration, profile roundtrip, malformed-input, rollback,
Advanced sync, footprint, support, unit-subject overlap and actual MRP1 emitter
geometry checks. Resized minimum/default/larger tables preserve shared
preview/photo/live geometry, supports and unchanged subject hash/camera.
All five touched JS modules pass node --check. Scoped whitespace check passed.

Python CPU gate: all 3 tests in test_meowdoku_table_lights.py and
test_meowdoku_marble_photo_room.py pass in 1.01 seconds using the established
C:/Users/alber/AppData/Local/Programs/Python/Python311/python.exe interpreter.
This covers actual MRP1 decoding, table/subject contact and Spectral's real
ten-emitter fixture reader. The first attempt in Turing's .venv passed the two
room/table tests but could not import Spectral because torch/nn is unavailable.
That venv also lacks pytest-timeout; the successful gate used an external
45-second subprocess timeout. No dependency installation or unrelated repair.

No browser, GPU job, production API, server restart, deployment, commit or push.
The active user live view was untouched. Browser QA/publication remain with
the integration owner. Percentile/retouch/live-display sections were not edited.
