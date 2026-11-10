# 2026-11-10 ISOLATED LIGHT FLOOR STAND GEOMETRY

Measured on Windows at f998b80f with the existing dirty working tree. Added only
marble_light_stands.js and its focused Node test. No renderer/profile/material
integration, browser session, publication, restart, commit, or push.

The pure layout derives a housing-side/rear arm, mast and three floor feet from
the existing fixture pose/dimensions, room floor and optional actual fixture
vertices. The existing triangle tuple contract carries one caller-resolved
shared material ID. Five capped twelve-sided pipes add 240 triangles per light.
The default material name is black_matte_paint; that named material still needs
a catalogue append and the existing catalogue-to-photo remap, not emitter IDs.

Passed: focused stand Node test, syntax checks, incumbent table-light and
fixture-component Node checks. Tested actual can/grid primitive vertices for
8-by-8 downward overhead with depth-1 grid, 4-by-4 upward lower fill, round
fixtures, oblique aim, small fixtures, side selection near a room wall, explicit
unavailable layouts, floor contact, outward winding/normal agreement, exact and
native-f32 closed per-pipe edges, and no mutation of input pose/mesh/bounds.

Not certified: combined pipe union topology, mechanical stability, intersections
with table/skirt, other lights, subject or props. Layout has finite candidate
mounts and returns a visible reason when none fit supplied room bounds; faces
throw on that unavailable result. A caller must surface it, not silently omit
the stand. Input fixture vertices must exclude this stand. Existing saved rigs
need explicit migration/opt-in policy before integration, and the existing
5000-triangle capture budget must be addressed before adding all ten supports
(2400 extra triangles). Parent froze this module pending those integration
decisions. No shared renderer files were edited by this worker.

## Source integration and narrow stage, 2026-11-10 03:03 UTC

Parent subsequently authorized integration. Added marble_light_stand_settings.js
with canonical lightStands version1 enabled/material settings. Fresh rigs enable
stands; absent legacy records disable them without moving poses or changing size.
Missing enabled material rows raise an explicit error, never a felt/gloss fallback.
Shared A UI and the existing B leaf resolver independently own enabled/material.

Source marble_studio.js now shares one stand builder among Phong, the photo
catalogue-material remapper and source-only static capture. Opaque/transparent
stands use the existing material classification; ordinary transparent stands join
the table/subject sort. Particle-mode stand glass is a separate sorted set, like
the existing cage glass. Frame errors are visible and the animation can recover
after editing controls. No failed stand layout returns partial geometry.

Measured all ten actual new-rig stands after subject-lock resolution at both
10mm and304.8mm units: every tested placement fits the room/floor. Current lower
fill is4x2 and overhead8x8 with depth1 grid. This does not certify table/skirt,
subject, other-stand collisions, mechanical stability or combined pipe union.
Those limitations are explicit in the UI.

Eleven focused Node suites passed, including real shared Phong/photo vertex
identity, exact material remap with zero-emission stand row, transparent finish
classification, byte-identical A photo payload under B stand overrides, UI
save/load and legacy preservation. Syntax and scoped whitespace checks passed.
The two Python photo-room/table upload wrappers passed after correcting legacy
VM stand-disabled adapters and JSON stdout pollution. The separate Python
ten-emitter wrapper cannot import _derive_lights_from_fixtures from the current
sibling Spectral worker; no sibling repair was attempted. Browser not tested.

Staged main starts from current published bytes, not source main. Reproduce with
python build/dot-light-stands-20261110/stage_main.py. Exact patch/report are in
that directory. Staged SHA256 is
083ec44498f267bbc05acc135e923485f81556eb4c15bc682d69acca77f614c3;
published base is
165ae0924a2aa503dd26039fe79f63b0a7892b6a2337d02ccc4552f1be20d1b4.
Bridge tests and syntax passed against this stage at both unit scales. It omits
unpublished DOF, viewNear, chase/player, capture and material metadata changes.
No scale module was copied. Parent owns remaining module copies, catalogue
assets, backend cap/restart, hashes and publication. No publish or restart was
performed by this worker.
