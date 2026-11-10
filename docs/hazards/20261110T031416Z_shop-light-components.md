# 2026-11-10 ISOLATED SHOP LIGHT COMPONENTS

Source-only marble_shop_lights.js, one Node geometry/config test and one Python
native-emission test. No main/profile/UI/material-table allocation edits, no
publication, browser, GPU render or server restart. Current shared code changes
belong to other workers.

Authored nominal models have real8-foot emitting tubes x2 or4-foot tubes x4.
Opaque housings add3inches overall; these are artistic fixture dimensions, not
a measured product claim. A selectable32-package LED array replaces each model's
tubes. Silver backing and black top/mounts reference metal_silver and
black_matte_paint in the shared catalogue. The mount reaches the actual ceiling.
Physical feet convert through the caller's unitMm. A model that does not fit
raises with its stable fixture ID; saved room scale and pose never move silently.

The existing writeStudioFixtureRows owns all emitter coefficient writes. Callers
must allocate distinct appended shared rows, one per shop fixture. Dimming writes
relative radiance, not calibrated lumens, and never changes the geometry. Exact
emitter triangle IDs/areas are returned for mesh emission. Photo off writes zero
emission while retaining tubes/housing. Burst exposure refuses explicitly until
the original timeline/timing-packet route is integrated; there is no fake flash,
averaged gain, timer or dummy burst UI. Absent old config becomes an empty rig.

Node checks passed real tube lengths/counts, mounting, backing/top materials,
LED source swap, dimming, migration, ID stability, immutable inputs, physical
unit conversion, complete triangle ownership, winding/normals, closed per-part
edges (including f32 tubes), actual emitting area and explicit failures.
Both default tube fixtures total360 triangles; both LED fixtures total3144.

Native CPU proof passed in2.99seconds through the incumbent Spectral
prepare_photo_geometry + native_emissive_sources path. All144 tube emitter
triangles and768 LED emitter triangles produce positive native spectral roots
on their actual surfaces, with outward directions. LED directions point down;
tubes emit on both sides. The adapter's spectral energy checks passed. Zeroing
emission retains the same scene triangles and generates zero roots. The first
test used10000 total triangles but the existing camera housing made the complete
scene45449 triangles; the proof now uses the standard250000 budget. No hidden
cap was bypassed and no source geometry was reduced.

Pending integration: preserve the old10 fixture/housing/label IDs and reserve
shop rows after the existing12 runtime slots; continue accepting saved+12/+9
table layouts. Add stable-ID profile leaves/UI after material-worker release,
collect actual shop parts as room group and use that shared compact-ID remapper.
Extend the single material editor with backing/top targets. The native source
path must keep exact emissive triangles. Phong may use explicitly labelled
surface-area quadrature within its100-light capacity, not replace native meshes
with guessed point sources. The2MiB config preflight remains authoritative.
Original shutter-timeline owner was requested from inspect_pluck_player.

Not certified: inter-fixture/prop collision, mechanical mounting strength,
combined component union, measured lamp spectrum/lumens, browser appearance or
time-dependent shutter exposure. No completion claim for those steps.

## Source integration and narrow release stage

Shop source integration completed 2026-11-10. New controls edit the same records
used by geometry and emission. A/B profile leaves are stable-ID keyed; removing a
fixture prunes only its obsolete B overrides. Old profiles without shopLights
remain empty. New rigs get both tube models. Global backing/top selects use the
existing shared catalogue and single room material editor, never runtime source
rows. Photos retain all source geometry with emission zero. No burst editor was
added; a restored unsupported burst request raises before writing any live row.

Runtime buffers reserve20 slots: unchanged original10 emitters/housing/label,
then8 shop rows. Saved +9 and +12 layouts remain accepted beside +20. Scene parts
join the existing room owner/compact map. Studio capture uses the full same mesh.
The existing shader receives at most90 deterministic area-weighted triangle
surface samples plus the10 original lights. Sample weights sum to exact emitting
mesh area and normals are outward. This is an explicitly labelled Phong
approximation, not native root truncation or measured radiometry. Native emission
still uses all144 tube or768 LED emitting triangles.

Fifteen focused Node suites passed: shops, shop controls, actual shop bridge,
materials, stands bridge, full Studio lighting, profiles, scale, photo room,
table lights, table, Studio scene, gobo scene, scene material engines and macro
framing. Python shop native emission and photo-room wrappers passed3 in4.08s.
Actual shop bridge proves canonical material/first12 rows unchanged, identical
Phong/ray vertices, zero photo emissions with intact tubes, room part ownership,
opaque/glass partition and byte-identical A photo payload after B source, dimmer,
filter and body-material changes. Actual default both fixtures fit room6/foot
scale. Unsupported placement is an actionable fixture-ID error.

Release staging script: build/dot-shop-lights-20261110/stage_release.py.
It narrow-patches current published main/lighting/materials/profiles, copies two
new shop modules and writes manifest.json plus readable patches. It never
publishes. The main stage excludes unpublished chase, DOF and scene-capture
changes. Material stage excludes unrelated authoring-key and append-only prefix
migration changes. Stage must be regenerated if the parent publishes its kicker
slice first, and base hashes checked immediately before copying. Shop and stand
bridge suites passed against the initial stage main. No browser/GPU appearance
claim or source/full-file deployment was made here.
