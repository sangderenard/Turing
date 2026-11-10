2026-11-09 TEN-LIGHT TABLE RIG (MEASURED, local Windows source):

Simple lighting exposes strength and filter for all ten roles; detailed pose,
key shape and debug coordinates move into collapsed Advanced. Fresh filters
are all white. The appended tabletop snoots default to strengths 6 and 6;
the rectangular gridded underlight defaults to 1.5. Original strength defaults
are unchanged. Studio/exposure remains linked unless explicitly split.

The two small round snoots have real tubes and closed base/stem supports.
Their bases rest at tabletop Y=-1, with emitters at X=+-1.85,Y=-.72,Z=.10.
The underlight is a 2.4 by 1.8 rectangle directly below the subject, emitter
Y=-1.4, with sixteen closed black grid bars forming 6 by 8 open cells. Its
grid remains below the .12-thick tabletop. Housing, grid, emitter and support
triangles are shared by Phong, photo and live-ray scene packing. The Phong
direct-light approximation does not ray trace table/grid occlusion; the UI
states that limitation. Ray underlighting depends on the tabletop material.

All ten roles have independent pose, subject lock, filter, Studio power and
optional exposure power. Legacy seven-role profiles validate first, retain
their exact seven poses/filters/controls and existing exposure overrides,
then append three white, linked, zero-strength roles. They do not enable the
new table or default backdrop. Storage is neither rewritten nor resaved on
read. Runtime material rows are twelve: ten emitters, housing and debug label,
after all authored catalogue/private rows. New catalogue fixture IDs 38..47,
housing 48, label 49 do not overlap the original saved IDs 32..34 or finishes 35..37.

Emitter direction correction: the existing quad/disc triangles had winding
opposite the explicit Phong normals. Spectral derives source direction from
winding. The shared fixture writer now reverses ONLY the emissive triangle
order for all ten roles, retaining vertex positions, dimensions, materials,
strengths and nonemissive housings. Old profiles retain their authored state
but their photo sources now point in the authored direction, not its inverse.

Limits measured: canonical/generated base_material MAX_LIGHTS is 100. The
photo explicit fallback light-list validator caps at 8, but Studio uploads use
fixtures, which supersede that list and derive all unique emissive material
sources without that cap. Actual ten-source decoder/derivation proof passes.
Fixture upload limits remain 5000 triangles and 64 materials. No limit was
silently bypassed or source omitted. No shader/export/restart/exposure occurred.

Tests: meowdoku_marble_light_strengths_check.mjs passes all ten real material
writer/direct uniform/payload lanes, independent zeros and split/link behavior,
invalid input and calibration isolation. Calibration now zeroes all ten and
relinks exposure levels before applying each single-light recipe. Studio
profile tests pass exact tinted seven-role migration, new ten-role save/load,
corrupt atomic rejection and surface/table state compatibility. New table-light
geometry test passes all ten emission normals, snoot opening/off-axis blocking,
table contact, closed grid-bar winding, open cells and shallow-ray blocking.
test_meowdoku_table_lights.py passes actual MRP1 decode and canonical Spectral
derivation with ten sources, correct all-ten directions, and all-off zero sources
(1 passed in 0.68s after final winding edit). Existing table, materials, room,
pose, profile and live lifecycle checks pass. JS syntax/scoped diff checks pass.
Parent owns combined browser/GPU validation. No production publication occurred.
