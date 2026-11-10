2026-11-10 UPPER SOFTBOX CEILING SUPPORT (MEASURED, Windows source only)

Added marble_light_ceiling_support.js and its isolated Node geometry test.
The parent owns the existing pipeFaces export, support/profile dispatch and
default upper height edit. This worker changed no main/profile/material files,
published nothing, restarted nothing and used no browser.

The pure layout requires the actual shared fixture builder's vertices and room
bounds. Four short housing attachment pins, four vertical cables and four small
ceiling plates reuse the existing capped pipe primitive and caller-resolved
shared catalogue ID: 12 components, 48 triangles each, 576 triangles total.
Rear/side mounting uses an upward-facing housing surface. Pins remain behind
the emitting plane. Complete cable/plate cylinders clear a supporting plane of
the entire supplied housing/emitter/tool mesh. Ceiling plates reach the actual
room ceiling. Invalid/incompatible layouts report actionable reasons; no pose,
size, profile or room correction is silently applied.

Measured with current production buildFixtureFaces, default 8-by-8 softbox,
one-unit-deep 8-by-8 grid, room scale 6, 304.8 mm per scene unit:
Y=3.2: mesh Y 1.7354842806059163 to 9.242255209397305.
Y=5.2: mesh Y 3.911565509887499 to 11.078686348866759.
Y=7.2: mesh Y 5.990632676111132 to 13.003297932878079.
Ceiling Y=23.099999999999998. Requested source change 3.2 to 5.2 is exactly
two physical feet and leaves 12.021313651133239 feet above the highest fixture
vertex. The parent confirmed this baseline and owns that default edit.

Passed both new-file syntax checks, new ceiling support suite, incumbent floor
stand suite and scoped diff whitespace check. New suite checks actual builder
vertices, default height alternatives, tilted/horizontal fixtures, small sizes,
all-triangle room bounds, actual tool exclusion, mount contact with ceiling,
immutable inputs, material/group IDs, component ownership, nondegenerate outward
winding and exact/native-f32 closed edge pairs. Explicit failures emit no partial
component callbacks. Browser, full repository suite, full bridge and publication
were not run by this worker. Other fixtures, table/skirt/props, intersecting
component unions and mechanical load capacity are not collision-certified.

hazards_log.py show rejected the existing malformed filename
20261109T_material-library-paused-checkpoint.md. Read newest receipts and legacy
opening directly before tests; did not edit or rename the unrelated receipt.
