2026-11-09 EXACT FIXTURE SOURCE COMPONENTS (MEASURED, canonical Windows):

Metadata-only refinement in buildFixtureFaces and tableLightModifierFaces.
Existing fixture/<role> IDs, body IDs, all rendered vertices, triangle order,
normals, materials and group IDs are retained. Nested components identify
housing back/individual wall strips, emitter, shroud strips, stand base/post,
and each local-X/local-Y grid bar by exact role-local triangle ranges.
componentGroups retain the complete housing/shroud/modifier range and original
construction arguments, including solidBox axes, center and half-extents.
All components report contactAdmission=unverified; no closedness, convexity or
embedding certificate is invented. Shared-body union ownership remains required.

Measured current default source fixture builders: 10 roles,660 faces,180 exact
component ranges. Every source triangle belongs to exactly one component. New
metadata and legacy primitive sequence compare identically, including signed
zero. Modifier outputs also compare identically for a rotated grid case.

Truthful blockers: each open box housing has4 same-direction paired edges;
each tube housing has12. These winding inconsistencies persist in nativef32.
The smaller emitter is a separate sheet, not a cap closing the housing. Round
closure arithmetic can leave source-coordinate differences; exact nativef32
indexing removes some but not all extra boundary edges. The underlight's192
modifier triangles form intersecting cuboid bars:4 nonmanifold exact edges
and13 edge-connected components at source and nativef32. Stand base/post are
separate24tri construction boundaries; their union/contact seams need proof.
No geometry was rewound, welded, flattened, omitted or repaired here.

node tests/meowdoku_studio_fixture_components_check.mjs passed and prints
source/nativef32 adjacency measurements. --json emits the full default role,
part/component and topology packet for native integration. Existing exact-scene
and table regression scripts also passed. A test VM initially included an
exported declaration in a script context; its extraction was corrected.

No active UI edits, frame behavior changes, production exports/restarts or
contact activation. Source callback regions are frozen for the native binder.
