# Removable table skirt

New rigs have a black-felt skirt around four table sides. The table controls provide an enable/remove checkbox and shared material selector for both sides. Existing saved profiles missing skirt data remain skirt-free. Explicit skirt settings roundtrip through lighting profiles, C/D and derived Room overrides.

Geometry is four thin closed cuboids outside the tabletop edges, from underside to 0.02 Studio units above the floor. Inner and outer faces share the selected material. The center/bottom remains open for the underlight; side-directed rays are intentionally blocked. The skirt is static geometry, not cloth simulation. It uses studioTableFaces, the existing source for Phong and ray scenes, with 48 extra triangles and normal scene material packing. Part metadata describes each panel separately.

Focused Node table tests verify normals on both sides, open center, material selection, disable/removal and validation. Actual full-scene harness verifies 48 extra ray triangles with black-felt PBR, and exact scene restoration when removed. Studio lighting tests verify controls and saved-profile roundtrip. Syntax checks pass. Python wrappers: 3 pass, 1 fails importing _derive_lights_from_fixtures from the current sibling spectral-analyzer module; no transport code changed here. Browser UI was not tested in this release.

Only marble_studio_table.js published. HTTPS hash verified 8207353cf0520db32373e23870611e45cc2346e41f428a8a601c01749f579370; 250 other published files unchanged. No restart, no user tab interaction, no active render interruption, no saved-object mutation. Receipt build/dot-table-skirt-20261110/publication.json. No commit/push.
