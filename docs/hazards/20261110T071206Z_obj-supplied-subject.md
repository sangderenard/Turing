# Supplied OBJ subject integration, 2026-11-10

Source-only work, not released. Studio now has a supplied OBJ library/upload/select
surface. Imported subjects reference server assets and retain the actual normalized
triangles. The existing Studio material editor is an explicit shell override; raw
OBJ/MTL materials remain retained in the source asset. Flat face normals preserve
facets. There is no geometry repair, decimation or procedural replacement.

Imported mass/inertia are unavailable, not estimates. Both Studio's sphere-contact
preview and the particle application's disk/sphere laws reject imported subjects.
The existing renderer and geometry digest path are used. Owner-isolated backend
asset access and photographic source binding are a separate coordinated patch.

Actual Stanford Bunny parser output: 69,451 triangles, 223 boundary edges in five
boundary loops; no zero-area float32 faces, nonmanifold edges or duplicated directed
edges in the measured source. These edge checks are not embedding/solid certification.
Its source SHA256 is 1eb35d1e21ce99e5ce911353b6be278990713448dd9e8f5c9387f9de39b32205.
Research and free redistribution with attribution are permitted by Stanford; no
commercial license is claimed. Source and terms are in the trial README.

Passed: imported object snapshot/tamper/mechanics tests; OBJ UI tests covering
active exposure/live rays/particle simulation, newer edits, invalid input, valid
load; existing immutable snapshot suite; eight existing download tests; Studio
checkpoint, physics and optical CPU checks. Actual full bunny snapshot construction
retained all triangles in a 1,344-byte reference snapshot in 262ms.

Two pre-existing failures remeasured against exact pre-edit source backups:
- marble object check: distant proxy retains nested triangle surfaces.
- Studio scene VM check: missing shopEmissionProfiles binding.
Neither is claimed as a regression from this patch. No full suite was run.

Narrow frontend pre-edit backups and patch: build/dot-obj-subject. Live source
contains unrelated unreleased work, so do not publish whole dirty source files.
Actual browser/render checks and owner authorization gates remain before release.

Actual isolated GPU reference now passed using the existing Studio-subject renderer
and canonical Studio draw functions: 69,451 transparent triangles, zero pixel
differences, 19,438 visible pixels, repeated draw, split phases and state restoration.
This is an authored OBJ reference fixture under an existing captured Studio rig,
not a claimed live Studio capture or ray-traced bake. Browser screenshot saved to
C:/Users/alber/scratch/stanford-bunny-trial/studio-bunny-reference.png. The separate
QA tab was closed after capture; user's active Studio was not changed.
