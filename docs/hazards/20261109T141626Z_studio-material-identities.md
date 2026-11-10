2026-11-09 STABLE STUDIO MATERIAL IDENTITIES (MEASURED, local Windows):

The material exporter now reserves private coating/paint IDs 32..34 before
appending the three new room finishes at 35..37. register_material_catalog(db)
is shared with export_assets and returns (material_index, material_extras),
allowing a materials-only staging pass without rebuilding existing glyphs.
The private records do not appear among public material picker names.

Studio reads explicit extras metadata, with the legacy max-ID fallback for
older assets. Named and private rows both precede all runtime fixture rows.
The concurrent ten-light extension uses twelve runtime rows, for 50 total
records with the new 38-row export; older 32-row assets reserve private rows
before allocating the current rig too. Saved material restoration validates
IDs, all four strides, complete known layouts and finite float32 values
before any mutation. It restores only saved named rows and coating/paint
rows, preserving newly added finishes and the current live fixture rows.
Legacy 41-row snapshots and previous nine-runtime-row snapshots are accepted
without renumbering, automatic resaving, or modifying stored source bytes.

Measured: tests/test_meowdoku_material_ids.py passes 2/2, checking real Spectral
MaterialDatabase registration and an existing local Store digest read-only.
tests/meowdoku_marble_materials_check.mjs passes actual Studio boot allocation,
legacy/current loadObject with real mesh and mass verification, unchanged
photo object digest and geometry-wire receipt, edited material restoration,
runtime updates during asynchronous verification, and nineteen atomic
malformed-material rejection cases. It also loads local c7d2d88acadfca7990cef
562aab48fb7c458763e7846aa36e452789bd09b2187.json without changing its bytes.

That local snapshot already has a mesh receipt mismatch under both the
pre-existing deployed build and current source: saved 4c28c9dd76db48032472dc74
e45624e2d42b8b27d052c1ec7a5fed7ca7ba001d, regenerated bc7571a7454e5cf09e4a53
cbac2a7c3a530bc0fabd98d386516e954e2e4fe514. Its existing inspection-only path
and warning remain intact; no false photo verification was introduced.

The pre-existing tests/meowdoku_marble_object_check.mjs fails its D6 proxy
smaller-than-low assertion: the current non-marble proxy implementation
returns the low mesh. Neither that test nor mesh generation was changed here.
Focused syntax and scoped whitespace checks pass. No full suite, full asset
export, saved-object rewrite, server restart, commit or publication was run.
