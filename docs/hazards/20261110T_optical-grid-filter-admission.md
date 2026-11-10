# Rotated optical grid filter admission, source only

2026-11-10: Per-cell finite filter slabs reuse the existing stage dielectric
contract and native material records. No independent RGB mask or alternate
attenuation model was added. Geometry and authoring UI remain owned by the
lighting controls worker and are not certified by this receipt.

The old world-axis bounds rejection prevented separated rotated slabs. Admission
now tries face-normal projections only after world bounds overlap. A strictly
positive gap exceeding a float32-scale margin proves separation, even for a
nonconvex mesh. No separating projection means rejection, not an inferred CSG
result. Genuine intersection, containment and ambiguous gaps remain rejected.
This does not add general overlapping-media ownership or scattering support.

System Python311, with both repository roots on PYTHONPATH:
python -m pytest tests/test_meowdoku_scene_material_engines.py -q
8 passed. This includes original native buffer/path absorption tests and new
rotated separation, intersection, containment and tiny-gap regressions. The
Turing .venv attempt failed before native loading because optional neural_assembly
imports nn without available torch; no environment or source workaround applied.

No server restart, asset publication, saved scene edits or render interruption.

Review follow-up: preserve existing exact world-AABB contact behavior per parent
instruction. The new separation proof applies only to strictly overlapping AABBs.
Exact touching table parts therefore remain admitted as before; this receipt does
not certify coincident dielectric boundary ownership. A proposed blanket contact
rejection was removed because it would change existing optical table semantics.
New grid filters must be authored with positive physical spacing; do not rely on
this legacy contact allowance for filters.
