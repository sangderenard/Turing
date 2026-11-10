2026-11-09 PLANAR SURFACE OWNERSHIP BOUND (MEASURED, canonical Windows):

Added planar_surface_contact to studio_contacts.py. Input triangles must share
authored body and surface ownership. Compile-time planar/convex-footprint
certificates come from construction, not tolerance-based geometry inference.
Folded groups are refused explicitly. Convex planar groups support their full
face/edge/vertex geometry. Nonconvex planar groups support face contacts only
where the perpendicular projection lies inside their triangle union. Returned
support is independent of geometric validity and must not silently gate away
a potentially touching unsupported particle.

This is the bounded safe case for the gentle-S course's inclined running
surface: its footprint is nonconvex even though the surface is authored planar.
A centre outside a concave footprint can touch multiple separate edges, so
global nearest-triangle selection is not a complete edge manifold. Broad-return
folds need explicit fold handling rather than a first-triangle plane estimate.
Floor and rail surface ownership stays separate to retain corner contacts.
Compound-solid inside/exit semantics remain unimplemented by this sheet helper.
No general collisionReady claim, native activation or UI wiring is made.

Final targeted checks: 20 passed in 2.41 seconds, installed Python311 running
tests/test_meowdoku_studio_contacts.py and
tests/test_meowdoku_studio_convex_contacts.py. New checks cover adjacent planar
face ownership, nonconvex L-footprint concave-edge refusal, rejected folded
groups, and independent floor/rail normals. Existing SSA/WebGPU translator
completely emits the projected-union support mask, nearest geometry, original
triangle identity, and DEM force/torque in one dispatch. The closed-convex
four-triangle path remains green. No GPU execution, compiler/main law/build
edits, restart, publication or live-view changes.
