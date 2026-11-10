2026-11-09 WHOLE STUDIO MESH BVH BAKE (MEASURED, canonical Windows):

Added examples/meowdoku/studio_mesh_bake.py, reusing incumbent
src/compiler/abstract_ui_bvh.py::build_packed_triangle_bvh. That packer is the
existing extraction of Spectral demo_pluck_gl.py::_build_gpu_bvh, with its
twelve-value node layout and stable original-triangle permutation. No new
spatial hierarchy, voxel proxy, SDF approximation, integrator or optical-to-
mechanical material conversion was introduced.

StudioMeshBake.from_faces accepts the exact shared [a,b,c,material,...] faces,
stable per-triangle source IDs and a common frame identity. It copies them into
an immutable snapshot with a content/frame/source/material SHA256 digest.
Caller mutations cannot alter this held snapshot. Degenerate source faces
remain present and explicit invalidity is returned by the existing
AbstractTensor closest_triangle primitive, rather than silently omitting faces.

Actionable native-host interfaces:
- to_data(): turing.studio-whole-mesh-bake.v1, exact float64 source triangles,
  original IDs/materials, incumbent packed BVH, digest and explicit numerical
  conversion/contact-manifold limitations.
- candidate_payload(point,radius,end=None): turing.studio-mesh-candidates.v1,
  complete source-index/ID/material/triangle payload plus query envelope and
  matching digest/frame. Conservative sphere/swept-sphere AABB traversal;
  capacity overflow refuses instead of dropping contacts. Not a time-of-impact
  computation and not a resolved force manifold.
- contact_geometry(): bounded CPU geometry proof using the same
  AbstractTensor closest_triangle function as the native law, with identity
  preserved on every touching candidate. No force sum or alternate solver.
- native_sheet_part(): explicitly selected single authored surface only;
  exact-coordinate edge indexing and the existing studio_environment._part
  admission prove native-f32 filled convex planarity/winding/edge topology.
  Returns the original source part packet, full source triangle map, and the
  measured coordinate-rounding error. This permits one table-top contact
  across its triangulation diagonal; it does not claim whole-table solid sign.
- require_digest(): refuses stale snapshot work before reveal/publication.

Combined focused gate: 32 passed in 2.95 seconds with installed Python311:
tests/test_meowdoku_studio_mesh_bake.py, tests/test_abstract_ui_bvh.py,
tests/test_meowdoku_studio_contacts.py, tests/test_meowdoku_studio_convex_contacts.py.
The new bake/BVH slice alone passed 12 tests in 0.78 seconds. It invokes the
actual shared JS room, table, backdrop and rotated elliptical-gobo builders,
then verifies every face enters one bake unchanged. A separate actual course
builder fixture proves a whole nonconvex compound mesh bakes without convex
proxy admission. These are test fixtures, not a read of the user's live scene.
The tabletop adapter enters real StudioEnvironment.contacts: six room planes
plus one tabletop contact, preserving source triangles/materials without a
double diagonal contact. Random BVH queries agree with exhaustive exact-
triangle queries. Folded and nonconvex sheet admission is refused. An initial
nonconvex fixture contained a T-junction and correctly hit the earlier boundary-
branches refusal; replacing it with a conforming L triangulation exercised the
intended convexity refusal. No production admission rule was weakened.

Remaining gaps: whole-scene producer/runtime upload wiring is owned by the
integration worker; BVH traversal is not yet compiled into the particle product.
General closed-volume sign, compound concave contact ownership, changing mesh
crossings and safe native first-frame handoff are not completed by this bake.
The sheet adapter cannot replace them. Full-source float64 coordinates are
retained; future native-f32 conversion must be declared and checked. No GPU,
long compilation, server/UI edit, publication, export or live-view action.
