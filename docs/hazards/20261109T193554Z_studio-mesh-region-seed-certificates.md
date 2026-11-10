2026-11-09 BOUNDED MESH REGION AND SEED CERTIFICATES (MEASURED, canonical Windows):

Extended StudioMeshBake.native_convex_part to retain complete authored solid
boundaries through incumbent studio_environment._part admission. Actual shared
table slab's twelve triangles admit as a closed convex part, preserving source
IDs/materials and reporting measured native-f32 coordinate rounding. Existing
inside/nearest-exit semantics remain available through that part. The actual
rotated elliptical gobo fixture refuses native convexity after f32 rounding;
no epsilon, fitted plane, hull or repaired boundary weakens that refusal.

admit_region accounts for every original source triangle returned by the
conservative query envelope. Admitted parts must reproduce source triangles,
materials, IDs, digest and frame exactly before their source coverage counts.
Room claims must lie on the exact shared bounded plane, wound inward. Duplicate
ownership and multiple same-body convex parts are refused. Nearby unaccounted
source IDs raise RegionAdmissionError with the complete coverage report.
The report never claims global scene support. Source triangles unaccounted
outside the proximity envelope remain listed; inside_outside_certified stays
false while any remain. Being far from an unknown boundary does not prove that
the particle is outside its enclosed volume. This is a candidate-accounting
report, not warm-step readiness, contact clearance or a time-of-impact proof.

Added studio_mesh_certificate.py::component_seed_certificate for independently
authored component indices. Exact rational binary-source predicates follow the
existing _part admission precedent. It checks paired directed edges, connected
boundary and vertex links, outward positive volume and exact ray parity.
Ray edge/vertex hits, coplanarity, boundary seeds, degeneracy and malformed
topology yield explicit blockers with original source triangle IDs. It never
perturbs the ray or uses a numerical tolerance. Open sheets explicitly claim
no volume. Source doubles are not confused with a rounded native product.

Watertight edge topology is NOT a self-intersection/embedding proof. A closed
source component stays blocked without an independently authored embedded-
boundary construction-proof identifier. The certificate does not invent one.
On the actual shared 192-triangle rotated elliptical gobo fixture, exact source
topology passed and the centre seed had odd/inside parity. It remained blocked
on missing-independent-embedded-boundary-proof, with all 192 source IDs. That
source observation does not override native-f32 convexity refusal. Even a
certified source seed reports contact_clearance_certified=false and
native_rounding_certified=false; radius clearance, subsequent crossings and
native parity remain separate responsibilities.

Final combined gate: 39 passed in 3.34 seconds using installed Python311:
tests/test_meowdoku_studio_mesh_certificate.py,
tests/test_meowdoku_studio_mesh_bake.py, tests/test_abstract_ui_bvh.py,
tests/test_meowdoku_studio_contacts.py, tests/test_meowdoku_studio_convex_contacts.py.
Bake/BVH/certificate subset passed 19 in 1.08 seconds. No compiler, shared
native law/build, UI or server edits. No GPU, long build, publication, active
scene inspection or live-view change. Parent owns runtime/scene wiring.
