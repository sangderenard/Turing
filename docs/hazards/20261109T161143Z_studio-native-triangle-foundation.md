2026-11-09 STUDIO TRIANGLE CONTACT FOUNDATION (MEASURED, canonical Windows):

Added examples/meowdoku/studio_contacts.py and its isolated test module. This
is geometry/contact groundwork, not an installed fountain product. No edits
to fountain3d.py, fountain3d_build.py, browser code, compiler or deployed ABI.

closest_triangle uses the exact face projection when inside the three edges,
otherwise the closest bounded edge segment (including endpoints). Distance
normals are two-sided, with winding normal only at exact surface coincidence.
Collapsed faces return validity zero and finite geometry; no epsilon changes
a small representable triangle into a degenerate one. room_planes explicitly
provides six inward signed half-spaces; outside centres retain restoring
contacts and a corner keeps three independent wall contacts. The mechanical
wrapper calls the existing fountain3d._contact with the existing bounded
overlap, infinite-wall effective mass/radius and explicit physical inputs.
No optical material is translated into friction or other mechanics.

Source-backed ABI route: fountain3d_build.trace uses named scalar () inputs
through SSATensorOperations.input and the generated product uniform_index.
triangle_input_names exposes nine coordinates per slot for this same route,
without altering the deployed argument list. Material/primitive identities
must remain host-side metadata associated with those slots; packing/upload
and identity ownership have not been implemented here.

The real SSA-to-WebGPU translator emits the isolated eight-particle triangle
slice completely in one dispatch, with all nine scalar vertex inputs in its
uniform map. Initial use of scalar face normals inside a vector where exposed
the existing unsupported broadcast_double primitive (SSA where explicitly
broadcasts all three operands). Promoting the scalar branch with the existing
wall-normal arithmetic idiom avoids that primitive without compiler changes.
Generated WGSL was not executed on a GPU; native/WASM execution was not run.

Final focused gate: 10 passed in 1.30 seconds with
C:/Users/alber/AppData/Local/Programs/Python/Python311/python.exe -m pytest -q
tests/test_meowdoku_studio_contacts.py. Covers faces/edges/vertices, both sides,
coincidence, arbitrary rigid orientation, reversed winding, shared diagonal,
collapsed and small triangles, room corners/outside points, force/torque
identity, named inputs and the existing translator. Repo .venv lacks llvmlite;
the installed Python311 interpreter runs the translator test. An intermediate
new-test assertion expected shape (3,) rather than the actual scalar backend
shape (3,1); fixed only the test's shape normalization.

Integration remains deliberately absent. A full mesh product needs triangle
capacity/binding budgeting, exact Studio-frame/unit packing, preserved material
and primitive identities, replacement of legacy slab contacts, and a contact
manifold/ownership policy. Summing one force per triangle doubles a coplanar
seam; globally choosing one triangle erases independent corner contacts. Closed
obstacles additionally need explicit sided/interior semantics. This module
does not silently choose either flawed reduction or infer closed interiors.
No long lowering, build, GPU job, restart, publish, commit or live-view action.
