2026-11-09 NATIVE LOCAL-FEATURE MESH CONTACT (MEASURED, canonical Windows):

Added studio_mesh_manifold.py. This is a real AbstractTensor geometry/manifold
extension with an incumbent wall-callback adapter, not a proxy solid, new
integrator or certificate-only path. Existing vehicle_balloon_tire vertex-
triangle CCD was inspected: it is a separate deformable-vertex impulse law,
not a reusable sphere boundary manifold, and is not substituted for DEM here.

NativeMeshBoundary retains actual float32 source vertices and indexed source
triangle identities. Strict triangle interiors own face minima; unique shared
edges/vertices own their local distance minima. Incident-face tangent-cone
tests reject false concave-edge/vertex radial forces while keeping independent
face contacts. Coplanar tessellation seams yield one edge/vertex contact rather
than doubled face stiffness. Closed embedded outer boundaries use normalized
face/edge and angle-weighted vertex pseudonormals at the nearest feature for
sign. Inside has one shortest outward exit; outside retains independently
contacting outward local minima. Separate bodies are never globally reduced.

Actual source float32 folds are retained: a microscopic concave fold has two
real minima and currently gets both contacts. No numerical flattening or
invented smooth-surface tolerance suppresses them. A complete rounded outer
boundary and independent embedding provenance remain required. Nonmanifold,
disconnected or internal construction-part soup is not reinterpreted as a
solid union. Native same-body table slab+legs still needs outer union assembly.

query returns FeatureContact records with original contributing source IDs,
distance, normal, point and validity, plus explicit geometry/finite/nearest/
sign/singularity diagnostics. apply calls the existing wall(overlap,valid,
normal); it neither integrates nor relocates. Optional named vertex inputs
must match the validated bake. nearest_distance is the signed closest-boundary
distance for contains/post-step checks. Parent owns failure gating and product
runtime wiring.

The final 9-test focused gate passed in 18.47 seconds including the finite and
signed-nearest diagnostic additions (12.96 seconds before those additions).
It includes actual 192-triangle rotated elliptical gobo native-f32 geometry,
inside shortest exit and outside orientation; exact source coordinates are
retained without convexity rejection. The actual course's 472-triangle exterior
union gives one contact at a real strip join and two at its floor/bank corner,
without construction-internal prism faces. Concave trihedral vertices retain
three faces, convex counterparts retain one radial feature, coplanar seam
force is exactly single-wall strength, reversed closed winding refuses, and
tiny folds remain explicit. Existing SSA/WebGPU emits the named-vertex feature
manifold plus DEM forces as one complete dispatch; no GPU execution.

No compiler, fountain law/build, Studio UI/server, live-view or publication
changes. New table union work is separately owned in an isolated module.
