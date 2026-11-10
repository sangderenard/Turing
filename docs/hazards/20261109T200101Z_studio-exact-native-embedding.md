2026-11-09 EXACT NATIVE-F32 EMBEDDING PROOF (MEASURED, canonical Windows):

Added studio_mesh_embedding.py::verify_native_embedding(vertices, triangles,
source_ids, max_triangles=4096, max_pairs=20000). This independently verifies
the actual rounded native coordinates; no user or producer assertion grants
the resulting proof ID. Geometry is rounded once to float32, then represented
exactly on a common dyadic integer lattice. The incumbent packed triangle BVH
provides conservative inclusive AABB pruning. Its bounds contain exact source
float32 coordinates promoted to float64, so no epsilon-expanded guess is used.

Every remaining triangle pair is checked by exact segment-to-triangle clipping
in the plane and three inward edge halfspaces. Noncoplanar and coplanar
intersection endpoints are rational. The triangle intersection is convex; its
extreme points lie on boundary edges of either triangle, so clipping all six
edges establishes whether the entire intersection is contained in the common
indexed edge/vertex. Nonadjacent touching, area overlap, crossings, duplicate
triangles and coincident geometry with unrelated indices all refuse. Distinct
noncoplanar triangles sharing an indexed edge can only meet on that edge's
line, providing an exact shortcut. No ray perturbation, floating tolerance,
flattened face or proxy hull is introduced.

The hard pair budget counts examined BVH leaf pairs before tighter triangle
AABB exclusion; exceeding it raises NativeEmbeddingError with source pair IDs
and incomplete-proof status. Degenerate rounded triangles also refuse with
their source ID. The result's digest/evidence ID binds exact native vertex
bytes, indexed triangles and original source identities. It reports source
rounding and pair counts. This is geometric embedding only: NativeMeshBoundary
still checks manifold topology, connected vertex fans, closedness, orientation
and arithmetic viability. A bowtie sharing an indexed vertex may pass pairwise
embedding but fails that subsequent topology gate; the artifact explicitly
does not claim manifold certification.

Actual rotated elliptical gobo: 192 shared builder triangles, 127 incumbent
BVH nodes, 7,356 examined pairs and 3,756 exact pair tests. It independently
proves, then enters NativeMeshBoundary with its generated evidence ID. Measured
maximum source-to-native coordinate rounding is 1.1410952271972974e-07 Studio
units for this fixture. No source material or vertex was fitted or discarded.

Targeted gate: 11 tests passed in 1.02 seconds with installed Python311,
tests/test_meowdoku_studio_mesh_embedding.py. Covers actual gobo admission,
coplanar and folded adjacency, crossings, containment/area overlap, unrelated
touching, same-index bowtie/topology refusal, representable subnormal separation,
rounded degeneracy, duplicates, source-bound digest changes, pair exhaustion
and BVH pruning. A subsequent actual-gobo-only proof printed the measurements
above. Scoped whitespace check passed.

Only isolated new module/tests and this receipt were edited. No scene binder,
environment, law/build, server/UI, GPU, activation, publication or active-session
change. Parent owns embedding-proof integration into feature_packet and native
runtime readiness.
