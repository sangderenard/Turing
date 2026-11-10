2026-11-09 EXACT SAME-BODY CUBOID OUTER UNION (MEASURED, canonical Windows):

Added studio_cuboid_union.py::build_cuboid_union(mesh, parts, max_cells=8192),
accepting the integration worker's actual StudioSourcePart dataclass or matching
field dictionaries. This resolves the authored table slab/legs as ONE rigid
body, rather than relabelling them as unrelated convex solids. Each complete
source part passes incumbent native-f32 closed/convex admission and explicit
axis-aligned eight-corner/twelve-triangle side verification. Rotated boxes and
incomplete, non-box or unrelated-body groups refuse.

The arrangement cuts only at existing authored native-f32 x/y/z face values.
An interval cell is included by exact interval containment in a source box,
not point sampling. Disjoint Cartesian cell interiors and occupied-to-empty
face extraction give the exact source cuboid union. Exterior quads share a
conforming global edge/vertex grid. The resulting NativeMeshBoundary edge and
vertex-fan checks reject edge-only, disconnected or nonmanifold unions before
an independent embedded-boundary evidence ID is issued. No sampled voxel grid,
proxy hull, shifted surface, rounded fit, new integrator or alternate force law.
The existing native 4096-feature bound remains in force after construction.

CuboidUnion retains material IDs, stable arrangement triangle IDs and original
supporting side-face triangle provenance. This is explicitly coplanar original
face support, NOT a false one-to-one claim after exact tessellation subdivision.
Internal interface rectangles retain their original source IDs separately.
Every original source triangle must be accounted as exterior support or an
internal interface. Source-side material splits requiring more clipping and
conflicting coincident exterior materials refuse; adjacent nonoverlapping side
material regions remain distinct. Same-body source box overlap is explicitly
reported and union volume is computed exactly with rational native coordinates.

Independent embedding evidence is named by a content-derived construction ID.
Proof metadata records pairwise-disjoint interval cells, exterior extraction,
paired edges/connected vertex fans, source accounting and source-box overlap.
The artifact exposes that proof, source map, exact volume ratio and measured
source-to-native coordinate rounding through to_data(). Parent owns assembly
into the complete scene environment and readiness gates.

Final targeted gate: five tests passed in 5.52 seconds using installed Python311,
tests/test_meowdoku_studio_cuboid_union.py. Actual shared JS table builder's 60
faces and actual StudioSourcePart records produce one conforming native outer
boundary. Four internal slab/leg interfaces are accounted. The native feature
manifold returns one top contact, two correct underside/leg-corner contacts,
and one negative-distance shortest exit inside a leg. All emitted coordinates
are existing native source-plane values. Touching/overlapping box volume,
single-cell capacity, duplicate/conflicting material, rotated-body, unrelated
body and nonmanifold edge-contact cases are covered. Scoped whitespace check
passed. No shared law/build/UI/server edit, GPU, long build or publication.
