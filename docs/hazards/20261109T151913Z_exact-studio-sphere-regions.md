2026-11-09 EXACT STUDIO SPHERICAL OPTICS (MEASURED, Windows RTX 3060):

The original Spectral SphericalSurface GLSL kernel now supplies exact saved
sphere intersections in both Studio photo directions. Authenticated saved
metadata, original face membership, materials, and explicit medium pairs
survive scene concatenation/filtering. The remaining interior/fixture
triangles still compete for every hit. No shell decimation or normal-only
substitute is used.

Hard triangle-budget admission uses the same CPU scene builder as rendering,
before stopping any existing process. Preflight is bounded to 8 seconds;
the nine-sphere fixture measured about 2.6 seconds. Invalid budgets, missing
metadata, unsupported shapes/coats/paint, or over-budget scenes leave the
existing exposure intact. Legacy omitted policy remains mesh. Initial exact
support excludes emissive shells and requires uniform uncoated spheres.

Turing targeted sphere admission, geometry lifecycle, and route tests:
55 passed. Actual forward/sensor GPU hit, media, TIR, silhouette, tangent,
and ordinary interior-occlusion outputs were bit-identical at 1 m and
0.01 m radii. Real 128 px / 16-epoch RGB Exact with Auto, monochrome Exact,
and RGB mesh photos completed. Both bubble GPU regression scripts passed.
The fixture changes 53,010 triangles to 24,466 triangles plus 9 spheres.
Auto correctly reports sparse pilot coverage rather than convergence.

Spectral handoff: MARBLE_PARAMETRIC_REGIONS_HANDOFF_2026-11-09.md.
Local receipts: spectral-analyzer/repo-cache/exact-spheres-20261109/.
No deployment, service restart, user exposure interruption, or full build.
Saved mechanical geometry and mass calculations remain triangulated.
