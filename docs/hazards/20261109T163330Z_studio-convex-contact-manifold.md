2026-11-09 STUDIO CONVEX-PART CONTACT REDUCTION (MEASURED, canonical Windows):

Extended studio_contacts.py with convex_part_contact. It reduces only one
explicitly authored closed outward-wound convex solid or convex planar sheet.
Outside a convex solid, its closest point is unique, so nearest-triangle
selection removes duplicate contacts within that solid without discarding a
second independent contact. Inside, nearest boundary gives signed negative
distance and the outward shortest-exit normal. Exact on-surface points retain
the winding normal. Equal distances preserve the first authored triangle index,
which is returned for source/material identity. Degenerate faces invalidate the
whole part rather than silently making a hole. No topology/winding inference,
coordinate welding, numerical tolerance, or optical-to-mechanical mapping.

Independent rigid parts and room planes remain separate contacts. Compound
body identity MUST remain in the producer's topology. Adjacent floor/rail
prisms are not automatically independent bodies: the new regression proves
that summing adjacent coplanar sheets doubles force exactly on their seam and
introduces a false lateral edge force nearby. Combining those triangles into
their actual convex planar sheet gives one correct contact. The general bent
compound-course union is not resolved here and must not be activated by simply
summing prism contacts. Internal construction-face suppression alone is not
enough; shared exterior-surface contact ownership is also required.

Coordinated with course geometry producer: proposed parts preserve exact
vertices, indexed triangles, authored convex/closed flags and material IDs.
Requested bodyId plus per-triangle surfaceId and exterior/internal boundary
role derived from construction, without tolerance-based inferred adjacency.
Those producer changes are owned separately and are not asserted tested here.

Final bounded gate: 18 passed in 2.49 seconds using installed Python311,
tests/test_meowdoku_studio_contacts.py and
tests/test_meowdoku_studio_convex_contacts.py. Added exact cube face/edge/vertex
and interior-exit checks, 200 independent analytic cube-distance comparisons,
triangle/material identity ties, duplicate-triangle invariance, separate corner
contacts, invalid topology, and explicit cross-part seam failure documentation.
Existing SSA-to-WebGPU pipeline emits both a runtime two-triangle sheet and a
runtime four-triangle closed-part reduction plus existing DEM force/torque,
each as one complete dispatch with all vertex uniform identities preserved.
This is generated-source proof, not GPU numerical execution or a full scene
product. No compiler, main law/build, live UI, server, export, restart or GPU
changes; no long lowering, commit or publication.
