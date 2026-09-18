# AbstractUI placement, custody, and subtractive inventory

Placement changes a representation without silently rewriting the program graph.
Every placeable payload retains its authored identity, semantic owner, source
container, filesystem relations, and other graph receipts while its
representation moves through three custody states:

`inventory -> preview -> placed`

Taking an object removes its current mesh and collider from the world but keeps
the complete geometry packet in inventory. Preview republishes that same
identity as non-colliding geometry. Placement publishes a revision and restores
collision at the selected transform. Semantic ownership changes only through a
separate explicit transfer operation.

## Placement gimbal and snapping

The placement policy exposes translation on X/Y/Z, yaw/pitch/roll, uniform
scale, and finite snap modes: free, grid, object face, object center, and
opening track. The initial browser realization provides X/Y/Z and yaw sliders,
grid snapping, and face/center snapping. Rotation is retained in the transform
packet; the current axis-aligned mesh baker does not yet realize yawed geometry.

## Additive and subtractive payloads

Additive placement moves or creates geometry. Subtractive placement creates an
identified opening owned by a boundary host. Doors, windows, and gates use the
same inventory, preview, placement, revision, and inspection lifecycle, but
commit as ordered boundary-opening records rather than positive prisms. Windows
carry a sill height, allowing wall geometry below the opening; doors and gates
default to the floor. Portals use the manifold placement described below and do
not cut wall geometry.

Subtractive identities are appended to the host world object's semantic-part
table and remain separate from the host identity. Rebuilding the mesh removes
only the corresponding interval from that host wall and preserves colliders on
the remainder.

## Probabilistic portal tube graph

Primary (left) action places a blue `in` splat and secondary (right) action
places an orange `out` splat. Placement raycasts the compiled mesh, records the
hit triangle and barycentric coordinate, gathers nearby coplanar triangle
memberships, and defines a radial local chart over those triangle subdomains.
The splat marks and divides the existing mesh; it is not a wall opening or an
independent portal rectangle.

All deployed splats are nodes in one directed many-to-many graph. Every `in`
has an edge to every `out`. Edge probabilities use a spatial Gaussian of the
node-center distance and are normalized independently for each input. Entering
an input samples that distribution, then moves the player or physics-ball over
time through the selected edge instead of teleporting immediately.

Edges are visible tube manifolds. Their centerlines are relaxed cubic curves,
and each tube's cross-section uses quaternion parallel transport along the
curve. This avoids frame snapping and gives traversal a continuous orientation
field. On emergence, local splat coordinates, velocity, and player facing are
mapped into the output chart. IN remains blue and OUT orange; the compiled
Pluck Phong adapter reserves both as exact palette materials. Portal
see-through rendering remains deliberately separate from this graph geometry.

The tube throat flares smoothly to the full radial-chart radius at both ends,
with its end rings immediately behind the visible splats. Player entry accepts
either a true plane crossing or forward contact at a collider-blocked portal
plane, because the player capsule must not have to penetrate the host wall to
activate the manifold. During transit the portal exclusively owns the player
pose: the entry offset is pulled into the centerline and the eye's facing is
parallel-transported through the same quaternion frame used by the tube mesh.
Ordinary locomotion and vertical support resume only after emergence.

The placement tool exposes `standard` and `mega` modes through the shared tool
mode control (`M`). Standard apertures are person-class. Mega apertures are
vehicle-class and scale the portal radius, tube throat, and cubic control
handles by four. Edges connect only matching aperture classes, preventing a
mega IN distribution from selecting a person-sized OUT. Each trumpet is part
of the tube itself: the radius sampler is constrained to equal the source
circle at the first ring and the target circle at the last ring, with a smooth
flare over the nearest 22 percent of the path at each end.

The graph explicitly publishes `backing: probabilistic-tube-graph`,
`distribution: normalized-spatial-gaussian`, and
`path_model: relaxed-quaternion-cubic`. These records are intended to accept
layered neural-network graphs and spatial Gaussian activation fields later
without changing deployed node identities.

## Inventory counts

Inventory entries publish `quantity`, `maximum_stack`, and `stack_key`. Unique
authored objects have a maximum stack of one because combining them would erase
identity. Recipe stock may stack: the reference world starts with eight doors,
twelve windows, four gates, and twelve portal splats. Placement decrements available
stock while minting a stable placed-instance identity. The inventory entry
remains the reachability and provenance record even when its available count is
zero.

## Outer representation skybox

The global envelope is a persistent, non-colliding twelve-unit skybox wall. Its
horizon represents the parent world map rather than another room. Crossing its
physics boundary requests `switch-map-representation`; it does not pretend that
the current document coordinates extend forever. The skybox deliberately has
no ceiling so it remains an orientation and transition surface rather than a
sealed building.
