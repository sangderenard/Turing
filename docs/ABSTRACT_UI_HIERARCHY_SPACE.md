# AbstractUI hierarchy space

The living data map treats spatial scale as an interpretation of program
hierarchy, not a direct conversion from CSS pixels. Local objects remain
legible while the empty distance between containment levels grows
nonlinearly.

The current reference metaphor is:

```text
global scope / representation envelope  -> broad sky floor
  class or namespace region             -> fortified courtyard
    structural members                  -> separated buildings
      methods, fields, nested values     -> rooms and interior objects
```

The metaphor makes basic object-oriented boundaries perceptible. A courtyard
states the authority and protection boundary of a class. Buildings group the
class's structural concerns. Rooms give individual members an inspectable
interior. Global observations and definitions occupy the much broader space
outside compounds instead of looking like accidental members of a class.

`document_geometry.hierarchy_space` publishes the formulas used for room
pitch, sibling gaps, region gaps, and global margin. These are backend-neutral
facts. The browser currently realizes them as WebGL geometry, but another
renderer can use the same hierarchy transform.

The outer envelope is also a representation boundary. Crossing it issues the
semantic action `switch-map-representation`; it does not extend one coordinate
system forever. A host can replace the current map with its parent-context map
while retaining the actor and focused identity.

## Linear traversal chart

Auto-locate does not search the visually stretched hierarchy coordinates
directly. The host derives a compact, piecewise-linear traversal chart from
the world bounds and authored opening landmarks. Ordinary spans retain their
scale while long hierarchy gaps are compressed with `2 + log1p(gap - 2)`.
The 64x64 A* grid, line-of-sight simplifier, and Catmull-Rom samples live in
that chart; an invertible piecewise projection maps every movement sample back
into nonlinear hierarchy space.

This makes path cost describe traversal topology rather than decorative empty
distance. The player is propelled through the chart at 5.2 traversal units per
second. Before accepting a route, the host maps every spline segment into world
space and performs continuous swept-clearance sampling against structural
walls and authored openings. Unsafe curves fall back to their clear linear
segments, and a route is rejected if the final continuous audit fails.

During traversal the div map draws the complete certified route as a local SVG
polyline. A second overlaid stroke reveals the completed portion and changes
hue with progress. The overlay belongs to the routed entity and is removed on
manual interruption, kernel reassignment, replacement, or arrival.

Route planning is isolated from graphics. A dedicated browser worker owns grid
rasterization, repeated unsafe-edge rejection, the WebAssembly A* invocation,
line-of-sight simplification, spline construction, and continuous collision
certification. It returns only certified route samples through structured
clone. The animation-frame thread installs those samples, interpolates the
actor pose, and paints the overlay; an `async` function on the graphics thread
is not treated as thread isolation.

## Deterministic document/world resynchronization

Structural div border boxes and game-world geometry boxes are paired by stable
identity. The browser converts each rendered border frame into map-root-local
coordinates, cancelling viewport translation and normalizing any rendered
scale, then binds all four local corners to the corresponding world corners.
Each context frame builds axis profiles from its own corners and the rendered
corners of its immediate children. Child boundaries therefore become shared
landmarks in parent and child transforms, making the piecewise-affine map
continuous while retaining nonlinear scale between hierarchy containers.

Entity markers, route overlays, and background click inversion all use this
same bidirectional transform. Scrolling therefore moves the containing map and
its overlays together and never becomes coordinate input. Initial layout,
element resize, window resize, and world-mesh revision mark the paired frames
dirty; the next simulation frame deterministically rebuilds them.

SVG route projection is adaptive rather than a sparse point conversion. Every
traversal segment is recursively subdivided until its transformed midpoint is
within 0.35 pixels of the projected chord and the chord is at most 10 pixels.
Consequently the visible path follows every nonlinear context bend instead of
drawing a straight shortcut between distant transformed samples.

## Embodiment and identity

The reference player embodiment is one quarter of the original demo scale.
Eye height, collision radius, and movement rate are published camera/control
facts rather than scattered renderer constants.

Authored string identities remain authoritative for persistence, editing, and
introspection. At world bake time the compiler additionally specializes
objects and semantic parts to deterministic, dense, 1-based `u32` IDs. Zero
is reserved for missing identity. The bidirectional table lives for one world
revision or mesh bake, allowing rendering, picking, collision, and compute
buffers to use compact records without sacrificing the self-referential map.

## Physics program boundary

World physics is an ordered selection of equation stages. The intended
executable is not a hand-written JavaScript solver: a SymPy equation set is
reduced through the canonical ProcessGraph and compiler SSA into WebAssembly. Dense
runtime IDs and typed arrays form its hot state layout; results cross back to
semantic identities when they publish poses, contacts, or actions.

The present stage records are `selected-unbound`. They define specialization,
static welding, broad phase, narrow contacts, player resolution, and pose
publication, but make no claim that the equation artifact or solver is already
bound.
