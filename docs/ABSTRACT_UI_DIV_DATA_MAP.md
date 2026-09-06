# AbstractUI top-down div data map

`abstract_ui_div_map.py` is the first concrete projection of the introspective
world graph. It returns the common `AbstractUI` object defined in
`abstract_ui.py`, rather than a page-specific result type.

By convention an `AbstractUI` object carries ordered, identified packets:

- HTML structure
- CSS presentation
- JavaScript behavior
- JSON neutral graph data

Packets declare their dependencies. The script packet for this map depends on
the HTML structure and neutral model packets. Other backends can add shader,
assembly, prose, game-world, or accessibility packets without expanding the
base object API. `AbstractUI.document()` is merely the conventional browser
assembly of the four initial packets.

The nesting is literal:

```text
world div
  region div                 module / document boundary
    building div             class
      room div               field, property, method, or nested class
  JavaScript district div
    source-line div          exact line of the executing renderer
```

CSS grid uses the deterministic `GridPosition` already assigned by
`abstract_ui_introspection`. It does not recalculate semantic positions in the
browser. Authored/member order and program identities consequently survive the
projection.

Every selectable div carries a graph identity. The inspector shows containment
dependencies and implied Python or repository-SSA code. The page's executable
script captures `document.currentScript.textContent`, splits that exact source
into identified line objects, and inserts them into a JavaScript district.
Those line objects link to their predecessor with `appears-after`, giving the
source its first deliberately simple dependency topology.

## Omnipotent event-host convention

An interactive AbstractUI node declares only:

```text
interaction type
destination node identity
```

For example, a room carries `inspect -> python:module.Class.member`. It does not
carry a callback, DOM event name, mouse rule, or keyboard rule. One delegated
event host owns input mechanics, resolves the destination through the identity
index, and interprets the interaction. The browser projection currently maps
both click and Enter/Space activation into that host. A game projection may map
gaze, proximity, a controller action, or a spoken request into the same two
semantic facts.

## Entity mezzanine

Above the document regions, but still inside the rendered system root, the map
now carries a viewer camera and shader viewport. The viewport consumes a
backend-neutral extrusion of the exact region/building/room grid and presents
it through an ordered palette/extrusion/first-person fragment chain. Moving the
player actor remains in data-world coordinates and the browser projects that
world pose back onto the div map. Mouse movement targets the UI but never
overwrites the player pose. See `ABSTRACT_UI_SHADER_VIEWPORT.md`.

The generated map also carries the initial entity mezzanine described in
`ABSTRACT_UI_ENTITY_MEZZANINE.md`. Its default projection contains one
`player-being` under the `players` organization. The former mouse-bound pointer
and four page followers remain available as explicit entity fixtures, but are
not spawned into the living map.

Clicking a semantic div routes to its geometry center; clicking map background
uses the local navigation-domain coordinate. Walls and authored openings are
projected into a compact, piecewise-linear traversal chart and rasterized to a
64×64 occupancy grid. A hot-swappable WebAssembly A* kernel returns a path in
that chart, so long nonlinear hierarchy gaps do not dominate path cost. The
host simplifies it by line of sight, accepts Catmull–Rom curve samples only
when their continuously sampled world projection remains clear, and performs a
final swept-clearance audit before movement begins. Orientation uses
shortest-arc quaternion slerp and the default propulsion rate is 5.2 traversal
units per second. Kernel assignment is per entity and can be changed at runtime
through `abstractUINavigation.assignKernel`. Manual movement interrupts only
that actor's current auto-locate route.

Successive map clicks append to a per-entity waypoint queue instead of
replacing the active destination. Each segment is planned from the certified
arrival pose after the prior segment completes. At every waypoint the entity
holds for at least 0.85 seconds and emits
`abstract-ui:navigation-presence`; game systems can also register an async hook
through `abstractUINavigation.onPresence`. Returned promises extend the hold,
so presence logic can finish before traversal continues to the next queued
location. Manual input and explicit cancellation clear both the active segment
and its pending waypoint queue.

A click on semantic geometry now preserves the exact clicked world point as
the first endpoint candidate. The geometry center and inside/outside opening
stand-offs follow as deterministic fallbacks. This prevents an unlucky room or
building center—especially one whose doorway falls between coarse grid cell
centers—from making the object appear non-navigable. Candidates are tried by
the navigation worker in order and only collision-certified results can become
active routes.

While a route is active, an SVG overlay shows its complete spline and colors
the completed length from amber toward cyan as traversal advances. The route
and all entity markers are children of the map's local positioning context,
not fixed viewport decorations.

The A* assembly module and its host-side grid/spline planning run together in a
dedicated navigation worker. Expensive retries and clearance sampling therefore
cannot block the animation-frame renderer. The main thread receives a compact
certified route and performs only route installation, pose interpolation, and
presentation.

The document/world bridge binds geometry corners to actual rendered div
border-box corners expressed relative to the map root. Viewport translation is
cancelled, so page scrolling never becomes world-position input. A
hierarchy-landmark transform makes the mapping continuous and exact at every
region, building, and room boundary; nonlinear context-container scale is the
only permitted deformation between those frames. Route polylines adaptively
subdivide their samples after this transform so the webpage line follows its
nonlinear projection rather than joining sparse points with linear shortcuts.

With the placement tool active, aiming at a mesh shows a projected viewport
bounding box and hovering its DOM counterpart highlights it. Clicking a
top-down object selects it for the x/y/z/yaw gimbal, publishes its dimensions,
and previews offset, elevation, and rotation in both mesh and DOM. **Apply
gimbal** persists the synchronized transform; **Cancel** restores the exact
pre-edit geometry.

The shader viewport has a procedural camera-centered upper-hemisphere sky. A
browser-local solar clock produces opposite sun and moon directions, the
visible half-dome gradient/discs, the native shader key direction, and two
view-space Pluck Phong emitter positions from the same state. Consequently the
Phong light is no longer the fixed `[0, 4, 2]` placeholder: daylight is keyed
by the sun and night is keyed by the moon.

Physics balls are minted as mezzanine entities as well as rendered projectile
geometry. Each publishes its compiled-physics pose to a smaller div-map marker;
expiry retains the marker at its last world position until spent history is
explicitly cleared.

Beside the organizations, a closed **Registered actions** disclosure contains
the source, interaction, destination, and issue count of every selectable map
edge. The AbstractUI JavaScript system timer delivers `update(actions)` once
per frame. A recently issued row lights briefly and then decays under timer
authority; event handlers never style the row directly.

The source district is location-scoped instead of mirroring every executing
JavaScript line into the DOM. It mounts one opening row for the smallest
region/building/room closure currently containing the player, plus the opening
of every closure the player has not entered. Entered non-current closures are
unmounted. The visible set is rebuilt only on a world-containment transition,
so normal animation frames do not churn source nodes and the page does not pay
for thousands of off-scope line elements.

This is a projection, not a second UI model. A future HTML backend can replace
the hand renderer, and a world/game backend can arrange the same identities as
terrain, without changing the `AbstractUIWorld` records or their code receipts.

## Example

```python
from src.compiler.abstract_ui_div_map import project_class_to_div_map

page = project_class_to_div_map(MyClass, depth_up=1, depth_down=1)
page.write("my-class-map.html")
```

The next step is to connect these semantic closure openings to compiler-emitted
JavaScript AST/SSA nodes. Then entering a closure can reveal its exact active
operand and control scope without restoring the full-source DOM burden.
