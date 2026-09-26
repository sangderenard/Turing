# AbstractUI viewer camera and shader viewport

The living data map now has a second view of the same subject directly beneath
the system root and above its document regions:

```text
system root
  viewer camera
  shader viewport
    resolve palette
    extrude document geometry
    first-person lighting
  document region
    courtyard
      building
        room / program object
```

`Viewport` is a bounded, backend-neutral view onto a subject. It owns a bbox,
camera, palette, presentation target, and explicit `views` dependency.
`ShaderViewport` inherits it and adds a geometry source, backend candidates,
and an ordered `FragmentOperation` chain. “Port” in the page label is shorthand
for this viewport, not a network or compiler-connection port.

Every `Viewport`, shader-filled or otherwise, may carry a
`ViewportControlPolicy`. The policy names the actor receiving control, the
highlight/focus activation rule, release conditions, captured device classes,
movement/look rates, gamepad selection rule, and mappings from native inputs
to semantic actions such as `move-forward`, `strafe-left`, `look`, and
`primary-action`. This keeps first-person control independent of HTML, document
geometry, and shader availability.

The browser realization highlights and focuses the viewport on click, requests
relative pointer lock when the browser permits it, routes WASD and mouse look
to the tracked actor/camera, and polls the first connected gamepad each entity
cycle. The left stick moves, the right stick looks, and button zero issues the
same identified `primary-action` edge as mouse button zero. Escape or focus
loss releases capture. Browsers still retain authority over pointer-lock and
gamepad permission/availability.

The first document-geometry adapter turns deterministic authored grid positions
into boxes in `data-world` coordinates. Regions become courtyard slabs,
buildings become foundations, and fields/methods/properties become room volumes
with different heights. Identity remains attached to every extrusion. The
WebGL2 backend expands each neutral box into a depth-tested triangle mesh.
The Canvas2D backend consumes that same triangle buffer. Neither realization
places a semantic box-count limit on the neutral geometry.

One `UIPalette` now carries named colors in addition to foreground/background,
spacing, radii, font, decoration, visibility, and locking. The CSS variable
root and shader material uniforms are generated from the same resolved palette
record. Neither backend owns canonical color values.

The native pointer actor carries a normalized `facing` vector. It is derived
from its previous and current control positions; followers derive facing from
their velocity. When the pointer lies inside the rendered document-region
bounds, its normalized position becomes the viewer camera's world position and
its facing becomes the camera direction. This avoids inventing a sixth-order
follower merely to infer orientation.

The canvas is a presentation target inside a semantic div-based viewport.
WebGL2 is the accelerated realization. If it is unavailable, Canvas2D uses a
perspective kernel authored in Python and compiled through captured numerical
IR to a browser WebAssembly module. The model retains that Python source, the
lowering receipt, the binary, and the generated parameter ABI. JavaScript
supplies camera and vertex arrays, depth-orders the resulting triangles, and
paints them with the palette lighting rule. The previous rectangle projection
remains only as a no-WASM fallback.

This establishes the cross-language boundary, but exact visual parity remains
a compatibility task. The viewer is intentionally dormant until the pointer
enters the document region; that interaction activates and positions its
camera. In that active state, the in-app browser successfully instantiated the
WASM kernel and reported 216 positive-depth triangles for the current self-map,
yet the surfaces were not visibly distinguishable during the last inspection.
The readout now reports positive-depth and on-screen triangle counts so later
browser/viewport clipping work has an observable contract.

Control capture initially regressed the Canvas path to an active black frame by
discarding the last actor-derived camera pose and starting from a generic
center-line camera. Capture now inherits the last inhabited camera. With no
prior inhabited pose it enters opposite the first room and looks directly at
that room; the current 216-triangle self-map places 190 triangles on the
Canvas surface from this deterministic entrance. Releasing capture also
recomputes active state instead of leaving it latched. If a future projection
still produces zero on-screen triangles, Canvas invokes the older rectangle
projection as a visible safety fallback while retaining the WASM diagnostics.

The initial procedural ray/box fragment realization proved too opaque to debug
across drivers and was replaced by conventional vertex-buffer extrusion with a
depth buffer. The fragment stage now has one narrow job: palette material
lighting and distance fog. Cameras beginning inside rooms see their inward
walls because rasterization does not discard containing volumes.
The shared palette has distinct `courtyard-face`, `building-face`, and
`room-face` material roles. Palette-owned warm light plus key/fill illumination
keeps those surfaces perceptually distinct from the dark UI palette while
leaving both material and light colors under palette authority.
