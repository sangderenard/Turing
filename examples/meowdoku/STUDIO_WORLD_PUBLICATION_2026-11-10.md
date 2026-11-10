# Studio objects in the existing MechanicalCreature world

## Verified reuse boundary

The original generated page is the world, not a shell for a replacement walker.
Its builder is `tools/build_mechanical_creature_page.py`, using class
introspection, the AbstractUI document generator, the original worker and seven
matched Wasm artifacts. The preserved original is in
`build/dot-living-map-runtime-20261110/matched/`. Its receipt binds all bytes.

`marble_world_house.py` appends the real
`camera_software.exposure_timing.SceneCameraCoordinator` class and its members.
The original registry accepts those additions without changing any existing
runtime identity. This is conceptual ownership only: the full Studio material
draw, door/portal and admitted contact geometry are not yet installed.

The existing shooter already owns entity membership, body state, impulses,
player/projectile and projectile/projectile contacts. In the matched behavior,
`firePhysicsBall`, `registerProjectilePhysicsMembership` and
`updatePhysicsBalls` are the presentation-import seam. Preserve them. The
archetype radius is global in the current contact paths; a selected imported
marble must use that physical radius or the original radius ownership must be
deliberately extended. Scaling only the visible mesh is incorrect.

Woodshop shares Machine/MachineSim/MachineSystem, managed Newton dt and Pluck
material rendering. Its inspected frontend is Python/Pygame, not a verified
replacement for the deployed generated web world.

## Geodesic appearance capture

`marble_capture_jig.py` uses the existing Turing geodesic generator and native
Spectral sensor basis to produce 12, 42 or 92 camera views. Each is an ordinary
single-frame photo configuration. Fixed scene geometry, light sources,
materials, focus, aperture and FOV are retained. This is not a second timing
loop or renderer. The existing authenticated MRP1 upload and PhotoJobs GPU
ownership remain the execution boundary.

The current Orbit implementation is horizontal, with 1/8/16/32 views. Its
existing `_snapshot` already retains completed RGB linear HDR files. The new
plan does not claim those horizontal frames cover a sphere, move a subject off
its table, or certify a suspended jig. Real jig preparation, clear camera
paths, actual full-mesh framing and per-view jobs remain to be executed.

Completed views can be registered through the existing RenderAssetCatalog and
RenderedAssetRecord after exact config/payload/status checks. They remain
IMAGE records until object coverage and projection semantics are realized.
Current Studio's `uFieldVolume` is an empty 1x1x1 texture with gain zero, not an
already baked light field.

A dedicated ideal jig must be constructed explicitly. Merely freezing the
current incidental room lighting is not verification of that requested rig.
The immutable plan does not consume game time-of-day or added game lights.
The table must remain available to lighting, shadows and reflected/refracted
paths. Primary-camera-only hiding for underside views and a separate shadow
receiver texture need explicit native transport support, not scene deletion.

A view-dependent marble material needs each original camera projection and
the game body's orientation. The bake depicts one fixed lighting/scene
condition. It cannot promise dynamic relighting. Refracted jig background
cannot be removed with an ordinary opaque silhouette alpha mask; that must be
an explicit capture/consumer limitation or a richer transport product.

## Exact Phong geometry publication

`marble_studio_world_packet.js` captures the actual linked shader sources,
uniform state and four exact material tables with the displayed scene. It
retains all triangles, source/material IDs and the current photographic camera
ownership. It is not ready to draw until sampler resources and both opaque and
transparent pass policy are realized by the original world renderer.

`marble_phong_visibility.js` binds actual primary triangle-ID observations to
the source mesh, material state, revision and whole camera/condition record.
It retains exact source-order triangles, normals and IDs, publishes immutable
source/draw remaps, and keeps the complete original mesh. This is a finite
sampled-view subset, not mathematical minimal expressions or proof for every
intermediate camera. It must not prune physics, shadows, glass/refraction or
transparent layers.

`marble_phong_id_pass.js` now instruments a detached copy of the actual linked
Phong shaders, retaining their original discards and material equations. It
uses Studio's existing opaque classifier, expanded triangle buffer, an integer
ID attachment and a GPU fence/readback. Glass is retained separately. An actual
Chrome WebGL2 test with the published shader and material tables observed only
the front two opaque triangles, excluded their occluded pair and the original
backface-discarded face, and retained the glass face. GL state was restored.
The receipt is `build/dot-map-engine-20261110/phong_id_browser_receipt.json`.
That test covers subject self-occlusion; real full-scene occluder replay and
geodesic execution remain unintegrated.

## Verification and publication status

Source-only tests cover the real class interpreter and original deployed
registry, exact geometry/table preservation, canonical geodesic directions,
native camera bases, immutable view identities and completed-image catalog
admission. Contract fixtures are not ray-traced captures. The separate GPU
visibility fixture is actual raster readback, not native ray-traced visibility.
No Studio main, live car page, renderer, server or compiled product is changed
by these modules. The independently restored playable tire page is owned by
its own release receipt and is not overwritten by this work.

## Native camera filtering and scene crops still outstanding

The actual photo SensorAccumulator first-hit seam is the camera-loop
`nearest_hit` call in `demo_pluck_gl.py`. It currently shares the complete BVH
with later bounces and shadow queries. A selected-owner filter must affect only
that primary query. The separate native image wrapper's global camera
visibility/X-ray modes do not implement it. Preserve analytic/region traversal
and carry source IDs through existing degenerate-triangle and sphere remaps.
Current photo flattening drops table part IDs, and `_config` must explicitly
admit and hash any new policy. Adding an ignored JSON field is not support.
Current native camera preparation always adds housing/iris/sensor geometry;
observer-only capture is a separate missing mode. Current HDR output has no
receiver/object/UV shadow attachment.

The request to place a scene crop on the product stage remains open. Existing
`render_object_library.InterfacePanelHarvest` and `_harvest_interface_panels`
retain rendered panel crops from complete holistic display/linear images.
`usd_layout_stage.recrop_reuse` requires matching optical and whole reflection
boundary signatures. These are reusable 2D capture products, not a verified
arbitrary 3D scene clipping/import operation. Select the intended source and
whether the crop is a rendered panel or a 3D subscene before implementing it.
