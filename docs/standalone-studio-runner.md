# Standalone Studio: scene plus camera state

`examples/meowdoku/web/standalone_studio.js` wraps the existing Studio renderer, without Studio's controls, physics, animation loop, or automatic initial frame.

The primary API is `await createStandaloneStudioScene({canvas, sceneAsset})`, followed by `runner.render({view, projection, width, height})`. Both matrices contain 16 finite numbers in the existing Studio column-major convention; view must be affine. Dimensions are 16..2048 pixels. The returned receipt contains the exact requested camera, verified scene identity, actual triangle counts, PNG data URL, and `simulation:false`. Call `dispose()` when done. Camera validation precedes framebuffer resizing; invalid state does not poison the previous valid frame.

The scene is an existing `turing.studio-scene-asset.v1` capture. Existing hydration verifies its content identity and binary geometry hash. Existing `createStudioSceneRenderer` renders captured room, fixtures, materials, lights, and fixed subject through Studio's actual shaders. Data is detached from the caller. The runner does not fetch URLs or execute module names from scene data. Material sampler/resource limitations remain those of the existing Studio adapter; unsupported captures are rejected, not approximated.

`createStandaloneStudio({canvas, packet, samplers, model, lightTransform})` separately supports an exact captured subject packet using the existing subject renderer. This deliberately omits the room/fixtures and is not the complete-scene API.

The optional `standalone_studio.html` contains only a canvas. A local caller uses `await window.standaloneStudio.load(sceneAsset)` then `window.standaloneStudio.render(camera)`. Loading does not draw. It contains no camera controls or parameter widgets. Serve it locally with its existing module dependencies; it is not a hosted deployment.

## Living Data is a separate input conversion

Class analysis creates an AbstractUIWorld semantic layout: module regions, class buildings, member rooms and declared relationships. Existing `document_geometry` and native map producers convert that layout into map geometry. The class is not instantiated, and no independent game engine is needed for the static map.

That existing map currently uses its own Pluck Phong path, not this Studio scene renderer. Canonical class/world/graph map exports are tested separately. Direct graph/class input, OBJ files, arbitrary triangles, and guessed material tables are not accepted by the Studio runner. The explicit composition adapter below accepts already-realized native map meshes, without pretending they are Studio captures. No unsupported form is silently substituted.

## Verification

- `node tests/standalone_studio_check.mjs`: camera validation and absence of input/animation machinery.
- `python tests/standalone_studio_browser_check.py`: real local headless Chrome WebGL2, verified complete scene capture, two changed-camera PNGs, malformed-state recovery, zero Workers/RAF/listeners. Fixture is `build/dot-studio-game-transfer-20261110/complete-studio.scene-asset.json`.
- Results: `build/dot-standalone-studio-20261111/receipt.json` and `studio-camera-0.png`, `studio-camera-1.png`.

The tested captured room is dark; outputs are actual renderer pixels, not enhanced or replaced illustrations. This test does not exercise new physics, a deployed REST service, arbitrary geometry conversion, or full Studio interactive controls.

## Explicit Living Data composition adapter

`living_data_studio_adapter.js` accepts the static host's realized mesh snapshot plus an explicitly supplied, hash-verified Studio scene asset as the shader/light/sampler rig. It does not regenerate geometry: it transfers all native triangle positions and normals, actual palette-derived material tables, native identity spans and separately identified overlays. It rejects gaps, overlaps, non-triangle ownership ranges and triangles whose vertices resolve to different materials.

`await createLivingDataStudio({canvas, mesh, studioSceneAsset})` returns the same explicit `render(camera)`/`dispose()` interface. The receipt labels this `turing.living-data-studio-composition.v1`, records the native map source and mesh hashes separately from Studio rig identity, and preserves object/overlay provenance. The existing Studio subject renderer performs the actual drawing. This is not a Studio capture of the map and does not rename map objects as Studio room parts.

The caller explicitly chooses the rig. Studio's captured lights replace native map celestial lighting, and Studio flat color/depth sampler feeds differ from the native map feeds. Therefore this is a clearly identified composition, **not appearance-faithful replay**. No implicit replacement materials or geometry are generated. Arbitrary external mesh formats remain outside this adapter.

Verified composition run: `python tests/living_data_studio_browser_check.py` reads the actual captured `realized-mesh.json`, renders StorybookEnvironment through Studio, and writes two PNGs plus receipt under `build/dot-storybook-studio-composition-20261111`. All 7,164 vertices / 2,388 triangles survive (2,136 opaque, 252 transparent, none omitted). The two target-camera outputs differ. The original Studio rig is near its captured origin while Storybook buildings are elsewhere, so these frames are nearly black; this test proves the rendering path and provenance, not satisfactory scene illumination. It is a standalone GPU test, distinct from the native-map REST test.

### Explicit rig placement

`createLivingDataStudio` and `composeLivingDataStudio` accept optional `lightTransform`, a scene-level immutable affine matrix validated by the existing Studio light-transform contract. Omitted means identity (original captured placement). Translation/rotation and uniform positive scale are supported; shear, reflections, nonfinite and nonuniform scaling are rejected. The exact copied matrix is recorded as `composition.light_transform`; geometry, materials, source/mesh hashes and camera state do not change when the rig is relocated.

Final bounded lighting test keeps baseline frames 0 and 2, and uses caller-selected rigid transforms for frames 1 and 3. Each transform rotates the captured rig toward the native building facade and translates to the corresponding native x coordinate. The doorway/floor and a strong localized bright reflection become visible; most of the scene remains dark. This demonstrates controllable rig placement, not polished environmental lighting. No light power, shader, exposure or material adjustments were made. The two transformed frames are `studio-camera-1.png` and `studio-camera-3.png`, with exact camera/rig matrices in the adjacent receipt. The unchanged mesh hash assertion passes.
