# Meowdoku tiles through spectral-analyzer's glyph maker

Date: 2026-11-05. A fork of the game's presentation away from the compiler:
tile sprites are authored and rendered by the sibling `spectral-analyzer`
repository's scene-order pipeline, in Python and GLSL, and composed into a
board by numpy. Nothing here has to lower. The compiled board shader
(`examples/meowdoku/shader.py`) remains the compiler-side presentation; this
is the asset-side one, and a web server can run it.

## What the glyph maker is

- `scene_orders.py`: a JSON scene order names font tokens, extrusion
  geometry, planes and spectral materials; `compile_job` replaces the lab
  scene's subject triangles with those and keeps the lab's sensor, lens,
  aperture, camera body, ring light and flash. The renderer is the native
  thick-lens BDPT (`exposure_render_demo.py`, `_spectral_kernels`).
- `camera_software/render_assets.py`: asset identity (`FontAssetSpec`,
  `ExtrusionAssetSpec`, `ExtrudedTokenAsset`, `LightFieldCondition`),
  `build_ink_on_slate_order` (the production look: a glossy extrusion half
  embedded in matte slate, 128x128 frame with an 80x80 content region,
  head-on lab camera), `RenderAssetCatalog` (atomic manifest of rendered
  artifacts keyed by target, condition and product) and `BakePlan`.
- `camera_software/sprite_compositor.py`: `extract_raytraced_sprite` splits
  a capture into a premultiplied body, an alpha from the authored outline,
  and a signed additive layer for spilled light, so the sprite composes over
  any background: `out = premultiplied + (1 - alpha) * destination +
  additive`. `CachedTokenStringComposer` composes strings from cached glyph
  sprites, which is the precedent for a board compositor.
- `camera_software/render_object_library.py` archives canonical style
  objects; `display_scene.py` holds fixed-camera progressive scenes.
- The Living Data Map inherited Pluck's world contract by adapter
  (`docs/ABSTRACT_UI_PLUCK_WORLD_CONVERGENCE.md`), and spectral-analyzer's
  `compile_base_material_webgl.py` compiles Pluck's desktop base-material
  shader to WebGL 2 through Turing's `glsl_project_ir`. Both directions of
  the sibling bridge already exist.

## What was built: `examples/meowdoku/tiles.py`

- `tile_order_package()`: one ink-on-slate job per role, cat, X mark and
  heart, from `build_ink_on_slate_order`, each in the font that has its
  outline and in its own material (`cat_orange`, `mark_slate`,
  `heart_red`). The cat is `U+1F431` from Segoe UI Symbol, which has real
  outline geometry for it (465 vertices); DejaVu Sans does not. Three
  renders serve every board: empty tiles are flat region colours from the
  board shader's own palette formula, and the compositor lays sprites over
  them.
- `compile_tiles` / `glyph_geometry`: the glyph builder resolves the fonts,
  extrudes and triangulates. Measured: cat 21,356 triangles, X 1,736,
  heart 3,160. spectral-analyzer's `--compile-only` imports the native
  kernels for the lab scene, so without them the module falls back to the
  builder alone and says so in its report.
- `render_tiles`: spectral-analyzer's `--render`; refuses by name when the
  native kernels are not built. `load_sprites` turns rendered linear frames
  into sprites through `extract_raytraced_sprite`.
- `compose_board(cat, x_mark, region, hearts, sprites)`: flat region tiles,
  sprites over cells, grid lines, a hearts row. Pure numpy over the state
  columns, so the pygame runner, a web server, or the dt-system session can
  all feed it.
- `tests/test_meowdoku_tiles.py`: 3 passed (package validates in the glyph
  fonts; tiles compile to glyph geometry; the compositor lays sprites and
  hearts), 1 skipped (rendering, until the kernels are built).

## What is not here yet

- **Rendering.** `_spectral_kernels.cp311-win_amd64.pyd` is not built in
  this checkout and no build driver is tracked in the repository; the only
  instruction found is "rebuild the native extension" in
  `BDPT_CPP_SIDEDATA_COMPANION.md`. Visual Studio 2019/2022, CMake and Ninja
  are installed, so the build is feasible; it is a build, so it waits for
  your word. Until then the three sprites do not exist and the compositor
  runs on synthetic ones in its test.
- **The server.** A Python web server that holds a `MeowdokuSession`,
  serves the composed board (or the compiled shader page) and takes moves is
  a day of plain Python once the sprites exist; the control record and the
  session API are already the server's interface.
- **A real cat mesh.** Scene orders accept planes and font tokens, plus
  display objects of kind text, plane, box and icon. The cat is a glyph
  today; a modelled cat would be a Pluck `WorldObject` form recipe, which
  the Living Data Map already adapts and the scene-order compiler does not
  yet take.
- **Puzzle generation at large n.** The generator is random arrangement
  plus grown regions, retried until a backtracking count proves uniqueness:
  0.2 s at n=8, 6 s at n=9, over a minute at n=10. The law and the dt round
  are fine at any n; a constructive generator is the fix.
