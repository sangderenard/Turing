# Exact full Studio scene transfer, reusable asset only

Measured on Windows DESKTOP-LCKQH36 in C:/dev/Powershell/turing.
New marble_studio_scene_asset.js serializes the original 37 parts/5,184 static
triangles and 72,960 subject triangles, shader sources, uniforms, four material
tables and source IDs. Source float64 geometry is retained because Studio's
shared transparentSet transforms subject vertices/normals before float32 GPU
upload. Derived ray arrays are omitted. Optional group-field presence and
numeric-key typed-array camera matrices survive capture normalization.

createStudioSceneAsset -> hydrateStudioSceneAsset -> createStudioSceneRenderer
uses the existing exact Studio subject renderer. The sole change to that
renderer is additive draw({emissionDirect:false}), enabling actual fixture/shop
opaque emission without changing subject defaults. Static opaque, fixtures,
subject opaque and shared furniture/subject transparent order match Studio.
The host still owns world, loop, framebuffer, viewport, contact admission and
update timing. This is a fixed captured scene, not a dynamic tick-sync system.

An explicit optional room/front 2m x 2.4m doorway changes only a derived draw
copy, preserves original source ownership and leaves the serialized asset
complete. 12 original front triangles intersect it; front count128 ->141,
whole static count5184 ->5197. Removed area matches exactly to 1e-10 Studio
square units in the focused test. No other source geometry is removed.

Passed:18 rejection cases, full source scalar/metadata roundtrip, existing
subject21 and game-asset27 rejection suites, GPU-fixture syntax. Artifact:
build/dot-studio-game-transfer-20261110/complete-studio.scene-asset.json,
16,065,996 bytes, identity
studio-scene:131e3cfc8de620cb8aea2d89a49cb7a0364eb83f1a5c3c3544aaa2617d4da44f.
GPU fixture tests/meowdoku_studio_scene_gpu_check.html is written but not yet
browser-run. No live publication, backend change or full repository suite.

Test hazard: an initial giant assert.deepEqual on the whole capture encountered
optional missing group fields vs explicit0 and attempted a huge formatted diff,
causing host memory pressure. Only that test process was stopped. The transport
now preserves group presence and tests compare bounded face tuples. Corrected
full-capture command completed with --max-old-space-size=1024. Avoid whole-scene
deep-diff assertions and concurrent full-capture processes on this shared host.
The hazards reader currently stops on pre-existing invalid entry filenames;
latest relevant entries and the baseline were read directly instead.
