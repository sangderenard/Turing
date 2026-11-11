# Static Living Data Phong host and real REST checks

Staged a separate derivative of the original compiled map at
build/dot-static-phong-rest-20261110/index.html. Exact original source SHA256:
297cda67f0c96cc2dada8278fcd1f559b279fd14ba3c8d31b62448e1f1fa82a1.
Reversible adapter preserves original model, shaders, mesh construction and
registry. It omits game/vehicle/audio/navigation startup and rejects physics
or game-cycle entry. No worker or physical update runs. Requested cameras only
parameterize the original updateShaderViewer, then read its real framebuffer.

New helper examples/meowdoku/web/living_data_static_host.js exposes ready,
snapshot() and render(). Stage tool tools/add_living_data_static_host.py checks
exact source hash and unique seams, and verifies exact reverse recovery.
Focused pytest: 1 passed. Node helper syntax check passed.

Actual Chrome headless/SwiftShader checks passed two distinct 320x240 Phong
views, native renderer identity, 103 registry objects and zero Worker instances.
The existing original scene lighting is dark; no fabricated light/mesh/frame
was substituted. Native images were inspected visually.

The second browser check uses real meowdoku.server Handler on an ephemeral
loopback port, temporary accounts/state and actual login. Register returned201,
inspect200, two render commands202, host poll/complete200, final receipts200.
No game Table was created. Receipts and actual returned PNGs are in
build/dot-static-phong-rest-20261110/static-rest-browser-receipt.json and
rest-phong-0.png / rest-phong-1.png. No production/public route was accessed;
no deploy, restart, firewall, auth-policy or physics change was made.

The first REST pass completed all calls but temporary SQLite cleanup failed;
fixture now explicitly closes its account connection before cleanup and the
rerun completed successfully. Only test-owned processes were stopped.


Final refinement: snapshot now clones all 103 actual registry objects intact,
including material_bindings, physics, persistence and extensions. Layout is
still compact and separate WASM modules are not transported. Native Chrome
asserted those fields survived. Derived stage SHA256 is now
057d2857275849402734205e7f9769261ddbc75116622fbe7b2fd3a756aedf1d.

Final actual REST rerun passed after server PNG decoding validation and camera
receipt validation. Requested position/facing remain explicit; actual_camera
reports the renderer's normalized facing and dimensions. The two final PNG
hashes are 08a27c431aee14726909849dbc5d883918add4115e5d0536b2d4bb658580c933
and edff4cc5ab447b44684aca9a0cef192152bd50b08643351efea9648924a76018.
No workers/game Tables; inspection returned full actual analytic object fields.
Focused patch pytest and JavaScript syntax checks passed again.


Static consistency guard now pins the exact serialized canonical map, actual
renderer geometry and shader identity on first snapshot/render. It checks before
and after drawing. Divergence permanently refuses that host; restoring fields
does not silently re-pin. Both real Chrome and full REST checks passed again,
including an explicit registry edit followed by rejected snapshot and render.
Final derived-stage hash: adda958f8300259d639a049467981f6a63b2b65ab617aee79cdf3e650aec4725.
The latest frame hashes are recorded in static-rest-browser-receipt.json;
earlier per-run PNG hashes above are historical. No engine or shader changed.


## Native Storybook static projection
Added canonical static_only early branch: existing graph/geometry, Phong, scene/software/transform helpers only; no entity spawn or game/vehicle/audio/physics builders. Public project_world_to_div_map(static_only=True) now applies the shared fail-closed static adapter before returning its AbstractUI, so opening the export alone is safe. Adapter implementation moved to src/compiler/abstract_ui_static_host.py with tools compatibility re-export. Source receipt hashes original native JavaScript packet; complete document hash is separately recorded. Semantic room fields retained in snapshot.

Stage: build/dot-storybook-static-phong-20261110. Actual Chrome and authenticated local REST tested genuine 33-object StorybookEnvironment, five class identities, two native geometry-derived camera renders, zero workers, no game Tables, immutable source refusal. PNGs are actual existing Phong output, visually dark in native solar lighting, not newly illustrated architecture. Eight focused tests pass; full suite not run. No deployment.


Final review fixes: static qualification gate is removed before renderer readiness, so a WebGL failure no longer hides the analytic map behind game startup. Static producer test now monkeypatches actual symbolic world/vehicle/wheel physics builders, living_map_loop_deployment, audio FFT and PCM producers (no hasattr skips). Eight tests and actual Storybook REST rerun passed. Latest complete HTML hash 1dfc8d6862fa1990920f9c697fc9765d90c39b490ba11b9b8e385a0c96be4624; native JS source hash d1e41474e347c459cbef219dd316678e23d4da063fe54175e11071562c7b1dd9. Receipt and PNGs in the same stage directory supersede prior frames.

Standalone Studio runner added separately in examples/meowdoku/web/standalone_studio.js and canvas-only standalone_studio.html. Reuses existing full-scene and subject renderers with explicit camera matrices, no automatic rendering or controls. Real GPU check verifies 37 captured scene parts, 5184 static and 72960 subject triangles, changed pixels from camera change, invalid-camera recovery, zero Workers/RAF/listeners. Local captures/receipts: build/dot-standalone-studio-20261111. No graph-to-Studio conversion claimed.

## Explicit native map / Studio composition

Added living_data_studio_adapter.js with distinct composition schema, verified rig input, native triangle/material transfer and object/overlay ownership checks. Existing Studio subject renderer accepts this explicit schema separately from actual Studio captures; its shader/resource/material validation remains intact. Exact Storybook realized mesh yields 2,388 triangles with no omitted triangles. Two camera PNGs and provenance receipt: build/dot-storybook-studio-composition-20261111. Frames are nearly black under the unmodified captured Studio rig, not polished lighting or appearance-faithful native map replay. Unit negative contracts and real GPU check passed. No deployment.

Final explicit-light placement test: added immutable caller lightTransform, validated through existing Studio light-transform rules and recorded in composition receipt. Unit check confirms input-copy detachment, unchanged faces/mesh hash and rejection of nonuniform scale. Real browser test compares identity-rig and facade-oriented rigid-transform frames at two authored targets; unchanged 2,388 triangles. A localized bright reflection and doorway/floor are visible, while most scene remains dark. Bounded test stopped without changing exposure, powers, shaders or materials. Final transformed frames are studio-camera-1.png and studio-camera-3.png in build/dot-storybook-studio-composition-20261111.
