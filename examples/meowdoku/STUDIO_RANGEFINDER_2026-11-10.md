# All-hit Studio mesh rangefinder

Live narrow static release, verified at 2026-11-10 04:18:10 UTC. Original camera, lens prescriptions, ray engine and exposure scheduler are unchanged. Receipt: build/dot-rangefinder-release-20261110/publication.json.

## Implementation

- `web/marble_rangefinder.js` packs the exact current `captureCurrentStudioScene()` triangle coordinates and stable source part/body/triangle/material identities. It uses the existing `StudioCompiledLawBatch`, not a JavaScript intersection implementation. Vertices are quantized first and edges formed in float32, matching the existing identity-scale native scene packer.
- A selected pinhole ray follows the actual displayed Studio view/FOV/aspect and physical metre frame. Every valid triangle hit is retained and sorted, including coincident owners and surfaces behind opaque ones. There is no nearest-hit reduction or arbitrary depth stepping that could skip a thin surface. Original compiled law tolerances remain unchanged.
- Both distance along the ray and axial distance along the camera forward axis are reported. The latter is the input to existing analytical focus controls. Neither is focal length or refracted apparent depth.
- `marble_rangefinder_panel.js` adds a hit selector and separate Apply button. It calls the existing `photo.focusLensAt`. Original mode requires a lens selection or explicit Custom analytical checkbox. Profile minimum focus is preserved. Custom application preserves FOV and pupil. Stale capture/camera changes reject application.
- Narrow toolbox/bootstrap hooks supply the displayed scene. Toolbox-only selection inputs no longer increment scene revision or invalidate the object/photo. Actual lens changes still do. The narrow release uses only reviewed capture/toolbox hooks and completed-HDR metering. It does not export the whole source bootstrap or enable native physics. A scoped lighting event guard preserves scene identity across the pointerdown/click measurement pair; real authoring still invalidates.

## Scope and limits

This measures all captured Studio triangle surfaces along a straight ray. It is not a refracted ray path, sensor-pupil sample, analytic sphere/curve query, diffraction result, or a full original mixed-geometry transport trace. Coincident triangle entries are deliberately not merged by a guessed tolerance. PBR transmission/opacity labels describe the captured surface row, not computed spectral throughput. Physics/toy mode currently fails the existing capture gate rather than substituting saved geometry.

The original `ParametricSurface.intersect` and native composed sensor `nearest_filtered` boundaries return nearest events, not a ready full-path range list. A future refracted/parametric mode must retain original transport and its branch/event identities. It must not reuse this straight-ray result under a refractive label.

## Measured tests

- `node tests/meowdoku_rangefinder_check.mjs`: actual original-law Wasm, four depths/owners including transparent, opaque, coincident and hidden hits; off-axis axial versus ray length; malformed input, cancellation, and 20,000 coincident hits across the 16,384-record batch boundary passed.
- `node tests/meowdoku_rangefinder_panel_check.mjs`: actual Wasm and actual lens-controls code with a minimal DOM adapter. All-hit selection, explicit custom opt-in, second-hit focus, FOV/pupil preservation, stale rejection, profile minimum and invalidation passed.
- Existing camera toolbox and DOF Node gates pass. Modified Python registration files compile.
- Real connected Chrome passes the standalone actual-Wasm/lens-control browser gate and the complete staged Studio. A selected central ray returned 13 hits from 78,144 real displayed triangles in 50.2 ms. Second-hit Custom focus applied 1050.145 mm; subsequent Frame correctly rejected the stale result. Repeated selection passed.
- Point and Gaussian metering read the generated completed profiled HDR fixture. Explicit Apply used the existing retouch API. Display exposure changed to -2.3122 EV, neutral RGB gains applied, and the raw NPZ SHA256 remained unchanged. No production exposure was changed.
- Fifteen regression suites run against the exact stage and published dependencies, including all 50 photo lifecycle cases, original scene builders, A/B isolation, scale, camera binding and DOF. Eight staged JavaScript syntax checks pass.

## Release gate

The tested existing artifacts are under `build/dot-client-ray-20261109/studio-compute/`:

- `studio_bvh_query.json`
- `studio_hit_tri_batch.json`
- `studio_hit_tri_batch.wasm` (SHA256 `4f75f4b0991a4b0fd3e2be10072f093fa8374d7b9b6af8326e261d74dfbd9f4a`)

The batch source text was compared with the current original `studio_bvh_compute.query_identity()` hit_tri and matched exactly. No retired compiler was invoked to regenerate it. Artifact digests are checked by the existing loader.

The published client-ray-plain artifacts now exactly match the tested source-bound JSON/Wasm. The release guard verifies them as dependencies without replacing them. Earlier missing-artifact and headless-startup findings are historical; actual connected Chrome QA completed. Never export the entire heavily modified bootstrap as a shortcut.

## Completed-HDR toolbox

Point and Gaussian options sample existing linear-sRGB exposure pixels, not new ray packets. Exposure targets a relative display white and changes display EV only. White point assumes the chosen subject should be neutral; it is not a calibrated illuminant measurement. Both show a proposal before a separate Apply and reject stale photos. The full-scene refractive Spot focus control remains explicitly unavailable. Held photographic cameras or different Exposure scenes require their own measurement; the mesh result belongs to the displayed Studio camera.
