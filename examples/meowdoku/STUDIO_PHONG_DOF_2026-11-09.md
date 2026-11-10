# Phong depth-of-field preview

Status: implemented and tested in source; not published. No server restart or
production exposure was performed. Do not export the entire dirty Studio
bootstrap: it contains independent, unpublished physics and chase work.

## Actual effect

`web/marble_studio_dof.js` renders the existing Phong scene into an RGBA8 plus
DEPTH_COMPONENT24 target, preserving up to four MSAA samples. It resolves both
attachments and composites the actual image with a fullscreen fragment shader.
It does not replace the material shader or alter ray-rendered images.

Depth is reconstructed using the same near/far planes as the scene projection.
In physical metres, the signed circle-of-confusion radius is

`pupilRadius * imageHeight / (2 * tan(verticalFov/2)) * (1/focus - 1/axialDepth)`.

Forty-eight equal-area disk taps use a stable per-pixel dither rotation. Taps
that would admit farther background through near foreground are rejected;
nearer samples must have a blur disk large enough to reach the pixel. Colour is
decoded from the already tone-mapped sRGB preview before averaging, then encoded
again. Sharp pixels are copied exactly. The preview explicitly caps radius at
64 device pixels. This is not an HDR optical integration or a compound-lens PSF.

`dofCameraFromView` derives the actual world-space eye and optical axis from the
displayed view, including the chase/player path. The existing
`photo.applyLensCamera` supplies the selected lens controls and updates the
focus-range warning. Invalid/incomplete input retains the last valid optics.
The toggle changes Phong only; photo/live-ray lens settings are not rewritten.

## Limits that remain visible

Transparent Phong glass does not write depth. Its blended colour therefore uses
the opaque depth underneath. The effect cannot infer apparent refractive focus,
reflection depth, hidden geometry or the background missing behind silhouettes.
The edge test verifies a specific far-colour rejection case, not universal
occlusion correctness. Stable dither is spatial noise, with no temporal denoiser.

The camera toolbox is implemented separately. Completed profiled HDR can be
metered with one accumulated pixel or Gaussian-weighted pixels. It is not a
fresh single-ray query. Full-scene refractive spot focus is disabled because no
supported complete reader endpoint exists. See `STUDIO_CAMERA_TOOLBOX_2026-11-09.md`.

## Verification

Actual Chrome WebGL2 on NVIDIA RTX 3060 / D3D11, 384 by 192 pixels, four-sample
render target, three checkerboard planes at 50, 100 and 200 mm:

- Focus at 100 mm with 6 mm pupil radius left the middle plane pixel-exact.
- Near/far variance fell from 16256.25 to 58.041 and 72.443 respectively.
- A 0.1 mm pupil radius restored the unblurred variance on those planes.
- Moving focus to 50 mm restored the near plane exactly and blurred the old
  focal plane to variance 58.934.
- Near red/far blue silhouette test: zero blue contamination in the tested
  foreground edge strip.
- Resize, cancel, framebuffer/program/VAO/depth state restoration and GL error
  checks passed. No GL errors occurred.

The repeatable GPU fixture is `tests/meowdoku_studio_dof_gpu_check.html` and its
module. Serve the repository locally and open that page, then Run GPU tests.
The loopback-only scratch server at `build/dot-studio-dof-20261109/qa_server.py`
was used for the measured run and served current source modules. It rejects
POST requests; no production API was called.

Integrated current Studio source rendered 72,960 triangles with the effect.
Browser checks verified changing focus, aperture, switching the effect off/on,
and an empty focus edit retaining the last valid preview with an explicit
warning. The mounted toolbox showed Hand selected and explained why focus/HDR
tools were unavailable with no completed photo. The isolated server had no lens
inventory endpoint, so Custom analytical was tested; full source-profile browser
integration remains part of release QA. CPU lens profile tests passed.

Focused Node gates passed: DOF geometry, camera toolbox, HDR sampler, all 50
photo-retouch lifecycle checks, raw calibration, lens controls, exact current
scene capture and hidden Phong partition. Modified JS syntax, Python server and
export syntax, and whitespace checks passed. Native ray rendering was not run.

## Release boundary

New production modules: `marble_studio_dof.js`, `marble_camera_toolbox.js`,
`marble_camera_toolbox_mount.js`, `marble_spot_sample.js`. The source export and
no-store lists include these names. The existing running server will not acquire
new Python no-store rules until its own coordinated release; no restart was made.

Shared changes are narrow imports/mount and frame hooks in `marble_studio.js`,
the DOF toggle/help in `marble_studio.html`, and the completed-HDR getter/Apply
contract in `marble_photo.js`. Rebase only these changes onto the latest
published material/macro assets. Never copy the full source bootstrap as a
shortcut. Re-run the GPU fixture and exact staged-asset tests before publishing.
