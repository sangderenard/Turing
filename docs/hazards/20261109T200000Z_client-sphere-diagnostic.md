# Local sphere-query diagnostic, 2026-11-09

## Delivered scope

Opt-in browser-local geometry queries reuse Spectral's compiled SphericalSurface kernel produced by client_ray_build.py. A collapsed advanced panel sends real pinhole batches from the current static Studio camera against the canonical analytic shell/bubble descriptors. This is a distance diagnostic, not client contribution to a server render. It does not trace triangles, rods, refraction, lighting, or spectral paths, and does not reduce server work.

Controls: 16–4096 rays per batch, maximum 0.1–10 batches/second, and 1–50% measured-wall-time duty scheduling. Duty is not a hardware utilization cap. Defaults off, no persistence. Disable, page hide and page exit cancel/discard pending work. No overlapping batches; old results reject when the geometry/camera changes. Invalid settings disable work. Unsupported WebGL2/float targets fail visibly. Static visual preview and uncoated spherical geometry are required. Bubble-only view deliberately omits the shell to inspect sphere intersections; it is not a glass transmission image.

## Verification

- node tests/test_client_ray_diagnostic.mjs passed: limits, camera ray launch, frequency/duty, pending cancellation, hidden pause, stale output rejection, repeated enable.
- Original builder/kernel tests and GPU readback are owned by the compiler survey worker.
- Isolated real Chrome module UI/GPU test passed: 256 shell hits; bubble-only view 7 hits; hide/resume, disable and invalid controls; no JS exceptions.
- Published-bootstrap overlay test passed at 440x900: actual Studio camera and geometry produced 80 sphere hits from256 rays, physical nearest distances24.58–31.35mm. Screenshot inspected. No exposure started or user browser opened.
- Earlier browser test exposed unbound setTimeout Illegal invocation; scheduler defaults now wrap browser timers, and both tests reran successfully.

## Release

Six scoped static assets staged under build/dot-client-ray-20261109/release. release-manifest.json pins current published bootstrap and staged hashes. Bootstrap was derived from the published file, adding only import/initialization; it does not include unrelated source-only physics/chase/volume changes. No server restart needed. Deployment status must be verified separately.

## Further work

Genuine client render participation still requires identical compiled transport/geometry contracts, scene-and-camera-bound task ownership, result validation and replacement of server work. Native branch controls and callback/chunk/file streaming sinks exist, but Studio has no distributed ray producer/consumer today. No savings claim has been made.

Published incrementally after restaging atop the optical-volume release. The first stale-bootstrap guard rejected before any write. Final published six asset hashes verified over trusted-CA HTTPS;219 other existing files unchanged. No restart. Browser overlay integration reran successfully after restage. Verification receipt: build/dot-client-ray-20261109/publication.json. Production browser navigation itself remains subject to the known scratch Chrome certificate trust limitation; no warning bypassed.
