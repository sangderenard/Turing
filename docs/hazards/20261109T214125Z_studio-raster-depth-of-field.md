2026-11-09 Phong depth-buffer DOF and completed-HDR toolbox (MEASURED, source only)

Implemented real offscreen Phong colour/depth rendering, analytical circle of
confusion and 48 stable dithered fragment samples. The current view, including
player/chase pose, feeds existing lens controls; the raster preview has an
explicit 64-pixel blur cap. Transparent glass has no raster depth, so this is
not refractive focus. Ray rendering and active exposures remain untouched.

Real RTX 3060 browser gate passed: in-focus pixels exact; near/far checkerboard
variance 16256.25 to 58.041/72.443 with wide pupil; pupil reduction restored
detail; moving focus switched the sharp plane; tested near silhouette had no
far-colour bleed. Resize/cancel/state restoration passed, GL error count zero.
Current source Studio rendered 72,960 triangles and passed focus/aperture,
off/on, and invalid-edit checks. No full-profile endpoint existed in the
isolated read-only QA server, so its UI test used Custom analytical.

The camera toolbox defaults to Hand, supports actual completed profiled HDR
point/Gaussian metering and explicit proposal Apply through the incumbent
retouch queue. Letterboxing, image dimensions, stale sources, canceled work and
delayed image decode are guarded. Full-scene refractive spot focus remains
disabled rather than substituting camera-target distance or a partial ray.

Eight focused Node suites passed, including all 50 photo lifecycle cases;
modified JS/Python syntax and diff whitespace checks passed. See
examples/meowdoku/STUDIO_PHONG_DOF_2026-11-09.md and
examples/meowdoku/STUDIO_CAMERA_TOOLBOX_2026-11-09.md for reproducible evidence.
No production files exported, no server restarted, no active user scene changed.
Shared source contains unpublished native physics/chase changes: release only
the narrow tested patch rebased onto the latest published material/macro assets.
