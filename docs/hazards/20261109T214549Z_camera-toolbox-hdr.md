# Camera toolbox, actual HDR metering and focus-reader boundary

Added isolated camera toolbox, completed-HDR sampler and mount modules. The
toolbox defaults to Hand, intercepts canvas gestures only in reticle mode,
checks capability/surface ownership and invalidates asynchronous measurements
on newer click, source revision, tool/spread change, Escape, hidden/blur or
disposal. Focus stays visibly unavailable because no verified arbitrary-scene,
mixed-geometry, refraction-aware reader was found. Existing target distance,
sphere-only queries or a fresh CPU ray solver were not substituted.

HDR point mode means one accumulated pixel, never one new ray. Gaussian sigma
uses the shorter image side, truncates at 3 sigma per axis, clips/renormalizes
at actual image boundaries and retains negative linear values. The reader uses
existing parseNpz to select only actual linear_srgb_exposure; raw sensor RGB
cannot silently stand in for it. White-point proposals assume a neutral target;
exposure proposals use explicit current recorded white and pre-tone linear
18-percent metering. Out-of-range corrections fail instead of clipping.

Narrow marble_photo.js hooks expose only the actually decoded completed final
profiled image, exact HDR/job/revision, detached parameters and recorded white.
Nonfinal orbit, playback, active job, pending/failed retouch and stale display
refuse. Pointer coordinates use the actual centered object-fit:contain image
rectangle, not letterbox pixels, and HDR dimensions must match the decoded
canvas. A proposed correction is only sent on the separate Apply button through
the incumbent retouch queue after rechecking exact source identity and ranges.
No second render pipeline or background exposure is started.

Passed node checks: meowdoku_spot_sample_check.mjs (real compressed NumPy archive,
Gaussian/edge/orientation/immutability/bounds, stale/cancel/error cases),
meowdoku_camera_toolbox_check.mjs (DOM-double input lifecycle plus real HDR
read/proposal and explicit Apply), meowdoku_calibration_raw_check.mjs, and all
50 cases in meowdoku_marble_photo_retouch_check.mjs, including five new getter/
apply lifecycle cases. No full repository suite, production HDR sampling,
visual browser QA, native autofocus, publication or restart was performed.

Initializer and exact existing scene/query audit are documented in
examples/meowdoku/STUDIO_CAMERA_TOOLBOX_2026-11-09.md. Shared main-frame/bootstrap
mount and the separate Phong DOF work remain with the coordinating worker.
