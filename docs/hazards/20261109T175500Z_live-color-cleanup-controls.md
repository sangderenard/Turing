# Live color and cleanup controls (source verified; not published)

Measured current live status ea5bacd190f1288375887874470f82eecef298f3b4b4cb1909f67af7fa60af64:
specular normalization, 95th percentile, linear luminance clipping, film matrix
condition 1.000628, no neutral fallback, firefly filter false. This is not proof
that this job is the exact screenshot the user supplied. The clipping policy is
intentional and remains unchanged. No claim of fully calibrated broadband color.

The real defect in control reachability: marble_live.py kept only tint and
normalization from display settings. Live UI had no exposure/white-balance or
color/cleanup controls, while the existing snapshot renderer already supported
them. Added live exposure, white balance, saturation, vibrance, black point, and
existing opt-in isolated-highlight cleanup (threshold and strength). All use
existing Spectral display math, unchanged. Display-only revisions retain the
ring/checkpoint and visible image. Incomplete/out-of-range display edits retain
the prior worker and accepted settings; fixing input recovers without restart.
Defaults preserve existing output. The filter is off by default, is not a general
denoiser, and the UI warns that real isolated glints can be reduced.

Verification: initial 19 Python tests and existing Node suite passed before edits.
Final 130 focused Python tests passed (marble_live, live_ray_batch,
display_transform), plus the full Node live lifecycle suite with new color,
filter, no-blank/no-restart, incomplete-edit and recovery checks. No new GPU job
or actual browser UI QA performed. Existing active user's job was not touched.

CPU-only check on latest persisted HDR b29b630cf106ade45eab6b3419bdd39e0fcf5333f12f35039ea364272c97fc48/exposure-000632.npz:
512x512 finite profiled values. Specular95 produces 5.0121% fully white pixels;
same raw data at -2 display stops produces 0.01869%. This demonstrates display
clipping recovery on a saved exposure, not correction of the latest live image.
Raw data and saved image files were not modified.

Publication staging: build/dot-live-color-release-20261109 contains the new live
module and HTML patched against published HTML; it deliberately excludes the
unpublished chase-camera controls. manifest.json records current published input
hashes and implementation source hashes. Server imports marble_live at startup,
so a guarded idle backend restart is required; existing compiled physics must
remain in retained-product mode. Page refresh ends the current live worker and
GPU accumulation, so do not claim history survives publication/refresh. New
controls preserve history after the new page/worker begins. Process inventory via
Get-CimInstance returned access unavailable; no idle claim or restart was made.


## Scoped publication completed

User subsequently requested everything move forward live as available. Rechecked
all nonterminal disk jobs via the running server; all were inactive (404), latest
jobs canceled. Get-Process showed no newer Python worker, so no active render was
stopped. Lifecycle restart saved sessions and closed old PID38320. Relaunched same
argv/env with existing retained-fountain-products receipt, PID45412 boot82a45c2d159f.
All products settled. Five scoped assets published: marble_live.js, marble_studio.html,
marble_light_strengths.js, marble_lighting_profiles.js, marble_calibration.js.
Last three belong to independently tested solo/quick-slot owner. All five HTTP
hashes verified using existing trusted CA; all 220 published files matched expected
hashes, including unchanged compiled products and excluded chase/physics UI.
HTML IDs unique and all added live controls present. Independent production browser
QA remains with quick-light owner. Existing loaded pages need refresh for new UI.
No active exposure interrupted. Noise owner's exact three-line NEE source was frozen
and GPU-verified before restart; newly launched workers import it as usual.
Release evidence: build/dot-live-color-release-20261109/verification.json.
