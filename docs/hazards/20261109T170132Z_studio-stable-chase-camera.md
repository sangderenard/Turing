2026-11-09 STABLE STUDIO CHASE CAMERA (MEASURED, canonical Windows source):

New marble_chase_camera.js filters completed displayed native positions, not
queued product state, and never changes physics. Stable Y-up horizon, body-
radius-relative distance, bounded damped aim/look-ahead, steering-locked auto
heading, fixed-heading option, drag orbit, scroll zoom and R/button recenter.
Run changes, seeks, teleports and long interruptions reset camera history.
The old pitch+pi/2 raw follow anchor is removed. Focused WASD/D-pad still uses
the incumbent gravity mapper, now with the active view's yaw. Holding a key
locks automatic heading so camera feedback cannot steer the body in circles.
Static Studio/photo orbit variables, units and pose function are unchanged.
No camera collision avoidance or shared-room physics is implied.

Source hooks: marble_studio.js/html; one module registration each in
web_export.py and server.py no-store list. New chase helper/test plus updated
existing drive adapter test. These registration files were frozen for the
separate camera-housing backend release. No export/publication was performed
for chase; the published static assets remain the previous visual release.

Passed Node: chase, drive input, Studio table, marble scale, live UI, ten
stage-course topology cases; node syntax checks and Python compile checks
for server/web_export; scoped git diff --check. Existing Spectral exposure
and managed-time bridge tests: 22 passed in 0.35s (scheduler/bridge fixtures,
not a Studio motion optical render).

Actual isolated Chrome opened current source Studio with retained published
compute.js, real generated WASM shell and actual compiled 4k toy products.
Displayed frame 10 advanced to 63; runtime had queued 408 native steps.
Verified held-W heading lock, blur drive release, scroll zoom, recenter,
fixed heading, case/visual return and exact unchanged photo camera. Browser
exception log was empty. Screenshot was inspected. This is Phong gameplay,
not ray output. Helper and scratch Chrome were closed; no object save,
photo/live job, production server restart or user browser tab was touched.
Artifacts: build/dot-studio-chase-20261109/browser-result.json and chase.png.
Second expanded Chrome run confirmed GpuProduct on NVIDIA Ampere, displayed
frames 7 to 57 and 372 queued steps. It also exercised actual pointer drag,
held/released D-pad and R-key recenter. All passed with no browser exceptions;
the result/screenshot artifacts now contain this expanded run.

Recorded motion architecture is documented in
examples/meowdoku/STUDIO_RECORDED_MOTION_HANDOFF_2026-11-09.md. Reuse thick
lens exposure slices, flash/sensor budgets, on_dt_request, SceneFrameProvider,
scene versioning, causal collector and managed-time commit/rollback. The
current callback's physics backend is a stub; the live static ray history is
not temporal motion. Native full archive, per-tick orientation convention,
exact tick boundary contract, whole-scene/compound admission, immutable BVH
version ownership, optical temporal weights and real GPU motion checks remain
required. No duplicate scheduler, interpolation-based physics or fabricated
motion render was added.
