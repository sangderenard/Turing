2026-11-13 Studio autofocus: original ray law + existing Phong DOF feedback (isolated, not deployed)

Owner requested default Gaussian ray casting, aperture/distance-aware scene focus,
and a real search driven by Phong image feedback.

Implementation in examples/meowdoku/web/marble_autofocus.js:
- Nine Gaussian-weighted image-plane rays use StudioRangefinder and its original
  compiled hit_tri Wasm. No new intersection or renderer implementation.
- Uses the first unobscured positive surface hit per ray, not hidden surfaces.
- Uses existing dofParameters/dofFocusAt/dofCircleRadius, including tilt/shift,
  pupil (therefore f-stop) and axial range. At the current preview resolution,
  enumerates reciprocal-focus intervals maximizing weighted samples within a
  one-pixel circle-of-confusion radius. Gaussian sigma is 0.28 image widths.
- Completed primary/secondary/DOF canvas frames supply Gaussian-weighted squared
  luminance gradients. Coarse contrast scan then two bounded reciprocal-focus
  refinements select within the maximum-coverage set. No ray-trace overlay.
- Only the winner reaches the existing lens-controls focusAt writer, shared by
  photos/ray exposures and Phong. Trial focus is preview-only. FOV/pupil do not
  change; Original settings becomes Custom analytical only when applying a win.
- Default is one shot after 750 ms stable scene; manual inputs release it.
  Focus now / Cancel remain available. Scene/camera/lighting changes invalidate
  the search. Thirty-second budget, no-contrast retention, hidden-page/context
  loss cancellation and pagehide disposal are included.

Measured in isolated worktree based on 8ff4bf80:
- node --experimental-default-type=module tests/meowdoku_autofocus_check.mjs: PASS
  Gaussian sampling, optical coverage, aperture, tilt/shift, image metric,
  one-shot controller, real feedback consumption via DOM adapter, final lens
  write, stale rejection, manual cancellation and disposal.
- meowdoku_marble_studio_dof_check.mjs: PASS.
- meowdoku_rangefinder_check.mjs and meowdoku_rangefinder_panel_check.mjs: PASS
  with the existing real compiled Wasm and actual lens controls. First attempts
  returned 404 because the new worktree lacked ignored build artifacts. Copied
  only studio_bvh_query.json, studio_hit_tri_batch.json and .wasm from canonical
  build/dot-client-ray-20261109/studio-compute; no compilation was run.
- Actual Chrome GPU fixture meowdoku_autofocus_gpu_check.html: PASS on existing
  Studio DOF shader and authored three-plane test scene. Maximum coverage
  0.4702433088389244; measured contrast selected 104.218569 mm from five candidates
  around the dominant 100 mm plane. Contrast 42155.6989 vs 10902.9763 at 92.5107 mm.
- Existing DOF GPU tests: PASS; ANGLE RTX3060 D3D11; zero GL errors, zero far-blue
  foreground leakage, pixel-exact in-focus region, aperture/focus/resize/cancel
  checks passed. The local stdlib server needed .mjs MIME explicitly set to
  application/javascript (Windows registry defaults to text/plain).
- JS parse checks and git diff --check pass. Full suite NOT run.

Limits: this is sampled visible-area coverage, not a mathematical optimum over
all scene geometry. Straight rays and the existing opaque-depth Phong DOF do not
model refracted apparent depth or guarantee focus through glass. Full Studio UI,
startup lens catalog timing, and a resulting ray-photo exposure have NOT been
accepted in a live browser. No production export, merge, push, restart or live
service modification. The isolated localhost test server was stopped and its
browser tab closed. Subsequent controller-only coarse-to-fine/cancellation edits
were retested in the DOM adapter; the original GPU geometry/metric remains as
measured. This receipt is not a claim that the complete feature is deployed.

## Isolated full-page QA follow-up

Actual Chrome Studio page with original published assets rendered 72,960 triangles.
Initial default autofocus completed at 1590.35 mm with 73.5% weighted sampled
coverage within one preview pixel. Existing photo controls changed to Custom
analytical and the same axial focus. Manual focus input disabled automatic mode.

Found and repaired an interaction race: Focus now must wait for the next completed
Studio frame, not try capturing from the click handler after an edit. It now queues
on the existing frame callback. Regression test checks no ray query before that frame.
After repair, actual manual search completed at 975.20 mm with 73.5% coverage;
Cancel focus search then preserved 975.196016622237 mm. A separate run hit the
30-second guard and retained its authored focus. The search is bounded, not guaranteed
to complete under every GPU load. UI screenshots were inspected, including scene
and cancellation status. Test tab closed and own static server stopped.

Backend limitation: attempted fresh-state dev server startup with --no-wasm still
entered prepare_lanes and began a damped_axis compilation. Stopped that owned process
promptly; no further backend start or compiler run. The actual UI test instead used
Python static serving on loopback with no mocked API responses: catalogue, accounts,
and photo jobs API were unavailable (404). Thus authenticated server integration and
ray submission/render acceptance remain UNTESTED. No live service was restarted or
modified. No ray-traced render was requested.

Shared focus path confirmed in source: photo.focusLensAt delegates to existing
lensControls.focusAt; photographic options use the same lensControls.applyCamera
before contract submission (marble_photo.js). Existing lens-control tests pass.
This is source/unit verification, not evidence of a rendered ray exposure.
