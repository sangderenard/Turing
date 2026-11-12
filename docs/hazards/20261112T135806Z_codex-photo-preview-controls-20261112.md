# Rendered photo preview controls

## Request and scope
User: "the rendered image preview needs minimize and close controls"

The existing floating photo panel gains Minimize/Restore and Close. Show photo
preview restores a closed panel from the exposure controls. Closing is view-only:
it does not cancel a render or delete its result. Progressive status updates and
late image decodes preserve dismissal; explicitly starting a new exposure opens
it again. Orbit playback stops when hidden and HDR sampling excludes minimized
content. No renderer, material, shader, service or API behavior was changed.

## Verification
Base: 10b60214. Isolated branch codex/photo-preview-controls-20261112.
- node tests/meowdoku_marble_photo_retouch_check.mjs: 56 cases passed, including
  six new preview lifecycle cases; network, image and timers are mocked.
- node tests/meowdoku_film_response_check.mjs: passed.
- node tests/meowdoku_marble_photo_camera_check.mjs: passed.
- git diff --check: passed.

On untouched base, exposure-start cases in both existing lifecycle fixtures fail
because they omit current exposure settings. Fixtures now supply the page's
substeps=1800, ISO=100 and highlight clip=25 defaults. Production validation is
unchanged.

Two broader checks fail identically on untouched base and this branch:
- meowdoku_marble_photo_room_check.mjs: base Studio emitter material profile
  unavailable (studio_scene_builder.js:400).
- meowdoku_photo_participation_check.mjs: missing expected Device intersections
  require exception (line 7).
These unrelated failures were not repaired or hidden.

## Limits
No live browser visual verification, render, GPU link, restart or deployment.
Canonical checkout has concurrent trigger-bus changes; it was not modified.
The controls have keyboard-native buttons, accessible names, expanded state and
focus restoration, with wrapping 40px-high toolbar buttons for narrow screens.
