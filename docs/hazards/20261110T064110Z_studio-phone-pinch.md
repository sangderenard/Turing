# Studio phone pinch repair, 2026-11-10

Published Studio used a single shared last-pointer position, so a second finger rotated the camera instead of zooming. Canvas touch-action already suppressed browser scrolling. Replaced only the camera-input block with the existing app.js convention: tracked pointer IDs and logarithmic two-finger span scaling. One finger still orbits. Releasing/canceling a pinch leaves its remaining finger idle until a fresh gesture. Blur, visibility loss and lost capture clear the gesture.

Source preserves its unpublished chase hooks. The live patch preserves the published particle-follow behavior and does not import that unfinished source. Lens minimum distance, zoom maximum, wheel behavior, held photographic camera, and unrelated scene code remain unchanged.

Validation: old real block fails new pinch regression; source and staged real blocks pass. Camera binding, chase, toolbox and syntax checks pass. Isolated full Studio page received DOM PointerEvents with pointerType touch: distance 3.45 to 6.9 and back, no pinch orbit, fresh orbit and cancellation correct, held photo pose unchanged, GL error 0. No hardware phone session was accessed; the test server rejected every API mutation.

Guarded one-file publication at 06:41 UTC: marble_studio.js SHA256 c7015c5eb269e10e2b50808e77a0c749ffadde015dc5404f4195ea7f20052155. Trusted TLS bytes verified and 279 unrelated published files unchanged. No restart or existing-tab refresh. Receipt and rollback bytes: build/dot-studio-pinch-20261110/. Do not run wholesale web export: source includes unpublished physics/chase work. The test tab and isolated server were closed.
