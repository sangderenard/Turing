# Studio rig and display release, 2026-11-09

Published the separately tested batch through the existing lifecycle restart,
after the user's 600-second exposure completed normally. No exposure was
cancelled, saved design rewritten, unrelated process stopped, commit or push made.

Production PID 66316, boot 8e845898b767. The existing user session restored 1/1.
Verified page: https://173.195.26.104:8766/marble_studio.html?v=rig-20261109

Delivered: ten-light simple UI with white fresh filters, per-light poses/locks,
optional independent Studio/exposure strengths, two table snoots and gridded
underlight; clear-glass photography table with catalog selection; simple room
finish matrix, recoverable Advanced settings, concentric backdrop and inversion;
creative saturation/vibrance/black-point display controls; live finite ray history;
nonlinear 1–200 bounce control with exact integer entry and default 4.

The emitter builder now aligns triangle winding with all ten authored directions.
This corrects previous reversed forward emission while preserving profiles' poses,
dimensions and strengths. Table-enabled Studio/photo/live scale uses the saved
physical radius, and vertical FOV matches the raster camera. Built-in and legacy
table-off framing retain their prior conventions. Material IDs 0–31 and private
extras 32–34 remain stable; new finishes use 35–37, fixture rows 38–49.

Verification: 15 combined JavaScript suites passed, followed by final bounce,
table, live and photo reruns (8 live lifecycle and 32 photo cases). An 84-test
Python set passed; final bounce/contact/source set passed 13 tests. Existing
historical object receipt mismatch and D6 proxy test failure remain documented
in their owners' receipts and were not hidden or rewritten.

Actual GPU combined scene smoke: 128×128, 4 bounces, 16 epochs, 99,472 total
triangles and all ten light source identities. Raw/profiled HDR were finite and
profile/PNG matched exact recomputation. A separate QA-only reduced 200-bounce
dispatch read both GPU uniforms back as 200 and completed with 256 camera paths
plus 160 forward rays. This is not full-resolution performance or watchdog
assurance; smoke images are noisy and do not establish convergence quality.

TLS-validated HTTP reads matched all 26 staged asset hashes. The rebooted CPU
Film Science endpoint returned the requested manual/linear/creative transform.
Actual Chrome page boot showed the expected table, ten white filters, material
matrix, live toggle and 200 maximum. Legacy seven-light profile browser test
kept old split strengths and loaded new lights off. No security warning bypass.

Artifacts: build/dot-light-color-20261109/release-manifest.json,
backend-source-manifest.json, publication-receipt.json, http-verification.json,
combined-ui-cpu-verification.json and release/ frozen assets. GPU receipts are
in build/dot-combined-studio-20261109/. Backups are recorded by publication receipt.

Limits remain: Phong sorted transparency/occlusion is approximate; intersecting
and nested media are unresolved; old unprofiled HDR uses its legacy display path.
Bubble, Advanced gobo and automatic bounce work are subsequent tasks, not this
release. Photo subprocesses import Spectral source on each launch, so next-batch
renderer work must remain isolated until tested even without a server restart.
