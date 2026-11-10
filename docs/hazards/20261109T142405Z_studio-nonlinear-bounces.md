2026-11-09 NONLINEAR BOUNCE CONTROL (MEASURED, local Windows source):

The Turing photo configuration boundary accepts integer max_bounces 1..200,
keeps default 4, and rejects out-of-range, fractional, boolean and nonfinite
values. The original photo-bounces field remains an actual integer count,
now a number input. A separate photo-bounces-slider uses position 0..100,
step .01, with count = round(1 + 199 * (position / 100)^2). No actual count
is reinterpreted as a position. Keyboard arrows change one bounce, Page keys
ten, and Home/End reach 1/200; accessible value text reports actual counts.

Both still-photo and live uploads read the same validated count. The live
view observes input from both controls and rebuilds its scene after a count
change. Invalid still-photo input is rejected before canceling another job.
The UI warns that high limits can substantially slow photos and live updates.

Measured before the change: the new admission tests failed on count 13 in
both photo and live configuration paths, at the old 1..12 validation bound.
After the change, 17 focused Python tests pass in .61s: the ten bounce cases,
six existing live-control tests, and the saved-mesh MRP1 decode test. No HTTP
server or GPU was started. One existing cffi imp deprecation warning remains.

The new JS check exhausts every .01 slider position, proves all 200 integer
counts are reachable without skips, inverse-roundtrips every count through
range-step rounding, and verifies wider low-count intervals, legacy/default
values, numeric validation, keyboard input and packed numeric values. The
production live lifecycle passes eight cases, including a 12-to-200 bounce
scene rebuild; photo/retouch passes 32 cases, including actual 200 upload and
201 rejection. Existing camera framing and authored-room packing checks pass.
Production JS syntax and scoped tracked whitespace checks pass.

Spectral validation is owned by the separate renderer worker. No Spectral
files, retouch implementation, saved assets, exports, server processes or GPU
jobs were changed by this task. Browser/GPU validation remains with the parent.
