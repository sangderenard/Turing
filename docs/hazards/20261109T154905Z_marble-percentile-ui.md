# Adjustable percentile and specular display controls, 2026-11-09

Source-only follow-up to the Studio analytic release. Photo retouch, Film Science
and live controls now route auto99, percentile, specular and manual normalization.
Auto99/exponential remains the original default. New percentile sliders span
1..100 in 0.1 increments. Entering Specular selects 95 and forces linear tone;
slider edits are retained, repeated selection does not reset them, and leaving
Specular restores the previously selected non-specular tone. Reset restores 99,
auto99 and exponential. Profile-less HDR retains its legacy path.

Film Science and live API validation use strict finite numeric percentiles,
reject booleans/strings/overflow and normalize Specular tone only after validating
the requested choice. Omitted percentiles default 95 for Specular and 99 otherwise.
Live display edits retain the existing revision route without scene recreation,
ring clearing or checkpoint changes. Manual white initializes from the actual
shown image's effective-white metadata, not a stale 99th-percentile fallback or a
separate retouched image's metadata.

Focused verification: 50 Film Science Python tests and 19 live Python tests passed.
Production-component Node checks cover photo retouch, Film Science, live rays,
and film-response compatibility. New cases include invalid/end-point percentile
values, debounce and repeated mode selection, previous-tone restoration, reset,
server metadata and exact returned pixels, legacy behavior and live revision
updates. Refreshed default/manual fixtures are captured from the actual CPU
pipeline: sensor/linear/input/output matrices match their previous fixtures
exactly. Added a current Specular 50 CPU fixture. No browser/GPU run, publication,
server restart or production request was made by this worker. Browser QA and
publication remain with the integration owner. Active user live view untouched.

Owned files: examples/meowdoku/film_science.py, marble_live.py; web/
marble_studio.html, marble_photo.js, marble_film_science.js and marble_live.js.
Focused tests and three Film Science fixtures accompany them. Starting copies
and final source hashes are under build/dot-percentile-controls-20261109/.
Backend colour math and snapshot metadata belong to the separate backend owner.
