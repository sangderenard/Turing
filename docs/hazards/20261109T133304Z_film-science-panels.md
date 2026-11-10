2026-11-09 DIRECT FILM SCIENCE PANELS (MEASURED, Windows local checkout):

Added source-only input/output 4x2 colour panels beneath the preserved six
Gaussian film-response controls. The component requests the CPU film_preview
endpoint only when Film Science is open or explicitly refreshed. It draws
server-returned display bytes, without another sensor matrix or tone mapper.
Sensitivity edits and current photo-correction inputs debounce for 400 ms;
new changes, invalid sensitivity, monochrome and closing abort old requests.
A generation guard rejects late results even when an aborted fetch completes.
No photo upload, cancellation, exposure or GPU request is made by this module.

Histograms count eight equal-area patches in 36 circular, centred 10-degree
encoded-sRGB HSV bins. Dark (max <= .01) and low-chroma (max-min < .02) patches
are excluded and counted separately. Both charts share the 0..8 y scale.
RGB ranges, pre-compression out-of-gamut counts, profile conditioning,
neutral-fallback warnings, limits and current corrections are visible. The UI
explicitly disclaims wavelength, perceptual-uniformity and ICC-calibration claims.

Verification: node tests/meowdoku_film_science_check.mjs passes eight production
component lifecycle cases plus hue accounting. The tests use native events with
DOM/canvas/fetch/clock boundaries doubled, not an actual browser. They cover
lazy/initial open, exact server-byte painting, debounce and latest-wins races,
invalid input, monochrome, close/reopen, reset/Apply/programmatic correction
sync, busy/timeout/network/bad-JSON/bad-matrix errors and disposal. Existing
meowdoku_film_response_check.mjs passes. JS syntax and scoped whitespace checks
pass. No export, live-page browser test, server restart or GPU render was run.

The initial handwritten test contract used lowercase patch names; review and a
real CPU preview_film(default response) call found the actual titlecase names,
including 'Gray 20%'. The UI now validates those exact names. The captured real
response is tests/fixtures/marble_film_preview_default.json; the Node suite
paints every unmodified input/output triplet from that fixture exactly.
Follow-up: the parent reported isolated-browser panel/histogram QA passed,
but noted 11px source histogram text shrank to about 5px in the sidebar. The
canvas font is now 20px, with bounded 50px x-axis tick slots. The Node test
checks all labels and bars stay within canvas dimensions. The eight patches
and readable nonzero-bin text legend are unchanged.

Confirmed profile display controls now feed the direct preview too:
normalization auto99/manual, unscaled numeric normalization_white in
[1e-6, 1e12], and tone_curve exponential/linear. Missing or disabled legacy
controls use auto99/1/exponential. Input/change, reset and programmatic sync
trigger the same request lifecycle. Actual server display_transform metadata
describes normalization, effective white, tone formula, transfer and gamut
mapping; no formula is executed in JS. Updated default and manual-white 2.5
linear fixtures were captured from the actual CPU helper. Ten production
component cases now pass, including exact manual/linear server-byte painting,
profile-control validation/bounds, disabled-legacy defaults and reset. Existing
film-response checks still pass. No export or live-server work was performed.
The manual/linear fixture reports clip(Y/white,0,1), which first failed a test
expecting the abbreviated Y/white caption. The assertion now checks the exact
server formula is preserved, including clipping bounds, rather than inventing
or abbreviating a client-side recipe; the component already rendered it intact.
