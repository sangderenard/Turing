2026-11-09 OPT-IN ISOLATED-HIGHLIGHT CONTROLS (MEASURED, local Windows source)

The existing photo retouch panel now exposes firefly_filter (boolean, off),
firefly_threshold (2..64, default 8) and firefly_strength (0..1, default 1).
The threshold numeric input preserves raw values; the strength slider scales
0..100 to 0..1. Options are disabled while the filter is off and retain chosen
values. Reset returns off/8/1 and the baked display gamma. All three controls
are unavailable for legacy unprofiled HDR and omitted from its retouch POST.
The UI states: May reduce genuine isolated glints or sharp highlight peaks.

Parameters enter the existing bounded serialized retouch queue. Invalid
options cancel pending valid edits before the API. Obsolete responses cannot
refresh the image or rendered metadata. Last-rendered metadata reports the
server's actual filter settings, method and caution rather than reconstructing
them from newer local edits. The raw HDR link stays unchanged.

Film Science validates and forwards the same three values through its existing
CPU subprocess. The actual eight flat-patch pipeline produces identical sensor,
profiled linear and display RGB values and the same white with the filter on.
The panel explains this absence of spatial effect; no fake filtered example is
drawn. The server's actual recipe fields are shown. Disabled legacy controls
use the synthetic preview's supported defaults rather than stale UI values.

Validation after final edits: 35 production retouch lifecycle checks and 14
Film Science component checks pass in Node. 36 Film Science Python tests pass,
including real sibling Spectral CPU pipeline and strict boolean/range rejection.
32 existing finished-retouch/display-transform Python tests pass. JS syntax,
Python compilation and scoped tracked diff whitespace checks pass. Before-edit
baseline was 32 retouch checks, 12 Film Science checks and 21 Python checks.

Only additive retouch HTML, marble_photo.js, marble_film_science.js,
film_science.py and their focused tests were changed by this worker. Concurrent
underlight/live/Auto/gobo changes remain present. Backend filtering and retouch
validation have a separate owner. No browser visual run, GPU exposure, server
restart, asset export, full suite, commit or publication was performed here.
Parent owns combined browser validation before publication.

## Actual completed-image browser/API verification

The coordinator restarted only its isolated 8883 fixture helper with current
backend imports, never the production server or active exposure. A generated
64×128 profiled HDR contained one synthetic isolated bright pixel in a gray
patch. Enabling the UI filter changed exactly that one pixel from [255,255,255]
to its neighbor value [155,155,155]. Disabling it restored every original image
byte. Threshold 64 and strength 0 reached saved retouch unchanged, and zero
strength preserved exact pixels. Reset restored off/8/1 with options disabled.
Raw HDR SHA256 stayed unchanged. Receipt:
build/dot-light-color-20261109/firefly-browser-verification.json.
