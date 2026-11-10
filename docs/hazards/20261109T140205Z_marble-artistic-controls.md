2026-11-09 ARTISTIC POSTPROCESSING CONTROLS (MEASURED, local Windows source):

The existing photo retouch UI now exposes saturation [0,2] default1,
vibrance [-1,1] default0, and black_point [0,0.5] default0. Slider integers
are hundredths; labels report actual backend values. Black point is labeled
as a fraction of normalized white before tone mapping. These are artistic
linear-sRGB display adjustments, not calibration or perceptually uniform
colour controls. Raw HDR download routing remains unchanged.

All three use the existing serialized/latest-only retouch request lifecycle,
reset to neutral without losing baked gamma, survive stale polling metadata,
and are disabled and omitted for legacy unprofiled HDR. Film Science reads
the same controls, forwards them to its existing CPU preview endpoint, and
paints the returned RGB bytes. Actual backend formulas and the artistic-control
disclaimer are displayed from response metadata; no colour transform was
implemented in the browser.

The photo-start camera selector now affects framing only: both built-in
close-up and Studio orbit use the callback's authored fixtures and lights.
Built-in camera and checker defaults are untouched. The callback is read once
before packing and before canceling an earlier job, so an invalid scene read
cannot cancel an active exposure.

Measured focused checks: 31 photo lifecycle cases and 12 Film Science cases
pass, including new neutral/reset, bounds/invalid values, legacy omission,
latest-only races, live status preservation, raw-HDR link stability, exact
server-recipe text and byte painting, plus both camera-mode payloads.
Existing film-response and photo-room checks pass. Both changed JS modules
pass node --check; production HTML has unique IDs and matched output elements;
scoped git diff --check passes. These are mocked network/DOM/canvas/clock
component checks, not a claim of real-browser layout or live-server testing.
No production export, restart, real exposure, existing-exposure cancellation,
commit or push was performed. Existing dirty work was preserved.
