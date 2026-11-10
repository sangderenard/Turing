# Percentile and specular display integration

Source is complete and frozen, but not published in this pass. Production live
view was active (fresh check PID 38448, digest 8c4b86a6..., state live). No user
worker was stopped or changed, and no production request was sent.

Shared backend: 119 Spectral tests and 96 Turing tests passed. UI/API worker:
four Node suites, 50 Film Science tests and 19 live tests passed. Integration
rerun of Film Science/live/display/finished-retouch passed 106 tests after the
new live percentile validator correctly rejected booleans, strings, nonfinite
values and huge overflowing integers. Existing defaults remain auto99 with
exponential tone; the new slider is explicitly adjustable percentile 1..100.

Actual Chrome tests used only the isolated 8883 helper and generated completed
HDR. Specular selection started at 95 and forced/disabled linear tone. A real
slider edit to 50 reached completed retouch, and the actual persisted PNG was
pixel-identical to the shared saved-HDR recipe. Specular 95 produced 1025 white
pixels of 8192; Specular 50 produced 4097; adjustable 50 with restored exponential
tone produced 1. The direct Film Science panel updated to the chosen percentile
and returned its actual recipe. Different percentile white between the 8-patch
panel and the full image is expected because their sample populations differ.

Reset restored the original image pixels exactly and defaults auto99/99/
exponential. File-byte identity was initially an incorrect test expectation:
the minimal fixture PNG has no sRGB chunk, while real retouch correctly adds
one. Pixel arrays were identical. All checks preserved the raw HDR SHA256:
29ba1b4778791a6b7caf10c41d139949c8f134b6e114d776e62b2689bf781f27.

The fake live fixture verified Specular 95 and then 60 as display revisions 1
and 2, with one start, no scene restart, unchanged history/checkpoint, and one
explicit stop at the end. This was UI/API wiring only, not a GPU live exposure.
The test tab was closed. The user live view was left running.

Artifacts: build/dot-percentile-display-20261109/ contains baseline, real
completed-HDR comparisons, reset and fake-live records. Tested assets and
backend hashes are frozen in build/dot-percentile-release-20261109/. The staged
release remains unpublished pending an authorized idle/restart window.
