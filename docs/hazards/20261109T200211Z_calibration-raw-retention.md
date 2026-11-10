# Raw-linear calibration retention (source only)

The previous calibration sweep stored segmented 8-bit PNG means after automatic
normalization, tone/gamut mapping and sRGB transfer. Its mean/255 report was labeled
a gain but was not a linear sensor measurement or fitted profile. That report now
states visual-only PNG ratios, with an explicit warning against calibration use.

Each completed calibration exposure now fetches its existing HDR NPZ and embeds
that original compressed archive byte-for-byte as base64 in the downloaded dataset,
with SHA-256 and byte count. Aggregate original archives are bounded to 64 MiB;
an unavailable/invalid/oversized archive is explicitly marked unavailable and the
UI reports image and raw-capture counts separately. No silent PNG fallback exists.
A selected-array option in the existing NPZ reader loads numeric sensor/profile
arrays while leaving Unicode metadata intact in the retained original archive.

Separate unnormalized entire-frame sums/means/min/max/negative counts are recorded
for raw sensor and profiled linear sRGB arrays, with dtype/shape. Recorded film,
profile, display, geometry, camera, elapsed/sample/ray counts accompany them.
Whole-frame statistics are intentionally not brightness-segmented. They still
include scene lighting/geometry and are not a matched neutral target or a matrix fit.
There is no division by wall-clock time, white percentile, display brightness or
sample count. The raw units are not claimed radiometrically calibrated. Future
fitting requires matched known target spectra, exposure normalization, dark/neutral
references and held-out validation. Existing A/B calibration routing is retained.

Tests: new Node suite creates a real numpy.savez_compressed archive with current
Unicode metadata, proves selected numerical parsing, exact original-byte/base64
roundtrip and SHA, data independent of display controls, finite shape checks,
missing raw/HTTP failure, header and actual-byte budget enforcement. All pass.
Existing lighting-profile/failed-calibration restoration suite passes. Read-only
capture of a real completed 512x512 HDR (8,786,767 bytes) roundtrips exactly;
receipt in build/dot-raw-calibration-20261109/real-hdr-receipt.json.
Existing light-strength suite fails before calibration assertions because current
unpublished Studio source calls writeStudioFixtureRows, which its VM harness does
not inject. Neither that source nor its test was modified here; current published
Studio does not use the extracted helper. No broad suite success is claimed.

Two static modules staged with prior published hashes under
build/dot-raw-calibration-20261109/release. No server restart, publication, user
calibration run, new exposure, or edit to saved raw/image data occurred.
