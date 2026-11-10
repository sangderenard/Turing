2026-11-09 MARBLE FIXED3 COLOR PROFILE (MEASURED, Windows RTX3060 source and CPU output):

Spectral camera_designer/marble_color_profile.py now converts the three-line
sensor output to CIE XYZ and linear sRGB before the display transform. The
650/550/450 nm CIE1931 values came from the official dataset, verified against
its published MD5 17cca777db64b17170f06f67ce9d3ab7. The fixed equal-power
three-line reference white is Bradford-adapted to sRGB D65. Sensor response
is inverted only below condition number 1000 and above minimum singular
value 1e-6. Unstable filters explicitly use neutral intensity fallback;
this is not recovered spectral luminance or calibrated broadband response.

Raw linear_rgb_exposure remains unchanged. New HDR files separately preserve
linear_srgb_exposure in float64 and exact color_profile_json. Live and saved
retouch share the same luminance tone/gamut/transfer recipe; no second sensor
matrix or sRGB transfer is applied. Old HDRs without profile fields retain
their legacy path. PNGs from the new path carry the sRGB chunk. Display
exposure still uses the 99th percentile and is not a radiometric calibration.

CPU profile processing of the earlier actual default/custom-film GPU renders
matched their output colors to relative L1 9.72199e-7. No new GPU exposure,
server restart, active-job cancellation, or publication occurred in this work.
Final focused runs: Spectral 67 passed; Turing photo/film contracts 23 passed
with one pre-existing cffi deprecation warning. Independent review found and
resolved a float32 persistence discrepancy (three one-code-value differences
in a seed-30 512x512 example); the new float64 roundtrip regression passes.

The direct sensor-pattern CLI uses the same profile and display functions,
not a scene renderer. It provides fixed eight-color sensor inputs and actual
output bytes for the Film Science UI. UI endpoint, histogram, terminal-state
retouch race fix, profile digest revision and publication belong to the
coordinated UI work; these results alone do not validate its browser behavior.

Existing source/material transport approximations, nested media, opacity
mixtures, absorption distance ownership, and historical PASS_SENSOR are not
corrected or scientifically validated by this output profile.
