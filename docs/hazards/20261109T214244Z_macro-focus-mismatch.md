# Macro focus mismatch and published repair

See examples/meowdoku/STUDIO_LENS_CONTROLS_2026-11-09.md for exact scope and limits.
The observed user exposure chose macro_50mm with 230 mm focus while the camera
was 23.126 mm from its target. Keep the source minimum rather than relabel this
as a physical 2 cm lens. New explicit custom target-focus clears source identity
and preserves camera pose, FOV and pupil. Frame & focus uses actual exposure
scale/FOV/profile bounds, with 1 mm numerical guard instead of the separate
21 mm framing floor. Dolly no longer has the unrelated 1.2-unit scale floor.

Published five static assets after the material editor release at21:42:09Z.
Trusted HTTPS hashes matched,224 unrelated assets stayed unchanged, server
boot775d6760ce04/session unchanged, no restart and no photograph interruption.
Eleven Node suites pass, including actual staged and served handlers. Browser
QA blocked before completion; no new GPU photo quality proof is claimed.

Read-only applyLensCamera now also updates the focus-distance warning DOM;
callers with an actual player pose should provide that pose in metres. Normal
focusAt still enforces source minima. Explicit focusAt(distanceM,{camera,
custom:true}) validates before changing controls and clears source provenance.
For off-axis surface picks use axial dot(hit-eye,forward), not slant distance.
A first-surface pick is not refractive autofocus of internal glass detail.

The prior cold export and native CPU trial remain reference evidence, not a
replacement renderer. Close rebuilt120Macro bench states failed their fixed-
back solve and tested poorly at the current image plate. Keep references gated.
