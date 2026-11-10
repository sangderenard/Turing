# Accessible underlight height, 2026-11-09

The requested underlight vertical control is now beside its simple strength and
filter. It edits the existing underlight pose Y, rather than storing a second
height field. Advanced edits and profile loads synchronize the shortcut. Invalid
or blank input leaves the last valid pose and scene unchanged.

Fresh default emitter Y is -1.60, lowered by 0.20 Studio units from -1.40.
The table top remains -1.00 and thickness 0.12. Existing ten-light profiles keep
their precise authored Y; legacy seven-light profiles still append new lights
off. No saved objects or existing profiles were rewritten.

Focused Node checks passed for pose-only edits, validation, Advanced synchronization,
old/new profile retention, all ten light directions and exact emitter translation
through the real photo wire packer. Split-strength tests passed unchanged. Real
Python upload/source and table/contact tests passed. No GPU exposure, export,
restart or publication was performed. Browser check is coordinated separately
with the live-controls worker on an isolated fixture server.

Actual Chrome fixture check subsequently passed: fresh -1.60, edit to -2.00,
Advanced Y synchronization, and restoration to -1.60 in both controls. Evidence
is in build/dot-live-ray-20261109/expanded-ui-fixture-verification.json.
