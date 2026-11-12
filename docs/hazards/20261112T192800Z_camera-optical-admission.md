# 2026-11-12 Studio native optical movements

Camera admission now preserves native `tilt_shift` and `lens_tilt` through still,
pose-sequence and live camera contracts. Shift uses normalized principal-point
coordinates; tilt is the existing analytical focal-plane slope angle in radians,
not a new compound lens prescription. Defaults are zero. Admission bounds are
+/-1 shift and +/-pi/4 focal-plane tilt. Lens/front-element mechanics are not
claimed by these controls. Spectral companion branch wires the same fields into
SensorAccumulator construction and live lens updates.

CPU verification with the existing service Python:
- test_meowdoku_camera_movements.py and test_meowdoku_marble_live.py: 89 passed.
- test_meowdoku_camera_pose_sequences.py: 28 passed.
- Broad marble_photo collection blocked by missing SpeakToMe sibling in isolated
  worktree path; not a product failure or a render verification.
- No GL context, render, server restart, deployment, or shader link attempted.

Owner request: honest tilt/shift in both Phong and ray tracing, free pointing and
tripod height. This change covers backend native optical-field transport only;
frontend controls, projection and pose work is coordinated separately.

Follow-up validation: with SPEAKTOME_REPO and PYTHONPATH explicitly selecting the
existing sibling dependencies, the marble_photo suite excluding the torch-only
trigger metadata test passes 26 tests. The excluded test fails on untouched
canonical too because service Python intentionally has no torch. One transient
HTTP fixture timeout occurred during an isolated run before explicit dependency
selection; it did not recur in the corrected 26-test run.
Spectral companion commit 1797d9f passes 205 analytical-focus/live CPU tests.
Its old test stub lacked _light_field_information after response-state support;
that failure was reproduced against canonical before adding the field to the stub.
