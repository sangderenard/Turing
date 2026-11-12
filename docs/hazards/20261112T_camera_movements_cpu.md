# Studio camera movement wiring (2026-11-12)

Reuses Spectral analytical tilt_shift (normalized principal point) and lens_tilt
(focus-plane slopes in radians). The latter is NOT physical front-element tilt.
No compound prescription, vignetting, glass depth reconstruction, or physical
tripod mesh is invented. Existing opaque-depth Phong blur remains approximate.

Shift changes the raster projection rather than panning the camera. Tilt changes
its per-pixel focal plane. Both values accompany the photo camera. Free pointing
retains the camera eye while drag changes aim. Floor-relative height updates the
same pose used by raster and photo. Frame & focus restores orbit. Particle chase
continues to own its own camera. A held photographic camera remains held.

Verification: node tests/meowdoku_camera_movements_check.mjs,
meowdoku_marble_studio_dof_check.mjs, meowdoku_marble_photo_camera_check.mjs,
meowdoku_camera_binding_check.mjs, meowdoku_camera_back_check.mjs all pass.
Node syntax checks pass. No GPU compile, render, browser visual proof or live
publication was run. Backend admission/worker wiring is in the companion camera
movements commits. UI values are page-local like the existing orbit controls.
