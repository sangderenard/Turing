# Source-derived analytical lens controls

## Release state

The source-derived controls and final backend validation are live through the
combined color/lens release. See build/dot-color-lens-release-20261109 for its
publication and detached-server recovery receipts. The macro framing/warning
increment was published at 2026-11-09T21:42:09Z after the material editor release.
Five exact assets were verified over trusted HTTPS, 224 unrelated assets were
unchanged, and boot 775d6760ce04/session remained unchanged. This increment did
not restart the server or interrupt its photograph. Publication used narrow
patches over served bootstrap/HTML, not unrelated source-only physics changes.

## Existing architecture retained

The existing component-built analytical shader remains the renderer. There is
no switch to the full native BDPT engine. `marble_lens_contract.py` imports the
existing cold Spectral contract generator and reads all current authored YAML
profiles; source-byte SHA256 identity is checked when accepting a selection.
`lens_contract_catalog.json` is the generated inventory, not a hand-authored
replacement for the source collection. Reproduce it with:

    python -m camera_software.lens_contracts --include-bench-presets --include-camera-presets --output C:/dev/Powershell/turing/examples/meowdoku/lens_contract_catalog.json

Six authored profiles are selectable as explicitly labeled approximations.
Thirteen compound reference states remain disabled pending their shader adapter
and calibration. A prime's focal-length slider is disabled. Custom analytical
mode exposes adjustable focal length without claiming a mechanical zoom lens.

## Controls and conventions

- Focus is the axial intercept of the mathematical focal plane from the
  entrance-pupil/camera eye, in millimetres in the UI and metres on the wire.
  Custom range 1 mm to 10 m is a numerical/application domain, not physical
  close-focus certification. Authored profile minima are enforced; their
  original focus datum remains unspecified and the UI explains that caveat.
- Focal length controls vertical FOV using an explicit 24 mm reference sensor
  height. Width follows the rendered image aspect. This is a reference format,
  not an inferred physical sensor or finite-conjugate calibration. The renderer
  keeps its 5-100 degree FOV bounds. Scrolling still dollies the camera.
- Aperture uses f-number and derives pupil radius as `f/(2N)`. The existing
  20 mm radius cap remains enforced. Profiles wider than that cap show their
  authored maximum and the stricter application bound. f/64 and 10 m are
  application travel limits, not invented authored limits.
- Distortion, vignetting and four-band lens transmission are retained in the
  source contract but not applied. The fisheye profile is still a rectilinear
  approximation. No full-field, spectral or radiometric equivalence is claimed.
- Original settings remain the default. Controls are page-local. Phong changes
  framing only; pupil/focus apply to photos and live rays. Captured lens settings
  are frozen before asynchronous exposure-snapshot hashing.

Still and live APIs use the configured Spectral checkout and the same selection
validator. The server derives FOV/pupil rather than trusting client conversion.
Changing or clearing a selection replaces its provenance. Incomplete edits
blank stale live pixels but retain the accepted worker and heartbeat; a valid
edit resumes through a control revision, not a scene restart. Live snapshots
publish the active camera settings and focus semantics after boundary acceptance.

## Analytical focus correction

The old normalized-ray construction focused on a sphere. The analytical branch
now intersects the intended axial/tilted plane using its unnormalized pinhole
vector, with the existing tilt signs and horizon guard. CPU entry points and
GLSL use the same 1 mm lower guard. Cache identity includes
`axial-planar-focus-min-1mm-v1`, so old photographs cannot satisfy the new job.
See Spectral `SENSOR_ANALYTICAL_FOCUS_2026-11-09.md` for the derivation.
The separate precomputed manifold bake is unchanged and is not enabled here.

## Verification

- Turing backend: 97 tests pass in 8.07 seconds, covering selection identity,
  configured source roots, units/bounds, malformed huge integers, still/live
  validation, explicit clearing and snapshot regressions.
- Nine Node suites pass, including controls, photo lifecycle, snapshot races,
  geometry/camera scale, live invalid-input retention and recovery.
- Spectral combined focused tests: 231 pass in 14.68 seconds.
- Real RTX 3060: both complete composed shaders compile; 540 GLSL camera
  samples pass. Final GPU setup/validation 0.875 seconds, process 3.969 seconds.
  Sub-20 mm maximum focal-plane residual 1.08e-8 m. All owned jobs exited.
- Isolated real Chrome at 440 x 1000 verifies 19 source entries, source-prime
  constraints, custom 5 mm focus, invalid edits, original-mode recovery and
  click-time lens capture. No horizontal overflow or JS exceptions. Screenshot
  `build/dot-lens-controls-20261109/lens-controls-mobile.png` inspected.
- The browser's real 72,960-triangle exposure-only snapshot successfully decodes
  through the backend with 50 mm, f/4, 5 mm axial focus, exact provenance and
  2 mm object radius. The test intercepts submission: no full-scene GPU photo
  was started, so this does not claim exposure quality or radiometric validation.
- Read-only trusted-CA HTTPS catalog request returned 200, six profiles and
  thirteen references. This verifies the initial API, not final publication.

## Macro focus diagnosis and repair

The user's saved exposure e88068387137be2f92887191b7751b0adcee8ce19b895e37bc6510c22eda902d
has macro_50mm, f/18.94125455, 230 mm selected focus and a camera-target distance
of 23.126041588 mm. The authored minimum is genuine; the close camera position
and profile focus were independent. For the shader's straight target plane this
produces roughly 110 pixels of defocus diameter at 512 pixels before refraction.
The saved photograph visibly has severe blur. This identifies a large input
mismatch, not a complete optical image-quality diagnosis.

- An amber warning compares actual aim-point distance and focus, including a
  badge on the collapsed lens section. It is not measured surface depth and
  does not pretend to autofocus a refracted glass interior.
- Focus camera target (Custom analytical) explicitly changes to the analytical
  mode, clears source-profile provenance, preserves current FOV/pupil/position,
  and sets focus to the existing target. The 230 mm profile limit is never
  silently weakened. Ordinary focusAt still enforces the selected profile.
- Frame & focus uses actual exposure object size, selected FOV and profile
  limits, then selects the Studio photo camera. The separate 21 mm framing floor
  and scale-dependent 1.2-unit dolly floor were removed. The 1 mm mathematical
  guard remains, and framing rejects distances outside the 10 m domain.
- These actions affect the next exposure/live camera, not an existing photograph.
  Centre focus is an initial choice, not a promise that an entire glass object
  is sharp. Refraction, depth variation and finite aperture remain real limits.

Eleven focused Node suites pass, including actual source and staged/published
bootstrap handlers, source minimum enforcement, transactional custom conversion,
unit-invariant framing, coherent photo snapshot scale, and live-edit recovery.
The new real-browser attempt did not finish: Chrome failed to start and the
installed alternative encountered a blocked external telemetry request. That
route was stopped. No fresh full-scene GPU photograph or image-quality signoff
is claimed for this static increment. The earlier real shader-camera validation
and browser validation above apply to their measured scopes.

The existing full compound 120Macro builder was also traced at requested 20,
300 and 500 mm. The 20 mm override is ignored by its >1.05*f gate, leaving about
1 m; close rebuilt 300/500 mm states have actual EFL 182.851/153.867 mm and
on-axis 5.62/3.60 mm RMS sensor spots in the tested pupil rays. Its internal
fixed-back solver reports failure for these close rebuilds. These are bounded
measurements of this configuration, not universal macro limits. Evidence is in
Spectral exposures/dot_macro_prescription_20261109/results.json and log.txt.
They do not justify enabling the physical compound reference in Studio yet.

Receipt: build/dot-macro-framing-20261109/verification.json. Phong depth-of-field
preview and reticle sampling are separate ongoing work and must preserve this
source/profile distinction and the selected camera's physical units.
