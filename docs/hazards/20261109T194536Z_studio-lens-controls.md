# Studio analytical lens controls, 2026-11-09

Source added on canonical Windows Turing f998b80f with pre-existing unrelated
dirty work preserved. Details and release caveats are in
examples/meowdoku/STUDIO_LENS_CONTROLS_2026-11-09.md.

The current composed shader consumes automatically source-derived authored
profile controls. Six profiles are selectable approximations; thirteen compound
states remain disabled references. Focus/zoom/aperture controls carry explicit
units, source hashes, genuine known bounds and unknown calibration limits.
Custom analytical focus accepts 1 mm to 10 m; the source macro minimum remains
230 mm. This is not a physical macro promise. The spherical analytical focus
bug was corrected to axial planar focus and job identity was versioned.

97 backend tests and nine Node suites pass. Spectral's combined focused gate
passes 231 tests. Both real RTX 3060 shaders compile and pass 540 camera samples,
maximum close-focus plane residual 1.08e-8 m. Isolated mobile Chrome QA passes,
including frozen lens settings during async object-snapshot hashing, source
prime limits, invalid edits and no horizontal overflow. Its actual 72,960-face
payload decodes successfully without launching a GPU photo. Independent review
found and fixed missing fresh-export dependency, configured-source mismatch,
invalid live edits canceling workers, and huge-integer rejection.

NOT PUBLISHED: five static assets are staged, retaining unrelated published
bootstrap state. Parent paused publication/restart pending user connectivity
confirmation. The recovery restart loaded the initial backend source; later
validation fixes require an idle final refresh. No full-scene image-quality,
color, radiometry or full-field compound-equivalence signoff is claimed.
