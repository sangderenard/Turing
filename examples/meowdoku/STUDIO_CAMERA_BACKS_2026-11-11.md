# Physical backs for the analytical Studio camera

The Camera back geometry selector provides Micro Four Thirds (17.3 × 13 mm),
APS-C (23.6 × 15.7 mm), 35 mm full frame (36 × 24 mm), medium format
(44 × 33 mm), nominal 6 × 6 film active area (56 × 56 mm), nominal 4 × 5
film active area (120 × 95 mm), and custom 1–300 mm dimensions per side.
The default is the existing 24 mm reference height, with width following aspect.
No back changes an offered output size, sampling budget, oversampling or tiling.

## Geometry and existing renderer

A selected back is center-cropped to the requested image aspect without
stretching. For aspect a and back width W, height H, the active gate is
h = min(H, W/a), w = a*h. The existing shader uses vertical FOV
2*atan(h/(2*f)) and horizontal ray slope a*tan(FOV/2), exactly w/(2*f).
Still aspect comes from complete output width/height, never a temporary tile.
Live retains its existing displayed viewport aspect contract; its rounded
integer raster is a sampling lattice for that viewport. The Phong preview
uses its own current viewport aspect through the same controls.

The existing reference focal-length and axial-focus conventions are retained.
Aperture radius remains f/(2*N). Selecting a back does not change pupil, focus,
or pose. This does not install a compound lens, finite-conjugate sensor
extension, distortion, vignetting, coverage/image-circle calibration, or a
physical close-focus guarantee. A 100-degree field bound still applies.
Larger back geometry is not itself an image-quality or resolution guarantee.

`camera_back` carries the schema `studio-camera-back-v1`, a preset key or
custom key, width_m, height_m, and fit=center_crop. Preset dimensions are
validated, not trusted from the client. `sensor_gate` is server-derived;
client-supplied gate metadata is ignored. `back_reference_fov_degrees` retains
the original 24 mm reference framing when no explicit lens is selected, so
repeated applications, Live revisions and clearing the back cannot compound
the crop. Existing v1 lens provenance remains unchanged.

The still MRP1 decoder and Live camera updates both use the same Python
validator. The existing Spectral camera path consumes the resolved FOV;
no alternate camera law or raster allocation is introduced. Invalid edits
preserve the previous Live control file and do not mutate an active exposure.

## Verification

Windows: 155 focused Python tests pass across back geometry, lens contracts,
Live controls and photo transport. Node back geometry/control/wire, lens,
photo camera and camera binding checks pass. The legacy photo room Node check
fails with `shopEmissionProfiles is not defined` both here and in a fresh
untouched 61a32c63 worktree; it predates this change. No full repository suite,
new GPU exposure, public deployment or server restart is claimed.

Real Chrome UI verification used the actual lens panel HTML/CSS and control
module in an isolated loopback fixture, not the live site. All six presets,
custom empty/valid dimensions, reference reset, and collapsing/reopening the
panel worked. The 2:1 camera output reported the correct physical crop and the
resolution remained 512 × 512. Visual inspection found a binary-float label
(23.599999999999998 mm); display formatting is now rounded without changing
physical arithmetic. Screenshot: build/camera-back-controls.png. The full
integrated WebGL scene and a live GPU exposure were not exercised by this UI
fixture. The temporary server and browser tab were closed.
