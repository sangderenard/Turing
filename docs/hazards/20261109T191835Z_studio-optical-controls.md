2026-11-09 STUDIO OPTICAL AUTHORING AND INVALID-IMAGE UI

Implemented opt-in Legacy (unchanged default), glass_rod and injection_dye
material controls. After transport reviewer clearance, explicit optical modes
are enabled as EXPERIMENTAL PHOTO-FIRST only. Browser live start rejects them
before any save or ray HTTP request. The parent owns backend admission/restart.
No restart, exposure, saved-user-state write, or port8767 action by this worker.

Each used material has internal transmittance450/550/650nm, thicknessmm,
concentration and host/rod IOR. Palette seeds are explicitly ILLUSTRATIVE
UNCALIBRATED, never measured spectra. Injection uses host IOR. The opted-in
snapshot copy changes PBR transmission to1 and opacity to0; legacy alpha is
not pigment absorption. Original editable PBR rows are retained in the authored
profile and restored on load or re-preparation. Default legacy captures do not
acquire optical metadata. Spectral edits participate in autosave identity.

prepareObjectSnapshot writes exact generated region_media and optical_media.
verifySavedMarbleObject checks their authoring recipe and the opted-in PBR rows.
Saved original rows/profile restore through actual Studio loadObject. Geometry,
LOD hashes, mass receipts and immutable original input remain checked. The
parent's optical-only bubble coexistence branch is covered with8cavities and
axial rods/dye; legacy overlap still rejects. Shared-volume mass/inertia are
explicitly approximate (massExact=false), not certified physical dynamics.

Separate prominent red INVALID IMAGE warnings consume
status.rod_bundle_diagnostic.unresolved_samples and warning in photo/live UI.
They remain across missing status, repeated image, errors, and stop. An explicit
zero diagnostic clears the warning only after a different valid image/revision
actually displays. Photo replacement decode is covered, not only status JSON.

Passed focused checks:
- New optical panel/save/load/re-save/tamper tests, both modes + bubble coexistence.
- Actual production loadObject test: legacy/current and optical profiles,
  original PBR restoration, fixture isolation, no write request.
- Live lifecycle/photo-first rejection and persistent invalid-image diagnostics.
- Photo retouch lifecycle42checks, including delayed valid replacement decode.
- Existing optical authoring, snapshot, axial defaults and bubbles checks.
- Autosave20/20; Python objects/bubbles/regions/optical suite31passed.
- JS syntax, source Python compile, scoped tracked diff whitespace checks.
- Isolated Chrome actual modular UI at1200px and430px: all18input fields,
  all3select options, repeated modes, no initial dirty event, visible warning,
  no horizontal overflow or JS exceptions. Screenshot inspected. This was an
  isolated no-write fixture, not a production WebGL exposure.

Known baseline: tests/meowdoku_marble_object_check.mjs fails its old d6
'distant proxy retains nested triangle surfaces' count assertion: d6 proxy
currently uses low detail. Identical failure reproduced with untouched
prepublication object/mesh/bubble modules. No attempt to alter unrelated LOD.
An initial materials harness failure lacked structuredClone/new optical helper
in its VM; corrected dependencies and added optical load coverage; now passes.
These are focused checks, not a full repository test result.

Publication authorized by parent after reviewer photo-first clearance:
11static assets published, every disk SHA256 and verified-CA HTTPS GET matches.
Main Studio JS/HTML were narrow-patched against published copies, preserving
unpublished chase/physics omissions and existing lighting. No full main copy.
Source exporter and server no-store lists include3new modules; parent restarts.
Receipt and backups: build/dot-optical-controls-20261109/release/manifest.json,
release/published-before/, published-get.json, result.json, controls*.png.
