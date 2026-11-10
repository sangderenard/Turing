# Studio camera targeting and pose ownership

## Live slice

The visual user orbit follows the actual displayed subject centre, including
current physical size/table offset and existing rolling X/Z displacement.
Framing uses the Studio transform instead of the Exposure profile transform.
The photographed subject has its own saved/current Exposure centre. Shared
yaw, pitch and logical zoom do not silently replace Exposure scale or FOV.

The controls appended near Exposure permit:
- Following the user orbit's angle and zoom.
- Copying that orbit to the photographic frame and holding its world-metre pose.
- Framing the photographic subject without moving the user view.

Held camera data is detached on input and output. Existing lens controls
remain authoritative. The live overlay bypasses held photo state. These are
page-local controls, like the pre-existing camera controls; reload relinks.
The built-in close-up option remains unchanged. Current Studio is selected
when the new controls are used.

## Target audit

- Default Studio orbit: corrected stale zero target.
- Frame marble: corrected Exposure/current Studio mix-up.
- Saved photographic orbit: corrected stale target, retained saved-radius independence.
- Automatic snapshot: uses current Exposure radius and centre.
- Built-in close-up: already centred on object_center_m; unchanged.
- Live overlay: explicitly follows linked orbit, not held photo camera.
- Calibration: follows the selected photographic pose.
- Particle camera: existing follow uses physical particle position; this release
  does not introduce the unpublished smoothed chase implementation.

## Existing walking code

`living_map_walk_controls.js` exposes 17 unchanged functions extracted from
`src/compiler/abstract_ui_div_map.py`, plus its exact state initializer and
authored bindings/policy. Its tests compare function bodies to source and
exercise walking/car input, collision clipping/support, jump/worker messages
and attach/blur/hidden/detach behavior. Python declaration parity and existing
viewport/control-record tests passed (12 tests).

This is not enabled in Studio. It requires actual world colliders, surface
support, physics records, snapshot registration, compiled stepping and action
handlers. There is no incumbent collision-free camera branch. Empty callback
substitutes would misrepresent the original controls. Server `control` assigns
pose; `wrench` is the independent persistent force/moment seam. Neither is a
verified reciprocal Studio player body yet.

## Verification and release

The exact staged main passed scale, table, full gobo/profile isolation and
macro-handler tests. New binding tests cover actual centres across supported
sizes/units, original camera factory equality, held pose immutability,
invalid input, real UI handler behavior and disposal. Real Chrome QA used a
loopback read-only server rejecting every POST. It rendered centred 10mm and
1mm subjects and exercised hold/copy/frame and reload without production
saves or exposures.

Publication receipt: `build/dot-camera-binding-20261110/publication.json`.
Three files, all TLS hashes verified, 268 unrelated assets unchanged, no
backend restart. Native housing/mount transport and the broader physical
player remain separate unfinished integration work.
