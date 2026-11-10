2026-11-10 ORIGINAL PLAYER OUTSIDE STUDIO ENTRY (MEASURED, offline Windows)

Latest user direction is walking nearby Marble Studio, with the imported
Springtail kept as a test case. The optional layer previously acknowledged a
visible Studio environment without entering it; the new focused host test
failed with zero entry calls before the change. Initial entry now follows the
first validated visible receipt only. Revisions and Return/reopen never reset
the player or recreate the iframe. The live original generated game is untouched.

The reversible entry patch adds an optional destination argument to the exact
original respawnViewportActor and a default resolver after Studio admission.
The existing native routine removes/re-registers the player and clears velocity.
Native clearActiveVehicle keeps the parked solver body/machine state. Native
navigationOpeningPoint, cameraGroundHeight and sampleContactSurface derive the
exterior doorway apron and support. Capsule/door dimensions, finite coordinates,
world boundary and existing collider overlap are checked before any mutation.
The immutable entry receipt records the original actor and support identities.
On-foot R resolves the current room again; mounted R uses the original vehicle
branch. The original authored pose remains unchanged. No new controller, timer,
physics, transport or camera is created. No tire adjustments were made.

Return uses original control release, primary/secondary release and blur;
reopen uses original requestViewportControls. Held keys/pointer lock are
released, and the continuing game remains in the same iframe. Photo camera
state is not written. The player status DOM exposes current native position,
entry location and controls, refreshed through the existing snapshot sink.

Passed: 15 focused pytest cases in 2.45 seconds (entry, layer patch and prior
negotiation aggregate), 21 host lifecycle cases, generated game/Studio syntax,
and actual native-entry extraction against the frozen new stage. Entry tests
use the exact native dismount/reset/registration/control helpers and real native
Studio/apron colliders with a controlled worker-message sink, not a physics
solver run. Five malformed/obstructed destinations cause no control/body writes.
R revalidates an edited doorway and refuses an invalid one before mutation.
The view test covers Return during initial load, repeated reopening, one
connection/iframe and original focus-release calls. Full repository tests,
native solver execution and browser/GPU QA were not run by this worker.

Frozen candidate: build/dot-studio-entry-20261110, 24 manifest files verified.
Receipt SHA256: 0180a5c8c7f5b2842a09dd743bf18f6b867af94dccae1a0397ec9ac72f34f646.
Fixture position: [32, 0.2875, -9.69268329297247], yaw -pi/2.
Stage now records original_game_path as well as original_game_sha256 for the
publisher's guard. Existing dot-studio-pose stage and QA server were untouched.
No live publication, server restart, commit or push. Parent owns browser QA
and any later guarded upgrade. The hazards reader refused the existing malformed
20261109T_material-library-paused-checkpoint.md name; newest receipts and legacy
baseline were read directly before tests.

Frozen source SHA256:
- mechanical_creature_studio_entry.js: 3ca1054cadb086f731cd94217eb52806d7c0a343c7fdaa74df673db6037f760e
- mechanical_creature_studio_layer_host.js: 9d634c70a8b167625f27d99c2edf80668637f9dabbd95a9ca4fd0735c8b67aed
- marble_studio_game_layer.js: 3a18bd6750353a6dc17707cab5a8a2528dbbd51c73e9100714b2ab4005b9a073
- add_studio_game_entry.py: f2adba0497b4a4d78111724bd43ebc86564e9d32d02f0dcf3527c188b20d135f
- stage_studio_game_layer.py: a65f758d0bbdcaf73a572c35b5a368545aa78fd7905093e54da74263eae698be
