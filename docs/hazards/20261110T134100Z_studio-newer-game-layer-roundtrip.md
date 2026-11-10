# Optional newer-game layer: bidirectional source integration

The newer complete MechanicalCreature page (SHA256
2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4)
is the canonical base. The original newer and older pages were not modified.
The previous older additive marble demo remains a separate historical release,
not the chosen base for this integration or a contact-quality certification.

Implemented source:
- marble_studio_game_layer.js mounts an opt-in same-origin game frame from the
  existing Studio engine section. It captures an authored displayed revision,
  transfers complete source geometry/materials/shaders, and consumes copied
  original game bodies on Studio's existing frame. Default is off. Photo camera
  controls and saved photo inputs are not changed.
- mechanical_creature_studio_layer_host.js consumes assets through the existing
  original-game appearance/ammo APIs. The original HUD and input remain intact.
- mechanical_creature_studio_bridge.js updates only registered Studio shapes
  before original control work, invokes the original mesh/collider rebuild,
  then commits a staged renderer. It publishes copied original body/camera
  state after original lifecycle/render work using two retained slots.
- mechanical_creature_studio_room.js stages a replacement renderer in the
  original callback, retains the old one until commit, and acknowledges only
  after both actual draw phases. Cancellation and mutable-input races tested.
- mechanical_creature_studio_return.js maps actual original tagged projectile
  state through inverse registered building placement and Studio unit scale.
  Complete accepted frames replace the previous slot. No interpolation,
  alternate simulation, invented orientation or new physics clock exists.

Ownership and limitations:
Studio owns authored static environment revisions. Existing game worker owns
projectile states. Original viewport frameDtSeconds is explicitly not native
solver dt. Collider receipts say posted-to-original-worker; they are not a
worker-applied barrier. Current admitted contacts are original room/table
shapes (10 identities,15 contacts), not reciprocal dynamic fixture bodies.
Returned bodies use the actual captured Phong appearance. They are not yet
inserted into ray jobs. Photographic pose remains intact. Glass ordering across
returned bodies and existing transparent furniture is approximate. The admitted
marble remains the original game's .11m sphere body, without published spin.
Initial ammo is the captured spherical uncoated Studio marble; later environment
updates do not refill inventory or replace bodies already owned by the game.
No general player spawn setter was invented; original spawn/navigation remain.

Measured checks:
- node tests/meowdoku_studio_game_layer_check.mjs:12 lifecycle checks.
- node tests/meowdoku_mechanical_creature_studio_return_check.mjs: actual original
  lifecycle/publisher, double-slot/environment/epoch gates, mapping and removal.
- node --experimental-vm-modules tests/meowdoku_mechanical_creature_studio_room_check.mjs:
  initial/update two-pass acknowledgment, commit/cancel/failure/disposal coverage.
- node --max-old-space-size=768 tests/mechanical_creature_studio_bridge_check.mjs:
  actual newer registry+mesh+launcher, queued table edit, original collider post,
  actual publisher to Studio return consumer, stale receipt and pickup removal.
  Worker also ran the legacy version for compatibility, not canonical selection.
- pytest tests/test_studio_game_layer_patch.py
  tests/test_mechanical_creature_studio_bridge.py
  tests/test_mechanical_creature_appearance_hook.py
  tests/test_mechanical_creature_studio_building.py:15 passed.
- Final staged newer behavior and Studio bootstrap parse with node --check.
The first aggregate invocation lacked --experimental-vm-modules and used an
incorrect pytest filename; corrected commands above passed. No full-repo pass.

Staging, not publication:
python tools/stage_studio_game_layer.py creates 23 bounded files under
build/dot-studio-sync-20261110. stage-receipt.json pins all hashes and original
reversibility. It never copies all dirty Studio source or changes live files.
The layer bootstrap is a narrow reversible patch over current published Studio.
The staged newer game retains its full embedded model, worker and feature set.
Any tire-worker candidate must be explicitly hash-reviewed and rebased, not
silently accepted by the staging guard.

Browser/visual end-to-end is NOT verified. The prior browser ID was unavailable;
current browser inventory returned an empty list, and native window inventory
showed no browser. No warning bypass, new screenshots, live publication,
production restart or browser-storage reset occurred. The unused owned QA
server was stopped. To resume when a browser is available:
python tools/serve_studio_game_layer.py
Open http://127.0.0.1:8898/marble_studio_layer.html. This is read-only merged QA,
not a proxy to production mutation APIs. Verify activation, current scene draw,
actual launcher motion, return to Studio, authored table/light updates, stale
updates, repeated hide/show, stop, and phone layout before publication.
