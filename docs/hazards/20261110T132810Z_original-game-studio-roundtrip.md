# Optional Studio exchange around the unchanged original game cycle

`mechanical_creature_studio_bridge.js` adds a bounded transport, not a second
engine. `add_mechanical_creature_studio_bridge.py` verifies either exact original
page hash (newer `2cd5c5e2...ef4`, legacy `aa6cb6fb...d5e2`), or an explicitly
supplied hash for a derived candidate. Three reversible seams consume queued
environment updates before existing controls, publish copied state after the
existing render/camera/lifecycle work, and mark the existing worker-ready
collider post. No worker bytes, compiled laws, player controls, timers or
persistence keys are replaced. Original source pages remain unchanged.

`prepareEnvironment` calls the existing captured-building builder with its
original registered identity, translation and entrance. `queueEnvironment`
accepts one complete, owned shape snapshot and monotonically ordered host
sequence. Source revision is retained, not invented as a physics clock.
Only the same ten registered room/solid identities can change dimensions,
placement, existing opening dimensions and capture identity. Labels, world
associations and runtime IDs remain fixed. One existing
`rebuildPortableSceneMesh` produces the native 15 contacts and existing collider
message. A staged renderer's synchronous commit runs at that frame boundary;
canceled/stale updates are rejected, and failed commits restore original
geometry. This is not a claim of atomic native-worker execution.

Small tagged-projectile state packets carry original viewport dt (explicitly
not native solver dt), original worker sequence, environment receipts, actual
active/sleeping/pickup poses and the original rendered camera. Two retained
slots publish fully copied, frozen packets on the same thread. There is no
SharedArrayBuffer/Atomics claim and no interpolation or synthesized spin.
Environment receipts mean `posted-to-original-worker`, not worker-applied.
If the worker was not ready, the receipt stays pending until the original
worker-ready handler actually posts colliders. Ready state alone never upgrades
that receipt.

The appearance patcher now supports newer `2cd5` without rolling back features:
its original opaque/transparent pass ordering is retained, and turret shots
without inventory remain untagged. Each patch reverses to the exact input.
The original player only exposes authored-spawn respawn, not a general target
camera setter, so the bridge leaves player spawn/control authority untouched.

Measured on DESKTOP-LCKQH36:

- Native bridge test passes against both complete original pages. It executes
  their actual world registry, mesh realization and native rebuild function,
  verifies ten IDs/fifteen contacts, applies table position/dimension changes,
  tests supersession/cancellation/rollback and copied publication.
- The return-adapter test extension executes actual launcher, slot reservation,
  environment edit, receipt consumption, stale receipt rejection and original
  expiry/collection lifecycle on both versions.
- `mechanical_creature_studio_prepare_check.mjs` confirms fresh capture uses the
  existing builder and fixed import placement/entrance.
- Ten targeted pytest checks pass for exact reversibility, unchanged embedded
  model/worker, refusal paths and appearance/bridge composition on both versions.
- Newer combined behavior syntax and actual launcher/pickup/vacuum tests pass.

These are native-source/CPU tests. Parent integration owns combined browser/GPU
and overlay validation. No publication, restart, tire tuning or full repository
suite was performed by this slice.
