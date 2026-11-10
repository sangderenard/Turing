# Studio manager renderer replacement at the original game boundary

Measured on Windows DESKTOP-LCKQH36, C:/dev/Powershell/turing. No live writes.
mechanical_creature_studio_room.js keeps installStudioRoom compatible and adds
room.ready plus room.update({asset,building,sequence,sourceRevision,beforeCommit}).
The initial ready and update promises resolve only after original opaque then
transparent callbacks complete. Receipts preserve source identity/revision,
manager sequence, original game snapshot sequence and the native environment
receipt. No independent animation, physics, polling or timer loop is added.

Hydration runs outside rendering, with one in flight and only the latest queued
complete batch. GPU allocation occurs inside the original identities callback.
beforeCommit receives {identity,sequence,sourceRevision,building,isCurrent,commit}.
The native host checks isCurrent immediately before changing geometry, applies
its existing environment/rebuild/collider publication at its normal pre-controls
boundary, then calls commit(receipt) synchronously. commit validates asset,
revision and sequence. Old rendering resources survive until both new visible
passes finish. Missing/mismatched proxy identity tags never hide native geometry.

The original collection adapter is retained and its identity Set is not mutated.
Sequence must strictly increase; authored sourceRevision can stay equal but
cannot decrease within one owner session. A new manager epoch requires a new
owner/page. Superseded/closed candidates cannot commit. Source class cannot
change. New content-hashed assets may replace old assets on the same native
building/proxy identities. GPU draw failure is not acknowledged and allows the
incumbent hook's native-proxy fallback.

Same-content updates reverify the serialized identity and reuse decoded
geometry; content identity alone does not bypass verification. Hydration now
detaches serialized input before its first async hash, preventing caller mutation
between validation and decode. Retired requests release decoded buffers/metadata.

Passed focused commands:
- node --experimental-vm-modules tests/meowdoku_mechanical_creature_studio_room_check.mjs
- node --experimental-vm-modules tests/meowdoku_mechanical_creature_studio_collection_check.mjs
- node tests/meowdoku_studio_scene_asset_check.mjs (18 rejection cases plus mutation race)
- node tests/meowdoku_studio_subject_renderer_check.mjs (21 rejection cases)
- node tests/meowdoku_marble_game_asset_check.mjs (27 rejection cases)
- node --check examples/meowdoku/web/mechanical_creature_studio_room.js

The new room tests explicitly double hydration/GPU/camera boundaries and cover
actual unchanged wrapper code: initial visible acknowledgment, delayed native
commit, latest-only hydration, stale revisions, canceled commits, wrong native
receipts, stage/draw failures, mutable identity cache hits, observer failures,
collection preservation and teardown. They are not full-page browser or real
worker execution tests. Combined integration belongs to the parent/host worker.
A colliderPublication value posted-to-original-worker means posted, not that a
worker executed or acknowledged the new collider state. No full repository run.
