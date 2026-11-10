# Original game layer negotiation and ownership

Offline source/test continuation only. No browser, server, production request,
publication, restart, native compilation, commit or push was performed. The
older screenshot proof applies to the earlier candidate, not these changes.

The controller previously accepted a fulfilled promise with the wrong scene
sequence. A regression reproduced that admission before editing. The shared
marble_studio_game_protocol.js now binds session epoch, source class, asset,
authored revision, request sequence, registered placement and completed two-pass
visibility receipt. It preserves the distinction between a collider message
posted to the original worker and actual execution. The host and controller use
the same validator. A negotiated epoch also allows a valid visible import to
start before the worker's first snapshot, without inventing body state.

The controller now discards obsolete captures before queueing them. Once the
game may have committed a request, its exact receipt is installed before the
next revision is sent. Failed or canceled generations cannot poison later
captures, and old connection callbacks cannot feed a replacement receiver.
Twenty-one controller checks cover these boundaries. Optional ammunition is a
separate capability: a complete scene can import without a compatible sphere
asset, while malformed geometry/material/shader captures still fail. The UI
labels a noncompatible subject as display-only instead of giving it invented
sphere physics. Later environment updates do not mint inventory.

Companion evidence:
- 20261110T173045Z_studio-host-negotiation.md: host lifetime and receipts.
- 20261110T173330Z_studio-original-worker-barrier.md: the actual original
  snapshot/recycle FIFO proves CPU collider-field consumption by N+2. It is not
  exact solver-tick or GPU completion evidence; workerAppliedBarrier stays false.
- 20261110T173415Z_studio-optional-ammunition.md: optional assets, exact
  backdrop-holder pass ownership, and authored revision checks.
- 20261110T173755Z_studio-registered-part-activation.md: native table removal and
  restoration, stable identities, contacts and rollback.

New structural identity admission uses an append-only method in the existing
canonical TuringWorldRegistry, not another registry or reconstructed world.
The complete bridge path passed against the corrected canonical-method host. Its complete
batch is cloned and validated before admission; existing object/part references,
runtime IDs, maps and arrays remain. The canonical AST-only Node test covers
21 atomic rejection cases and monotonic IDs. Independent review reproduced
two additional boundary bugs before correction: implicit root identity aliasing
and capacity equality at the JavaScript array limit. Both now reject before
mutation; the corrected file was rechecked against the exact owned-block hash.
New IDs remain registered but inactive if a later renderer/mesh activation fails; IDs are never recycled.
The exact startup Studio object constructor is shared with runtime admission.

The generated-host adapter reads only the authored Python literal using AST;
it does not import src.compiler and load the native DLL. It copies the exact
canonical method, verifies all other original class bytes, updates the existing
inlined utility content key, and proves reversible restoration of the original
page. Six patcher tests cover both preserved original versions. The newer
2cd5 game remains the canonical host and its source file is unchanged.

The user warned that identity and web exposure safety were changing. The
canonical registry file was frozen and reread after its narrow approved edit.
No auth, proxy-origin, route-cookie, operator credential, network or server
lifecycle setting was changed. Existing deployment assumptions must be reviewed
again against the current operator/proxy contract before any later publication.
Do not run the older default publisher or stage into an actively served QA path.

Remaining product boundaries: native dynamic light bodies/toppling, ray-job
insertion of returned bodies, a solver/GPU-applied barrier, continuous Studio
preview-pose streaming, transparent room/fixture admission and checker-off
(no captured native room bounds) are not established by this slice. Current
ammunition is the initial captured compatible marble, not a full saved-library
selection UI. Fresh browser/GPU regression and publication remain held by the
user's offline-only instruction. Targeted offline tests are not a full-repo pass.

Final focused aggregate at this checkpoint: 31 pytest cases passed in 6.89s;
original collection/room, asset 27, scene 18 and actual return-lifecycle tests also
passed. The corrected complete worker/native-mesh/launcher bridge test passed
against registry-game.html SHA256 6d3016268e4069e9411f4c23819c716330e659fb731a65e6488cda5b21a1052b.
The 24-file stage at build/dot-studio-negotiation-20261110 has verified hashes
and passing full game/Studio syntax checks. It remains offline and unpublished.

Final independent bridge review found the analogous document-array capacity
edge. A bounded sparse-array reproduction showed partial canonical admission
before document retention failed. The importer now rejects equality/overflow
before constructor/append. Both regressions leave every ownership table and
array unchanged. The final stage was regenerated after that guard correction.
