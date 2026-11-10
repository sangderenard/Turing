# Scene transfer is independent of sphere ammunition admission

Offline-only source/tests on Windows DESKTOP-LCKQH36. No browser, server,
restart, live publication, native compile, commit or push occurred.

marble_game_asset.js exports prepareStudioAmmunition(packet,options). It returns
{supported:true,asset,reason:null}, or {supported:false,asset:null,reason,
code:'UNSUPPORTED_STUDIO_AMMUNITION'} for the existing sphere-body limits only.
createStudioGameAsset stays strict. Shader/material/geometry validation and
float32 attribute overflow are checked before optional sphere admission, so
malformed non-marble captures remain errors. Dice/OBJ/bunny metadata, coatings,
bands and non-unit exteriors do not silently acquire false spherical physics.

mechanical_creature_studio_return.js now admits zero to eight appearance assets.
An empty set still validates original environment receipts, clock, worker/frame
ordering and camera, and rejects bodies naming unregistered appearances. Its
empty draw performs no GPU allocation. The host/controller/collection split is
owned by the parent and host worker, not changed in this slice.

The full scene validator now admits backdrop-holder/ parts emitted by the real
capture. Current rebuildStandMeshes merges these with light stands, so the
transfer uses that same ordinary/transparent stand group, not direct emission.
All original part/body/triangle identity metadata remains preserved.

Revision audit: main input/change/lens handlers and current lighting cache
invalidation hooks cover room, table, light, backdrop and material edits.
Scene material edits invalidate the lighting reference; geometry replacement
also changes the displayed revision. Actual incoming game body store updates
leave those authored references and revision unchanged. Continuous Studio
preview pose is not an authored-revision clock and remains fixed-capture scope.
Checker-off has no captured room bounds for native building import; transparent
room/fixture materials still have explicit unconditional-opaque admission limits.

Passed focused existing game-asset27, scene18, room lifecycle and actual original
return lifecycle/bridge-publisher suites. New checks cover seven optional-ammo
reasons, five malformed non-marble captures, source backdrop-holder pass ordering,
empty return state/renderer and actual revision/invalidation hooks. Non-spherical
scene tests use bounded small geometry fixtures with dice/bunny/OBJ metadata,
not a live full-model browser import. No visual or native physics claim.
New tests: tests/meowdoku_studio_empty_return_check.mjs and
 tests/meowdoku_studio_authored_revision_check.mjs.
