# Additive Studio building in the original MechanicalCreature world

The helper `examples/meowdoku/web/mechanical_creature_studio_building.js`
prepares 10 native document-geometry objects from the captured Studio packet:
one ordinary building boundary with a north/front entrance and the nine
explicitly authored, axis-aligned table cuboids. It appends their world objects,
semantic parts, dense runtime identities and a discoverable map building before
the existing TuringWorldRegistry constructor. Existing class interpretations,
player, controls, world objects and worker modules remain unchanged.

Metre translation is `[32, 1.544, -18.00788329297247]`; source feet retain
`unitM=0.3048`. The lot is inside the original north flat apron, east of the
original car/player spawn and north of the elevated skill road. The derived
front entrance is 2 m wide by 2.4 m high, centered at source x=0 and starting
at the source floor. The capture is not edited. The material-rendering worker
owns the corresponding derived visible opening.

Native shell realization yields six wall contact boxes around that doorway
plus nine exact authored solid table boxes: 15 static contacts. Native floor
and ceiling behavior is unchanged; no new ceiling-contact law is added.
Fixtures and stands remain captured appearance geometry, not invented dynamic
bodies or lights. Proxy identity tags admit only these new objects to the
separate color-rendering adapter. Original native depth, shadow and picking
meshes retain the proxies.

`tools/add_mechanical_creature_studio_building.py` adds the startup descriptor
after original JSON parsing, without reserializing the embedded model. It
asserts exact byte reversibility and refuses to overwrite the source or either
existing playable page. No original generator changes, compilation, live
writes, server restarts or publication were performed here.

Passed on Windows:

- `node --check examples/meowdoku/web/mechanical_creature_studio_building.js`.
- `node tests/mechanical_creature_studio_building_check.mjs build/dot-studio-game-transfer-20261110/studio-building-definition-v2.json`.
  Executes the original TuringWorldRegistry, native mesh builder, triangle
  picker and player horizontal-contact functions. Confirms 10 objects,
  15 contacts, nonzero runtime identities, clear lot bounds, doorway passage,
  wall blocking, table picking, map discovery and unchanged source capture,
  original world entries, player and Wasm modules.
- `.venv/Scripts/python.exe -m pytest -q tests/test_mechanical_creature_studio_building.py`:
  3 passed in 0.51 s on the final patcher.

Applied definition is
`build/dot-studio-game-transfer-20261110/studio-building-definition-v2.json`.
The patcher exposes frozen `globalThis.mechanicalCreatureStudioBuilding`
containing `assetIdentity`, rigid `placement`, renderer `entrance` and
`proxyIds`, avoiding separate placement inference by the material adapter.

Combined material/browser validation belongs to the parent integration and
was not performed by this placement slice. The full repository suite was not
run. Brief concurrent desktop resource pressure stalled otherwise read-only
commands; both bounded test processes eventually completed successfully.
