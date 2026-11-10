# Additive chosen-marble ammunition, original playable game

`tools/add_mechanical_creature_appearance.py` hash-guards the published/restored
playable page SHA256 `aa6cb6fb897d24e6bf769ca2dbfde3a0d3d0dbcdd64ddb87fd30173bc0f0d5e2`.
It writes a separate candidate, never the source, published game, or restored
backup. No compiler imports, full compilation, model serialization, Wasm relink,
server restart or publication occur. Canonical `abstract_ui_div_map.py` is not
modified. Eight explicit behavioral seams are reversible to the exact original
script bytes; the entire embedded model, worker and compiled binaries remain
byte-identical.

The added classic-script helpers are
`examples/meowdoku/web/mechanical_creature_appearance_hook.js` and
`examples/meowdoku/web/mechanical_creature_asset_ammo.js`. The existing render
cycle reads an optional appearance adapter. Only selected, explicitly tagged
projectile identity spans are suppressed in the visible color draw. Original
camera depth, shadow proxies, geometry, picking and colliders stay present.
The imported renderer must restore every GL state it changes. No reciprocal
lighting or imported triangle collider claim is made.

Chosen collection items are ordinary existing-inventory ammunition stacks.
The compact rendered-image picker changes the original ammo slot 8 and calls
the existing tool selection. Each shot retains its source stack and asset ID.
Original collection/vacuum functions credit that source stack after selection
changes. Refreshing an imported asset does not refill it or select it. Default
physics balls stay untagged. No launch law, radius, worker membership, movement
controller or physics clock changes.

Measured checks on Windows:

- `node tests/mechanical_creature_appearance_hook_check.mjs`: PASS.
- `.venv/Scripts/python.exe -m pytest -q tests/test_mechanical_creature_appearance_hook.py`: 4 passed.
- `node tests/mechanical_creature_asset_ammo_check.mjs build/dot-studio-game-transfer-20261110/game-appearance-candidate-v2.html`: PASS. Executes actual patched launcher/selection/expiry/pickup/vacuum function bodies against separate stacks.
- `node --check` for both helpers and the extracted full candidate behavior: PASS.
- Published game and preserved restored page hashes remain unchanged.

Candidate v2 SHA256:
`bd346aee51eff087674b343079858595efc6357d0fe352f1be0b5f740b0cfc82`.
The candidate contains the hook, not the actual asset loader/renderer. Parent
integration owns that separate source and combined browser/GPU validation.
Hardware phone behavior and the combined game/asset render have not been
verified by these source/CPU checks. Full repository suite was not run.
