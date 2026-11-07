# Marble Madness first playable course

The `marble_madness` catalog entry runs the existing `fountain3d` DEM/SDF law with the `sdf_toy` track. The 4k product is unchanged: id 0 emits normally, and `emit_rate = 1e9` leaves the other ids parked for the duration of a game. Arrow keys, WASD, and touch controls change the law's recorded runtime gravity input. The page reads id 0 from the normal packed frame and draws its layered cat's-eye mesh at that measured position and physical radius. The server checks the goal against the same analytic SDF chamber and records elapsed simulation time.

## Integration dependency

This branch has **no compiled product**. The existing scratch 4k WebGPU SDF product and its `fountain3d.manifest.json` are in `turing-claude-integration-20261107`, not canonical `main` as of 2026-11-07. Integrate or regenerate that product through the repository's sanctioned build path before exposing this game as live. The current product is WebGPU only; a non-WebGPU client cannot play this lane until a corresponding Wasm product is built. No compiler build was run for this branch.

The server reports a win from a client-submitted displayed position and validates that coordinate against the SDF goal. It does not replay the recorded gravity history on the server, so this is not an authoritative score. The packed frame has position and radius but not spin; the cat's-eye inclusion is position-correct and visibly lit but does not rotate with the physical angular state yet. A later product/presentation ABI can expose orientation without adding another physics model.

## Checks

- `C:\dev\Powershell\turing\.venv\Scripts\python.exe -m pytest -q tests/test_meowdoku_marble_madness.py`
- `node tests/meowdoku_marble_mesh_check.mjs`
- `node --check examples/meowdoku/web/app.js`

Browser play remains unverified until the SDF product and the material catalogue are together in a served build.
