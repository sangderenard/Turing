# Marble Madness course integration

The `marble_madness` catalog entry uses the compiled 3-D fountain DEM law in its dedicated `fountain3d_toy` variant. Id 0 emits into the shared `toy_design.toy_parts` SDF course; `emit_rate = 1e9` parks the other 4k ids. The shared design describes the ramps and their return-dribble holes, pegs, pool, hourglass, and free hinged wheel. The toy product compiles the existing body law together with SDF walls. The case-frame vector law, including X/Y/Z motion, remains the current fountain law.

Arrow keys, WASD, and touch controls drive the documented runtime gravity vector. The page reads id 0 from the packed frame, draws its layered cat's-eye mesh at the simulated position and radius, and checks a finish against the analytic SDF chamber. A selected Studio design instead draws its complete verified opaque and transmissive mesh with its saved material tables and applies supported simulation parameters through `fountain_param`. Contact remains spherical, and the packed frame does not expose marble orientation. The server validates a submitted goal coordinate but does not replay the gravity history, so scores are not authoritative.

`prepare_fountain3d_toy` builds WebGPU products only. Set `MEOWDOKU_TOY_SIZES=4k` for the playable course; the default is every wheel-supported size. No eager Wasm build is requested. A non-WebGPU client needs a separately requested fallback product before this course can start. The startup readiness map tracks the dedicated toy product.

Focused checks: `python -m pytest tests/test_meowdoku_marble_madness.py -q`, the SDF geometry and waterwheel tests in `tests/test_meowdoku_fountain3d.py`, and `node --check examples/meowdoku/web/app.js`.
