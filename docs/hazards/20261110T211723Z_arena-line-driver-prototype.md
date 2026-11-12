ARENA LINE DRIVER PROTOTYPE (2026-11-10, local Windows)

Added data-only arena paths plus uploaded graph masks/weights through the existing
meowdoku graph_policy_forward and TuringGraphProgram. Five hidden gates, three
outputs. Reversible HTML adapter samples original declared controls, calls original
updateActiveVehicle/physics worker, and paints through original navigation mesh.
No vehicle pose writes or alternate physics clock. Existing game source, embedded
model and worker are not replaced. Matched base a7b75ff09b7fcd903408623cd33ef91e12b5262df0be7ce4d676190ca71ebaf5
patch seam checks and reverse-byte recovery passed; extracted patched inline JS
passes node --check. No deployment by this task.

Tests: node tests/arena_line_driver_check.mjs = 28 checks passed; targeted Python
patch/toolchain tests = 4 passed. Existing compiled Wasm compared against existing
AbstractTensor eager graph forward, three seeded random 15-feature rows: max
absolute error 1.1102230246251565e-16. No full repository suite run.

Compiler repro: system Python python tools/build_arena_line_policy.py
build/dot-arena-line-policy-20261110. Repo .venv lacks llvmlite. System Python Zig
0.16 rejects emitted pilot.ll with line1 Unexpected, including -c. llvmlite parse
and verify succeed. Installed clang20.1.7 builds the SAME .ll successfully.
Added explicit TURING_WASM_CLANG driver selector only in existing shared
llvm_training_runtime._compile_artifact_to_wasm; default Zig invocation unchanged.
Configured clang successful build and numerical parity above. No handwritten
network evaluator fallback. Build assets under build/dot-arena-line-policy-20261110.

Limits: UI, colored mesh and actual physical driving require browser QA; starter
weights are hand-seeded and uncalibrated, not trained. Terrain oval marks are not
collision-certified roads. Named absent bridge supports reject; rotated supports
reject pending native oriented support contract. Surface edits require stop/reload.
Packet schema and reproduction are in docs/arena_line_driver.md.
