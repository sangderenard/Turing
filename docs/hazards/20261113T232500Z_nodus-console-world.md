# 2026-11-13 NODUS CONSOLE WORLD FIRST SLICE

Base 763e8d27. New adapter consumes actual GRAPH schema, projects stable IDs,
selected subgraphs, row/Panel metadata, edges and existing paired doors.
Seven focused unittest checks pass; py_compile and git diff --check pass.

Full project_graph HTML generation fails in existing navigation WASM compiler:
`navigation_astar.c:1:1: error: Unexpected`. Reproduced by calling existing
compile_astar_navigation_kernel directly in untouched C:/dev/Powershell/turing.
No compiler/navigation fallback added. No GPU/visual run, deployment or restart.
An isolated native Nodus scalar run did not yield expected 8 after one step;
no native code changed and that execution result is not claimed as verified.

This is a partial read-only conversion, not completed native console controls,
telemetry subscriptions, relaxed wires, dial/treadmill runtime, or service UI.
See docs/NODUS_CONSOLE_WORLD.md for integration boundaries.
