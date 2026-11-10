2026-11-09 LIVE CHECKPOINT REAL GPU GATE (PASSED, SOURCE NOT PUBLISHED)

The pending OpenGL gate in 20261109T145800Z_live-history-checkpoint-controls.md
is now complete. Immediately before launch, exact CIM enumeration found zero
camera_designer.marble_preview_worker processes, and the newest user job,
c6ca15008155ab40b6a639aef6673299ffdb107aa549e4c051eea33b6d7d5ac6,
was complete. No user exposure was canceled or interrupted. The isolated test
created no scene and did not call the production server.

Spectral tests/marble_history_gpu_check.py exited 0 on NVIDIA GeForce RTX 3060.
It ran the real compute shader against CPU expectations for finite ring means,
checkpoint-plus-ring all-sample means, eviction counts, 1e20 bright inputs,
pixel/channel independence, camera/scene reset validity, and capacity changes.
All comparisons passed, including exact float32 output equality. The current
full production sensor shader also linked on the actual driver.

The production allocation helper then allocated and resized real buffers:
- 1024 samples, checkpoint on: 3,170,304 bytes.
- 4096 samples, checkpoint on: 12,607,488 bytes.
- 16 samples, checkpoint off: 55,296 bytes.
It verified allocated sizes and released all buffers. The driver's reported
SSBO per-buffer limit was 2,147,483,647 bytes. Near-2 GiB allocations were not
stress-tested; software/device bounds, available-memory checks where supported,
and allocation-failure cleanup remain the safeguards for those requests.

Machine-readable receipt is in the Spectral sibling repository at
build/dot-live-history-20261109/gpu-check.json. GPU ownership was handed back to
the geometry kernel coordinator after the test process exited. There was no
production export, restart, commit, or publication in this task. The geometry
policy worker owns the additional same-session replacement/HTTP400 recovery
change and its separate lifecycle regression checks.
