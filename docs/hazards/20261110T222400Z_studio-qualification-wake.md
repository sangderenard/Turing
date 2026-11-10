2026-11-10 STUDIO EMBEDDED QUALIFICATION WAKE (MEASURED, Windows)

The reviewed oriented original game is build/dot-oriented-arena-candidate-20261110/index.html,
SHA256 297cda67f0c96cc2dada8278fcd1f559b279fd14ba3c8d31b62448e1f1fa82a1.
Standalone qualification passed, but the Studio iframe stayed at 0.47 simulated
seconds while both documents reported visible. No native error was displayed.

The exact original worker can reach 180 low-speed physics ticks while fewer
than 30 snapshots have been recycled. It then sleeps for
integration-only-quiescence before the original qualifier's next neutral
vehicle-control wake. With eight-tick snapshot recycling, the actual worker
and its matched Wasm modules stopped at tick 229, snapshot 28. With immediate
recycling they completed 500 ticks and 500 snapshots. No vehicle law was
replaced in this regression. GPU parity is not covered by this Node test.

The optional reversible entry adapter now notifies a helper from the existing
engine-state handler. Only an active, incomplete original qualification with
its original active vehicle and the measured sleep reason resends the existing
zero-throttle, zero-steering, zero-brake vehicle-control message. There is no
new timer, worker, simulation, gate decision, or shorter qualification period.
The same delayed recycling now completes 500 ticks, 62 snapshots and two
native wake events. Tests reject wakes for completed/failed gates, unrelated
sleep reasons, absent qualification and dismounted vehicles.

Final stage: build/dot-studio-oriented-awake-v2-20261110. Prior frozen stages
were not overwritten. The 24 manifest hashes verify; original raw model,
worker and kernels remain byte-identical (model SHA256
1359fa8e524ff03dd12c4e4c6c28460f44597165a06b465096de63c856851920).
Stage receipt SHA256:
1372e2771951fee7ff4532d00f45faafb047fb38d0746ed9e6d5ab47d1da5bc1.

Actual Chrome iframe QA passed the original 20-second gate, with no failures.
Native player spawned at [32, 0.2875, -9.69268329297247], walked with W to
[32, 0.2875, -10.123543818389816], and fired a Studio marble (12 to 11).
The visible synchronization receipt returned original worker projectile ball-1
at sequence 5453. Return hid the iframe and preserved all captured photo/camera
inputs; reopen retained the exact moved pose and inventory without a new gate
or respawn. browser-qa-receipt.json and browser-return-proof.png are beside the
stage receipt. No live deployment was performed by this worker.

Focused checks: test_studio_game_entry.py and test_studio_current_game_adapters.py
10 passed; layer patch, negotiation and stage review suites 11 passed; native
host suite 23 passed. Final oriented native entry check and generated JavaScript
syntax check passed. A Windows helper newline mismatch found during testing
was corrected by preserving raw helper bytes; tests now verify exact inclusion.
The test extractor now includes the oriented game's original support helpers.

Limits: full driving course, generic oriented player/projectile contacts, and
GPU/Wasm numerical parity remain unverified. The known browser adblock extension
checksum error is separate from native qualification. The full repository suite
was not run. hazards_log.py remains blocked by the pre-existing malformed
20261109T_material-library-paused-checkpoint.md name; latest entries and the
legacy baseline were read directly instead.
