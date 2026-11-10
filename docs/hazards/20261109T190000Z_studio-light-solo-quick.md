2026-11-09 LIGHT SOLO AND QUICK LIGHTING SLOTS (canonical Windows source)

Per-light Solo controls cover all ten STUDIO_LIGHTS. Solo is an exclusive,
reversible emission mask in the existing getStrengths owner, shared by preview
fixture/PBR emission and photo/live fixture serialization. The original linked
or independent levels are never overwritten. Click the active Solo or All
lights to restore the complete rig. Selecting another Solo switches directly.
Profile state intentionally saves the complete rig, not temporary solo state;
loading a profile/slot clears solo. Calibration clears solo before its declared
single-light case. Fixture geometry remains present when emission is masked.

Four A/B/C/D recall buttons with explicit Store A/B/C/D use the existing full
profile capture, validation and atomic apply path. Same serialization/storage
failure preservation, under a separate browser-local key. Empty slots are
inactive. Storing changes no render. Recall uses normal scene invalidation.
No credentials, server state or user-browser storage were touched in testing.

Passed: Node lighting profiles (including slot corruption/persistence/invalid
capture), real fixture light-strength/PBR/photo wire tests (all ten solo masks,
restore/switch/clear, calibration), complete Studio profile lifecycle and repeated
quick recall; node syntax checks; pytest test_meowdoku_table_lights.py 1 passed.
The focused tests are not a full repository gate.

Actual isolated browser QA was attempted with the repository's Chrome CDP
helper and read-only source QA server. Chrome failed to start within helper
startup timeout; no browser behavior or screenshot was verified. The owned QA
server was stopped in finally; no production process was stopped or restarted.

Not published, committed or pushed. Staged only the three changed modules in
build/dot-light-quick-20261109/release, with published-before/source SHA256s.
No HTML or studio bootstrap change; no unpublished physics/chase release.
Files: marble_light_strengths.js, marble_lighting_profiles.js,
marble_calibration.js and corresponding focused Node tests.

Follow-up browser verification: scratch Chrome launched successfully without
optional WebGPU/Vulkan override flags. The actual source page at mobile width
430x930 passed four-slot store/recall, repeated solo/clear, recall clearing solo,
and persistence after reload. No browser exceptions. Screenshot inspected and
slot help made full-width for a readable 2x2 A/B/C/D layout; rerun passed.
No user browser/tab/storage was used. Updated final profile module staged with
new SHA256. Evidence build/dot-light-quick-20261109/{result.json,controls.png}.

Combined publication follow-up: color owner published the three staged modules.
Independent HTTPS GET using the configured CA verified every light module against
the final source SHA256. Strengths/profiles return no-store; calibration no-cache.
Production scratch Chrome did not trust the server certificate and displayed
Privacy error. No warning bypass attempted. Root explicitly said not to bypass;
production browser UI is unverified. Prior isolated source browser test passed.
No exposure started, saved object changed, or user browser session interrupted.
