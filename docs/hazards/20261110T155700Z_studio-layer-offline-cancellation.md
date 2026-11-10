# Offline Stop and reopen lifecycle hardening

The user restricted work to offline preparation while another agent compiles
the real Nodus link. No browser, server, restart or publication action was taken
after that instruction. The earlier browser proof and screenshots remain the
evidence for the initial functional roundtrip, not for these later fixes.

Controller reproduction: start a deferred capture, Stop, and connect a new host.
The old code kept busy=true, preventing the new capture. An old rejection and
finally also had unconditional access to the shared failure/busy state. The new
generation now owns those flags. Stop releases its own busy state immediately;
an old catch/finally cannot poison or clear a later capture. A generation check
after scene packing prevents further stale work. The regression failed at the
busy assertion before editing and passes afterward. No new clock or simulation.

The companion host fix is documented in
20261110T155455Z_studio-host-async-disposal.md. It disposes late installer handles
before assignment and rejects late successful acknowledgments after Stop,
while preserving original rejected errors.

Offline checks: 13 controller lifecycle groups, 9 host lifecycle tests, syntax
checks for the complete staged game and Studio bootstrap, and 19 targeted
pytest cases passed. Publisher tests include a stopped server causing zero
live writes. These are focused checks, not full repository or browser coverage.

New candidate: build/dot-studio-sync-offline-20261110, 23 files, all manifest
hashes verified. It is separate from the currently served QA stage and has not
been browser-tested or published. Controller SHA256:
d14b3ede3b35481f8614a9dd0cf61747ff481753e2a06bb2e634aa5a721ffa43.
Host SHA256:
131581d58876d50320e8feda1ff195f9c1fdcc31b80909d55ff721bb046d78d4.
The complete original newer game remains byte-identical at
2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4.

Do not run the previous publisher default against the older QA stage. Once
live testing is allowed, rebase the fixed source on the then-current published
Studio, repeat bounded activation/Stop/reopen browser checks, and only then
publish through the guarded additive flow. No automatic restart is permitted.
