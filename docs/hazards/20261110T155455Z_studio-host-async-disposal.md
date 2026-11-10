2026-11-10 Optional game host async disposal (MEASURED, offline Windows)

The actual mechanical_creature_studio_layer_host.js module assigned awaited
collection/room handles even after dispose(), then accepted late successful
ready/update receipts. Four focused cases reproduced those paths against the
unchanged source. Five normal-flow/error-preservation cases already passed.

The host now keeps each returned candidate local, disposes it if the host
closed during its installer, and only then permits assignment. A closed check
after the existing ready/update await refuses late success without replacing
an original rejection. No API, controller, clock, worker or game code changed.

Offline checks: nine Node lifecycle cases pass, host syntax passes, the
existing controller reports 13 checks passed, and the host pytest wrapper
plus Studio patch checks pass 3/3 in 0.60 seconds. Tests execute the actual
host with controlled installer/DOM dependencies; they do not claim GPU,
browser teardown or full-suite coverage. The initial fixture run left one
mock ready promise unsettled and was canceled; after settling that fixture,
the unchanged host completed with the four expected failures.

Host SHA256 before: 934b5e54fdb8fe172dc5f7240a538fce63b48431663c4402b928997e6f578351
Host SHA256 after: 131581d58876d50320e8feda1ff195f9c1fdcc31b80909d55ff721bb046d78d4

The existing stage still has the before hash. Original newer game remains
2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4.
No restaging, publication, server change or browser execution occurred. This
repairs the demonstrated source race, not a claimed cause or cure for the
previous clean-stage browser reload stall.
