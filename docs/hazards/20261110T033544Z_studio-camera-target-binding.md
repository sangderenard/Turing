2026-11-10 Studio camera target/pose ownership (MEASURED, Meowdoku)

The old Studio orbit target stayed at [0,0,0] after a foot-sized table moved
its displayed subject to approximately Y=-0.967. Frame marble used the
Exposure transform rather than the currently displayed Studio transform.
The photo bridge then combined the same stale target with its own scale.

The shared displayed-centre helper now follows the exact renderer's centre
and rolling X/Z displacement. Frame uses current Studio radius/scale. The
photo bridge retains its own saved/current Exposure centre and scale, so
unlinked Studio-only edits do not mutate protected Exposure camera bytes.
A detached hold/copy/frame binding owns the optional fixed photo pose in
metres; lens controls still apply. The live overlay deliberately bypasses
held photo pose. Invalid current-radius edits cannot break a saved camera.

Source and exact staged-main tests passed for scale, table, gobo profile
isolation, macro handlers, binding validation, immutable held poses and UI
teardown. Chrome read-only local QA exercised rendering, 10mm/1mm framing,
copy/hold, held user framing, independent photo framing and reload. No
production design, profile, save or exposure was modified during QA.

Published only main, new binding module and one explanatory HTML sentence.
Receipt: build/dot-camera-binding-20261110/publication.json. All three TLS
hashes matched; 268 unrelated files were unchanged; no backend restart.
The broader unshipped physics/chase/DOF/rangefinder source was not exported.

Walking is NOT enabled by this release. The unchanged Living Map controller
needs real world/collider/support/compiled-step dependencies absent from
Studio. Position-driven walking is not reciprocal player rigid-body physics.
Camera mount still needs authoritative native housing vertices and transport.
