2026-11-10 Canonical Studio part admission (MEASURED, offline Windows)

The bridge now admits first-time captured cuboids through the canonical
TuringWorldRegistry.appendObjects method and the building module's factored
startup object constructor. It does not define another object schema, allocate
runtime IDs, or replace a registry. The existing world objects and exact capture
box extensions are the retained-part authority. Prospective descriptor copies
normalize asset tags for the next capture; authoritative inactive admission
records retain their actual origin tags until a later activated update.

New rows must belong to the original building/source class and exact authored
source-part identity and use the supported static solid cuboid recipe. Initial
room identity, units and placement remain fixed. Preparation and enqueue have
no admission effects. Application validates the entire requested batch, invokes
the canonical append, obtains canonical cloned box refs, and retains them in the
original document geometry before active native mesh/contact rebuilding.

A failed later renderer commit restores the prior active geometry and receipt
but keeps new canonical object/part IDs inactive. Future preparation discovers
those retained objects through the existing world records. Retry and later
disable/re-enable reuse the same refs and IDs, without growing the registry.
No activated-environment receipt is issued for inactive admission alone.

Focused tests passed against the separately patched candidate
build/dot-studio-negotiation-20261110/registry-game.html, SHA256 at this run
05980ca88f8780d260eee0464837dc326bd1d94c419a62736a8159c42a67f2ad.
Tests prove late renderer failure, stable inactive IDs, no-growth retry, mixed
old/new refs and native contact/picking IDs, unchanged player/fired projectile,
and rejection before admission for invalid batch/parent/recipe/source class.
Both pinned unpatched originals still pass the complete existing bridge tests
and explicitly refuse first-time parts without the canonical append API.
Bridge syntax and six patcher pytest checks passed (1.29 seconds).

Bridge SHA256: 7cdfb704732c3e276c7d7d48f0315c9bcae397b9248ab74e67434aab05eb2a61.
Test SHA256: 1648dbfa7e02ac5396a766b07b6c449f91760aa7f35a1edb58efae4d2d1017b0.
This slice is frozen. Independent review subsequently identified two boundary
bugs in the new canonical append API (implicit-root identity and array-capacity
equality). The registry owner is correcting that block and the parent will
regenerate the separate candidate and rerun this combined test. The above run
does not claim coverage of that not-yet-regenerated candidate.

No shared authentication/exposure files, server, live page or publication were
changed. No compiler import, physics tick, GPU/browser test, commit or push was
performed. The original worker and dynamics remain unchanged. These are bounded
offline integration checks, not a full repository test claim.

Parent final retest: canonical guards were corrected after exact own-block/file
hash verification and root clearance. Regenerated registry-game.html SHA256
6d3016268e4069e9411f4c23819c716330e659fb731a65e6488cda5b21a1052b.
The complete combined bridge test passed again on that corrected candidate,
including first-time admission, inactive late-failure retention and no-growth
retry. Full optional layer staged to a new offline-only directory with24files;
no browser, server, live publication or native compilation occurred.
