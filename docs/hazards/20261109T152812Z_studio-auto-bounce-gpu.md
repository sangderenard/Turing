2026-11-09 AUTO BOUNCE FINAL GPU GATE (MEASURED; PUBLICATION OWNER SEPARATE)

The current canonical forward shader's three continuation sites are all
instrumented, including explicit glass/air boundary reflection/transmission.
The guard also requires the explicit boundary branch, and the existing analytic
sphere-count uniform is forwarded; currentMedium is never replaced or reset by
instrumentation. Nine probe CPU tests and Auto UI tests pass.

After a fresh check showed zero production preview workers and the newest user
job complete, the direct Python311 GPU test finished with exit0. A fresh
post-check also showed zero preview or probe workers. Four tiny real canonical
shader cases passed, with8x8x8 fields and256 pilot paths each:
- Opaque diffuse room: chosen depth8, no residual paths at200 or1000.
- Small slow-mirror room: chosen189,96/256 paths reach interaction201;
  the observed tail beyond200 is0.618%, not zero.
- Trapped dielectric room:61/256 paths still carry energy after1000;
  render cap200 retained and mirror-room warning reported.
- Explicit glass host and triangulated air cavity:3/256 paths still carry
  energy after1000; render cap200 retained and mirror-room warning reported.

Every case preserved the original field bytes and production depth uniform.
The measured individual pilots took16..47ms. This bounds this tiny fixture,
not arbitrary production scenes or full-resolution render timing. The probe
horizon1000 and render cap200 remain independent. Sampling is explicitly a
limited transport-energy pilot, not an image-error convergence guarantee.
Receipt: spectral-analyzer/build/dot-auto-bounces-20261109/gpu-result.json.

The first GPU pass exposed an overstrict test expectation, not a renderer
failure: the small mirror fixture's actual tail passes at189 rather than beyond
200. Its test now checks a long depth and surviving paths after200 instead of
assuming a particular scene must cross200. Long-lived1000/cap behavior is proved
by the two dielectric cases. No transport/selection threshold was changed.

The film-response JS fixture now supplies the real Auto and geometry defaults
and performs explicit Cancel before another photo, preserving the new no-silent-
replacement contract. The complete film-response JS check passes. No production
bundle publication or server restart was performed by this worker. Product
source is frozen; publication belongs to the integration coordinator.
