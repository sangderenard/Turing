# Named shop emitter bridge checkpoint

Sources are the original fluorescent LightSpec YAML and persisted LED EmitterProfile YAML. Saved references contain source key, source SHA256, gain and filter, not process-local profile IDs. Native triangle emission registration remains authoritative.

2026-11-10 review corrections: spectral filtering occurs before transport-bin integration with no restoration of absorbed energy. Browser preview uses a signed 3x3 filter-response matrix derived by the original profile_to_rgb conversion; negative channels are clamped only after combining the filter. Native emission uses that same filtered full profile. Off sources retain their authored surface coefficients, and oversized numeric input is rejected as a normal validation error.

Verification: 20 Spectral named-source tests and 3 actual Turing photo-admission tests pass. Actual shop geometry native registration recheck: fluorescent gain 0 produced 0 roots; gain 1 produced 1152 roots and power 2.3083148 for 360 triangles. LED gain 0 produced 0 roots; gain 1 produced 2304 roots and power 0.092903085 for 3144 triangles. These totals follow actual emitting area and existing source gain, not a new lamp brightness.

Limits: fixed1/fixed3 transport bins approximate the full source spectrum. LED bridge samples the original normal-direction spectral shape, not all phase, polarization or angular behavior. No calibrated-watt claim. Current-scene native transition is separate from the photo capture path and is not integrated here. Conductor experiments remain runtime-disabled. Publication awaits independent review and scoped browser QA; this checkpoint does not authorize a broad build or service interruption.
