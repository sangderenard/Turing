# Camera-only backend publication, retained compiled products

The requested physical camera fix is live after a guarded lifecycle restart.
Old PID 36180 / boot 712c18d8b10f was stopped only after the fresh exact photo
worker inventory was empty. New PID 38320 / boot 3efb43f7a1ed uses the recorded
Python311, original environment and TLS settings, with the supported new
--retain-fountain-products flag and the receipt below. No user photo/live
history was interrupted, and no object was saved or exposure started by QA.

Camera implementation and GPU evidence are recorded separately in
20261109T170313Z_studio-camera-alignment.md and the Spectral repository's
build/dot-camera-geometry-20261109. The native body is behind the analytical
thin lens and follows still/live/orbit pose. Both actual intersection shaders
proved pupil clearance and body hits from reflected directions. A bounded
128-pixel render of the user's saved scene completed, but is noisy and is not
a converged image-quality result. No second compound lens was introduced.

The separate native physics source changes would invalidate all four 3-D
fountain product source digests during ordinary startup. They were not built
or activated. Receipt-validated retained startup keeps the accepted published
products and original digests; it does not claim current-source parity or new
GPU validation. Receipt: build/dot-retained-fountain-20261109/fountain-receipt.json,
SHA256 7f3f6c2177b322099d3d0df02c2d64fae33b638fddefe6244efbe3b8b04cb93f.
All eight GPU families reported retained/ready. Only the actually advertised
one-size fountain and fountain-wheel CPU fallbacks reported ready; the other
six CPU families reported skipped. All 83 fountain files (270,132,287 bytes)
remained byte-identical after startup and final verification. A future
deliberate physics deployment must validate its new products and update the
launch/receipt explicitly, not refresh the receipt to hide a mismatch.

No static assets were copied. Thirty-five retained web assets, including the
published compute runtime, matched their prior release hashes over verified
TLS HTTP. The actual Chrome production page booted with seven canvases and
its saved-object status Ready. The QA tab was closed. Film Science's CPU
endpoint returned the expected specular percentile50 linear display profile.
The unpublished chase and shared-room physics UI remain excluded.

Verification: camera worker 54 Spectral CPU tests passed plus one optional
skip, 22 Turing tests passed, and the bounded actual GPU gates passed. Retained
startup added 30 passing tests and six existing startup regressions. Coordinator
independent reruns passed 16 camera tests, 22 photo/live/version tests, and 31
retained/version tests, plus actual receipt validation. These are focused
checks, not a full repository baseline.

Two coordinator QA issues were isolated from production: the first retained
inspection repeatedly scanned each multi-megabyte WGSL for each entry point
and was stopped by its verified owned PID. A single function-name scan made
validation approximately five seconds. The first HTTP verification flattened
the shader subdirectory and received a 404; preserving the relative path fixed
the harness, after which all asset checks passed. Neither issue changed a
compiled product or required another server restart.

Complete release guards, source/static/product manifests, launch records,
readiness and browser receipts: build/dot-camera-release-20261109.
No commit, push, merge or native physics build was performed.
