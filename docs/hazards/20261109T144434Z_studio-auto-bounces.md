2026-11-09 AUTO BOUNCE PILOT (SOURCE ONLY; GPU/PUBLICATION PENDING)

Auto is the new UI default; the existing numeric manual count and quadratic
1..200 slider remain separate and survive Auto/manual switches. New requests
carry bounce_mode. Legacy requests with an explicit max_bounces stay manual.
The probe horizon is 1000 interactions, independently of the render cap 200.

The isolated Spectral module instruments the canonical forward shader; it does
not implement replacement intersection/material physics. It emits four
independent isotropic unit-energy samples per 4x4x4 spatial cell (256 paths),
records every observed hit's transport energy through depth1000, and records
paths still carrying energy after1000 separately. Diagnostic sources/buffers
are owned; canonical scene SSBOs are read-only; splats are disabled and source
bindings restored. The render field is cleared after its initial warmup before
using the chosen limit. No render path gains a new termination rule.

Selection uses the entire observed remaining energy tail, with at least64
further depths available. It requires all four batches and their t-based batch
spread to be below1%, at least16 first hits per batch, and no survivor after1000.
The minimum automatic choice is8; a required choice above200 is reported and
rendering stays200. Sparse coverage, missing/failed pilot and survivors at1000
are explicitly inconclusive. Persistent paths trigger a mirror-room warning.
This is forward transport-energy guidance, not a bound on camera-image error.
Small objects/rare caustics can be missed; existing forward cutoff.002 and the
different sensor roulette/throughput cutoffs remain and are disclosed.

Measured CPU/static checks:9 Spectral probe tests pass, including late returns,
closed-tail samples, one rare survivor, empty coverage, spatial stratification,
canonical instrumentation and host binding restoration. glslangValidator
accepts the generated compute shader.13 Turing Python tests pass for Auto
admission, photo API, room packing and gobos. Auto/manual JS checks pass;32
photo lifecycle/retouch cases pass; gobo geometry/scene, camera framing, room
and material-ID checks pass. Python/JS syntax and scoped whitespace pass.
The final combined Turing gate now passes19 Python tests and11 live lifecycle
cases plus the memory boundary matrix after the2GiB worker updated its tests.
Lighting profiles and exact ten-light strengths also pass. No full suite ran.

GPU QA is not yet executed: the user's production live then180-second photo
jobs remained active throughout implementation. No job was interrupted and no
server restarted or static export published. The opt-in tiny12-triangle GPU
check is tests/marble_bounce_probe_gpu_check.py in Spectral. It validates diffuse
selection, slow opaque mirrors beyond200, glass total-internal-reflection
survivors at1000, unchanged field bytes and unchanged production depth uniform.
Run only after verifying all production GPU photo/live work is idle.

Expanded regression gate:100 Spectral CPU tests pass across probe, color and
film-response code, including new worker Auto-success and failed-pilot cases.
This gate caught two integration issues, now fixed: old direct worker jobs
without bounce_mode keep their old manual default, and every preparation/probe
status retains the same film/transport/frame metadata. New Studio Auto requests
remain explicit. Latest user exposure is1200seconds, so the GPU gate is still
pending. No production job was interrupted or UI bundle published.
