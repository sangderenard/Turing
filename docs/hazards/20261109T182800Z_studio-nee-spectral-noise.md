2026-11-09 CAMERA NEE SPECTRAL VARIANCE (MEASURED, Windows source)

Observed current two-white-snoot scene ea5bacd...: fixed 4096 baked packets,
2048 per source. Per-source wavelength counts (R/G/B) are 653/702/693 and
717/685/646. Existing camera NEE chooses one such packet at each opaque hit,
so neutral lights generate single-band samples. Independent 150-sample means
would retain about 11-12% per-band relative standard deviation from this
choice alone. This is an isolated estimator measurement, not attribution of
all image noise. Forward pump reuses these baked primary directions; a
65536 ray batch visits them 16 times, not 65536 fresh initial directions.

Added fixed3_nee_records to spectral_sensor_basis.py. It computes the exact
conditional mean of existing packet-energy * amplitude-weight by identical
float32 position and fixed3 wavelength. The same record count/order/source
selection and visibility remain. The sensor-only derived buffer replaces
source direction slots with mean 650/550/450nm weights; it is never submitted
to the forward tracer. SensorAccumulator owns and releases it, rebinds it
only for fixed3 and restores original sources for other film modes. Original
forward pump always rebinds its original records. No film/color/BSDF change,
spatial blur, pixel deletion or glint clipping. Existing baked-spectrum bias,
centroid-light approximation and geometrical/path variance remain.

Verification:
- 35 focused CPU tests passed, 1 opt-in GPU skipped.
- Real RTX3060 isolated 65536-sample shader test: original means
  [49.853,49.800,50.347], SD [70.657,70.664,70.814]; new means exactly
  [50,50,50], SD [0,0,0]. Full production sensor shader linked.
- Real worker ran copied current 95216-triangle scene at128x128, two epochs,
  8 bounces, completed, 5.03s setup/1.19s exposure. This is a lifecycle smoke,
  not evidence of whole-image convergence or a final photographic comparison.
- Expanded CPU gate: 131 passed, 13 failed, 1 skipped. All 13 failures are
  old film-worker mocks missing the camera geometry update method, before
  reaching new NEE setup. Running same tests with only the new NEE call removed
  in memory reproduced the same13 failures (50 pass in that file). No source
  rollback or live job was used. Full gate must not be described as passing.
- Python compile and tracked scoped whitespace checks passed.

No production request, active exposure cancellation, export, server restart,
commit or push. Existing already-running workers are unchanged; newly started
workers import this source. GPU slot checked idle from recent status/leases
immediately before each isolated run; latest jobs canceled, leases expired.

GPU proof: spectral-analyzer/build/dot-nee-spectral-20261109/gpu-check.json.
Worker smoke: same directory/worker-smoke. New tests:
tests/test_studio_nee_spectral.py and tests/studio_nee_spectral_gpu_check.py.
