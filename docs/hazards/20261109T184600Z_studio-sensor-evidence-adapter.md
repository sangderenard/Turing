2026-11-09 STUDIO SENSOR EVIDENCE ADAPTER (MEASURED, Windows, opt-in only)

Added camera_designer/studio_sensor_evidence.py in Spectral. This composes the
EXISTING unmodified csrc/shaders/sensor_mip_accumulate.comp.glsl in Studio's
current GL context, not the native thick-lens transport engine. It maps each
fixed sensor pixel to an ABI-compatible direct leaf node. No alternative
moment math, scheduler, reconstruction, hierarchy subdivision or transport.

SensorAccumulator.enable_sample_evidence() opt-in only. Actual sampled global
UV, local pixel UV, sample index and weight1 are carried in the native lineage
ABI; actual post-film linear spectral values (including zeros) are written
before sensor_store changes ring/image accumulation. Native shader accumulates
sum/squared-sum/weight/count. The adapter never replaces the image or deletes
bright pixels. Original sensor bindings are restored on each subsequent strip.
No production worker invokes enable_sample_evidence and no UI activates it.

Side-channel semantics: evidence since enable/reset, not finite-window noise.
Camera/scene/history clear also clears evidence. Display adjustments do not.
Only fixed:3 unique row dispatches with one sample per pixel are admitted;
random-collision/spp/mode changes reject. Buffers release with accumulator.
At512x512,16-row strips: 24,477,736 bytes (~23.34MiB) additional GPU allocation;
this is not counted as image history and is NOT yet offered on the live page.
Readback is explicit debug only, never in the frame loop.

Tests:
- 51 targeted CPU tests passed, 1 opt-in GPU skipped. Covers native leaf ABI,
  rectangular UVs, invalid bounds, original history/coverage, NEE and mipmap.
- Existing native accumulator on RTX3060: 6pixels x3bands x2samples (including
  zero) exact sums/squared sums/counts/weights, reset, cleanup; 856bytes.
- Full production sensor shader linked with the optional bridge disabled.
- Actual copied95k-triangle128x128 worker with bridge enabled: all16384 pixels
  had count1 after full sweep; moment sums exactly match saved raw sensor RGB.
  Raw image is byte-for-byte identical to prior run with side channel disabled.
- Native float32 squared moments flush some subnormals:110/49152 components
  differed from NumPy square by at most2.2452e-41 (CPU retains subnormals).
  The strict CPU relative comparison failed on those only. Do not call the
  squared moments universally ULP-identical or conceal this with tolerance.
- New GPU proof: Spectral tests/studio_sensor_evidence_gpu_check.py and
  build/dot-nee-spectral-20261109/evidence-gpu-check.json.
- No live API call, production restart, exposure cancellation, export or push.

NEXT GATES BEFORE CONFIDENCE/ADAPTIVE UI:
- This is raw evidence, not a confidence percentage. At one sample, or all
  zero/identical samples, rare unobserved paths remain unknown.
- Current repeated4096-ray source bake creates shared/correlated evidence;
  ordinary variance/count is conditional on that bake, not proof of complete
  physical accuracy. Need genuinely refreshed/independent batches or measured
  batch-level uncertainty before claiming effective independent counts.
- Add native score/frontier/select with reserved broad coverage; preserve node
  lineage and correct sample weights, never hard-stop dark regions simply
  because they have not yet hit a caustic. Pixel rendering currently stays
  dense rows; no adaptive operation has been enabled.
- Native float32 sums/squares have underflow/overflow/cancellation limits.
  Any diagnostics must report nonfinite/undersampled state, not certainty.
- Per-band squares omit cross-band covariance. Color-profile uncertainty needs
  covariance or independent developed-image batch comparisons, not just
  applying a matrix to three standard deviations.
- Optional display confidence blend must remain separate from physical HDR;
  validate color bias, rare glints and step/reset/window semantics, allocation
  near2GiB history and actual GPU cost before exposing a default.

2026-11-09 continuation: native scorer integration (not production enabled)
- Opt-in adapter now compiles the unmodified native sensor_mip_score shader;
  score_conditional_variation uses existing moment variance with uncertainty
  weight only. Image, learned, requested inputs are disabled (sensor_res=0),
  so no square-image assumptions are imposed on rectangular Studio frames.
- Explicit debug readback includes conditional_variation_priority. No sampler,
  display, stopping condition or production caller consumes it.
- This is NOT physical confidence: no/single/zero-valued samples return zero,
  bake correlation/bias and unseen rare caustics remain outside this score.
- CPU gate51 passed1 skipped; Python compilation passed. Native scoring GPU
  check added with float64 reference but NOT RUN: latest user live job
  1cfb0b66894cc968077bab7b0d5c311c8ecc28808ccd28db07dee89ec7fabde3
  active and GPU busy at preflight. Deferred to avoid competing with user.
- Prior native moment and transport proof remain valid; new scorer has not
  been GPU verified. Keep evidence disabled until this gate and resource/
  statistical semantics are addressed. No production live-loop overhead.

Independent-pilot design guard (2026-11-09):
- Added deterministic200,000-trial rare-event benchmark. Stopping on a dark
  first sample and otherwise extending to10 estimates0.019 instead of0.1;
  independent pilot choosing fixed1-or10 fresh final samples estimates0.1
  within0.003. Pilot excluded from final estimator. Two tests pass.
- This is a synthetic statistics benchmark, not scene convergence or a new
  production sampler. Preserve minimum coverage; dark pilots miss1% events
  with probability>92% in8 samples. Need independent source bakes/seeds and
  caustic/emitter-edge scene comparisons before any adaptive stopping release.
- Root supplied references for research context (not reviewed here):
  https://cseweb.ucsd.edu/~henrik/papers/adaptive_sampling/
  https://www.researchgate.net/publication/220721115_Unbiased_sampling_techniques_for_image_synthesis
- GPU scorer gate still deferred: newest user job d19be4c... live at12:27
  local, GPU90%; no background GPU test or interruption performed.

GPU verification update (2026-11-09 after region worker released slot):
- Both latest live jobs confirmed canceled at preflight12:47/12:46 local.
- python tests/studio_sensor_evidence_gpu_check.py completed exit0 in3.7s
  on NVIDIA RTX3060. Native accumulator AND unchanged native scorer pass
  against float64 reference:6pixels,3bands,2samples including zeros; reset
  and cleanup pass. Fixture856bytes. Full production sensor shader linked.
- Receipt: spectral-analyzer/build/dot-nee-spectral-20261109/evidence-gpu-check.json.
- Combined CPU gate53passed1skipped. This supersedes the earlier deferred
  GPU gate status only; all statistical limits and default-off status remain.
- No restart, active exposure cancellation, production enablement or changed
  displayed output. Scorer still reports diagnostic priority, not certainty.
