# Directional bundle and immediate secondary material passes

The v3 twelve-direction release was published on 2026-11-11 using the existing
supervisor's graceful save/restart path. The authoritative receipt is
`build/dot-secondary-release-20261111/verification.json`: 37 changed assets,
all 108 served hashes verified, 74 retained products preserved, and sessions
restored on boot `28e70e0353c1`. No forced termination or other instance restart
was used. TLS-verified local and same-host public-name HTTP checks passed;
these do not establish external ingress or a production browser-render check.
The subsequent v4 denser-grid/replay-normalization work is separate and is not
published merely because its source is now present.

## Existing systems remain authoritative

The Spectral forward tracer records actual propagation on its existing traced
exterior segments. It adds no rays and leaves the scalar field bytes unchanged.
The directional estimator integrates actual entry packet energy times exact
distance inside each crossed voxel, then divides by voxel volume. The immutable owned
`/api/marble_photo/field` artifact now optionally contains `studio.forward-field.v3`:
the old scalar prefix, configurable directional moments, original server triangle IDs,
material classifications, baked source records, native PBR16, and exact submitted
PBR16-to-native-ID lineage. See Spectral's
`camera_designer/STUDIO_DIRECTIONAL_FIELD.md` for the binary contract.
The default is twelve ordered icosahedral directions; six axis directions remain
selectable for comparison. These classify measured rays, not replacement lights.
The opted-in job cache identity includes schema and stencil. V1/v2 remain readable.

The browser verifies hash, shape, finite data, section spans, job/revision identity,
and exact authored Float32 material bytes before installing a complete replacement.
Native engine IDs never stand in for browser material or triangle IDs. Material
eligibility is remapped through the recorded authored lineage. Unmatched local rows
receive no field contribution and retain their ordinary direct Phong specular.
Optical glass-rod lookup uses the existing canonical snapshot projection on a
cached copy; both original and projected bytes are checked for staleness. The
displayed material rows are never rewritten to make identity matching succeed.
The complete triangle source remains intact for transparent layers, shadows,
refraction and physics. Nothing is accumulated a second time by the browser.

## Same-frame rendering

`marble_secondary_field.js` runs synchronously in the existing Studio animation
frame, using the existing scene meshes, transforms, shared material tables and
canonical `base_material.frag.glsl` equations:

1. Primary opaque geometry writes linear premultiplied RGBA16F and opaque depth.
2. Secondary material-volume response draws with depth EQUAL and class bit 1.
3. Secondary reflectance draws with depth EQUAL and class bit 2.
4. Retained sorted transparent geometry draws primary plus field glass response
   together, depth-tested against opaque geometry without writing depth. This
   preserves attenuation between transparent layers rather than adding all glass
   light after its covering layers have already blended.
5. The canonical shader resolves exposure/ACES/sRGB exactly once.

Steps 2-4 are the three requested material passes. There is no visibility readback
or animation-frame delay between them. Per-fragment material eligibility AND the
actual primary depth controls the opaque work. A nearest-triangle-ID mask is not
used to erase transparent layers. Invariant clip positions guarantee matching
depth across the separately linked primary and secondary programs.
This is a depth-and-material gate, not an exact winning-triangle-ID gate.
Coincident coplanar opaque triangles at identical depth can both pass EQUAL,
even if only the first won the primary LESS test, and can accumulate twice.

The optional Phong checkbox retains direct-source specular instead of field-driven
specular. It reuses authored shininess, strength and enamel properties, and avoids
counting both specular sources. Primary field-spec replacement is per material,
not a global deletion of highlights on unsupported materials.
The field includes direct as well as bounced paths, so matched material diffuse,
body and transmission responses must likewise have one owner; adding ordinary
direct response on top of them would count zero-bounce illumination twice.

The ordinary primary program remains separate, retaining the old sampler ABI for
saved Studio subject captures. When inactive the existing draw path remains in use.
DOF uses its single-sample color/depth target while this pipeline is active;
ordinary DOF keeps its existing multisampling. Primary/secondary GL state and
framebuffer ownership are restored even when a draw fails.

## Limits that are not hidden

- The atlas has 32 cubed cells and either six axis or twelve geodesic bins. It retains measured mean
  world propagation within each bin, not a fabricated axis vector. Shading uses
  the opposite vector as incoming light. Bin order, bounds and direction space
  remain explicit future alignment points for geodesic texture packs; no such
  pack baker is implemented here.
- The receiver reconstructs weighted propagation moments from actual spectral
  energy. Optional hardware trilinear filtering blends energy and moments
  together, with clamping inside each angular tile. Twelve bins use two X tiles
  and six Z slabs (64x32x192), within WebGL2 minimum texture limits. Devices
  without float-linear filtering explicitly fall back to nearest cells.
  Filtering does not create new ray samples or eliminate Monte Carlo noise.
- Directional values are fixed3 track-length throughput density, not calibrated
  irradiance or a complete angular radiance function. They are not normalized by
  ray count or solid angle. This is not lossless mirror scene reconstruction.
  The explicit estimator tag rejects the earlier volume-splat experiment: actual
  scene testing showed legacy fog extinction and integer quantization reduce
  those deposits to zero more than roughly a meter from the ceiling emitters.
  Recording real travelling packets independently does not change that primary
  fog estimator or invent additional illumination paths.
- Volumetric means the existing material thickness/translucence/remission
  response. It is not a newly invented participating atmosphere.
- Glass uses the existing RGB transmission and Fresnel model. It is not a new
  traced refraction or nested-medium solver. The producer does not invent
  directional evidence for directionless emission or interior media.
- No crepuscular beams, fog, humidity, dust or other atmospheric scattering is
  added or amplified. Atmospheric appearance needs its own declared particle
  model. The record of travelling energy is not a scattering model.
- Signed accumulator overflow invalidates the directional section, not the
  primary exposure. The API continues publishing the unchanged scalar prefix as
  v1 with an explicit reason. The next ordinary field clear restores v3.

## Validation

Focused Python/Node gates cover owned API streaming, hashes, malformed data,
lineage remapping, stale material/camera state, cancellation, atomic texture
replacement, same-frame draw order, depth/blend state, one display resolve,
resizing, DOF and restoration. Existing Studio boot, loading, subject renderer,
triangle-ID instrumentation and Node-only scene product checks also pass.

The native GPU checks compare canonical scalar output bitwise, record an actual
oblique direction, test overflow/recovery, and cover direction response, material
masks, depth visibility, clear and high-opacity glass, two-layer attenuation,
specular override/fallback and legacy disabled parity. A concentrated 131072-ray
fixture is a stress check, not a representative scene performance benchmark.

The isolated real Studio browser/API/worker fixture and its per-frame GPU probe
are `tests/meowdoku_secondary_browser_check.py` and
`tests/meowdoku_secondary_probe.js`. Browser receipts are in
`build/dot-secondary-browser-20261111/`; canonical temporary shader exports are
in `build/dot-secondary-field-20261111/shaders/`. Final browser receipt status,
including any pending acceptance, is authoritative rather than an empty log.

The completed six/twelve-direction comparison is
`build/dot-secondary-browser-20261111/room-comparison-receipt.json`.
Its five 1530x855 images share a raised 24 mm camera, unchanged lights, gain 1,
DOF disabled and 131072 actual traced rays per field. The only producer setting
that differs between fields is the directional stencil. All three material
passes plus the final resolve ran immediately after primary without GL errors.
Weighted interpolation visibly softens the coarse cell boundaries. Twelve
directions change fewer pixels than spatial interpolation and do not resolve
the strongly saturated field color or the largely dark tabletop. The ordinary
primary reference is cream/white, so these colors are not authored room paint.

Preliminary GPU timer-query medians were 20.7 ms for primary alone, 33.2/54.0 ms
for six-bin nearest/weighted, and 36.2/79.8 ms for twelve-bin nearest/weighted.
Chrome was background-throttled to about one animation frame per second; these
are GPU timings, not an interactive frame-rate benchmark. Sample counts and
percentiles are retained in the receipt. No brightness, source or atmosphere
adjustment was used to improve this comparison.

This is not a full-repository-suite claim. Sixteen broader Spectral mock-contract
failures were reproduced unchanged against HEAD copies; their native-PBR and
geometry-update interfaces predate this change. No baseline source was stashed or
replaced. The Windows symlink test remains an existing environment skip.

## Publication boundary

Ship backend, worker, canonical vertex/fragment exports, receiver/orchestrator,
Studio UI, transitive module imports and the current material catalogue together.
Do not publish a checkbox against an older worker, API or shader. Existing jobs,
concurrent source changes and service lifecycle must be reconciled before any
production restart. The authorized coordinated release preserved concurrent
Phong capture/hosting edits and the current module graph. A future source edit
still requires a coherent asset/backend release rather than assuming the prior
publication includes it.

## Follow-up radiometry finding

The actual twelve-bin job had 4464 baked source roots and 131072 dispatched jobs.
Root beta already sums to emitter triangle area times authored spectral emission.
Each 8192-ray pump starts its source index at zero: roots 0-3727 were reused twice
per pump and roots 3728-4463 once. Across sixteen pumps this is 32 versus 16
repeats, producing about 29.44 times cumulative spectral energy. Dividing the
atlas by total ray count would also be wrong because the root weights already
include emitter sampling normalization. The v4 correction is being implemented
as exact per-root replay-multiplicity compensation and a mean of complete
root-covered pump estimates, with explicit incomplete-coverage handling.

There is a second distinct limit: 18619 of 66655 nonempty bins in that job held
only one fixed spectral line. Baked root wavelengths/directions stay fixed
across repetitions. Their valid signed linear sRGB conversion followed by high
magnitude clipping explains saturated red/yellow/magenta cells. Signed field
contributions are summed before the display transform; there is no per-bin tone
mapping to remove. A denser spatial grid can make this sparse chromatic sampling
more visible, not cure it. No exposure-based or artistic color correction is
being substituted for the sampling/accounting work.

## Subject-region acceptance and actual coverage

The v4 replay correction and v5 independent subject recording region are now
implemented and tested locally. The live publication described above remains
the earlier v3 release until a new coherent publication receipt supersedes it.
The v5 release was staged, not silently substituted into the running service.

The actual subject capture in `build/dot-subject-field-20261111/` uses 12
directions, a 64-cubed field at exactly 0.5 mm per cell, and a 32 mm cubic
recording region around the full 20 mm subject. Requested minimum margin is
1 mm; cell alignment and cubic padding yield actual per-face padding of
5.5-6.5 mm. The full scene and its physical light fixtures remain in the trace.
Subject receives field response; Stage and Room do not. Region and division
settings affect recording, whereas category opt-ins affect reception.

This exposure recorded **zero occupied directional cells**. Its 4464 actual
baked source roots produced 8192 recorded jobs in one complete normalized
sweep; 128 initial partial jobs were excluded. Gain 0 versus gain 1 changes
zero pixels. The dark subject in `subject-phong-field.png` is therefore an
honest missing-coverage result, not evidence of successful subject lighting.
The separate `subject-native-photo.png` is the actual 256-square native sensor
exposure, not the Phong field presentation. The accepted pass order is primary,
volume, reflectance, glass, then resolve, with no GL errors.

The primary-only comparison is saved with identical framing. Stage/Room image
corners differ slightly between the linear field pipeline and ordinary primary
path (maximum 11/255); exact whole-frame baseline parity outside the region is
not claimed. Canonical shader tests do verify inclusive region boundaries and
ordinary material fallback on all six outside faces.

Source provenance is concrete: `prepare_photo_geometry` calls
`studio_native_emitters.native_emissive_sources` with seed 42 and requested
budget 4096; `RayTracer.submit_emissive_triangles` produces the 4464 actual
weighted records uploaded as `SourceRec12` to SSBO 5. There are 144 assembled
emissive fixture triangles using fixture-local material IDs 17/18, mapped to
native materials 16/17 (not similarly numbered subject catalogue entries).
They use fluorescent cool-white profiles, gain 400, and a white filter. Because these emitters exist, the
configuration's preview-stage point light is ignored. Increasing dispatch
batch size alone only replays this fixed source population and cannot repair
its absent subject coverage.

The requested follow-up is field-only live streaming with exposed canonical
source budget/seed controls, refreshed angular samples, and a weighted running
mean across complete source populations. This v6 implementation is now locally
source-frozen and native-tested, with real browser acceptance still pending at
this receipt. Native seed changes randomize angular sampling, not every source
dimension; fully independent IID paths are not claimed. Existing native
interaction-target sampling has an inconsistent proposal PDF/energy weight
and is not enabled as a shortcut to make the subject look illuminated.

The v6 recorder uses a declared fixed-point divisor of 16777216 and decodes with
that same divisor; scalar and legacy recording remain at 65535. This improves
numerical resolution rather than increasing light energy. At requested source
budget 32768, the actual fixture allocation is 32832 roots. With a 65536-job
dispatch, representative 0.5 mm crossings would contribute only 0.31-0.62 old
counts, versus about 80-158 new counts. Signed overflow still rejects the
population. The native proof checks three distinct angular populations,
unchanged energy weights, coherent forward/camera source buffers, cumulative
raw moments, matched decoding and explicit history reset. Focused gates report
96 passes and one environment skip. The browser must still measure actual
subject coverage; a small controlled fixture's nonzero field is not that proof.

The actual subject also contains explicit region media: clear host material 14
and glass-rod materials 41/42. Their transmittance at 450/550/650 nm, referenced
to 1 mm, is [1,1,1], [0.91,0.43,0.10], and [0.39,0.73,0.08] respectively.
The ordinary Phong reference has dark inward-facing rod areas too. No causal
diagnosis of that shadow follows from the empty directional capture. Existing
native boundary transport already performs Snell/Fresnel/TIR and Beer
attenuation. An independent recorder helper for actual internal segments is
being tested; it is not yet part of the frozen v6 live core. It must observe
entry throughput and integrate attenuation inside each cell, not reuse a
post-attenuated exit packet or invent participating fog inside clear glass.

## Completed live acceptance, including internal media

The internal observer is now integrated under explicit
`light_field_internal_media=true`, restricted to refreshed v6. It uses the
canonical material/region/dye Beer coefficient and records actual refracted
segment entry throughput. Legacy versions and scalar transport are unchanged.
Internal recording combined with an external `ExtinctionGrid` is explicitly
unsupported and rejected, rather than multiplying two incorrect independent
integrals. The helper and integrated fixed24 GPU proofs verify clear and
absorbing glass, Snell directions and scalar byte parity. Combined focused
producer gates report 106 passes and one environment skip; API gates report
265 passes and validate the actual internal-enabled native artifact.

The first real live start exposed a save identity defect: canonical optical
projection and raw authoring state were incorrectly compared. Preparation now
retains the exact raw authoring key as a local receipt. The field gate checks
that receipt against both the settled key and a fresh raw capture, including
edits before the queued update. Stored object geometry and stale checks are not
bypassed. Snapshot/optical/load regressions cover the fix.

`build/dot-live-field-20261111/live-receipt.json` records the accepted real Studio
run: subject 0.5 mm, 12 bins, 1 mm minimum margin, Subject-only reception,
unchanged lights, gain 1, DOF off, requested source budget 32768, actual source
count 32832, and 65536 dispatched jobs per complete population. Internal-media
recording was enabled. The worker completed 22 fresh populations, seeds 42-63,
and 1441792 recorded jobs; the normalization divisor was 22. Observed source
hashes changed and sampled artifact cadence had a 2.09-second median. Phong and
all secondary passes remained active with no GL errors and no ray-image API
requests. The live UI was then stopped and its job canceled to release the GPU.

All 262144 subject cells remained zero in every observed atlas. This verifies
streaming and correct metadata, not successful subject relighting or a complete
internal field. `subject-phong-after-stream.png` is explicitly ordinary Phong
after stopping, because stopping disables field reception; it is not a field-on
acceptance image. The earlier dark field-on capture is retained separately.

Coherent release stage: `build/dot-live-field-internal-release-20261111`.
It passes six staged Node gates and internal-enabled native artifact decoding.
Publication is held while shared gobo/backdrop source dependencies are being
finished for the user's expanded artwork request. The server-side scene product
uses those source builders, so mixing old served builders with unfrozen new
server builders is not an accepted release. The running service remains the v3
publication until a new explicit deployment receipt records otherwise.

## Coherent live release, 17:37 UTC

The release is now deployed through the existing authenticated save/exit and
supervisor lifecycle. `build/dot-studio-coherent-release-20261111/verification.json`
records boot `80a815f06ed5`, 114 verified served assets, 18 changed assets, and
preservation of all 74 retained products, saved sessions, launch configuration,
TLS configuration, and the other three service instances. The source-only SQL
host bindings written after this boot are not part of this release.

The live UI now exposes the opt-in shared opaque transport model
`studio-pbr-mixture-v1`. It evaluates and samples the same authored material
mixture in forward and camera paths. The correction removes the legacy use of
dimensionless material residual as air-distance absorption and the absolute
packet cutoff from this opted-in lane. Native glass and conductor transport are
unchanged. Corrected source weights remain invariant under source-root splitting;
the field remains the average of completed source populations, not a brightness
sum. Focused producer and API checks passed (109 plus one skip, and 278,
respectively). A single corrected population still had no subject coverage;
cumulative real-service visual acceptance is not yet established.

Final acceptance uses the actual Studio REST/browser controls and active
field-on canvas capture. Standalone arrival ledgers remain opt-in diagnostics,
not ordinary bake payloads. No diagnostic archive is required by the live path.

The first actual production run after this release produced 1,631 nonzero
subject cells out of 262,144 after 56 populations (3,670,016 recorded jobs),
with original lights unchanged. This is measured forward coverage, not full
subject convergence. The browser did not apply an atlas during that run:
Apache recorded seven successful HTTP 200 field responses, each interrupted
after 40.6-76.5 MB of an 89,279,056-byte payload at 15-16 second intervals.
The receiver's fixed 15-second timeout caused these repeated partial downloads.

`build/dot-field-delivery-hotfix-20261111/receipt.json` records a verified
single-file client publication. The transfer now has a size-bounded deadline,
visible download/error status, and retains the last valid atlas on transport
failure. Scene, job, revision, material, and payload validation remain intact.
The service boot is unchanged; no backend reload was needed. Real-browser
applied-atlas acceptance is the next check.

## Actual applied-atlas acceptance

The hotfix was verified in a separate signed-in tab on the real deployed Studio,
with the original user tab and its edits untouched. Ordinary browser controls
started the same live worker with corrected opaque transport and internal-medium
recording enabled, Subject-only reception, 0.5 mm cells, 12 directions, a 1 mm
minimum margin, source budget 32768, and forward batch size 65536. Lights and
field gain were unchanged. The ray-image overlay remained off.

The receiver applied successive real atlases; observed transfer transitions
were 1, 4, 7, 10, then 13. While transferring, the preceding valid atlas remained
active. The 1530 x 855 canvas capture
`build/dot-real-field-20261111/subject-field-final-active.png` was made before
stopping the live job. The first known nonzero applied bundle had six populations
and 212 occupied cells. At stop the producer reported 68 populations, 4,456,448
recorded jobs, and 2,034 occupied cells (0.776%). That final producer count is
not a claim about the exact atlas displayed in the final image.

The central subject remains dark. Streaming, receipt validation, upload, and
successive canvas application are verified; complete internal illumination and
useful visual convergence are not. This production check used no injected
instrumentation, so it does not add a fresh all-pass GL-error assertion to the
earlier shader tests. The live job was canceled through its existing control,
and the GPU was released. No diagnostic archive was delivered for this check.

## Current correction: copy existing responses, do not reconstruct them

Further renders, exposures and screenshots are prohibited until the requested
information path is complete. The prior active-field image did not establish
physical darkness. Its primary shader removed selected Phong contributions
from material/category flags without sample availability; the secondary shader
then supplied zero from missing data. Its so-called volume pass sampled only a
surface position. The browser also reconstructed lighting and material response
from direction moments, contrary to the clarified consumer contract.

The required implementation copies the existing native ray-tracer response at
its original evaluation site, immediately before the same integration. The
second pass applies those recorded results, with actual availability alpha to
Phong. It must not reconstruct transport, material response, paths, or geometry
from the package. Empty input means no information, not physical black. All
additional copies and availability data are explicitly opt-in; ordinary tracing
must remain unchanged.

Source-only work now observes the original camera lum additions and throughput
multiplications through identity-return copy functions. Original scalar deposit
visitation and rounding/clipping/invalid flags accompany the matching query;
evaluated zero and missing input remain separate. Source-selection and shadow
outcomes are copied from the original NEE branches without another ray or RNG
call. The existing sensor-evidence infrastructure owns these extra rows.

Recipient identity now retains the existing original-face and primitive-owner
lineage through packing, filtering and compaction without changing transparent
sorting. The final response product, transport, and direct screen application
are not yet complete. New large-domain observers and duplicated scene-export proposals
were disconnected from the active path. Optional new per-band continuation is
not automatically enabled by response-copy opt-in. No new render has validated
the pending correction.

The retained information must be attached to the scene, with location, direction,
material/volume identity, sample availability and the original path semantics.
It includes relevant existing forward, camera, direct-light, reactive and volume
results. It is not limited to forward rays. Camera contributions remain useful,
but a camera-pixel response image alone is not the requested scene-attached
field. Their accumulated path throughput cannot silently be relabeled as a
local material response, and different estimators cannot be blindly added.
The source-only camera component publisher is therefore inactive. Work is
continuing on bounded native response copies and explicit progressive retention;
discarded or unavailable records must remain unknown rather than become black.

## Source implementation and current execution hold

The current source connects opt-in native terminal responses, original scalar
cell accumulation columns, and completed camera samples to a bounded typed
scene store and the existing authenticated field route. Camera samples retain
the original `lum` and filtered sensor result before pixel integration. Their
component columns use the same already-evaluated native film coefficients.
The consumer applies producer-side linear RGB and sample availability directly;
it does not reconstruct lights or material optics from direction moments.

The three secondary slots use a separate pre-first-hit volume target, opaque
downstream response, and first-glass downstream response. Nearest-recipient
and depth matching prevent a complete downstream response from being applied
again through another layer. Eviction, invalid ownership, unsupported prefixes,
and missing samples remain unavailable rather than becoming black. Native-only
and unlocated records have a bounded raw partition, separate from surface
application eligibility. See `STUDIO_SCENE_RESPONSE_CONSUMER.md` and Spectral's
`docs/STUDIO_EVALUATED_SCENE_RESPONSES.md` for the current contracts.

Execution is explicitly held before GL setup. CPU and offline source checks
are not a claim of completed hardware or visual acceptance. A permitted
context/query/link-only check on NVIDIA GeForce RTX 3060, driver 610.88, found
16 compute storage blocks and 96 binding points. Its first full forward
compile/link call did not return and reached 13,940,441,088 bytes of process
working set. Only that verified preflight process was stopped. No ray dispatch,
draw, exposure, screenshot, field store, or stream outlet ran during the check.
The precise driver subphase and allocation cause remain unproved.

The following source-only repair specializes the opt-in prebuilt program for
its actual forward entry, without changing that entry's operations or the
separate camera kernel. Unprocessed reactive work is still identified as such.
Exception cleanup and artifact retirement/lease cleanup were also repaired;
those separate defects do not explain the blocked first compiler call.
The service was verified healthy afterward on port 8766, with an independently
started PID 61124. This work has not been deployed or visually accepted.

The next, protected link-only attempt used the frozen specialized source
`8718336e6144a90a1da9d40c87b29a9a8457558908fe2f133eadcaf41832048c`.
It returned from shader compilation in 0.175 seconds, then failed program
linking in 6.251 seconds with NVIDIA `C9999: out of memory - internal malloc
failed`. Suspended-child JobObject admission verified 4 GiB process and job
limits, kill-on-close, a 90-second independent watchdog and an 8 GiB free-RAM
floor before execution. Minimum sampled free RAM remained about 13.4 GiB.
Program/context cleanup completed and the job had no active processes.
Windows reported aggregate peak accounting 28 KiB above its configured cap;
the receipt preserves that discrepancy rather than rounding it away.

This isolates the observed failure to native program linking, before any
ray-data allocation, stream outlet or render. It does not establish the
driver's internal allocation cause. The complete receipt is in Spectral's
`build/dot-shader-link-only-20261111/protected-attempt/RESULT.md`.
Execution and application readiness remain held.

Native queue availability is now carried into camera context. The prebuilt
owner does not drain queued reactive/diffraction work. Pending, dropped or
unobserved work therefore prevents claiming a complete response; clearing
scalar epochs cannot erase that uncertainty. All extra recording remains
explicitly opt-in. The existing field REST route passed 204 CPU/API checks,
including exact native-only/virtual/unlocated record retention and no extra
response assets for ordinary requests. These checks are not hardware or
visual acceptance.

The separately user-authorized observed repeat of the same frozen source ran
from 20:16:32.718 to 20:16:40.241 UTC. It returned from compilation in 0.053
seconds and failed linking in 5.965 seconds with the same C9999 internal
malloc error under the same verified 4 GiB caps. Total child lifetime was
7.516 seconds. Minimum free RAM remained about 13.18 GiB and recovered after
normal program/context cleanup; the job had no active processes. No further
attempt was started. Receipt: Spectral
`build/dot-shader-link-only-20261111/observed-repeat-20261111T201632640Z`.

The separately authorized single-write epilogue candidate
`5d40745209cb00a65aff91d1d003618126f1974de8216e2169e7de76db130097`
reduced the full terminal-record writer from nine sites to one. Its 18
self-contained CPU/static tests preserve native expressions, outer branch
outcomes, nested controls, snapshot values, and helper-side state lifetime.
This is not executed-ray equivalence. The candidate is separate and inactive.
Its protected native link also failed with C9999 internal malloc at the 4 GiB
cap: compilation 0.065 seconds, linking 13.974 seconds, total 15.468 seconds.
Minimum free RAM remained about 12.6 GiB; all owned processes and GL objects
were cleaned. Receipt: Spectral
`build/dot-shader-link-only-20261111/single-epilogue-20261111T201945725Z`.
The reduction did not establish a working or lower-memory native program.

The queue application guard now has 33 passing CPU tests. The server applies
copied camera/sensor responses only at compatible scene-recipient/view
contexts. Forward packet power and directionless native scalar rows are
retained with their original semantics, not relabeled display radiance. This
is not yet unrestricted view-independent repainting from every forward ray.
No render, screenshot, exposure or compute dispatch has validated the repair.
