# Shop-emitter diagnostic, 2026-11-10

No product source edits or publication made during this diagnostic.
Independent localhost8898 serves live build assets with test-only WebGL instrumentation.
User's original Studio tab was not changed or reloaded.

Completed-frame readPixels after the render callback, 890x743, GL error zero:
- Original ten light strengths zero; both shop dimmers 1: RGB sum2160623, max2/255, nonzero channels1782429, hash3732255354, active light count100.
- Both shop dimmers0: sum0, max0, nonzero0, hash3297398989, active light count10.
- Single long fixture1, wide0: sum724599, max1/255, nonzero724599, hash3495874328.
Native arrow-key edits confirmed Room updated and shader uniform changes. Earlier automation fill-only values did not commit, and earlier drawArrays-only instrumentation sampled an intermediate draw. Those preliminary reads are not causal evidence.

Current shop material rows53/54 are PBR runtime rows written by writeStudioFixtureRows, same low-level writer as ten disk rows41..50, but not authored MaterialDatabase registration. Both get emit_gain1; disks get texstack profile1/lobe2.2, shops profile0. Shop sources manually use90 area-stratified triangle centroids/cones. This differs from original automatic material registration.

Existing full-system reference: spectral-analyzer/test_basic_gl_cpp_window.py registers source materials into MaterialDatabase, passes actual verts8/groups/db to BaseGLRenderer.derive_emissive_area_lights at1271 and pack_emissive_area_rays at1335. BaseGLRenderer method at497 reduces emissive material groups using PBR emission and exact area-weighted centroid, up to100 groups. Do not clone it into another bespoke JS reducer.

Existing Studio native photo path: camera_designer/studio_photo_geometry.py appends fixtures with append_saved_studio_mesh_to_gpu_scene, detects MAT_FLAG_EMISSIVE and feeds studio_native_emitters. It does not use the shop90 source approximation. Current runtime rows omit named emission-profile/material identity.

Integration unresolved: browser Studio currently constructs lights manually even for disks. No existing JS call to automatic Python method found. A sanctioned existing source export/compiled or backend binding must connect that method and complete MaterialDatabase registration; no substitute material or invented radiance applied.

## Existing lamp profile search
Actual lamp families were found; no claim of absence is warranted.
- configs/lights/fluorescent_cool_white.yaml: 4100K triphosphor tube, existing LightSpec.load/sample_freq_hz ingestion in demo_pluck_gl.py.
- configs/lights/fluorescent_warm_white.yaml: 2900K tube, same ingestion.
- camera_designer/emitter_profile.py EMITTER_CATALOG and presets/emitters: led_cool_white and led_warm_white, die/phosphor components; existing EmitterSpec profile resolution in camera_preset.py and RayOrder.from_emitter_specs.
These registries are not the same type as MaterialDatabase's EmissionProfileDatabase. No current named binding from these source profiles to shop geometry or automatic browser light registration was found. Do not substitute tungsten_filament or basic_led_display.
Choice-independent work can wire actual source geometry/material identity to existing registration and expose named existing profiles. A warm/cool selection and intensity calibration must not be invented.

## Narrow shared-loop correction (source/staging only)
Removed shopLightPhongSamples entirely. Shop material groups enter the same existing fixture registration loop as disk/panel roles. Mesh area-weighted centers and exact emitter area are retained; no90 triangle sample lights or forced hemisphere cones. Original disk behavior preserved.
Independent completed-frame GPU test of narrow current-build patch, original10 source strengths0:
- Shops RGB1/dimmer1:12 fixtures, each area4.141104721640375; sum9003080,max9,nonzero1795097,hash4048450281; GL errors0.
- Shops0:sum0,max0,nonzero0,hash3297398989; GL errors0.

Named profile source/UI/transport integration is staged but NOT released. Fields: fixture.emitterProfile, assets.emitterProfiles.records, photo fixtures.emitter_profiles with named key/source revision/gain/filter/compact material ID. Missing profiles fail closed. Geometry/gain tests use explicit white test fixture profiles, not runtime fallback.
Six focused suites pass: shop lights, shop controls, shop bridge, lighting profiles, Studio lighting, pointer identity. Source main contains unrelated unpublished dependencies, so narrow build transplant is staged separately using tests/stage_shop_registration_probe.py. No full source-main overwrite of live build.

Current blocker: raw fluorescent LightSpec wavelength PDF is not absolute radiance. Initial peak-normalized and then raw-PDF projections are both held from release pending correct original power ownership. Raw-PDF named projection at dimmer1 produced RGB[.0053227553,.0051637986,.0017106512],12 sources, but all-black8-bit completed frame with original sources off. No brightness repair is claimed. The profile worker is correcting this boundary. User's original session remains untouched.

## Correct source-power conversion, local QA
Existing LightSpec chooses wavelength while original stage packet energy stays1; profile worker preserved this power owner and integrated probabilities rather than treating PDF heights as radiance. Regenerated profile export uses original native profile_to_rgb on relative SPD; no peak normalization. Four existing names and source SHA256 identities retained. Local browser results (original ten sources off, full completed-frame readback, GL errors0):
- Fluorescent cool white, both dimmers1, default poses: sum15574725,max21,nonzero1835784,hash2699451074. ShaderRGB[1.9694193602,1.9106055498,.6329409480], area4.1411047216 each.
- Fluorescent dimmers0: exact zero.
- LED cool white, both dimmers1, default high ceiling: 8-bit framebuffer rounds tozero despite positive registered RGB and actual emitterarea.1666666667 each.
- Same real LED geometry lowered10ft in isolated QA: sum282198,max1,nonzero282198,hash1301769539.
- Lowered LEDs dimmers0: exactzero.
This proves source participation and on/off behavior, not calibrated lamp output. Named LED area is much smaller than tubes and default ceiling is23ft away. No arbitrary brightness multiplication was added.
Native worker reports11focused tests passed incl original RayTracer source sampling; full actual tube/LED geometry proof pending. No live publication/restart made by this worker yet.

## Final signed spectral filter QA
Native reviewer independently passed20tests and3 real upload-admission tests. Export now carries signed filter_rgb_matrix, native source spectrum is filtered before coarse bins, and browser clamps only after matrix/filter combination. No componentwise RGB shortcut remains.
Actual completed-frame browser proof,890x799, original ten lights0, both shop dimmers1:
- Longred [1,0,0] and wideblue [0,0,1] retain material/group53 and54.
- GPU colors exactly[1.3278942108,.2019255906,0] and[.0788159519,.0929974094,.8609548211], matching corresponding signed native matrix columns with final clamp.
- Sum3195323,max6,nonzero1516050,hash3542542684; GL errors0.
- Both filters black: sum0,max0,nonzero0,hash2895986189; GL errors0.
Final five static files staged with static_release_hashes.json. Live publication/backend reload still coordinated by parent; user session untouched. QA tabs closed after testing to release GPU resources.


## Scoped live release, 2026-11-10 06:21 UTC

Controlled lifecycle save/exit and exact recorded hidden relaunch completed. New PID 67640, boot 21637d216896. The existing session was restored (sessions 1, restored 1); no user browser reload or settings edit. Previous photo was terminal HTTP409 with no active worker. Original74fountainfiles retained; no wholesale web export.

Published exactly five reviewed emitter files plus client_ray_settings.js, client_ray_controls.js, client_ray_participation.js. All eight served HTTPS SHA256 values match the staged/source release hashes. Backup and machine-readable receipt: build/dot-emitter-release-20261110T062109Z. Backend housing/mount/named admission focused tests18passed immediately before restart. Prior final completed-frame GPU and native emitter evidence above applies to the exact now-served emitter bytes; a new production exposure was not started.

Client TLS certificate_unknown handshake alerts appeared after restart while jobs requests still returned200. CA-verified localhost HTTPS succeeds; no certificate or security settings changed. Existing open Studio must reload to load changed JavaScript; it was deliberately left untouched.


## Source level exposed, 2026-11-10 06:37 UTC

User tab still displayed old area-sample help and lacked named profile selectors; a fresh independent production tab loaded new controls successfully. Original tab was not reloaded. A table-level view does not include the ceiling; fixture mesh visibility from an upward camera remains unverified in this pass.

Added sourceLevel0..400 in existing Studio strength units, default1 for new and saved legacy rigs. Effective gain=sourceLevel×dimmer, shared between original PBR writer and existing named native reference. No watts/lumens calibration, new light source, default brightness increase, main bootstrap change or restart. Fresh B already has existing .5 dimmer overrides, while A displays1; preserved.

Six focused suites passed, plus source gain/native reference parity, legacy1, invalid-value rejection and UI rollback checks. Completed-frame890×743 actual linked shader test with other lights unchanged: source0sum13962759/hash763061744; source1sum21575425/hash3113139380;source80sum304415236/hash338253855. Both shop RGB uniforms scale80×, GLerrors0. Source80 visibly lights room. Only the two shop modules published, served HTTPS hashes verified. Backup/receipt build/dot-source-level-20261110T0637Z. User active tab and rig untouched.
