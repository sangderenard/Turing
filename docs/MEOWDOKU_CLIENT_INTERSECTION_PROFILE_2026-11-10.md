# Client intersection performance profile, 10 November 2026

Scope: Node's WebAssembly runtime on the user's Windows computer, actual saved stage with95,217triangles and65,535BVH nodes. Larger batches repeat the same256captured region-primary rays, so they establish throughput and parity, not additional geometric coverage. No browser or WebGPU performance claims.

## Measured causes

Original256ray query:6,104.5ms total. Explicit setTimeout(0) after each of401traversal rounds costs5,913.2ms (97%). Compiled slab wrapper139.4ms,triangle wrapper11.2ms,other host40.7ms. This is our scheduling overhead, not inherent WebAssembly slowness.

Changing cooperative yielding to occur after8ms of work reduces the same query to273ms. This keeps cancellation checks and returns to the event loop without forcing a Windows timer wait for every node wave.

Removing Array.from/spread argument arrays and fresh memory-view allocation for every scalar Wasm call further improves medians across three warm queries:

-256rays:57.17ms.
-1024rays:124.88ms.
-4096rays:499.06ms.

All primitive identities remain exact; maximum t difference from original GPU reference remains1.4901161193847656e-8.

The last4096query spent60.89ms in slab wrappers,3.89ms in triangle wrappers,197.01ms in traversal/packing/reduction and201.05ms yielding. It made2,542,704scalar Wasm calls,10,046slab batch calls and1,034triangle batch calls. Temporary feed arrays allocated77.60MB cumulatively (not peak memory). This shows both substantial remaining orchestration cost and useful compiled-law throughput.

The35.11MB JSON scene read takes13.37ms and parsing157.42ms; validation/packing39.22ms. These are one-time costs when scene residency is maintained. Result serialization is under1.4ms at4096rays. Restarting a Node process and resending/reparsing the entire scene for each job would reintroduce avoidable costs.

## Evidence and release scope

Receipts: `build/dot-client-ray-20261109/studio-region-compute/profile-baseline.json`, `batch-profile-yield-only.json`, `batch-profile.json`. Harness: `build/dot-client-ray-20261109/profile_stage_batches.mjs`.

Optimized runtime is isolated in the three `web/studio_*_profile_scratch.js` modules pending coordination with the ongoing whole-image baseline. A short scheduling-only change reached the shared file before the other worker's freeze request arrived; it was restored immediately and disclosed. Do not treat that whole-image run as a clean single-version timing baseline.

Next: apply these semantically identical host fixes after the image test stops; increase engine strip batching with exact ownership/sample retention; retain scene across batches; then move the actual original full BVH traversal into the common compiler rather than substitute handwritten geometry. The pure GLSL frontend still lacks general loop/storage lowering, so full-traversal compilation remains unfinished. Native C scalar float32 widening is a separate correctness boundary, documented in the existing shader report.

## Varied-ray numerical follow-up

A second distinct actual aperture-ray strip (the full-photo strip that initially stalled at7424samples) passes strict primitive identity after the optimized host fixes, taking76ms. One ray differs by12float32ULPs. The exact Wasm result is reproduced by separately rounded float32 cross/dot arithmetic; the exact GPU result is reproduced by fused multiply-subtract and fused-dot evaluation. A float64 reference lies closer to Wasm for this sample. This establishes a concrete contraction/rounding explanation, not a different triangle law. Receipt `rounding-report.json` records input triangle/ray and both matching arithmetic variants.

The engine owner tested a bounded32ULP acceptance gate with strict identity, then resumed using the original GPU records already obtained by its1:1validator; the failed strip now completes with bitwise image parity. This is not a claim that32ULPs universally bounds every ill-conditioned intersection. Larger disagreements must remain explicit failures.

## Reviewed promotion

After the whole-image test stopped, the three optimized runtimes were promoted to shared source, not deployed by this worker. Review found no numerical blocker and requested elapsed-budget yield checks inside slab/leaf groups as well as between node waves; these were added so a large leaf cannot monopolize the UI task indefinitely. The 8ms threshold is a cooperative target, not a hard deadline for an individual group.

Final repeated-fixture medians with these extra checks:256rays58.77ms,1024rays162.43ms,4096rays545.47ms. Eighteen targeted tests pass. The engine owner is now testing4096distinct aperture rays and the entire image; these results must be reported separately from the repeated-fixture throughput profile.

## Distinct rays and compiled candidate batches

The actual4096-distinct-aperture-ray fixture now has a three-run profile: median480ms for the optimized scalar-call runtime. One of4096primitive identities differs from native GPU because the original barycentric tolerance admits both neighboring coplanar triangles near a shared edge; their distances differ1ULP. Float64 reference confirms identical plane distance and one candidate barycentric coordinate -4.8e-6, within the original -1e-5 tolerance. Do not report strict full GPU primitive parity. The engine validates the alternative candidate explicitly before selecting canonical GPU records; no blind ID relaxation is warranted.

The compiler worker's new candidate-batch artifacts are integrated through `studio_compiled_law_batch.js`, enabled explicitly with `compiledBatches:true` on the Wasm runtime. Each law batch runs one exported call containing a compiler-generated counted loop. Host BVH traversal remains unchanged. The runtime verifies original source identity and artifact hash, lays out typed SoA buffers after compiler-reserved memory, and checks count/capacity. It does not call Wasm for each scalar candidate.

On the actual4096distinct rays, JS/Wasm crossings fall from2,028,312 to1,788. Exact scalar-versus-compiled-batch result equality passes for every field, including the known GPU rounding ambiguity. Reused input feeds reduce cumulative299MB allocation to344KB scratch. Median query time473ms, with approximately31ms in compiled-law copy/call wrappers; host traversal and cooperative yields still dominate. This intermediate is therefore a crossing/ABI improvement, not a demonstrated large end-to-end speedup. Twenty-four combined tests pass. Full original traversal compilation is a separate ongoing compiler task.

Receipts: `distinct4096-profile.json`, `distinct4096-compiled-batch-profile.json`, `shared-edge-report.json`; exact runtime parity harness `verify_batch_runtime.mjs`, all under the existing build evidence directory. Additional optional runtime dependencies must be copied with the main JS runtime during any coordinated release; this worker did not publish them.

## Whole original traversal in one Wasm execution

The structured frontend and compiler batch owner now emit the actual original `nearest_filtered` source, its stack/loops and original helpers as a single Wasm batch. `studio_traversal_compute.js` loads verified artifacts, retains native-packed geometry by primitive owner, and invokes the exported function once per batch. It records explicit sphere-count0, region-filter-1 and boundary-only-false specialization. Original server sphere/partition/transport remain outside this candidate-only kernel.

A fair experiment reuses the SAME4096distinct captured rays, not regenerated RNG rays: one compiled batch83.50ms;16compiled batches of256rays81.48ms; previous host traversal504.34ms. Every output field is exactly equal across all three paths. Thus removing JS traversal is the demonstrated improvement; merely enlarging a compiled batch is not shown to improve throughput here. The native GPU shared-edge ambiguity remains as documented, not erased.

Caller near clipping is fail-closed: when the original global-nearest result is excluded by caller tMin, the runtime rejects rather than report a miss that could hide a farther admissible hit. A test verifies this rejection. Scene shapes are explicitly bounded by compiled node/triangle capacity. This experimental runtime is staged, not activated in the live worker.

Receipt `studio-region-compute/full-traversal-fair-profile.json`, harness `verify_full_traversal.mjs`. Full nested-CFG WebGPU emission is still being implemented; no claim that Wasm proof demonstrates WebGPU execution.
