# Python, native and browser deployment survey — 2026-11-09

## Answer first

The compiler-to-browser pipeline is substantial and already carries real products. It is not a universal automatic migration of the Python application. There are several distinct, working lanes, with different frontends, ABIs, numeric contracts and limits. The current Marble Studio ray tracer is still a server-side native/OpenGL worker; its browser receives developed images. A small client intersection product is a credible next vertical slice, not an existing switch.

The best initial ray harvester is a batched, resident-scene nearest-hit query: rays in, distance plus surface identity and hit coordinates out. It must reuse the original geometry/intersection owners and compiler contracts. It should not become a second hand-written JavaScript renderer or silently substitute a depth image for hidden refractive boundaries. Measure full transfer and scheduling cost before deciding it should feed a server bounce loop.

## Scope and evidence

Read current local Turing and Spectral Analyzer sources on DESKTOP-LCKQH36, prior Wasm handoffs, the Meowdoku web report and the Nodus service-backbone audit. Turing HEAD observed f998b80fff3bfefa2eb6a444cb1f4c623d95e304; Spectral HEAD cff63425be11e9a1b97987f623b729a7bd2c2f7d. Both have extensive concurrent uncommitted work. These hashes are orientation, not exact release identities. This report edits notes only, does not launch rendering, replace products, restart services, push or deploy.

Historical reports are evidence of prior work, not fresh acceptance. Fresh checks below are explicitly separated from source inspection and recommendations.

## The actual lanes

### 1. Whole authored Python to repository SSA to native or direct Wasm

The sanctioned public source entry is `src.compiler.fortran_c_shell.lower_ast_source_to_ssa` (currently near line 67734). Concordance and extraction contracts remain authoritative. C, LLVM and Fortran are native backend targets; a native DLL is not itself a browser artifact.

There is a direct repository-SSA Wasm backend (`src/compiler/ssa_wasm_backend.py`, `emit_ssa_module_to_wasm_core`, near 692), with a real binary writer, explicit supported operations/shortfalls, scalar and buffer publication. `examples/meowdoku/entity_motion.py:1134–1183` uses the sanctioned source entry for the actual authored/printed laws, then writes Wasm and an ABI. The browser `TuringWasmProgram` consumes the descriptor instead of rewriting the law in JS.

Fresh proof: `tests/test_wasm_core_product.py::test_the_page_shell_binding_runs_the_module_from_bytes` passed in 6.71 seconds. This test lowers an authored batched drift law, emits a new Wasm core, loads it with the page binding under Node and checks results. This establishes a real end-to-end vertical slice, not arbitrary Python or an optics kernel.

### 2. AbstractTensor graph forward to native LLVM or LLVM-targeted Wasm

`examples/meowdoku/graph_wasm.py:66–109` traces the existing law through SSATensorProgram and records named IDs, shapes, dtypes and extents. `src/compiler/llvm_training_runtime.py:260` takes the same traced forward used by native execution, re-emits it for `wasm32-freestanding`, then invokes installed Zig/LLVM with `-nostdlib -O2` (191–223). Libm imports are supplied by the existing host Math table. This is not a DLL-to-Wasm conversion.

Products include `law.wasm` and `pilot.wasm`; the generic `TuringGraphProgram` uses the buffer-table ABI. `native_forward.py` applies the native graph path to chess inference with current weights as inputs. Reverse/training is a separate frontier; the Nov 5 report records later ownership/evaluation-point refusals. I did not rerun full reverse compilation and do not assert those historical refusals are still current.

### 3. Tensor/SSA to browser WGSL compute

`fountain_build.py:88–169` traces the authored law, calls `ssa_webgpu_backend.emit_tensor_program_passes`, records a deployment cost verdict and emits explicit arena, slots, bindings, dispatches and uniforms. This is genuine generated compute, not handwritten fountain physics in WGSL. The WebGPU lane declares the fast work contract and f32 legalization; Wasm uses develop. That distinction must remain visible for glass geometry and numerical parity.

`web/compute.js` checks adapter/buffer limits, keeps the arena resident, executes generated dispatches, records state and uses Wasm only when the product exists. It already budgets bounded work rather than tying simulation ticks to display frames. WebGL2 presentation cannot consume its WebGPU buffers directly: it packs 20 bytes per displayed particle and copies them back, then uploads for WebGL presentation. This real seam is both a reusable pattern and a transfer cost.

Fresh inventory of current served-directory manifests: eight fountain-family manifests, 62 size records in total; only two advertise Wasm products. Most size records require WebGPU in practice, even when the cost verdict prefers Wasm, because the fallback was not built. Extra old .wasm files on disk are not proof of an advertised compatible fallback. The current source comments say some size-specialized Wasm builds can take minutes; no large rebuild was run here.

### 4. GLSL project ingestion and browser graphics

`src/compiler/glsl_project_ir.py` ingests authored shader structure/resource declarations into project IR and lowers read-only scalar SSBOs to texture-backed WebGL2 reads. `web_export.py:55` uses it for the actual Pluck base-material shader. It is a working graphics portability route.

The planner labels some functions compute-candidate, but its own comment at 541 explicitly says extracting a compute pass still requires cross-stage transport/reuse proof. This is not evidence that the entire native ray pipeline can already be translated to WGSL automatically. Desktop OpenGL compute/SSBO assumptions, append queues and atomics need a declared browser deployment, not a relabelled shader.

## Fresh verification and a current gap

- Current sanctioned AST → direct Wasm → Node page-loader test: 1 passed.
- Node validates published law, pilot and ballistic modules. Their sizes are 1,280,653; 10,044; and 3,648 bytes. Actual Wasm magic/version is correct.
- Invoked current ballistic product through TuringWasmProgram: x=0, vx=2, dt=.1, zero gravity/air gave x=.2 and vx=2. Speed/energy include the authored tiny regularization. This is a smoke, not a complete physical parity suite.
- Current legacy test `test_ast_generated_float32_program_emits_wgsl_compute` fails in 2.83 seconds. Its helper uses deprecated `compile_ast_aot`; direct inspection of its artifact reports `WGSLShortfall(function='kernel', operation='outputs', reason='compute module has no named output')`. This local failure is a measured test-path gap, not a diagnosis of all WebGPU products. No compiler patch was attempted and no clean-tree baseline was run.
- A broader selected WebGPU/Wasm test run produced failures and was interrupted before completion to avoid a long survey-side compilation. No total pass/fail claim is made for that run. No GPU/browser hardware performance benchmark was run.

Older August Wasm documents describe working class-memory coordinators and worker splitting, but also warn about collective reductions: splitting a collective graph into independent element tiles previously changed semantics. Their old speedup was explicitly invalid. Reuse the extent-effect/independence proof, not the timing.

## Current Studio and the harvester boundary

`web/marble_live.js:124,175` posts the saved photo request and fetches developed image frames. It does not trace rays in the browser. `marble_photo.py:3–7,289–336` authenticates MRP1 against the immutable saved object: positions and normals are f32, IDs u32, geometry digest and count validated. MRP1 carries 76 bytes per triangle before config/fixtures. A 95,000-triangle mesh is about 7.22 MB in that wire form; a compact BVH representation is a different artifact and must have its own verified identity.

Spectral already owns BVH/AABB/triangle intersection (`csrc/shaders/ray_bvh_intersect.comp.glsl:185–239`), exact parametric sphere kernels, and the new opt-in region boundaries. The worker's composed shader is not identical to the full native T1 pipeline. Existing fixed thresholds in either lane must not be copied blindly into a new client kernel, especially with millimeter marbles and thin bubbles.

A useful hit result requires more than raw distance: validity, ray ID, scene revision, exact surface/region identity, distance in declared units, barycentric or parametric coordinates, geometric normal/orientation, and explicit overflow/unresolved-root status. Otherwise the server cannot safely apply the correct material, boundary ownership or spectral attenuation. Reflection and refraction depend on geometric normals, not merely display normals.

### A bounded next experiment

1. Select one existing authoritative intersection family (start with exact spheres or the existing triangle/BVH owner), keeping original source/concordance and numerical contract. Do not reimplement it in JS. Establish which sanctioned frontend can ingest that specific source. No completed client ray compilation has been demonstrated by this survey.
2. Publish a compiler-built product and ABI for a bounded ray batch; load it with the existing Wasm shell first. A worker owns the persistent geometry/BVH buffers. Compile WGSL from the same declared law only if the emitter admits it without omitted operations and numeric policy is explicit.
3. Compare actual nearest hits with the existing server oracle: miss, inside origin, grazing, two roots, tiny features, transformed objects, tie/coincident ownership, dynamic revision and exhausted budgets. Keep optical material processing on the established path initially.
4. Benchmark the whole operation: setup/upload, queue wait, execution, result readback, server transfer and revision churn. Include a browser without WebGPU. Do not claim desktop GPU measurements cover iPhone.
5. Only then choose deployment. Local camera-ray visibility/depth/picking and preview diagnostics are natural initial consumers. Remote batched intersection assistance is possible, but one network round-trip per bounce is a poor default. Geometry resident plus larger coherent batches is more plausible than streaming every individual distance.

Illustrative bandwidth, not a measurement: a 32-byte ray request and 32-byte result at one million rays is 64 MB per bounce round-trip before transport framing. Multiple bounces can dwarf the math saved. If all rays originate on the client and hits stay there, much of that cost disappears, but then the next material/queue stage also needs a browser product. This is why a small harvester is a good proof slice and not automatically the final responsibility split.

## Ranked opportunities

1. **Use the existing product system, not per-page numerical glue.** Add an optics hit product through the compiler-authored ABI/manifest, bounded runtime and identity checks. This is the most direct answer to the requested client distance work. Feasibility high for a narrow family; end-to-end optics product unproven.
2. **Move image development closer to display.** Existing pure color transforms, histogram and direct calibration pattern are plausible compiled client products over an occasional raw linear buffer, avoiding repeated server image development/encoding for every slider. Needs raw image provenance, precision, memory and frame cadence contract. No such endpoint/product was built in this survey.
3. **Keep scene/BVH and ray queues resident.** Avoid full mesh/config resend and readback of intermediate data. Use existing digest/revision and buffer ownership rather than another cache scheme. Measure local vs remote data costs before distributing bounces.
4. **Expose existing deployment evidence.** The source digest, ABI, precision contract, actual selected backend, fallback availability and refusal should be visible per product. Current cost verdicts are models, not universal measured speedups. They must include transfer, memory and compilation amortization for ray work.
5. **Finish missing fallback products selectively.** Current manifests have little CPU/Wasm coverage. Choose small high-value sizes by device needs, not an indiscriminate long build. Retained-product restart already protects verified artifacts from unfinished source rebuilds.
6. **Do not move session/service semantics first.** The prior Nodus audit identifies useful native/CPython row and FIFO mechanisms but also scheduler/ownership/persistence questions. It is a different responsibility boundary from a pure intersection kernel. Those old source findings need remeasurement before being called current defects.

## What is still bespoke

The application still has substantial hand-authored JS for scene controls, geometry generation, camera/UI state and orchestration. Some numerical helpers are compiler products; others remain JS. Server HTTP/session/product build ownership stays Python. The ray worker composes existing native/GLSL components and local adapters; it is not a product emitted end-to-end from the Python application. This is not inherently wrong, but it must not be described as universally translated.

The priority audit follow-up is a responsibility inventory per callable/component: authored owner, sanctioned compiler entry, admitted targets, ABI, state authority, source/product digest, tested parity, data crossing cost and honest refusal. Start with nearest-hit queries and display development rather than attempting to lower the entire Studio server at once.

## Prior records consulted

- `docs/WASM_SHELL_HANDOFF.md` (August historical source, not current gate).
- `docs/WASM_CLASS_COORDINATOR.md`.
- `docs/WASM_STATIC_COLLECTIVE_DEPLOYMENT_HANDOFF.md`.
- `docs/MEOWDOKU_WEB_2026-11-05.md`.
- `docs/NODUS_MEOWDOKU_SERVICE_BACKBONE_AUDIT_2026-11-07.md`.

No universal compiler readiness percentage or performance gain is supported by this survey. The strong result is that real native/browser product lanes already exist, and a narrow geometry product can be pursued through them without replacing the optics engine.


## Follow-up: original shader conversion and bounded client query

The existing GLSL-to-WebGPU route is GLSL source ingestion to SSA followed by the WGSL emitter, not a full unrestricted text converter. Passing the current Spectral `csrc/shaders/ray_bvh_intersect.comp.glsl` to `lower_glsl_source_to_ssa` returned `complete=False`, with `GLSL_LVALUE` (only direct-name assignments supported) and `GLSL_OUTPUT` (no declared output) at line 153. These are frontend failures, not an exhaustive assessment of the remaining shader or its resource bindings. No complete original compute-shader WGSL conversion is demonstrated.

A smaller original kernel does work through the existing GLSL project-to-WebGL compiler. `examples/meowdoku/client_ray_build.py` embeds `SphericalSurface.glsl_intercept_fn` verbatim, wraps it in a bounded resident-sphere batch interface, and compiles the shader pair with `compile_glsl_project_to_webgl`. `web/client_ray_kernel.js` verifies compiled artifact hashes, validates float32 records and scene revisions, and returns distances, owner IDs, validity, and face-forward normals. The implementation has no JavaScript replacement intersection arithmetic. Bounds: 65 exact spheres, 4096 rays per dispatch. Other studio geometry is not silently approximated by this interface.

Validation on this machine: builder pytest 1 passed; Node contract checks passed; isolated headless Chrome WebGL compiled, dispatched, fenced, and read back original-kernel results. Outside hit distance 2, inside exit distance 1, miss, and scaled-radius/nonunit-direction hit distance 2 matched expectations; stale scene was rejected. Receipt: `build/dot-client-ray-20261109/gpu-result.json`. One four-ray query took about 55.8ms including first dispatch/readback; this is not a throughput benchmark or evidence of server savings. Float32 grazing intersections, very small features, full meshes/BVH, and optical transport remain outside this validation.

This backend alone is not distributed rendering. An opt-in UI must perform a useful clearly labeled local geometry diagnostic; it cannot say it reduces server load. Actual participation requires a server-issued scene/kernel-bound task lease, original work queue integration, stale/duplicate result rejection, trusted validation policy, and measured reduction in local server dispatches. The current forward spectral shader has no such boundary. Full shader reuse should extend the compiler's real storage/control-flow lowering and resource ABI rather than replace transport mathematics.

## Updated implementation result: actual Studio law, common Wasm/WebGPU

The full-shader GLSL frontend limitation was not treated as fundamental. A bounded pure-function extension now ingests the *actual active Studio* hit_tri into common SSA, with original text provenance. The two existing compute backends now execute that same law: Wasm float32 and WebGPU compute. See `docs/hazards/20261109T210000Z_studio-triangle-shared-compute.md` for files, tests, artifact hashes and limits. Exact 256 real sensor-launch rays were tested, with identical hit identities on both routes, WebGPU distance max delta zero and Wasm max delta 2.38e-7. This supersedes the earlier sphere-only feasibility boundary, but does not establish whole-shader translation or large-scene default offload. A separate retained-ray continuation and relay integration is required and is being verified against the actual sensor image.
