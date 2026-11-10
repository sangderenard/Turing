# 2026-11-10 COMPILED STUDIO CANDIDATE BATCHES

The original GLSL triangle/slab laws now have a single exported Wasm batch
entry built through the shared SSA module backend. `ssa_scalar_batch.py`
posts the batch contract, generated CFG/value identities and derivations into
the existing concordance. The numerical function is called, not reimplemented.
Shared scalar and span arguments can be broadcast unchanged to every item.
The source builder emits separately hashed original plain/region laws and WGSL
parallel kernels at capacity 16,384. Wasm uses a compiled scalar loop; SIMD is
not claimed. This is candidate batching, NOT complete compiled BVH traversal.

Measured failures repaired in `ssa_wasm_backend.py`:
- Posted float32 scene buffers were rejected by GEP/Load/Store layout admission.
  The backend now honors four-byte f32 addresses and loads/stores.
- The Node host executor ignored `buffer_dtypes` and packed buffers as f64.
  After enabling f32 emission this produced wrong values (25.60000041 instead
  of 0.2). It now follows the emitted ABI for inputs as well as outputs.
- Explicit f32/int32 casts are emitted directly, and boolean conversion compares
  f32 against f32 zero. A signed int32 return previously became 4294967294 for
  -2; its ABI widening is now signed, while bool remains unsigned.

Tests: source-derived triangle batch equals scalar execution exactly for 24
varying origins/nonunit directions/sizes, both hits and misses. Counts 0, -1
and above capacity never write. Shared scalar arguments and an indexed shared
f32 span execute correctly. CorrelationTable audit reports zero findings.
Final focused gate: 31 passed in 7.80s (batch/source triangle/source slab,
f32 storage/casts, multiblock). Additional int64 regression gate: 14 passed
in 15.37s after the signed-return fix. No full-suite claim. WGSL is emitted
and byte hashes checked; native WebGPU execution was not run (wgpu absent).

Files: examples/meowdoku/studio_batch_compute.py; src/compiler/ssa_scalar_batch.py;
tests/test_studio_compiled_batch.py; tests/test_wasm_float32_storage.py.
Build command: python -m examples.meowdoku.studio_batch_compute

Limits: no production restart/deployment, no claim of full original shader
transport compilation. Full traversal needs structured GLSL CFG/storage
lowering. Existing WGSL frontend emission does not yet support arbitrary
nested traversal CFG. The sibling runtime experiment measured exact parity on
4096 distinct rays, crossing count 2.03M to 1788; see runtime profile receipt.

## Full traversal follow-through

`glsl_control_ingestion.py` now lowers original composed `nearest_filtered`,
its slab/AABB/triangle helpers, local stack, indexed scene storage, while/for,
short-circuit conditions and out/inout into existing SSA CFG/Phi/storage ops.
Specialization, full16-f32 TriGeom/12-f32 Node layouts and uniforms are recorded
with the original source. The batch wrapper calls this complete traversal once
per ray inside Wasm; the host makes one exported call for the whole batch.

`studio_traversal_compute.py` builds the explicit triangle-candidate subset
(boundary_only=false, sphere_count=0, region_filter=-1). Analytic-region
competition and spectral transport remain in the server continuation. Original
near-distance law and initial best=1e30 are retained; arbitrary caller tMin
must not silently change this contract. No SIMD or whole-WebGPU claim.

Measured real saved stage:95,217 triangles,65,535 BVH nodes,256distinct captured
rays. Source lowering/emission0.99s,7,122-byte Wasm, one call11.29ms excluding
packing/startup. Primitive IDs all match the GPU fixture; maximum distance
difference1.4901161e-8. Result receipt is stage-full-traversal-result.json in
build/dot-client-ray-20261109/studio-region-compute. Larger distinct-ray/runtime
acceptance is coordinated separately, not claimed by this receipt.

Final targeted gate40passed10.75s across control/source/batch/float32 storage/
multiblock tests. Includes original right-first equal-distance ordering,
misses/nonunit rays/repeated batch execution, short-circuit inout, local stack,
source mutation and explicit unsupported-call rejection. The direct-return
identity test exposed a separate backend bug: returned scalar input IDs were
mistakenly skipped as if already buffers. They now receive an output pointer
without replacing the scalar formal's local binding. No production deploy.
