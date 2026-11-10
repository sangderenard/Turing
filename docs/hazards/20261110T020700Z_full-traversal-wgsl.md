# 2026-11-10 FULL TRAVERSAL WGSL EMISSION

The existing WGSL backend now has emit_cfg_module for typed, source-derived
inlined CFGs. It reuses the backend scalar expression emitter. It emits a
per-invocation CFG dispatcher, simultaneous Phi edge copies, local fixed arrays,
indexed shared buffers, explicit typed casts and output stores. No BVH law or
function name is recognized by this compiler path. Precision obligations are
checked and unsupported operations fail explicitly. Emitted module text is
posted to the concordance with its source value cells.

studio_traversal_compute.py now emits both Wasm and WGSL for the same original
nearest_filtered source specialization. Eight WebGPU storage bindings cover
runtime item count, per-ray SoA columns, node count, three shared scene buffers,
primitive IDs and distances. The manifest records exact source-formal IDs,
typed layouts, capacities, workgroup dimensions and byte-level hashes.

Both plain and region actual-stage artifacts built successfully. Seventeen
focused source/control/batch tests passed in6.67s with this emission active.
These tests execute Wasm and inspect successful source generation; they are
NOT a WebGPU execution pass.

Native WebGPU runtime validation could not run: wgpu was absent and installation
was blocked by network policy at pypi.org:443. No alternate download route was
attempted. The user then explicitly requested immediate live publication without
waiting for a render test. Runtime/release workers have the emitted artifacts
and must retain explicit shader/device errors. No claim of tested GPU numerical
parity, GPU speed, or production activation is made by this compiler receipt.
