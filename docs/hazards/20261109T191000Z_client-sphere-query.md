# Original-kernel client sphere query

New isolated files: client_ray_build.py, web/client_ray_kernel.js, test_client_ray_build.py, test_client_ray_contract.mjs. No Studio deployment or server task handoff occurs in this change. Original Spectral sphere generator is compiled through existing project IR WebGL route. Runtime validates scene/ray ABI and rejects stale revisions; artifact SHA256 checked before shader compilation.

Checks: pytest builder 1 passed (0.31s), Node contract checks passed, isolated Chrome GPU hit/miss/inside/scale/stale checks passed. GPU receipt build/dot-client-ray-20261109/gpu-result.json. Full GLSL ray BVH source frontend is incomplete; no WGSL conversion claim. No speedup/server savings claim. Known limit: sphere-only float32 query, no general material/region transport and no grazing/small-feature certification. Preserve current Studio controls; UI integration belongs to coordinating worker.
