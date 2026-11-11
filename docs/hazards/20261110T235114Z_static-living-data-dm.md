# Static Living Data DM REST and original Phong (2026-11-10, Windows)

User narrowed this task to static map/Phong only, with no game engine.
Added owner-scoped authenticated static-map registration, analytic inspect and
render-phong commands. Browser host uses original registry, mesh builder and
Pluck Phong renderer; original model bytes are unchanged. Reversible adapter
suppresses physics, vehicle/navigation/audio startup and entity-cycle dispatch.
Static canonical map, geometry and shader are pinned; mutations latch refusal.
No source compiler or physics law was replaced. No deployment, live restart,
account grant outside temporary fixtures, network settings or public route.

Actual Chrome HTTP flow: login 200, register 201, inspect 200; two camera render
commands 202, host poll/complete 200, result retrieval 200. Actual returned PNGs
fully decoded. Zero browser Workers and no game Tables. All 103 registry objects
retained their analytic fields. Requested and normalized actual cameras are both
recorded, with exact position/dimensions. Original lighting is dark, preserved.

Final focused suite: static DM, static adapter, cursors, query, worldmap and issue
tickets: 53 passed plus 10 subtests, one cffi deprecation warning. Syntax checks
passed. Independent service review identified PNG verify-only acceptance; fixed
with bounded single-frame full decoding, corrupt-IDAT and APNG regressions.
Full repository suite not run. Receipt source provenance remains browser-host
reported, explicitly not independent server attestation of live game state.

Stage/PNG/REST receipts: build/dot-static-phong-rest-20261110.
API guide: docs/LIVING_DATA_STATIC_DM_API.md. This is the original mechanical
map, NOT the concurrent new storybook class. Future physical participation,
MCP, arbitrary DM mutations and shared player permissions remain outside scope.

Verified source SHA256: 297cda67f0c96cc2dada8278fcd1f559b279fd14ba3c8d31b62448e1f1fa82a1
Stage SHA256: adda958f8300259d639a049467981f6a63b2b65ab617aee79cdf3e650aec4725
Returned PNG 4ff93fd2037361f27db9c4f3907f642bc2d07d1e0998bb771c5e240b08e94073
Returned PNG 907411c2d8a94363f53597dd912639e2da656a36485e7cad792a072ecf2254b0
