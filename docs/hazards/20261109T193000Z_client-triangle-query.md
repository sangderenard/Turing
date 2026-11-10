# Original triangle kernel browser query

The standalone triangle batch backend reuses `moller_trumbore` verbatim from Spectral `ray_bvh_intersect.comp.glsl`, including original T_SELF=1e-4 and determinant epsilon=1e-9. It compiles with the existing project-IR WebGL compiler. This deliberately does not claim watertight intersections or scale-independent robustness.

Build: `python -m examples.meowdoku.client_triangle_build --out <asset-directory>`.
Runtime: `web/client_triangle_kernel.js`, ClientTriangleQuery.create(assetURL), setScene({revision, triangles:[{v0,e1,e2,owner}]}), query(rays,revision). e1/e2 are edge vectors, not second/third vertices. Owner is a unique primitive identifier, not a shared material ID. Records retain the sphere ray ABI (origin xyz/min distance/direction xyz/max distance); directions normalized internally. Returned hits are distance/owner/valid/triangle index. Barycentrics are b0/b1/b2/front-facing sign. Both original kernel and requested interval must accept the hit. Degenerate/parallel triangles miss by the unchanged source law.

Scene is resident; queries reject stale revisions and overlapping dispatch. Artifacts are hash verified. Bound 4096 rays and 4096 triangles, brute-force O(rays*triangles), no BVH yet. Input unsupportedGeometry entries reject rather than disappearing; caller must enumerate unsupported scene objects rather than sending an incomplete scene as complete. This is not the full Studio optical renderer and is not wired as a replacement when server budget expires.

Validation: builder tests (sphere + triangle) 2 passed in 1.18s; Node sphere and triangle contract checks passed. Browser GPU validation is tracked separately in build/dot-client-ray-20261109/triangle-gpu-result.json only if the isolated browser run succeeds. Initial Chrome harness failed at startup, before executing the shader. No live deployment or server restart.

Required next integration: existing server continuation must retain all ray energy/material/medium/random state, issue scene+kernel-bound expiring leases, reject stale/duplicate results and consume exactly-once before retiring matching local work. Budget exhaustion must produce pending work, not lost rays or a fake local-success status. Current Studio monolithic forward transport cannot consume this geometry result without that real boundary.
