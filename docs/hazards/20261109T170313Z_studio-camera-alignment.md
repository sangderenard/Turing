# Native camera housing follows the analytical thin lens

2026-11-09, canonical Windows checkouts, source-only pending coordinator publication.

Studio now uses the existing CameraPreset body, iris, sensor and disabled projector-back geometry at the actual sensor camera pose. The native 139 x 101 x 77 mm body dimensions stay in metres, independent of marble/rig scaling. CameraPreset optical +Z maps through (-screen-right, up, forward), preserving a proper rigid rotation and triangle winding. The sensor and housing use the identical extracted SensorAccumulator look-at basis, including vertical views. Finite iris front surfaces are at least one authored iris thickness (0.1 mm) behind the analytical pupil. No second refractive lens is added.

The opt-in native body port is radially resolved. Existing grid-centre culling sealed a 2 mm pupil inside 4.34 x 3.16 mm body cells. Default unrelated CameraDesigner construction is unchanged. Native pupil radius/blade rotation follows the current exposure pupil.

Live updates rebuild the native geometry/BVH at an accepted sensor-sweep boundary, retaining shared GL buffer identities and replacing geometry, BVH nodes/IDs, contexts, materials/shading and exact-sphere buffers together. Both forward and sensor counts/bounds update before the new view launches a ray. Old photon field and camera history are cleared; display-only updates neither rebuild geometry nor clear history. Orbit frames use the same update before constructing each sensor; still exposures do not reset. Full geometry/BVH preparation is synchronous, not a frame-rate guarantee. The original large saved scene's CPU geometry preparation measured 1.08 seconds.

Turing's normalized config carries server-owned camera_geometry_version=native-housing-behind-thin-lens-v1 to avoid completed-cache aliasing with old fixed-origin images. Worker effective config and per-image geometry metadata identify the current convention even for direct legacy configs.

Verification: 54 Spectral CPU tests passed, one optional GPU test skipped in pytest (10.43 s). Targeted Turing provenance/photo/live tests: 22 passed (1.67 s). Independent parent camera suite: 16 passed. CPU tests cover five arbitrary/vertical poses, determinant/winding, complete forward half-space clearance and sampled pupil rays, real side/back intersections, 10 um through 20 mm open pupils, native pupil polygon orientation, physical size/world source preservation, all buffer/bounds updates, atomic live versus display-only changes and real worker eight-frame orbit/still orchestration.

Explicit idle-GPU tests ran on RTX 3060. Both production forward and sensor intersection shaders proved 16 pupil misses and side/back reflected-direction hits at 0.1305/0.173 m, at the supplied user's pose and a distant vertical pose. Actual worker 128x128 still completed. Actual 16x16 RGB live accepted revision 1 with moved eye and pupil radius 2->4 mm, produced a finite history and stopped at the test epoch limit. All GL/source processes exited 0. Initial probe harness launched 16 sensor lanes for 18 rays; its two unwritten sentinels caused an assertion. The harness dispatch was corrected to two groups; renderer code was not changed for that failure.

The original canceled user's payload/config were read only. A separate 128x128 RGB/manual4 exposure completed 16 epochs, 131072 forward jobs, 2.078 s preparation and 0.344 s exposure. The former in-frame orphan box is absent; this bounded image is very noisy, not a converged quality comparison. Body reflection visibility is independently established by the actual shader hit tests. No Phong camera mesh was introduced; no browser preview geometry, material optics, frozen display profile or saved object bytes changed.

No server restart, static export, production API mutation, exposure cancellation, commit or push. Existing physics/product changes in shared checkouts remain owned by their other tasks. Publication/restart boundary belongs to the coordinator.

Receipts: spectral-analyzer/build/dot-camera-geometry-20261109/{gpu-receipt.json,source-sha256.json}; saved-user-scene/{config.json,status.json,preview-000016.png,exposure-000016.npz}. Run CPU gates with installed Python311 and PYTHONIOENCODING=utf-8; run tests/studio_camera_geometry_gpu_check.py directly only after a fresh exact process check and idle-GPU coordination.

## Prompt History

> Make sure the camera exists where the camera exists so it exposes right in reflections

> It will have to be behind the thin lens i think not on it
