2026-11-11 STUDIO TILED GPU RECEIPT (MEASURED, Windows):

Requested default scene: 8192x8192 output, 2x linear oversampling, 16384x16384
sensor image, sixteen 4096x4096 windows, 30 exposure seconds per window.
Submitted through the existing authenticated PhotoJobs owner, no second renderer.
Digest: 3227e23ebf0b1c41410b2f0db8c1653b05c82fadc7c1f5afd487f426483b2895.
MRP1 SHA256: 4e6eeb63d881b2bd13cd74b62ef7f2c2b2157dc67cd0276e4b872786ee014413.

The real 4096x4096 sensor allocated 805306368 texture bytes; driver reported
GL_MAX_TEXTURE_SIZE=32768. The default effective scene had 98912 triangles,
one analytic sphere and 4464 emissive root samples. The first window completed
three full sensor passes and wrote its PNG/HDR. A supervisor restart by another
operator interrupted that first job; the retry used the normal same owner.

The retry committed six windows, then Windows Application event 1000 at
2026-11-11T06:02:16Z reported worker PID 79492 (0x13684) crashing in
nvoglv64.dll 32.0.16.1088, exception 0xc0000409. Status/log stopped during window
index 6. No completed 8K image exists. A successful allocation/first window is
not evidence that the entire 4K-window run is stable.

Before that crash the user superseded the large render with 1024x1024 output,
2x oversampling and four 1024 windows. Normal cancellation/new submission was
blocked by the duplicate-claim account transaction documented in
20261111T055639Z_photo_duplicate_owner_transaction.md. No cancel POST executed,
no ownership gate was bypassed, and no small test render has yet been submitted.
Its verified default-scene MRP1 is prepared under build/dot-studio-agent-20261111.

Integrated collation/explicit-camera-pose/tile/account/ownership tests:
75 Turing Python tests, 57 Spectral CPU tests, Jobs Node checks and camera-pose
Node checks passed. These are not a replacement for the pending smaller GPU test.
New Jobs grouping/pose controls need activation after account-service recovery.
