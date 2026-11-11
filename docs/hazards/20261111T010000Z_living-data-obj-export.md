# Native Living Data OBJ export (2026-11-11, Windows)

Added meshSnapshot to the existing static host. Two reversible, unique seams
capture the actual uploaded Float32 material resources and existing appended
route/portal mesh ranges. Geometry realization and shaders remain unchanged.
Separate dynamic/solver draw buffers are refused. Actual Storybook browser
capture has 33 objects, 7,164 vertices, 2,388 triangles and 199 semantic spans;
zero Workers. Repeated capture is stable; mutation and restored-mutation are
both refused by the latched guard.

New serializer and CLI export exact world vertices/normals, object and semantic
group names, OBJ face materials and MTL. The sidecar retains the complete
original registry, transforms, native material arrays and mesh. Formats do not
preserve executable behavior or native optical/shader semantics. Source receipt
is host supplied, not independent attestation. No deployment, server restart,
network setting, production state, source compiler law or physics changed.

No existing OBJ exporter was found in Turing or the sibling Spectral camera
pipeline. The existing Spectral importer is reused for a compatibility test;
its intentional unit-radius normalization is checked explicitly. Exact native
position/normal/name round-trip is tested with an independent OBJ reader.

The repo .venv lacks yaml, so initial restaging failed before writing output.
System Python has the required dependencies and completed canonical staging.
First focused collection lacked examples on sys.path; new test now sets it.
One new test initially read UTF-8 output using the Windows default encoding;
explicit UTF-8 fixed the test. Final results are recorded below.

hazards_log.py still refuses the pre-existing malformed
20261109T_material-library-paused-checkpoint.md name. Latest dated receipts and
the legacy baseline were read directly. Full repository suite not run.

Final focused run: 19 passed, one existing cffi imp deprecation warning, in
5.88 seconds. Python syntax compilation and Node helper syntax check passed.
Actual native OBJ + MTL + manifest + README bundle validates all file hashes
and ZIP entries byte-for-byte. Bundle contains no hazard notes.
