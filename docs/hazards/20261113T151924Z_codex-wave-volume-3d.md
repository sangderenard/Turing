2026-11-13 SPECIALIZED 3D WAVE VOLUME LAW (MEASURED, Windows):
`src.compiler.wave_volume_3d` compiles a centered, damped 3D scalar wave
stencil from SymPy through the existing SSA/native and SPIR-V emitters. A
5x5x5 stationary impulse test updates its 27 interior voxels and verifies
the center and six axial neighbors. `tests/test_wave_volume_3d.py`: 5 passed;
the SPIR-V product assembled and validated. The shader was published under
`C:/Users/alber/nodus-build/wave-volume-3d-artifact/` in user build space.
This is a 3D *interior arithmetic* kernel with explicit seven-sample inputs;
grid indexing, boundary policy, persistent two-time-level state, and volume
rendering remain with its future host. It is separate from Spectral T4's
transverse angular-spectrum optical arena. The local test ran without the
Nodus tensor core shared library, so it does not establish state-library
integration or GPU execution by itself.
