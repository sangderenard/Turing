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

2026-11-14 REBASE RECEIPT (Windows, base `ac15a210`, branch head
`8968ba00`): the same five focused tests now give **4 passed, 1 failed**.
The native CPU qualification fails during the sanctioned
`lower_ast_source_to_ssa` entry in `publish_formal_actual_occurrences`:
`formal_actual_occurrence_concordance` has recorded operand 1 but proposes
operand 3 for `binary_scalar_double` in the planned 3D region. The GPU shader
product test and CFL validation pass. The earlier 5-pass receipt describes
the older compiler base, not this rebased branch. Do not land this branch as
fully green or change the SymPy wave law to silence the identity failure.
The unchanged existing `tests/test_symbolic_law_batch_materialization.py`
also gives 3 failures on this base (two `source_call_actual_formal`
concordance disagreements and a persisted-row gate). Thus the compiler
baseline has a broader identity problem; the 3D failure is not evidence that
the 3D stencil equation is wrong.
