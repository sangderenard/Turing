"""N-frame parity of authored symbolic programs against C, LLVM and Fortran.

Each program compiles in seconds; nothing here rebuilds a vehicle.  The
programs run closed-loop: every backend feeds its own outputs back, so the
numbers below are accumulated drift over the whole run, not one frame.
"""

from __future__ import annotations

import pytest

from src.compiler.ssa_fortran_backend import fortran_compiler
from tools.frame_parity import run_parity


FRAMES = 48
BACKENDS = ("python", "c", "llvm") + (("fortran",) if fortran_compiler() else ())


@pytest.mark.parametrize("program", ("roller_fixture", "member_material"))
def test_symbolic_program_frames_agree_across_backends(program, tmp_path):
    report = run_parity(program, frames=FRAMES, backends=BACKENDS, directory=tmp_path)
    assert report["feedback"], "closed-loop parity needs at least one fed-back state"
    assert "c" in report["backends"] and "llvm" in report["backends"]
    for backend, row in report["backends"].items():
        assert row["non_finite"] == 0, (backend, row)
        # float64 products of the same SSA against the sympy authority over
        # 48 fed-back frames: ULP-level (measured ~1e-14 relative; the
        # material's fracture gate was the one thing that broke this and it
        # now uses an exact Max, see vehicle_mechanical_material._positive).
        assert row["max_rel_error"] < 1.0e-12, (backend, row)
    # C and LLVM are two spellings of one SSA through one toolchain.
    assert report["backend_vs_backend_max_abs"]["c/llvm"] < 1.0e-12, report


def test_symbolic_program_cache_serves_the_parity_programs():
    """The fixture and material compiles must be persistent-cache hits."""

    from src.compiler.vehicle_mechanical_material import compile_vehicle_member_material_ssa
    from src.compiler.vehicle_native_deployment import compile_vehicle_roller_fixture_ssa

    for compile in (compile_vehicle_roller_fixture_ssa, compile_vehicle_member_material_ssa):
        compile.cache_clear()
        assert compile().cache_hit is True, compile.__name__
