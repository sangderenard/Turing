import json
import os
from pathlib import Path
import subprocess

import pytest

from src.compiler.native_fortran_display import (
    compile_image_probe,
    compile_voxel_mac_display,
)
from src.compiler.ssa_fortran_backend import fortran_compiler


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_compiler_fortran_probe_paints_non_black_rgb_through_common_shell(tmp_path):
    directory = tmp_path / "probe"
    artifact = compile_image_probe(directory, width=16, height=8)

    # Launch outside the artifact directory, as Explorer/direct launch does.
    # The common shell must locate its initial arena beside the executable.
    completed = subprocess.run(
        [artifact.executable_path, "1"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"),
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(completed.stdout)

    assert payload["frames"] == 1
    assert payload["outputs"]["red"]["sum"] > 0.0
    assert payload["outputs"]["green"]["sum"] > 0.0
    assert payload["outputs"]["blue"]["sum"] > 0.0
    source = artifact.executable.c_source_path.read_text(encoding="utf-8")
    assert "turing_open_artifact(argv[0]" in source
    assert "module fortran_image_probe_fortran" in artifact.module.source
    assert "subroutine fortran_image_probe(" in artifact.module.source
    compiler = Path(fortran_compiler())
    objdump = compiler.with_name("objdump.exe")
    if objdump.is_file():
        imports = subprocess.run(
            [objdump, "-p", artifact.executable_path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.casefold()
        assert "libgfortran" not in imports
        assert "libquadmath" not in imports
        assert "libgcc_s" not in imports
        assert "libwinpthread" not in imports


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_compiler_fortran_voxel_mac_advances_preallocated_fluid_arenas(tmp_path):
    artifact = compile_voxel_mac_display(
        tmp_path / "voxel", width=12, height=8, pressure_iterations=3,
    )

    completed = artifact.run(frames=2, capture_output=True)
    payload = json.loads(completed.stdout)

    assert payload["frames"] == 2
    assert payload["outputs"]["next_time"]["first"] == pytest.approx(0.008)
    assert payload["outputs"]["next_dye"]["sum"] > 0.0
    assert abs(payload["outputs"]["next_pressure"]["sum"]) < 1.0e-10
    assert payload["outputs"]["red"]["sum"] > 0.0
    assert payload["outputs"]["green"]["sum"] > 0.0
    assert payload["outputs"]["blue"]["sum"] > 0.0
    assert "subroutine voxel_mac_rgb_step(" in artifact.module.source
