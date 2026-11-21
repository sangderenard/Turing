"""Native and GPU qualification of the independent 3D wave stencil."""

import shutil
import subprocess

import numpy as np
import pytest

from src.compiler.wave_volume_3d import (
    compile_volume_wave_compute_shader,
    compile_volume_wave_kernel,
    volume_wave_cfl,
)


def test_center_impulse_propagates_to_all_six_3d_neighbors(tmp_path):
    grid = np.zeros((5, 5, 5), np.float64)
    grid[2, 2, 2] = 1.0
    interior = np.s_[1:-1, 1:-1, 1:-1]
    samples = {
        "current": grid[interior],
        "previous": grid[interior],
        "x_minus": grid[:-2, 1:-1, 1:-1],
        "x_plus": grid[2:, 1:-1, 1:-1],
        "y_minus": grid[1:-1, :-2, 1:-1],
        "y_plus": grid[1:-1, 2:, 1:-1],
        "z_minus": grid[1:-1, 1:-1, :-2],
        "z_plus": grid[1:-1, 1:-1, 2:],
    }
    batch = 27
    for name in ("dx", "dy", "dz"):
        samples[name] = np.ones((3, 3, 3), np.float64)
    for name, value in (("dt", 0.5), ("speed", 0.5),
                        ("damping", 0.0), ("source", 0.0)):
        samples[name] = np.full((3, 3, 3), value, np.float64)

    assert volume_wave_cfl(0.5, 0.5, 1.0, 1.0, 1.0) < 1.0
    piece = compile_volume_wave_kernel(batch, directory=tmp_path)
    arguments = tuple(np.ascontiguousarray(samples[name].ravel())
                      for name in piece.argument_names)
    (result,) = piece(*arguments)
    volume = np.zeros_like(grid)
    volume[interior] = result.reshape(3, 3, 3)
    assert volume[2, 2, 2] == pytest.approx(0.625)
    for axis in range(3):
        for direction in (-1, 1):
            index = [2, 2, 2]
            index[axis] += direction
            assert volume[tuple(index)] == pytest.approx(0.0625)
    assert np.count_nonzero(volume) == 7


def test_3d_wave_kernel_has_separate_valid_gpu_product(tmp_path):
    artifact = compile_volume_wave_compute_shader()
    assert artifact.complete, artifact.shortfalls
    assert artifact.name == "wave_volume_3d_step_f32"
    assert artifact.bindings and artifact.component_abi
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        pytest.skip("SPIR-V assembler/validator unavailable")
    source = artifact.write(tmp_path)
    binary = tmp_path / (artifact.name + ".spv")
    assembled = subprocess.run(
        ["spirv-as", "--target-env", artifact.target_env, str(source),
         "-o", str(binary)], capture_output=True, text=True,
    )
    assert assembled.returncode == 0, assembled.stderr
    validated = subprocess.run(
        ["spirv-val", "--target-env", artifact.target_env, str(binary)],
        capture_output=True, text=True,
    )
    assert validated.returncode == 0, validated.stderr


@pytest.mark.parametrize("bad", [
    (-1, 0.1, 1, 1, 1),
    (1, 0, 1, 1, 1),
    (1, 0.1, 1, 0, 1),
])
def test_invalid_3d_time_or_spacing_is_rejected(bad):
    with pytest.raises(ValueError):
        volume_wave_cfl(*bad)
