"""The same spring equations run as a deployable GPU kernel and native row."""

import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.compiler.spring_network_law import (
    compile_spring_network_compute_shader, compile_spring_network_step,
)
from src.compiler.native_law_kernels import law_kernel
from src.compiler.shader_deployment import publish_spirv_product
from tools.build_spring_network_shader import build as build_shader


def _columns(count):
    compiled = compile_spring_network_step()
    values = {
        name: np.zeros(count, dtype=np.float64)
        for name in compiled.function.metadata["argument_names"]
    }
    values["occupied"][:] = 1
    values["position"][:] = np.linspace(-0.75, 0.75, count)
    values["velocity"][:] = np.linspace(-0.2, 0.3, count)
    values["neighbor_0"][:] = values["position"] + 0.1
    values["neighbor_1"][:] = values["position"] - 0.1
    values["edge_0"][:] = values["edge_1"][:] = 1
    values["stiffness"][:] = 10
    values["inverse_mass"][:] = 1
    values["drag"][:] = 0.2
    values["dt"][:] = 0.01
    values["external_acceleration"][:] = -9.81
    values["pinned"][1] = 1
    values["anchor"][1] = 2
    values["occupied"][2] = 0
    values["contact_enabled"][3] = 1
    values["contact_height"][3] = 0.5
    return compiled, values


def test_spring_shader_product_validates(tmp_path):
    shader = compile_spring_network_compute_shader()
    assert shader.complete, shader.shortfalls
    assert "Float64" in shader.capabilities
    assert len([item for item in shader.bindings if item.role == "feed"]) == 24
    product = publish_spirv_product(tmp_path, shader)
    assert product.module_path is not None
    assert product.deployment["target"]["client"] == "vulkan"
    direct = build_shader(tmp_path / "direct")
    assert direct["glsl_emitter"].endswith("emit_ssa_scalar_to_glsl_compute")
    assert (tmp_path / "direct" / direct["glsl_compiled_spirv"]).is_file()
    assert (tmp_path / "direct" / direct["resident_spirv"]).is_file()
    resident=(tmp_path / "direct" / direct["resident_glsl"]).read_text()
    assert "feed17.data[element - control.row_stride]" in resident
    assert direct["resident_work_contract"]["steps_per_loadout"] == 4


@pytest.mark.parametrize("shader_source", ["ssa_spirv", "direct_glsl"])
def test_spring_shader_device_matches_native_row(tmp_path, shader_source):
    probe = os.environ.get("TURING_VULKAN_COMPUTE_PROBE")
    if not probe or not Path(probe).is_file():
        pytest.skip("set TURING_VULKAN_COMPUTE_PROBE to the repository Vulkan probe")
    count = 128
    compiled, values = _columns(count)
    shader = compile_spring_network_compute_shader()
    product = publish_spirv_product(tmp_path, shader)
    assert product.module_path is not None
    shader_module = product.module_path
    if shader_source == "direct_glsl":
        direct = build_shader(tmp_path / "direct")
        shader_module = tmp_path / "direct" / direct["glsl_compiled_spirv"]
    names = dict(zip(
        (argument.id for argument in compiled.function.args),
        compiled.function.metadata["argument_names"],
    ))
    files = []
    outputs = {}
    for binding in shader.bindings:
        path = tmp_path / f"binding_{binding.binding}.bin"
        if binding.role == "feed":
            data = values[names[binding.value_id]]
        elif binding.role == "output":
            data = np.zeros(count, dtype=np.float64)
            outputs[binding.binding] = path
        else:
            data = np.array([count], dtype=np.uint32)
        path.write_bytes(data.tobytes())
        files.append(path)
    result = subprocess.run(
        [probe, str(shader_module), str(math.ceil(count / shader.local_size[0])),
         "1", "1", *(str(path) for path in files)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    native = law_kernel(compiled, "spring_network_step", count, "llvm")(values)
    for binding, name in zip(sorted(outputs), ("position_next", "velocity_next")):
        observed = np.frombuffer(outputs[binding].read_bytes(), dtype=np.float64)
        np.testing.assert_allclose(observed, native[name], rtol=1e-12, atol=1e-12)
