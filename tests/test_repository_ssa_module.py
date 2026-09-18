from __future__ import annotations

import json

import numpy as np

from src.compiler.repository_ssa_module import assemble_repository_ssa_module
from src.compiler.ssa_wasm_backend import prepare_wasm_core_execution
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


def test_one_ssa_module_owns_native_wasm_and_webgpu_under_one_abi(tmp_path):
    left = SSAValue(0, "float32")
    right = SSAValue(1, "float32")
    result = SSAValue(2, "float32")
    function = Function("add", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Add", [left, right], result),
            Instr("Ret", [result], None),
        ]),
    })
    function.metadata["argument_names"] = ("left", "right")
    function.metadata["output_names"] = ("sum",)

    assembly = assemble_repository_ssa_module(
        IRModule({function.name: function}),
        function.name,
        entry_name="add",
    )
    manifest_path = assembly.write(
        tmp_path, compile_native=False, emit_diagnostic_shell=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert tuple(item["name"] for item in manifest["abi"]["inputs"]) == (
        "left", "right",
    )
    assert [item["name"] for item in manifest["abi"]["outputs"]] == ["sum"]
    realizations = manifest["realizations"]
    assert set(realizations) == {
        "windows-native", "webassembly", "webgpu-compute",
    }
    assert {
        item["abi_digest"] for item in realizations.values()
    } == {manifest["abi_digest"]}
    assert (tmp_path / "add.c").is_file()
    assert (tmp_path / "add.wasm").is_file()
    assert (tmp_path / "add.compute.wgsl").is_file()
    diagnostic = tmp_path / "add_diagnostic.html"
    assert diagnostic.is_file()
    html = diagnostic.read_text(encoding="utf-8")
    assert assembly.abi_digest in html
    assert "WebGPU compute" in html

    execution = prepare_wasm_core_execution(assembly.wasm_artifact, {
        0: 2.0, 1: 3.0, 2: np.zeros(1, dtype=np.float64),
    })
    try:
        execution.run()
        assert execution.buffers[2].tolist() == [5.0]
    finally:
        execution.close()


def test_graphics_shader_members_are_copied_into_the_same_module(tmp_path):
    shader = tmp_path / "source.vert.glsl"
    shader.write_text("#version 330 core\nvoid main() {}\n", encoding="utf-8")
    value = SSAValue(0, "float32")
    function = Function("identity", [value], {
        "entry": BasicBlock("entry", [Instr("Ret", [value], None)]),
    })
    function.metadata["argument_names"] = ("value",)
    function.metadata["output_names"] = ("value",)
    assembly = assemble_repository_ssa_module(
        IRModule({function.name: function}), function.name,
        entry_name="identity", graphics_shaders=(shader,),
    )
    output = tmp_path / "module"
    manifest_path = assembly.write(output, compile_native=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = manifest["realizations"]["graphics-shaders"]["files"][0]
    assert member["path"] == shader.name
    assert (output / shader.name).read_bytes() == shader.read_bytes()
