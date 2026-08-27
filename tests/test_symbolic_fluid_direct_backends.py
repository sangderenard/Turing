import json
from pathlib import Path
import shutil
import subprocess

import pytest

from src.compiler.ssa_c_backend import emit_ssa_function_to_c
from src.compiler.ssa_fortran_backend import emit_module as emit_ssa_module_to_fortran
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
from src.compiler.symbolic_fluid_model import compile_symbolic_fluid_step
from src.compiler.work_contract import set_active_contract
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


@pytest.fixture()
def deploy_contract():
    """The four-lane verification runs under ``deploy`` deliberately.

    ``deploy`` is the preset whose documented meaning is "inexact identity
    set, stable across hosts": the shared identity pass reduces every
    constant-exponent Pow before emission, so all four lanes receive the
    same SSA and none needs a private spelling. Under exact-only contracts
    scalar WASM has no pow instruction and refuses honestly instead.
    """

    set_active_contract("deploy")
    yield
    set_active_contract(None)


def _uniform_inputs(compiled):
    values = {name: 0.0 for name in compiled.function.metadata["argument_names"]}
    for name in values:
        if name.startswith("height_"):
            values[name] = 1.0
    values.update({
        "dt": 0.01,
        "dx": 0.25,
        "gravity": 1.0,
        "minimum_height": 1.0e-4,
    })
    return values


def test_reference_directed_cast_is_repository_ssa_in_all_four_lanes():
    value = SSAValue(0, "float64")
    reference = SSAValue(1, "int32")
    result = SSAValue(2, "int32")
    function = Function(
        "cast_like_kernel",
        [value, reference],
        {"entry": BasicBlock("entry", [
            Instr("CastLike", [value, reference], result),
            Instr("Ret", [result], None),
        ])},
        metadata={
            "argument_names": ("value", "reference"),
            "output_names": ("result",),
        },
    )
    module = IRModule({function.name: function})

    c_artifact = emit_ssa_function_to_c(module, function.name)
    llvm_artifact = emit_ssa_function_to_llvm(module, function.name)
    fortran_artifact = emit_ssa_module_to_fortran(
        module,
        name="cast_like_fortran",
        outputs={function.name: (result,)},
    )
    wasm_artifact = emit_ssa_function_to_wasm(module, function.name)

    assert c_artifact.complete, c_artifact.shortfalls
    assert llvm_artifact.complete, llvm_artifact.shortfalls
    assert fortran_artifact.complete, fortran_artifact.shortfalls
    assert wasm_artifact.complete, wasm_artifact.shortfalls


def test_sympy_fluid_emits_and_runs_direct_repository_ssa_c(tmp_path, deploy_contract):
    compiled = compile_symbolic_fluid_step()
    artifact = emit_ssa_function_to_c(compiled.module, compiled.function.name)

    assert artifact.complete, artifact.shortfalls
    assert artifact.output_publications
    artifact.compile(tmp_path)
    outputs = dict(zip(artifact.output_names, artifact.run(_uniform_inputs(compiled))))
    assert outputs["height_next"] == pytest.approx(1.0)
    assert outputs["momentum_x_next"] == pytest.approx(0.0)
    assert outputs["momentum_y_next"] == pytest.approx(0.0)
    assert outputs["tracer_next"] == pytest.approx(0.0)
    assert outputs["wave_speed"] == pytest.approx(1.0)
    assert outputs["height_violation"] == pytest.approx(0.0)
    assert outputs["tracer_violation"] == pytest.approx(0.0)


def test_sympy_fluid_emits_and_runs_direct_repository_ssa_wasm(tmp_path, deploy_contract):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the emitted WebAssembly")
    compiled = compile_symbolic_fluid_step()
    artifact = emit_ssa_function_to_wasm(compiled.module, compiled.function.name)

    assert artifact.complete, artifact.shortfalls
    assert artifact.output_publications
    _wat, wasm = artifact.write(tmp_path)
    ordered = [_uniform_inputs(compiled)[name] for name in artifact.input_names]
    script = """
const fs = require('fs');
(async () => {
  const bytes = fs.readFileSync(process.argv[1]);
  const instance = await WebAssembly.instantiate(bytes, {});
  const memory = new Float64Array(instance.instance.exports.memory.buffer);
  const inputs = JSON.parse(process.argv[2]);
  // The artifact publishes where its values live; the tables occupy the
  // front of linear memory, so this layout is not implicit.
  const inputSlot = Number(process.argv[3]) / 8;
  const outputSlot = Number(process.argv[4]) / 8;
  memory.set(inputs, inputSlot);
  instance.instance.exports.symbolic_fluid_step(0);
  const start = outputSlot;
  console.log(JSON.stringify(Array.from(memory.slice(start, start + 11))));
})().catch(error => { console.error(error); process.exit(1); });
"""
    completed = subprocess.run(
        [
            node, "-e", script, str(wasm), json.dumps(ordered),
            str(artifact.input_offsets[0]), str(artifact.output_offsets[0]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    outputs = dict(zip(artifact.output_names, json.loads(completed.stdout)))
    assert outputs["height_next"] == pytest.approx(1.0)
    assert outputs["wave_speed"] == pytest.approx(1.0)
    assert outputs["height_violation"] == pytest.approx(0.0)
    assert outputs["tracer_violation"] == pytest.approx(0.0)


def test_fluid_semantic_outputs_are_identical_in_all_four_direct_lanes(deploy_contract):
    compiled = compile_symbolic_fluid_step()
    outputs = compiled.function.blocks["entry"].instrs[-1].args
    c_artifact = emit_ssa_function_to_c(compiled.module, compiled.function.name)
    llvm_artifact = emit_ssa_function_to_llvm(
        compiled.module, compiled.function.name,
    )
    fortran_artifact = emit_ssa_module_to_fortran(
        compiled.module,
        name="symbolic_fluid_fortran",
        outputs={compiled.function.name: outputs},
    )
    wasm_artifact = emit_ssa_function_to_wasm(
        compiled.module, compiled.function.name,
    )

    assert c_artifact.complete, c_artifact.shortfalls
    assert llvm_artifact.complete, llvm_artifact.shortfalls
    assert fortran_artifact.complete, fortran_artifact.shortfalls
    assert wasm_artifact.complete, wasm_artifact.shortfalls
    expected = c_artifact.output_publications
    assert expected == llvm_artifact.output_publications
    assert expected == fortran_artifact.api.metadata["semantic_outputs"]
    assert expected == wasm_artifact.output_publications

    native_surfaces = c_artifact.output_surfaces
    assert native_surfaces == llvm_artifact.output_surfaces
    assert native_surfaces == fortran_artifact.api.metadata[
        "semantic_output_surfaces"
    ]
    assert native_surfaces["target_family"] == "native"
    web_surfaces = wasm_artifact.output_surfaces
    assert web_surfaces["target_family"] == "web"
    assert any(
        surface["adapter"] == "webgl.scalar-field"
        and surface["outputs"] == ("height_next",)
        for surface in web_surfaces["surfaces"]
    )
    velocity = next(
        surface for surface in web_surfaces["surfaces"]
        if surface["id"].endswith(":fluid.velocity")
    )
    assert velocity["adapter"] == "canvas.vector-field"
    assert velocity["outputs"] == ("velocity_x", "velocity_y")
    assert any(
        surface["adapter"] == "dom.live-metric"
        and surface["outputs"] == ("wave_speed",)
        for surface in web_surfaces["surfaces"]
    )


def test_exact_contracts_forbid_the_private_sqrt_spellings():
    """prove/develop promise exact-only emission, backends included.

    C keeps a faithful ``pow()`` spelling; scalar WASM has nothing exact to
    fall back on and must refuse rather than silently substitute sqrt.
    """

    base = SSAValue(0, "float64")
    exponent = SSAValue(1, "float64")
    result = SSAValue(2, "float64")
    function = Function(
        "half_power_kernel",
        [base],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], exponent, attributes={"constant": 0.5}),
            Instr("Pow", [base, exponent], result),
            Instr("Ret", [result], None),
        ])},
        metadata={"argument_names": ("base",), "output_names": ("result",)},
    )
    module = IRModule({function.name: function})
    set_active_contract("develop")
    try:
        c_artifact = emit_ssa_function_to_c(module, function.name)
        assert c_artifact.complete, c_artifact.shortfalls
        assert "sqrt(" not in c_artifact.source
        assert "pow(" in c_artifact.source
        wasm_artifact = emit_ssa_function_to_wasm(module, function.name)
        assert not wasm_artifact.complete
        assert any(
            "no exact scalar WASM spelling" in shortfall.reason
            for shortfall in wasm_artifact.shortfalls
        )
        set_active_contract("deploy")
        c_deploy = emit_ssa_function_to_c(module, function.name)
        assert "sqrt(" in c_deploy.source
        wasm_deploy = emit_ssa_function_to_wasm(module, function.name)
        assert wasm_deploy.complete, wasm_deploy.shortfalls
    finally:
        set_active_contract(None)


def test_wasm_reciprocal_sqrt_is_an_explicit_deploy_identity():
    base = SSAValue(20, "float64")
    exponent = SSAValue(21, "float64")
    result = SSAValue(22, "float64")
    function = Function(
        "reciprocal_sqrt_kernel",
        [base],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], exponent, attributes={"constant": -0.5}),
            Instr("Pow", [base, exponent], result),
            Instr("Ret", [result], None),
        ])},
        metadata={"argument_names": ("base",), "output_names": ("result",)},
    )
    module = IRModule({function.name: function})

    set_active_contract("develop")
    try:
        exact = emit_ssa_function_to_wasm(module, function.name)
        deployed = emit_ssa_function_to_wasm(
            module, function.name, work_contract="deploy",
        )
        assert not exact.complete
        assert deployed.complete, deployed.shortfalls
        assert "f64.sqrt f64.div" in deployed.wat
    finally:
        set_active_contract(None)
