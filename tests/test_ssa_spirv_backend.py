import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.compiler.ssa_spirv_backend import emit_function, emit_module
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def _scalar_elementwise_function(dtype="float32"):
    a = SSAValue(0, dtype, ())
    b = SSAValue(1, dtype, ())
    scaled = SSAValue(2, dtype, ())
    summed = SSAValue(3, dtype, ())
    activated = SSAValue(4, dtype, ())
    block = BasicBlock(
        "entry",
        [
            Instr("mul", [a], scaled, attributes={"right_scalar": 2.5}),
            Instr("add", [scaled, b], summed),
            Instr("tanh", [summed], activated),
            Instr("Ret", [], None),
        ],
    )
    return Function("axpy_tanh", [a, b], {"entry": block}), activated


def test_scalar_elementwise_chain_emits_spirv():
    function, activated = _scalar_elementwise_function()
    module = emit_function(function, outputs=[activated])

    assert module.complete
    assert "OpCapability Shader" in module.source
    assert "OpCapability Float64" not in module.source
    assert '%glsl = OpExtInstImport "GLSL.std.450"' in module.source
    assert "OpMemoryModel Logical GLSL450" in module.source
    assert "OpFMul" in module.source
    assert "OpFAdd" in module.source
    assert "Tanh" in module.source
    assert "OpReturnValue %t4" in module.source
    assert "%axpy_tanh = OpFunction" in module.source
    assert "OpFunctionEnd" in module.source


def test_float64_transcendental_reports_shortfall():
    # GLSL.std.450's trig/exp/log/pow subset is specified for 16/32-bit
    # float only; spirv-val rejects a double operand outright. float64 is
    # this repo's default dtype, so this is a real, expected case -- caught
    # honestly rather than silently narrowed to float32.
    function, activated = _scalar_elementwise_function(dtype="float64")
    module = emit_function(function, outputs=[activated])

    assert not module.complete
    assert any("16/32-bit float" in s.reason for s in module.shortfalls)


@pytest.mark.skipif(shutil.which("spirv-as") is None, reason="no spirv-as on PATH")
def test_emitted_module_assembles_and_validates():
    function, activated = _scalar_elementwise_function()
    module = emit_function(function, outputs=[activated])
    assert module.complete

    workdir = Path(tempfile.mkdtemp(prefix="turing_spirv_test_")).resolve()
    source = module.write(workdir)
    binary = workdir / f"{module.name}.spv"
    assembled = subprocess.run(
        ["spirv-as", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert assembled.returncode == 0, assembled.stderr

    validator = shutil.which("spirv-val")
    if validator is not None:
        validated = subprocess.run(
            [validator, str(binary)], capture_output=True, text=True
        )
        assert validated.returncode == 0, validated.stderr


def test_comparison_and_cast_ops_resolve():
    a = SSAValue(0, "float64", ())
    b = SSAValue(1, "float64", ())
    lt = SSAValue(2, "bool", ())
    as_int = SSAValue(3, "int32", ())
    block = BasicBlock(
        "entry",
        [
            Instr("Lt", [a, b], lt),
            Instr("FpToSi", [a], as_int),
            Instr("Ret", [], None),
        ],
    )
    function = Function("compare_and_cast", [a, b], {"entry": block})

    module = emit_function(function, outputs=[as_int])

    assert module.complete
    assert "OpFOrdLessThan" in module.source
    assert "OpConvertFToS" in module.source


def test_multi_block_control_flow_reports_shortfall():
    entry = BasicBlock(
        "entry",
        [Instr("Br", [], None, attributes={"target": "exit"})],
        successors=["exit"],
    )
    exit_block = BasicBlock("exit", [Instr("Ret", [], None)])
    function = Function("branchy", [], {"entry": entry, "exit": exit_block})

    module = emit_function(function)

    assert not module.complete
    assert "control flow" in module.shortfalls[0].reason


def test_array_argument_reports_shortfall():
    a = SSAValue(0, "float64", (4,))
    block = BasicBlock("entry", [Instr("Ret", [], None)])
    function = Function("arrfn", [a], {"entry": block})

    module = emit_function(function)

    assert not module.complete
    assert "array" in module.shortfalls[0].reason


def test_region_call_reports_shortfall():
    a = SSAValue(0, "float64", ())
    aggregate = SSAValue(1, "ssa.aggregate")
    block = BasicBlock(
        "entry",
        [
            Instr(
                "Call",
                [a],
                aggregate,
                attributes={
                    "callee": "numerical_region_0",
                    "result_convention": "ssa.aggregate",
                },
            ),
            Instr("Ret", [], None),
        ],
    )
    function = Function("calls_region", [a], {"entry": block})

    module = emit_function(function)

    assert not module.complete
    assert "region calls" in module.shortfalls[0].reason


def test_emit_module_shares_type_declarations_across_functions():
    a1 = SSAValue(0, "float64", ())
    b1 = SSAValue(1, "float64", ())
    scaled = SSAValue(2, "float64", ())
    total1 = SSAValue(3, "float64", ())
    block1 = BasicBlock(
        "entry",
        [
            Instr("mul", [a1], scaled, attributes={"right_scalar": 2.5}),
            Instr("add", [scaled, b1], total1),
            Instr("Ret", [], None),
        ],
    )
    fn1 = Function("axpy", [a1, b1], {"entry": block1})

    a2 = SSAValue(10, "float64", ())
    b2 = SSAValue(11, "float64", ())
    total2 = SSAValue(12, "float64", ())
    block2 = BasicBlock(
        "entry",
        [Instr("add", [a2, b2], total2), Instr("Ret", [], None)],
    )
    fn2 = Function("plain_add", [a2, b2], {"entry": block2})

    module = emit_module(
        {"axpy": fn1, "plain_add": fn2},
        outputs={"axpy": [total1], "plain_add": [total2]},
    )

    assert module.complete
    assert module.source.count("OpTypeFloat 64") == 1
    assert "%axpy = OpFunction" in module.source
    assert "%plain_add = OpFunction" in module.source
