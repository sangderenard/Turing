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


def test_array_elementwise_chain_unrolls_and_validates():
    a = SSAValue(0, "float32", (4,))
    b = SSAValue(1, "float32", (4,))
    total = SSAValue(2, "float32", (4,))
    block = BasicBlock(
        "entry",
        [Instr("add", [a, b], total), Instr("Ret", [], None)],
    )
    function = Function("vec_add", [a, b], {"entry": block})

    module = emit_function(function, outputs=[total])

    assert module.complete
    # One OpAccessChain/OpLoad pair per operand per element, plus one store.
    assert module.source.count("OpFAdd") == 4
    assert module.source.count("OpStore") == 4
    assert "OpTypeArray %float32 %cu" in module.source
    assert "%vec_add = OpFunction" in module.source

    if shutil.which("spirv-as") is None:
        return
    workdir = Path(tempfile.mkdtemp(prefix="turing_spirv_test_")).resolve()
    source = module.write(workdir)
    binary = workdir / f"{module.name}.spv"
    assembled = subprocess.run(
        ["spirv-as", str(source), "-o", str(binary)], capture_output=True, text=True
    )
    assert assembled.returncode == 0, assembled.stderr
    validator = shutil.which("spirv-val")
    if validator is not None:
        validated = subprocess.run([validator, str(binary)], capture_output=True, text=True)
        assert validated.returncode == 0, validated.stderr


def test_array_broadcast_against_scalar_operand():
    a = SSAValue(0, "float32", (3,))
    scaled = SSAValue(1, "float32", (3,))
    block = BasicBlock(
        "entry",
        [
            Instr("mul", [a], scaled, attributes={"right_scalar": 2.0}),
            Instr("Ret", [], None),
        ],
    )
    function = Function("scale_vec", [a], {"entry": block})

    module = emit_function(function, outputs=[scaled])

    assert module.complete
    assert module.source.count("OpFMul") == 3


def test_array_rank_mismatch_reports_shortfall():
    a = SSAValue(0, "float32", (2, 3))
    b = SSAValue(1, "float32", (4,))
    total = SSAValue(2, "float32", (2, 3))
    block = BasicBlock(
        "entry",
        [Instr("add", [a, b], total), Instr("Ret", [], None)],
    )
    function = Function("bad_broadcast", [a, b], {"entry": block})

    module = emit_function(function, outputs=[total])

    assert not module.complete
    assert "broadcast" in module.shortfalls[0].reason


def _reductions(dtype):
    """Every whole-array reduction of one static (2, 3) operand."""

    a = SSAValue(0, dtype, (2, 3))
    start = SSAValue(1, dtype, ())
    float_dtype = dtype if dtype.startswith("float") else "float32"
    results = {
        "sum": SSAValue(2, dtype, ()), "prod": SSAValue(3, dtype, ()),
        "max": SSAValue(4, dtype, ()), "min": SSAValue(5, dtype, ()),
        "mean": SSAValue(6, float_dtype, ()), "all": SSAValue(7, "bool", ()),
        "any": SSAValue(8, float_dtype, ()), "sum_start": SSAValue(9, dtype, ()),
    }
    instrs = [
        Instr(op, [a], value) for op, value in results.items() if op != "sum_start"
    ]
    instrs.append(Instr("sum", [a, start], results["sum_start"]))
    instrs.append(Instr("Ret", [], None))
    function = Function(
        f"reduce_{dtype}", [a, start], {"entry": BasicBlock("entry", instrs)}
    )
    return function, list(results.values())


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
def test_whole_array_reductions_assemble_and_validate(dtype):
    function, outputs = _reductions(dtype)
    module = emit_function(function, outputs=outputs)
    assert module.complete, module.shortfalls
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        pytest.skip("no spirv-as/spirv-val")
    workdir = Path(tempfile.mkdtemp(prefix="turing_spirv_test_")).resolve()
    binary = workdir / f"{module.name}.spv"
    assembled = subprocess.run(
        ["spirv-as", str(module.write(workdir)), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert assembled.returncode == 0, assembled.stderr
    validated = subprocess.run(
        ["spirv-val", str(binary)], capture_output=True, text=True
    )
    assert validated.returncode == 0, validated.stderr


def test_axis_reduction_and_empty_max_report_shortfalls():
    a = SSAValue(0, "float32", (4, 2))
    total = SSAValue(1, "float32", ())
    block = BasicBlock(
        "entry",
        [Instr("sum", [a], total, attributes={"axis": 0}), Instr("Ret", [], None)],
    )
    module = emit_function(Function("reduce_axis", [a], {"entry": block}), outputs=[total])
    assert not module.complete
    assert "no axis" in module.shortfalls[0].reason

    empty = SSAValue(0, "float32", (0,))
    block = BasicBlock(
        "entry", [Instr("max", [empty], total), Instr("Ret", [], None)],
    )
    module = emit_function(Function("reduce_empty", [empty], {"entry": block}), outputs=[total])
    assert not module.complete
    assert "empty" in module.shortfalls[0].reason


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
