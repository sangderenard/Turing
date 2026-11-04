"""An authored float32 view uses typed Call bridges to double kernels."""

import contextlib
import io
from pathlib import Path

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_call_input_adapters import _numeric_physical_name


def test_numeric_alias_admission_uses_declared_layout():
    assert _numeric_physical_name("float") == "float32"
    assert _numeric_physical_name("double") == "float64"
    assert _numeric_physical_name("int64") == "int64"
    assert _numeric_physical_name("opaque_record") is None


def test_authored_float_result_crosses_selected_double_calls():
    source = """def root(values: AbstractTensor) -> AbstractTensor:
    mask = values > 0.0
    narrowed = mask.to_dtype('float')
    return narrowed * values
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"values": [{
        "function": "root", "parameter": "values", "storage": "span",
        "dtype": "float64", "rank": 1, "shape": [4],
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }], "records": {}, "bindings": []})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        lowered = lower_ast_source_to_ssa(
            source, "root",
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            name="authored_float_call_bridge",
            extraction_contract=contract,
            progress_bars=False,
        )
    module = lowered[0] if isinstance(lowered, tuple) else lowered.module
    assert not CorrelationTable.build(module).findings(module)
    regions = tuple(module.functions.values())
    conversion_calls = [
        (function, instruction)
        for function in regions
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "cast_double_to_float_values"
    ]
    bridges = []
    for function, call in conversion_calls:
        block = next(block for block in function.blocks.values() if call in block.instrs)
        position = block.instrs.index(call)
        if position + 1 < len(block.instrs):
            bridge = block.instrs[position + 1]
            if bridge.op == "Cast" and bridge.attributes.get("target_dtype") == "float":
                bridges.append((function, call, block, bridge))
    assert bridges
    function, call, block, bridge = bridges[0]
    assert bridge.op == "Cast"
    assert bridge.attributes["physical_call_output_conversion"] is True
    assert bridge.attributes["source_dtype"] == "float64"
    assert bridge.attributes["target_dtype"] == "float"
    assert bridge.args == [call.res]
    assert call.args[call.attributes["ssa_output_argument"]] is call.res
    assert call.res.id != bridge.res.id
    assert any(
        instruction.op == "Cast"
        and instruction.attributes.get("source_dtype") == "float"
        and instruction.attributes.get("target_dtype") == "float64"
        for function in regions
        for block in function.blocks.values()
        for instruction in block.instrs
    )
