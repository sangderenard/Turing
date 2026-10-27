"""Structural iterable handles retain exact native C element storage."""
from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_to_c


@pytest.mark.parametrize("annotation,literal,expected", [
    ("int", "[0, 1]", 1),
    ("int", "[1, -1]", 0),
    ("bool", "[False, True]", 1),
    ("float", "[1.5]", 1),
    ("int", "[9007199254740993]", 1),
])
def test_specialized_iterable_predicate_native_c(tmp_path, annotation, literal, expected):
    source = f'''def validate(offsets: list[{annotation}]) -> int:
    previous = -1
    for offset in offsets:
        if not isinstance(offset, {annotation}):
            return 0
        if offset < previous:
            return 0
        previous = offset
    return 1
def root():
    return validate({literal})
'''
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="c_static_iterable_predicate",
        extraction_contract=Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path)
    result = outputs[exports[0]][0].id
    assert artifact.prepare_execution({}).run().buffers[result].item() == expected


@pytest.mark.parametrize("dtype,payload", [
    ("ssa.aggregate", (False, True)),
    ("ssa.aggregate", (1.5,)),
    ("ssa.aggregate", (2**53 + 1,)),
    ("uint8", (255,)), ("uint16", (65535,)),
    ("uint32", (2**32 - 1,)), ("uint64", (2**64 - 1,)),
])
def test_constant_gep_load_exact_native_c(tmp_path, dtype, payload):
    from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule

    element_dtype = ("bool" if type(payload[0]) is bool else
                     "float64" if type(payload[0]) is float else
                     "int64") if dtype == "ssa.aggregate" else dtype
    data = SSAValue(0, dtype, () if dtype == "ssa.aggregate" else (len(payload),))
    index, address, result = SSAValue(1, "int64"), SSAValue(2, "ptr"), SSAValue(3, element_dtype)
    module = IRModule({"root": Function("root", [], {"entry": BasicBlock("entry", [
        Instr("Const", [], data, attributes={"value": payload}),
        Instr("Const", [], index, attributes={"value": len(payload) - 1}),
        Instr("GetElementPtr", [data, index], address),
        Instr("Load", [address], result), Instr("Ret", [result], None),
    ])})})
    artifact = emit_ssa_to_c(module, "root")
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path)
    actual = artifact.prepare_execution({}).run().buffers[3]
    assert actual.dtype.name == element_dtype
    assert actual.item() == payload[-1]


def test_mixed_structural_constant_refuses_without_element_contract():
    from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule

    data, index = SSAValue(0, "ssa.aggregate"), SSAValue(1, "int64")
    address, result = SSAValue(2, "ptr"), SSAValue(3, "float64")
    module = IRModule({"root": Function("root", [], {"entry": BasicBlock("entry", [
        Instr("Const", [], data, attributes={"value": (1, 1.5)}),
        Instr("Const", [], index, attributes={"value": 1}),
        Instr("GetElementPtr", [data, index], address),
        Instr("Load", [address], result), Instr("Ret", [result], None),
    ])})})
    artifact = emit_ssa_to_c(module, "root")
    assert not artifact.complete
    assert any("no exact scalar element storage contract" in shortfall.reason
               for shortfall in artifact.shortfalls)
