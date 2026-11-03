"""A scalar truth test must not replace the value carried by a break edge."""

import inspect
from pathlib import Path

from src.common.tensors.compression import octet_lz
from src.compiler.concordance_declarations import CONTROL_VALUE_BINDING
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    ConcordanceRefusal, CorrelationTable, current_identity_book,
)
import src.compiler.precompile_to_ssa as precompile_to_ssa


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts/program_extraction.yaml"
)


def _lower(source: str, name: str):
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name=name, extraction_contract=CONTRACT,
        progress=lambda _message: None,
    )
    return module, module.functions[f"{name}__root"]


def test_authored_header_break_keeps_the_integer_binding(monkeypatch):
    captured = []
    original = precompile_to_ssa.lower_control_program_to_ssa

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if "_header" in str(kwargs.get("function_name", "")):
            captured.append((result, current_identity_book()))
        return result

    monkeypatch.setattr(precompile_to_ssa, "lower_control_program_to_ssa", capture)
    source = (
        inspect.getsource(octet_lz._header)
        + "\ndef root(x: int) -> bytes:\n    return _header(x)\n"
    )
    try:
        lower_ast_source_to_ssa(
            source, "root", name="authored_header_break",
            extraction_contract=CONTRACT, progress=lambda _message: None,
        )
    except ConcordanceRefusal as error:
        # This exact source currently reaches a later native element-layout
        # audit refusal; the control-stage result is the regression subject.
        assert "loop_body: Load int64 from column 1" in str(error)
        assert captured
    assert captured
    (root, shortfalls), book = captured[-1]
    assert not any(item.name == "break-edge-value" for item in shortfalls)
    truth_tests = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding") == "scalar_truthiness"
    ]
    assert len(truth_tests) == 1
    test = truth_tests[0]
    assert test.op == "Ne"
    assert test.args[0].dtype == "int64"
    assert test.res.dtype == "bool"
    assert test.res.id != test.args[0].id
    source_id = int(test.attributes["source_value_id"])
    binding_rows = tuple(
        row for row in book.page(CONTROL_VALUE_BINDING).rows()
        if row[-1] == source_id and str(row[0]).startswith(root.name + "@control:")
    )
    assert len(binding_rows) == 1
    cell = book.latest_ref(CONTROL_VALUE_BINDING, binding_rows[0])
    assert cell is not None
    bound = book.page(CONTROL_VALUE_BINDING).cell(cell.row, cell.column)
    assert bound.ssa_value.row[1] != test.res.id


def test_boolean_condition_keeps_its_existing_predicate_value():
    module, root = _lower(
        "def root(flag: bool) -> int:\n"
        "    if flag:\n"
        "        return 1\n"
        "    return 0\n",
        "boolean_condition_truth",
    )
    assert not any(
        instruction.attributes.get("binding") == "scalar_truthiness"
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    assert CorrelationTable.build(module).findings(module) == []
