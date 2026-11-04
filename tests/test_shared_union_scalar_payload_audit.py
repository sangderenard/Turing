"""A tagged optional scalar keeps one material payload without losing None."""

from dataclasses import replace
from pathlib import Path

from src.compiler.concordance_declarations import (
    SSA_VALUE, VALUE_TYPE_ABI, VALUE_TYPE_TAG,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_record_return_state import function_scope_of


CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"


def _optional_module():
    module, *_ = lower_ast_source_to_ssa(
        "def f(x: int | None, n: int) -> int:\n"
        "    if x is None:\n"
        "        return n\n"
        "    return x + n\n",
        "f", extraction_contract=ExtractionContract(CONTRACT),
        progress=lambda _message: None, progress_bars=False,
    )
    name, function, payload = next(
        (name, function, arg)
        for name, function in module.functions.items()
        for arg in function.args
        if (arg.accounting or {}).get("native_value_union_shared_parameter")
        and name in module.value_union_tables
    )
    book = module.metadata["identity_book"]
    scope = function_scope_of(function)
    return (module, book, scope, int(payload.id),
            int(module.value_union_tables[name].by_id(
                int(payload.id)).tag_value_id))


def _disagreements(module):
    return [finding for finding in
            CorrelationTable._ssa_abi_physical_findings(module)
            if finding.kind == "ssa-abi-physical-disagreement"]


def test_shared_optional_payload_requires_tag_and_exact_material_arm():
    module, book, scope, value_id, tag_id = _optional_module()
    assert _disagreements(module) == []

    tag_page = book.page(VALUE_TYPE_TAG)
    tag_ref = book.latest_ref(VALUE_TYPE_TAG, (scope, value_id, tag_id))
    original_tag = tag_page.cells.pop((tag_ref.row, tag_ref.column))
    try:
        assert any(finding.value_id == value_id
                   for finding in _disagreements(module))
    finally:
        tag_page.cells[(tag_ref.row, tag_ref.column)] = original_tag

    abi_page = book.page(VALUE_TYPE_ABI)
    abi_ref = next(source for source, _ in book.edges_into(
        book.latest_ref(SSA_VALUE, (scope, value_id)))
        if source.page == VALUE_TYPE_ABI)
    abi = abi_page.cells[(abi_ref.row, abi_ref.column)]
    arms = tuple(replace(arm, dtype="float64")
                 if arm.storage == "scalar" else arm
                 for arm in abi.physical_alternatives)
    abi_page.cells[(abi_ref.row, abi_ref.column)] = replace(
        abi, physical_alternatives=arms)
    try:
        assert any(finding.value_id == value_id
                   for finding in _disagreements(module))
    finally:
        abi_page.cells[(abi_ref.row, abi_ref.column)] = abi
