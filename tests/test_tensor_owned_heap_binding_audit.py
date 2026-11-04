"""A dynamic tensor has a distinct native carrier and exact caller owner."""

from dataclasses import replace
from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    TENSOR_HEAP_CONSUMER_NONESCAPE, TENSOR_HEAP_OWNER_CALL_BINDING,
    TENSOR_HEAP_RELEASE_BINDING,
    TENSOR_OWNED_HEAP_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book


def test_owned_arange_tensor_and_caller_owner_are_audited():
    source = """def root(lengths: AbstractTensor, bit_count: int) -> float:
    return AbstractTensor.arange(bit_count, cls=type(lengths)).to_dtype("float64").sum().item()
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "lengths", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "bit_count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ], "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="owned_arange_audit",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    owned_row = next(row for row in book.page(TENSOR_OWNED_HEAP_BINDING).rows()
                     if row[2] == 6)
    call_row = next(row for row in book.page(TENSOR_HEAP_OWNER_CALL_BINDING).rows()
                    if row[4] == owned_row[2])
    assert CorrelationTable._tensor_owned_heap_binding_findings(module) == []
    assert CorrelationTable._tensor_heap_owner_call_binding_findings(module) == []

    table = module.tensor_tables[owned_row[1]]
    descriptor = table.tensors[owned_row[2]]
    table.tensors[owned_row[2]] = replace(
        descriptor, heap_owner_value_id=descriptor.data_value_id)
    try:
        assert CorrelationTable._tensor_owned_heap_binding_findings(module)
    finally:
        table.tensors[owned_row[2]] = descriptor

    table.tensors[owned_row[2]] = replace(
        descriptor, element_count_value_id=descriptor.heap_owner_value_id)
    try:
        assert CorrelationTable._tensor_owned_heap_binding_findings(module)
    finally:
        table.tensors[owned_row[2]] = descriptor

    caller = module.functions[call_row[1]]
    call = next(instruction for block in caller.blocks.values()
                for instruction in block.instrs
                if instruction.op == "Call" and instruction.res is not None
                and int(instruction.res.id) == call_row[2])
    original = call.attributes["callee_input_ids"]
    call.attributes["callee_input_ids"] = (*original[:-1], original[0])
    try:
        assert CorrelationTable._tensor_heap_owner_call_binding_findings(module)
    finally:
        call.attributes["callee_input_ids"] = original


def test_private_owned_tensors_release_after_last_use():
    source = """def root(lengths: AbstractTensor, bit_count: int) -> float:
    return AbstractTensor.arange(bit_count, cls=type(lengths)).to_dtype("float64").sum().item()
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "lengths", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "bit_count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ], "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="private_arange_audit",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    rows = tuple(book.page(TENSOR_HEAP_RELEASE_BINDING).rows())
    assert len(rows) == 2
    assert CorrelationTable._tensor_heap_release_binding_findings(module) == []
    caller = module.functions[rows[0][1]]
    block = caller.blocks[book.page(TENSOR_HEAP_RELEASE_BINDING).latest(rows[0])[2]]
    release = next(instruction for instruction in block.instrs
                   if instruction.attributes.get("native_heap_abi") == "release_owner"
                   and instruction.args[0].id == rows[0][2])
    original = release.args[0]
    release.args[0] = next(value for value in caller.args
                           if value.id == rows[1][2])
    try:
        assert CorrelationTable._tensor_heap_release_binding_findings(module)
    finally:
        release.args[0] = original

    # A missing release receipt and its missing instruction must still be
    # caught by the owner-binding inventory, not just by the release audit.
    release_page = book.page(TENSOR_HEAP_RELEASE_BINDING)
    removed = {
        key: value for key, value in release_page.cells.items()
        if key[0] == rows[0]
    }
    release_index = block.instrs.index(release)
    try:
        for key in removed:
            del release_page.cells[key]
        release_page.row_columns.pop(rows[0], None)
        block.instrs.pop(release_index)
        assert CorrelationTable._tensor_heap_owner_call_binding_findings(module)
    finally:
        block.instrs.insert(release_index, release)
        release_page.cells.update(removed)
        release_page._reindex()

    nonescape_page = book.page(TENSOR_HEAP_CONSUMER_NONESCAPE)
    nonescape_row = next(row for row in nonescape_page.rows()
                         if nonescape_page.latest(row)[1] == 6)
    target_name, _tensor_id, formal_ids, layouts = nonescape_page.latest(
        nonescape_row)
    output_id = layouts[0][0]
    target_table = module.tensor_tables[target_name]
    output = target_table.by_id(output_id)
    target_table.tensors[output_id] = replace(output, alias_of=formal_ids[0])
    try:
        assert CorrelationTable._tensor_heap_consumer_nonescape_findings(module)
        assert CorrelationTable._tensor_heap_release_binding_findings(module)
    finally:
        target_table.tensors[output_id] = output
