"""A private tensor owner survives a booked branch until the sole return."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_CONTROL_EDGE, TENSOR_HEAP_CONSUMER_NONESCAPE,
    TENSOR_HEAP_RELEASE_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.ssa import Instr, SSAValue


def _lower_branching_private_tensor():
    source = """def root(lengths: AbstractTensor, bit_count: int, flag: bool) -> float:
    total = AbstractTensor.arange(bit_count, cls=type(lengths)).to_dtype("float64").sum().item()
    if flag:
        adjustment = 1.0
    else:
        adjustment = 2.0
    return total + adjustment
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
            {"function": "root", "parameter": "flag", "storage": "scalar",
             "dtype": "bool", "rank": 0, "shape": [],
             "python_type": "builtins.bool"},
        ], "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="owned_tensor_cfg",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    return module, module.functions[exports[0]]


def test_private_owner_release_at_dominated_merge_and_native_parity(tmp_path):
    module, root = _lower_branching_private_tensor()
    book = identity_book(module)
    rows = tuple(row for row in book.page(TENSOR_HEAP_RELEASE_BINDING).rows()
                 if row[1] == root.name)
    assert len(rows) == 2
    for row in rows:
        producer_id, tensor_id, return_block_name = book.page(
            TENSOR_HEAP_RELEASE_BINDING
        ).latest(row)
        producer_block = next(block for block in root.blocks.values()
                              if any(instruction.op == "Call"
                                     and instruction.res is not None
                                     and instruction.res.id == producer_id
                                     for instruction in block.instrs))
        assert producer_block.name != return_block_name
        assert producer_block.name in CorrelationTable._dominators(root)[
            return_block_name]
        assert any(instruction.attributes.get("callee")
                   == "turing_heap_release"
                   and instruction.args[0].id == row[2]
                   for instruction in root.blocks[return_block_name].instrs)
    assert not CorrelationTable._tensor_heap_release_binding_findings(module)

    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path)
    returned = next(instruction.args[0].id
                    for block in root.blocks.values()
                    for instruction in block.instrs if instruction.op == "Ret")
    for count, flag, expected in ((0, False, 2.0), (7, True, 22.0),
                                  (7, False, 23.0)):
        execution = artifact.prepare_execution({0: np.int64(count),
                                                1: np.bool_(flag)})
        execution.run()
        assert float(execution.buffers[returned][0]) == expected


def test_private_owner_release_rejects_unbooked_control_and_late_use():
    module, root = _lower_branching_private_tensor()
    book = identity_book(module)
    row = next(row for row in book.page(TENSOR_HEAP_RELEASE_BINDING).rows()
               if row[1] == root.name)
    _, tensor_id, return_block_name = book.page(
        TENSOR_HEAP_RELEASE_BINDING
    ).latest(row)
    edge_page = book.page(SSA_CONTROL_EDGE)
    edge_row = next(edge_row for edge_row in edge_page.rows()
                    if edge_row[1] == root.name)
    removed = {key: value for key, value in edge_page.cells.items()
               if key[0] == edge_row}
    try:
        for key in removed:
            del edge_page.cells[key]
        edge_page.row_columns.pop(edge_row, None)
        edge_page._reindex()
        assert CorrelationTable._tensor_heap_release_binding_findings(module)
    finally:
        edge_page.cells.update(removed)
        edge_page._reindex()

    producer_block = next(block for block in root.blocks.values()
                          if block.name != return_block_name and any(
                              instruction.op == "Call"
                              and any(argument.id == tensor_id
                                      for argument in instruction.args)
                              for instruction in block.instrs))
    later_block = next(block for block in root.blocks.values()
                       if block.name not in {producer_block.name,
                                             return_block_name})
    fake = Instr("Add", [SSAValue(tensor_id, dtype="float64")],
                 SSAValue(999999999, dtype="float64"))
    later_block.instrs.insert(0, fake)
    try:
        assert CorrelationTable._tensor_heap_release_binding_findings(module)
    finally:
        later_block.instrs.remove(fake)

    nonescape_page = book.page(TENSOR_HEAP_CONSUMER_NONESCAPE)
    nonescape_row = next(candidate for candidate in nonescape_page.rows()
                         if candidate[1] == root.name
                         and nonescape_page.latest(candidate)[1] == tensor_id)
    target_name, _, formals, layouts = nonescape_page.latest(nonescape_row)
    output_id = layouts[0][0]
    table = module.tensor_tables[target_name]
    original = table.by_id(output_id)
    table.tensors[output_id] = replace(original, alias_of=formals[0])
    try:
        assert CorrelationTable._tensor_heap_consumer_nonescape_findings(module)
        assert CorrelationTable._tensor_heap_release_binding_findings(module)
    finally:
        table.tensors[output_id] = original
