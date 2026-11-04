"""A singleton runtime shape tuple feeds its exact scalar member."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    NUMERICAL_CONSTRUCTOR_DIMENSION, REGION_SIGNATURE,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book


def test_dynamic_singleton_constructor_dimension_has_exact_region_feed():
    source = """def root(like: AbstractTensor, count: int) -> AbstractTensor:
    return AbstractTensor.zeros((count,), cls=type(like))
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "like", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="dynamic_constructor_dimension",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert not CorrelationTable.build(module).findings(module)
    book = identity_book(module)
    rows = tuple(book.page(NUMERICAL_CONSTRUCTOR_DIMENSION).rows())
    assert len(rows) == 1
    row = rows[0]
    receipt = book.latest_ref(NUMERICAL_CONSTRUCTOR_DIMENSION, row)
    member, layout, operands = book.page(
        NUMERICAL_CONSTRUCTOR_DIMENSION
    ).cell(receipt.row, receipt.column)
    assert row[1] == "arg:0" and member.row[-1] == 0
    assert layout.page.name == "source_aggregate_layout"
    assert operands.page.name == "process_graph_operands"
    signatures = tuple(
        (signature, book.page(REGION_SIGNATURE).latest(signature))
        for signature in book.page(REGION_SIGNATURE).rows()
        if receipt in {
            source for source, _stage in book.edges_into(
                book.latest_ref(REGION_SIGNATURE, signature)
            )
        }
    )
    assert len(signatures) == 1
    assert member.row[-1] in signatures[0][1].feeds
    root = module.functions[exports[0]]
    calls = tuple(
        instruction for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and "__planned_region_" in str(instruction.attributes.get("callee"))
    )
    assert len(calls) == 1
    assert tuple(value.id for value in calls[0].args) == (member.row[-1],)
