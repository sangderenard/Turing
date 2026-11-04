"""Exact source shape projections retain their receiver and literal axis."""

import ast
import contextlib
import io
from types import SimpleNamespace
from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import SOURCE_TENSOR_SHAPE_DIMENSION
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_shape_dimension_source_receipt_rejects_wrong_receiver_and_axis():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.python_bindings = {"AbstractTensor": AbstractTensor}
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def dimension(values: AbstractTensor):\n"
                "    return values.shape[0]\n"
            ))
        reduce_abstract_tensor_topology(graph)
        page = book.page(SOURCE_TENSOR_SHAPE_DIMENSION)
        rows = tuple(page.rows())
        assert len(rows) == 1
        row = rows[0]
        ref = book.latest_ref(SOURCE_TENSOR_SHAPE_DIMENSION, row)
        original = page.cells[(ref.row, ref.column)]
        receiver, shape, index, axis = original
        assert axis == 0
        module = SimpleNamespace(metadata={"identity_book": book})
        findings = CorrelationTable._source_tensor_shape_dimension_findings
        assert findings(module) == [], (original, tuple(type(item) for item in original))

        for forged in ((index, shape, receiver, axis),
                       (receiver, shape, index, 1)):
            page.cells[(ref.row, ref.column)] = forged
            assert len(findings(module)) == 1
        page.cells[(ref.row, ref.column)] = original
    finally:
        end_identity_book(token)


def test_live_shape_extent_uses_authored_receiver_result_and_axis():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "octets", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }, {
            "function": "root", "parameter": "count", "storage": "scalar",
            "dtype": "int64", "rank": 0,
            "type_expectations": ["builtins.int", "builtins.NoneType"],
        }], "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        "def root(octets: AbstractTensor, count: int | None = None) -> int:\n"
        "    if count is None:\n"
        "        limit = octets.shape[0]\n"
        "    else:\n"
        "        limit = int(count)\n"
        "    return limit + octets.sum().item()\n",
        "root", name="shape_extent_audit",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    findings = CorrelationTable._source_tensor_shape_dimension_findings
    assert findings(module) == []
    extents = [instruction
               for function in module.functions.values()
               for block in function.blocks.values()
               for instruction in block.instrs
               if instruction.attributes.get("source_shape_dimension_cell")]
    assert len(extents) == 1
    extent = extents[0]
    axis = extent.attributes["axis"]
    extent.attributes["axis"] = axis + 1
    assert any("live extent call" in item.detail for item in findings(module))
    extent.attributes["axis"] = axis
    receiver = extent.args[0]
    other = next(value for function in module.functions.values()
                 for value in function.args if value.id != receiver.id)
    extent.args[0] = other
    assert any("live extent call" in item.detail for item in findings(module))
    extent.args[0] = receiver
    assert findings(module) == []
