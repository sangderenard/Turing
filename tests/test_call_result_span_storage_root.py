"""A returned tensor member is not a record-row payload candidate."""

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.identity_concordance import SHAPE_EDGE_PAGE, identity_book
from src.compiler.concordance_declarations import TENSOR_ELEMENTWISE_EXTENT_BINDING
from src.compiler.tensor_ssa_lowering import _matching_elementwise_metadata_cause
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("indexed_view", (False, True))
def test_indexed_and_scalar_used_returned_span_keeps_call_result_storage(
    tmp_path, indexed_view,
):
    # The same returned tensor is indexed and consumed by sum. The former
    # storage survey classified this as a possible row payload and demanded
    # a local numerical producer for the structural call-result projection.
    source = """def derive(x: AbstractTensor):
    return x.to_dtype('int64'), x + 1

def root(x: AbstractTensor, indices: AbstractTensor):
    weights, active = derive(x)
    if_view
""".replace(
        "    if_view",
        "    leaves = weights[indices]\n    return leaves[0] + leaves.sum().item()"
        if indexed_view else
        "    return weights[0] + weights.sum().item()",
    )
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": parameter, "storage": "span",
            "dtype": "int64", "rank": 1, "shape": shape,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        } for parameter, shape in (("x", [4]), ("indices", [2]))],
        "records": {}, "bindings": [],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"call_result_span_storage_root_{indexed_view}",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert exports
    assert CorrelationTable.build(module).findings(module) == []
    # derive's Add result owns a runtime metadata triple even though its
    # displayed shape remains the authored static (4,). Its first writer
    # books that exact extent before enrichment revises the shape edge.
    book = identity_book(module)
    extent_page = book.page(TENSOR_ELEMENTWISE_EXTENT_BINDING)
    add_rows = tuple(
        row for row in extent_page.rows()
        if len(row) == 3 and row[1] in module.functions
        and len(extent_page.latest(row)) == 6
        and extent_page.latest(row)[0] == "add"
    )
    assert add_rows
    causally_enriched = 0
    for row in add_rows:
        function = module.functions[row[1]]
        results = tuple(
            instruction.res for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None and int(instruction.res.id) == row[2]
        )
        assert len(results) == 1
        accounting = results[0].accounting or {}
        receipt = _matching_elementwise_metadata_cause(
            book, function, row[2], accounting,
        )
        assert receipt == book.latest_ref(TENSOR_ELEMENTWISE_EXTENT_BINDING, row)
        shape_edges = tuple(
            book.latest_ref(SHAPE_EDGE_PAGE, edge_row)
            for edge_row in book.page(SHAPE_EDGE_PAGE).rows()
            if row[2] in (edge_row[1], edge_row[3])
            and edge_row[4] == "repository_ssa_enrichment"
        )
        if any(receipt in {source for source, _ in book.edges_into(edge)}
               for edge in shape_edges if edge is not None):
            causally_enriched += 1
        forged = dict(accounting)
        forged["tensor_element_count_value_id"] += 1
        assert _matching_elementwise_metadata_cause(
            book, function, row[2], forged,
        ) is None
        class MissingExtentBook:
            def latest_ref(self, page, key):
                if page == TENSOR_ELEMENTWISE_EXTENT_BINDING and key == row:
                    return None
                return book.latest_ref(page, key)
        assert _matching_elementwise_metadata_cause(
            MissingExtentBook(), function, row[2], accounting,
        ) is None
    assert causally_enriched > 0
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "call_result_span")
    names = dict(module.functions[exports[0]].metadata["parameter_names"])
    feeds = {
        int(names["x"]): np.asarray((1, 2, 3, 4), dtype=np.int64),
    }
    if indexed_view:
        feeds[int(names["indices"])] = np.asarray((1, 3), dtype=np.int64)
    buffers = artifact.prepare_execution(feeds).run().buffers
    result = tuple(value for key, value in buffers.items() if key not in feeds)
    assert len(result) == 1
    assert result[0].item() == (8 if indexed_view else 11)
