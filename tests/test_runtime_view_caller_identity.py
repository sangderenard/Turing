"""A runtime view's metadata Call returns its semantic caller identity."""

from dataclasses import replace

import pytest

from src.compiler.concordance_declarations import (
    TENSOR_METADATA_RESULT_BINDING, TENSOR_RUNTIME_VIEW_GEOMETRY,
)
from src.compiler.identity_concordance import EDGE_PAGE, identity_book
from src.compiler.ssa_record_return_state import (
    function_scope_of, metadata_result_caller_value,
)
from tests.test_dynamic_unsqueeze_geometry import _lower


@pytest.fixture(scope="module")
def projected_flatten():
    module, _outputs, exports = _lower(
        """def root(bits: AbstractTensor) -> AbstractTensor:
    flat = bits.flatten()
    return flat + 1
""", rank=2,
    )
    caller = module.functions[exports[0]]
    calls = tuple(
        instruction for block in caller.blocks.values()
        for instruction in block.instrs if instruction.op == "Call"
        and "__planned_region_" in str(instruction.attributes.get("callee"))
    )
    producer = calls[0]
    callee = module.functions[producer.attributes["callee"]]
    semantic = int(producer.attributes["output_ids"][0])
    carrier = int(producer.attributes["callee_output_ids"][0])
    assert semantic != carrier
    book = identity_book(module)
    result_rows = tuple(
        row for row in book.page(TENSOR_METADATA_RESULT_BINDING).rows()
        if row == (function_scope_of(caller), int(producer.res.id),
                   callee.name, semantic)
    )
    assert len(result_rows) == 1
    result = book.latest_ref(TENSOR_METADATA_RESULT_BINDING, result_rows[0])
    view = book.latest_ref(
        TENSOR_RUNTIME_VIEW_GEOMETRY,
        (function_scope_of(callee), callee.name, semantic),
    )
    assert view is not None
    return module, caller, callee, producer, semantic, result, view


def test_runtime_view_result_uses_exact_caller_projection(projected_flatten):
    module, caller, _callee, call, semantic, result, view = projected_flatten
    assert metadata_result_caller_value(
        module, caller, call, result, book=identity_book(module),
    ) == (semantic, (view,))


def test_runtime_view_projection_refuses_wrong_carrier(projected_flatten):
    module, caller, callee, call, semantic, result, _view = projected_flatten
    book = identity_book(module)
    table = module.tensor_tables[callee.name]
    descriptor = table.by_id(semantic)
    table.tensors[semantic] = replace(descriptor, data_value_id=semantic)
    try:
        assert metadata_result_caller_value(
            module, caller, call, result, book=book,
        ) is None
    finally:
        table.tensors[semantic] = descriptor


def test_runtime_view_projection_requires_cited_view_cell(projected_flatten):
    module, caller, _callee, call, _semantic, result, view = projected_flatten
    book = identity_book(module)
    direct = tuple(
        (source, stage) for source, stage in book.edges_into(result)
        if source == view
    )
    assert len(direct) == 1
    edge = (result.key, view.key, direct[0][1].name)
    page = book.page(EDGE_PAGE)
    fact = page.cells.pop((edge, 0))
    stamp = page.stamps.pop((edge, 0))
    columns = page.row_columns.pop(edge)
    page.scopes[result.key].pop(edge)
    try:
        assert metadata_result_caller_value(
            module, caller, call, result, book=book,
        ) is None
    finally:
        page.cells[(edge, 0)] = fact
        page.stamps[(edge, 0)] = stamp
        page.row_columns[edge] = columns
        page.scopes[result.key][edge] = None
