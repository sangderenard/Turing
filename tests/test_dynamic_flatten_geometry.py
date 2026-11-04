"""A runtime flatten preserves count and publishes one logical axis."""

from src.compiler.concordance_declarations import TENSOR_RUNTIME_VIEW_GEOMETRY
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_record_return_state import function_scope_of, posted_ssa_value_fact
from tests.test_dynamic_unsqueeze_geometry import _lower


def test_rank_two_dynamic_flatten_uses_count_as_rank_one_extent():
    module, _outputs, exports = _lower("""def root(bits: AbstractTensor) -> AbstractTensor:
    flat = bits.flatten()
    return flat + 1
""", rank=2)
    assert not CorrelationTable.build(module).findings(module)
    root = module.functions[exports[0]]
    calls = [instruction for block in root.blocks.values()
             for instruction in block.instrs if instruction.op == "Call"
             and "__planned_region_" in str(instruction.attributes.get("callee"))]
    producer, consumer = calls[:2]
    view_id = int(producer.attributes["output_ids"][0])
    physical_id = int(producer.attributes["callee_output_ids"][0])
    assert view_id != physical_id
    view_region = module.functions[producer.attributes["callee"]]
    view = module.tensor_tables[view_region.name].by_id(view_id)
    source = module.tensor_tables[view_region.name].by_id(physical_id)
    assert view.metadata_state == source.metadata_state == "dynamic"
    assert view.shape == ()
    assert (view.alias_of, view.data_value_id) == (physical_id, physical_id)
    assert view.element_count_value_id == source.element_count_value_id
    assert view.shape_value_id != source.shape_value_id
    book = identity_book(module)
    row = (function_scope_of(view_region), view_region.name, view_id)
    fact = book.page(TENSOR_RUNTIME_VIEW_GEOMETRY).latest(row)
    assert fact == (
        "flatten", physical_id, source.shape_value_id,
        source.rank_value_id, source.element_count_value_id,
        view.shape_value_id, view.rank_value_id,
    )
    posted = posted_ssa_value_fact(view_region, view_id, book=book)
    assert (posted.physical_dtype, posted.storage, posted.rank) == (
        "int64", "span", 1,
    )
    consumer_region = module.functions[consumer.attributes["callee"]]
    consumer_source = module.tensor_tables[consumer_region.name].by_id(view_id)
    assert consumer_source.metadata_state == "dynamic"
    assert consumer_source.rank_value_id == int(consumer_region.args[2].id)


def test_known_rank_two_flatten_keeps_exact_static_product():
    module, _outputs, exports = _lower("""def root(bits: AbstractTensor) -> AbstractTensor:
    return bits.flatten()
""", rank=2, shape=[2, 3])
    assert not CorrelationTable.build(module).findings(module)
    root = module.functions[exports[0]]
    call = next(instruction for block in root.blocks.values()
                for instruction in block.instrs if instruction.op == "Call"
                and "__planned_region_" in str(instruction.attributes.get("callee")))
    callee = module.functions[call.attributes["callee"]]
    assert call.attributes["result_convention"] == "void"
    result_id = int(call.attributes["aggregate_output_passthrough_bindings"][0][1])
    descriptor = module.tensor_tables[callee.name].by_id(result_id)
    assert descriptor.shape == (6,)
    assert descriptor.metadata_state == "static"
    book = identity_book(module)
    assert not book.page(TENSOR_RUNTIME_VIEW_GEOMETRY).rows()
