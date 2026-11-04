"""A runtime reshape must retain its authored geometry requirement."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.concordance_declarations import (
    TENSOR_ELEMENTWISE_EXTENT_BINDING,
    TENSOR_RUNTIME_RESHAPE_VALIDITY,
)
from src.compiler.identity_concordance import identity_book
from src.compiler.ssa_record_return_state import (
    function_scope_of, posted_ssa_value_fact,
)


def _contract(*, shape=None):
    bits = {
        "function": "root", "parameter": "bits", "storage": "span",
        "dtype": "int64", "rank": 2,
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }
    if shape is not None:
        bits["shape"] = shape
    return ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [bits, {
            "function": "root", "parameter": "width", "storage": "scalar",
            "dtype": "int64", "rank": 0, "shape": [],
            "python_type": "builtins.int",
        }],
        "records": {}, "bindings": [],
    })


def _lower(source, *, shape=None):
    return lower_ast_source_to_ssa(
        source, "root", name="reshape_geometry_probe",
        extraction_contract=_contract(shape=shape),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )


def test_runtime_inferred_axis_keeps_distinct_view_and_returned_metadata():
    source = """def root(bits: AbstractTensor, width: int) -> AbstractTensor:
    shaped = bits.reshape(-1, width)
    return shaped + 1
"""
    module, _outputs, exports = _lower(source)
    assert not CorrelationTable.build(module).findings(module)
    root = module.functions[exports[0]]
    calls = [instruction for block in root.blocks.values()
             for instruction in block.instrs
             if instruction.op == "Call"
             and "__planned_region_" in str(instruction.attributes.get("callee"))]
    producer, consumer = calls[:2]
    view_id = int(producer.attributes["output_ids"][0])
    storage_id = int(producer.attributes["callee_output_ids"][0])
    assert view_id != storage_id
    assert int(consumer.attributes["feed_ids"][0]) == view_id
    assert int(consumer.args[0].id) == view_id
    assert all(int(actual.id) not in producer.attributes["output_ids"][1:]
               for actual in producer.args)
    callee = module.functions[producer.attributes["callee"]]
    descriptor = module.tensor_tables[callee.name].by_id(view_id)
    assert descriptor.metadata_state == "dynamic"
    assert descriptor.alias_of == storage_id
    assert descriptor.data_value_id == storage_id
    book = identity_book(module)
    assert book.latest_ref(TENSOR_RUNTIME_RESHAPE_VALIDITY,
                           (function_scope_of(callee), callee.name, view_id))
    consumer_callee = module.functions[consumer.attributes["callee"]]
    result_id = int(consumer.attributes["output_ids"][0])
    result_descriptor = module.tensor_tables[consumer_callee.name].by_id(
        result_id
    )
    assert result_descriptor.metadata_state == "dynamic"
    assert (
        result_descriptor.shape_value_id,
        result_descriptor.rank_value_id,
        result_descriptor.element_count_value_id,
    ) == tuple(int(value.id) for value in consumer.args[1:4])
    assert book.latest_ref(
        TENSOR_ELEMENTWISE_EXTENT_BINDING,
        (function_scope_of(consumer_callee), consumer_callee.name, result_id),
    )
    result_fact = posted_ssa_value_fact(consumer_callee, result_id, book=book)
    assert (result_fact.physical_dtype, result_fact.storage, result_fact.rank) == (
        "int64", "span", 2,
    )


def test_left_scalar_subtraction_keeps_order_and_tensor_extent():
    source = """def root(bits: AbstractTensor, width: int) -> AbstractTensor:
    shaped = bits.reshape(-1, width)
    return 1 - shaped
"""
    module, _outputs, exports = _lower(source)
    assert not CorrelationTable.build(module).findings(module)
    root = module.functions[exports[0]]
    consumer = next(
        instruction for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and str(instruction.attributes.get("callee", "")).endswith(
            "__planned_region_1"
        )
    )
    callee = module.functions[consumer.attributes["callee"]]
    result_id = int(consumer.attributes["output_ids"][0])
    binding = identity_book(module).page(TENSOR_ELEMENTWISE_EXTENT_BINDING).latest(
        (function_scope_of(callee), callee.name, result_id)
    )
    assert binding is not None
    assert binding[0] == "sub"
    result_fact = posted_ssa_value_fact(callee, result_id,
                                         book=identity_book(module))
    assert (result_fact.physical_dtype, result_fact.storage, result_fact.rank) == (
        "int64", "span", 2,
    )
    assert tuple(int(argument.id) for argument in next(
        instruction for block in callee.blocks.values()
        for instruction in block.instrs if instruction.op == "Sub"
    ).args) == (binding[2], binding[1])


def test_exact_static_reshape_keeps_view_and_downstream_tensor_shape():
    source = """def root(bits: AbstractTensor, width: int) -> AbstractTensor:
    shaped = bits.reshape((2, 2))
    return shaped + 1
"""
    module, _outputs, exports = _lower(source, shape=[2, 2])
    assert not CorrelationTable.build(module).findings(module)
    root = module.functions[exports[0]]
    regions = tuple(
        function for function in module.functions.values()
        if function.name.startswith(root.name + "__planned_region_")
    )
    view = next(
        descriptor for function in regions
        for descriptor in module.tensor_tables[function.name].tensors.values()
        if descriptor.storage == "view"
    )
    assert view.shape == (2, 2)
    assert view.alias_of is not None
