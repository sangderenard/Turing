"""Runtime unsqueeze keeps one exact view geometry across region calls."""

from pathlib import Path
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    TENSOR_ELEMENTWISE_EXTENT_BINDING,
    TENSOR_METADATA_CALL_BINDING,
    TENSOR_RUNTIME_VIEW_GEOMETRY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.precompile_to_ssa import SSALoweringIncompleteError
from src.compiler.ssa_record_return_state import function_scope_of, posted_ssa_value_fact


def _lower(source, *, rank=1, shape=None):
    bits = {
        "function": "root", "parameter": "bits", "storage": "span",
        "dtype": "int64", "rank": rank,
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }
    if shape is not None:
        bits["shape"] = shape
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [bits],
        "records": {}, "bindings": [],
    })
    return lower_ast_source_to_ssa(
        source, "root", name="unsqueeze_geometry_probe",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )


def test_dynamic_unsqueeze_crosses_call_as_rank_two_view_before_binary():
    module, _outputs, exports = _lower("""def root(bits: AbstractTensor) -> AbstractTensor:
    column = bits.unsqueeze(1)
    return column + 1
""")
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
    assert view.metadata_state == "dynamic"
    assert (view.alias_of, view.data_value_id) == (physical_id, physical_id)
    book = identity_book(module)
    assert book.latest_ref(
        TENSOR_RUNTIME_VIEW_GEOMETRY,
        (function_scope_of(view_region), view_region.name, view_id),
    )

    binary_region = module.functions[consumer.attributes["callee"]]
    source = module.tensor_tables[binary_region.name].by_id(view_id)
    assert source.metadata_state == "dynamic"
    assert (
        source.shape_value_id, source.rank_value_id,
        source.element_count_value_id,
    ) == tuple(int(arg.id) for arg in binary_region.args[1:4])
    assert tuple(int(arg.id) for arg in consumer.args[1:4]) == tuple(
        int(arg.id) for arg in binary_region.args[1:4]
    )
    assert any(row[4] == binary_region.name for row in book.page(
        TENSOR_METADATA_CALL_BINDING
    ).rows())
    result_id = int(consumer.attributes["output_ids"][0])
    result = module.tensor_tables[binary_region.name].by_id(result_id)
    assert result.metadata_state == "dynamic"
    assert (
        result.shape_value_id, result.rank_value_id,
        result.element_count_value_id,
    ) == (
        source.shape_value_id, source.rank_value_id,
        source.element_count_value_id,
    )
    assert book.latest_ref(
        TENSOR_ELEMENTWISE_EXTENT_BINDING,
        (function_scope_of(binary_region), binary_region.name, result_id),
    )
    fact = posted_ssa_value_fact(binary_region, result_id, book=book)
    assert (fact.physical_dtype, fact.storage, fact.rank) == (
        "int64", "span", 2,
    )


def test_out_of_range_dynamic_unsqueeze_axis_refuses_before_geometry_mint():
    with pytest.raises(SSALoweringIncompleteError, match="runtime unsqueeze axis is outside the proven source rank"):
        _lower("""def root(bits: AbstractTensor):
    return bits.unsqueeze(-3)
""")
