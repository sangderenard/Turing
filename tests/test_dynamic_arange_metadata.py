"""A computed arange extent crosses its planned-region result call."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    TENSOR_METADATA_RESULT_BINDING, TENSOR_OWNED_HEAP_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book


def test_dynamic_arange_count_and_call_metadata_are_exact():
    source = """def root(lengths: AbstractTensor, bit_count: int) -> AbstractTensor:
    return AbstractTensor.arange(bit_count, cls=type(lengths))
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
        ],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    root = module.functions[exports[0]]
    region = next(function for name, function in module.functions.items()
                  if name.startswith(root.name + "__planned_region_"))
    region_ops = [instruction for block in region.blocks.values()
                  for instruction in block.instrs]
    count_cast = next(instruction for instruction in region_ops
                      if instruction.attributes.get("binding")
                      == "arange-count-narrowing")
    assert (count_cast.op, count_cast.args[0].dtype,
            count_cast.res.dtype) == ("Cast", "int64", "int32")
    assert (count_cast.attributes["conversion_kind"],
            count_cast.attributes["overflow"]) == ("integer_conversion", "trap")
    safe_count = next(instruction for instruction in region_ops
                      if instruction.attributes.get("binding")
                      == "arange-safe-count")
    kernel = next(instruction for instruction in region_ops
                  if instruction.attributes.get("callee") == "create_arange_int64")
    descriptor = module.tensor_tables[region.name].by_id(5)
    assert descriptor.metadata_state == "dynamic"
    assert safe_count.res.id == descriptor.element_count_value_id
    assert kernel.args[2].id == safe_count.res.id
    assert region_ops.index(count_cast) < region_ops.index(safe_count) < region_ops.index(kernel)

    book = identity_book(module)
    assert not tuple(book.page(TENSOR_OWNED_HEAP_BINDING).rows())
    rows = tuple(book.page(TENSOR_METADATA_RESULT_BINDING).rows())
    assert len(rows) == 1
    fact = book.page(TENSOR_METADATA_RESULT_BINDING).latest(rows[0])
    assert fact[1] == (descriptor.shape_value_id,
                       descriptor.rank_value_id,
                       descriptor.element_count_value_id)
    call = next(instruction for block in root.blocks.values()
                for instruction in block.instrs
                if instruction.attributes.get("callee") == region.name)
    assert tuple(call.attributes["callee_output_ids"][1:]) == fact[1]
    assert tuple(call.attributes["output_ids"][1:]) == fact[2]
    root_result = next(instruction.res for block in root.blocks.values()
                       for instruction in block.instrs
                       if instruction.op == "Load" and instruction.res is not None
                       and instruction.res.id == 5)
    assert tuple(root_result.accounting[key] for key in (
        "tensor_shape_value_id", "tensor_rank_value_id",
        "tensor_element_count_value_id",
    )) == fact[2]
    assert not CorrelationTable._tensor_metadata_result_binding_findings(module)
    original = tuple(call.attributes["callee_output_ids"])
    call.attributes["callee_output_ids"] = (original[0], original[2],
                                           original[1], original[3])
    assert CorrelationTable._tensor_metadata_result_binding_findings(module)


def test_dynamic_arange_cast_and_sum_keep_runtime_count_across_regions(tmp_path):
    source = """def root(lengths: AbstractTensor, bit_count: int) -> AbstractTensor:
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
        ],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    root = module.functions[exports[0]]
    cast_region = next(function for name, function in module.functions.items()
                       if name == root.name + "__planned_region_1")
    sum_region = next(function for name, function in module.functions.items()
                      if name == root.name + "__planned_region_2")
    cast_ops = [instruction for block in cast_region.blocks.values()
                for instruction in block.instrs]
    sum_ops = [instruction for block in sum_region.blocks.values()
               for instruction in block.instrs]
    cast_call = next(instruction for instruction in cast_ops
                     if instruction.attributes.get("callee")
                     == "cast_int64_to_double_values")
    sum_call = next(instruction for instruction in sum_ops
                    if instruction.attributes.get("callee") == "sum_double")
    cast_input = module.tensor_tables[cast_region.name].by_id(6)
    cast_output = module.tensor_tables[cast_region.name].by_id(8)
    assert cast_input.metadata_state == cast_output.metadata_state == "dynamic"
    assert (cast_output.shape_value_id, cast_output.rank_value_id,
            cast_output.element_count_value_id) == (
                cast_input.shape_value_id, cast_input.rank_value_id,
                cast_input.element_count_value_id,
            )
    assert cast_call.args[2].id == cast_input.element_count_value_id
    sum_input = module.tensor_tables[sum_region.name].by_id(8)
    assert sum_input.metadata_state == "dynamic"
    assert sum_call.args[1].id == sum_input.element_count_value_id
    # The computed producer already returns this exact geometry. Its three
    # local extent queries were promoted to bound input formals; retaining
    # another query would give one tensor two shape/rank/count identities.
    assert not any(instruction.op == "extent" for instruction in cast_ops)
    assert not any(instruction.op == "extent" for instruction in sum_ops)
    assert tuple(value.id for value in cast_region.args[1:4]) == (
        cast_input.shape_value_id, cast_input.rank_value_id,
        cast_input.element_count_value_id,
    )
    assert tuple(value.id for value in sum_region.args[1:4]) == (
        sum_input.shape_value_id, sum_input.rank_value_id,
        sum_input.element_count_value_id,
    )

    rows = tuple(identity_book(module).page(TENSOR_METADATA_RESULT_BINDING).rows())
    assert len(rows) == 2
    assert not CorrelationTable._tensor_metadata_result_binding_findings(module)
    from src.compiler.ssa_storage_requirements import module_storage_requirements
    requirements = module_storage_requirements(module)
    for value_id in (6, 8):
        assert requirements[root.name][value_id].dynamic
        assert requirements[root.name][value_id].element_count is None
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.concordance_declarations import (
        TENSOR_OWNED_HEAP_BINDING, TENSOR_HEAP_OWNER_CALL_BINDING,
        TENSOR_HEAP_RELEASE_BINDING,
    )
    book = identity_book(module)
    assert len(tuple(book.page(TENSOR_OWNED_HEAP_BINDING).rows())) == 2
    assert len(tuple(book.page(TENSOR_HEAP_OWNER_CALL_BINDING).rows())) == 2
    assert len(tuple(book.page(TENSOR_HEAP_RELEASE_BINDING).rows())) == 2
    assert not CorrelationTable._tensor_owned_heap_binding_findings(module)
    assert not CorrelationTable._tensor_heap_owner_call_binding_findings(module)
    assert not CorrelationTable._tensor_heap_release_binding_findings(module)
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    import numpy as np
    import pytest
    from src.compiler.native_faults import NativeIntOverflow

    artifact.compile(tmp_path)
    for count, expected in (
        (0, 0.0), (1, 0.0), (7, 21.0), (16, 120.0),
        (127, 8001.0), (-3, 0.0), (7, 21.0), (0, 0.0),
    ):
        execution = artifact.prepare_execution({0: np.int64(count)})
        for _repeat in range(3):
            execution.run()
            assert float(execution.buffers[10][0]) == expected
    with pytest.raises(NativeIntOverflow):
        artifact.prepare_execution({0: np.int64(1 << 31)}).run()
