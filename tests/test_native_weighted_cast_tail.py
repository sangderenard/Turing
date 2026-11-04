"""The selected double-backed integer cast keeps its physical output ABI."""

from pathlib import Path
from dataclasses import replace

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import SSA_VALUE, TENSOR_OWNED_HEAP_BINDING
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import (
    function_scope_of, posted_ssa_value_fact,
)


def test_weighted_axis_cast_then_scalar_sum_uses_proved_double_storage(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "bits", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    shaped = bits.reshape(-1, 8).to_dtype('float64')
    return (shaped * weights.unsqueeze(0)).sum(dim=1).to_dtype('int64').sum().item()
"""
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="native_weighted_cast_tail",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    cast_regions = tuple(
        (function, instruction)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "cast_double_to_int_values"
    )
    assert len(cast_regions) == 1
    function, cast = cast_regions[0]
    assert cast.res is not None
    result_id = int(cast.res.id)
    book = identity_book(module)
    kernel = module.functions["cast_double_to_int_values"]
    assert kernel.metadata["ssa_output_argument"] == 1
    assert kernel.metadata["named_outputs"] == ((
        "out", int(kernel.args[1].id),
    ),)
    assert posted_ssa_value_fact(
        kernel, int(kernel.args[1].id), book=book,
    ).physical_dtype == "float64"
    result_fact = posted_ssa_value_fact(function, result_id, book=book)
    assert result_fact.dtype == "int64"
    assert result_fact.physical_dtype == "float64"
    owned = book.page(TENSOR_OWNED_HEAP_BINDING).latest((
        function_scope_of(function), function.name, result_id,
    ))
    assert owned is not None and owned[3] == "float64"
    assert int(cast.args[1].id) == owned[0]
    assert module.tensor_tables[function.name].by_id(result_id).dtype == "float64"

    # The old int64 allocation was self-consistent across these four
    # records, but disagreed with the selected kernel's typed output formal.
    owned_page = book.page(TENSOR_OWNED_HEAP_BINDING)
    owned_ref = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, (
        function_scope_of(function), function.name, result_id,
    ))
    owned_key = (owned_ref.row, owned_ref.column)
    result_page = book.page(SSA_VALUE)
    result_ref = book.latest_ref(SSA_VALUE, (
        function_scope_of(function), result_id,
    ))
    result_key = (result_ref.row, result_ref.column)
    descriptor_table = module.tensor_tables[function.name]
    descriptor = descriptor_table.by_id(result_id)
    allocation = next(
        instruction for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Alloca" and instruction.res is not None
        and int(instruction.res.id) == int(owned[0])
    )
    original_element_dtype = allocation.attributes["element_dtype"]
    original_carrier_accounting = allocation.res.accounting
    original_result_fact = result_page.cells[result_key]
    try:
        owned_page.cells[owned_key] = (*owned[:3], "int64")
        result_page.cells[result_key] = replace(
            original_result_fact, physical_dtype="int64",
        )
        descriptor_table.tensors[result_id] = replace(descriptor, dtype="int64")
        allocation.attributes["element_dtype"] = "int64"
        allocation.res.accounting = {
            **dict(original_carrier_accounting or {}),
            "heap_element_dtype": "int64",
        }
        assert CorrelationTable._tensor_owned_heap_binding_findings(module)
    finally:
        owned_page.cells[owned_key] = owned
        result_page.cells[result_key] = original_result_fact
        descriptor_table.tensors[result_id] = descriptor
        allocation.attributes["element_dtype"] = original_element_dtype
        allocation.res.accounting = original_carrier_accounting
    assert CorrelationTable._tensor_owned_heap_binding_findings(module) == []

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "weighted_cast_tail")
    input_id = int(dict(
        module.functions[exports[0]].metadata["parameter_names"]
    )["bits"])
    weights = np.asarray((128, 64, 32, 16, 8, 4, 2, 1), dtype=np.float64)
    for count in (0, 8, 16, 64):
        data = np.arange(count, dtype=np.int64)
        feeds = {input_id: data}
        buffers = artifact.prepare_execution(feeds).run().buffers
        actual = tuple(value for key, value in buffers.items()
                       if key not in feeds)
        assert len(actual) == 1
        expected = np.trunc(
            (data.reshape(-1, 8).astype(np.float64) * weights).sum(axis=1)
        ).astype(np.int64).sum(dtype=np.int64)
        assert actual[0].dtype == np.dtype("int64")
        assert int(actual[0].item()) == int(expected)
