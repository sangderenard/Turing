"""A dynamic axis sum publishes its typed, owned span before a later cast."""

from pathlib import Path

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_OWNED_HEAP_BINDING, TENSOR_REDUCTION_DOMAIN,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import function_scope_of


def test_runtime_axis_sum_is_typed_before_cast_and_native(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "float64", "rank": 2,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = (
        "def root(values: AbstractTensor) -> int:\n"
        "    rows = values.sum(dim=1)\n"
        "    return rows.to_dtype('int64').sum().item()\n"
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="runtime_axis_sum_cast_native",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(TENSOR_REDUCTION_DOMAIN)
    rows = tuple(row for row in page.rows()
                 if row[0].startswith("runtime_axis_sum_cast_native")
                 and (fact := page.latest(row))
                 and fact[0] == "axis-runtime")
    assert len(rows) == 1
    row = rows[0]
    function = module.functions[row[0]]
    scope = function_scope_of(function)
    result_id = row[1]
    fact = page.latest(row)
    assert fact[3:5] == (1, False)
    result_ref = book.latest_ref(SSA_VALUE, (scope, result_id))
    assert result_ref is not None
    result_fact = book.page(SSA_VALUE).cell(
        result_ref.row, result_ref.column,
    )
    assert (
        result_fact.dtype, result_fact.physical_dtype,
        result_fact.storage, result_fact.rank,
    ) == ("float64", "float64", "span", 1)
    descriptor = module.tensor_tables[function.name].by_id(result_id)
    assert descriptor is not None and descriptor.metadata_state == "dynamic"
    assert (
        descriptor.shape_value_id, descriptor.rank_value_id,
        descriptor.element_count_value_id,
    ) == fact[8:11]
    owned_ref = book.latest_ref(
        TENSOR_OWNED_HEAP_BINDING, (scope, function.name, result_id),
    )
    domain_ref = book.latest_ref(TENSOR_REDUCTION_DOMAIN, row)
    assert owned_ref is not None and domain_ref is not None
    assert book.page(TENSOR_OWNED_HEAP_BINDING).cell(
        owned_ref.row, owned_ref.column,
    ) == (
        descriptor.data_value_id, descriptor.heap_owner_value_id,
        fact[10], "float64",
    )
    kernel = module.functions["reduce_dim_double"]
    typed_formals = tuple(book.latest_ref(
        SSA_VALUE, (function_scope_of(kernel), int(formal.id)),
    ) for formal in kernel.args[:2])
    assert all(ref is not None for ref in typed_formals)
    causes = {ref for ref, _ in book.edges_into(result_ref)}
    assert {owned_ref, domain_ref, *typed_formals} <= causes
    assert all(
        book.latest_ref(SSA_VALUE, (scope, value_id)) in causes
        for value_id in fact[5:11]
    )

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "runtime_axis_sum_cast")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    for data in (
        np.empty((0, 3), dtype=np.float64),
        np.asarray([[1.25, 2.5, -0.5]], dtype=np.float64),
        np.asarray([
            [1.25, 2.5, -0.5],
            [-3.5, 1.25, 0.5],
            [0.0, 0.0, 4.75],
        ], dtype=np.float64),
        np.asarray([[float(2**53)], [1.0], [-float(2**53)]], dtype=np.float64),
    ):
        expected = np.sum(data, axis=1).astype(np.int64).sum(dtype=np.int64)
        actual = np.asarray(
            artifact.prepare_execution({parameter: data}).run().buffers[output]
        )
        assert actual.dtype == np.dtype("int64")
        assert actual.item() == int(expected)
