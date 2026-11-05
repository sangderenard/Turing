"""A runtime axis sum of int64 data keeps every integer limb."""

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


def test_runtime_axis_int64_sum_uses_typed_owned_output(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 2,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = (
        "def root(values: AbstractTensor) -> int:\n"
        "    columns = values.sum(dim=0)\n"
        "    return columns.sum().item()\n"
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="runtime_axis_int64_sum_native",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(TENSOR_REDUCTION_DOMAIN)
    matches = tuple(
        row for row in page.rows()
        if row[0].startswith("runtime_axis_int64_sum_native")
        and (fact := page.latest(row))
        and fact[:3] == ("axis-runtime", 0, "sum")
    )
    assert len(matches) == 1
    row = matches[0]
    fact = page.latest(row)
    function = module.functions[row[0]]
    scope = function_scope_of(function)
    result_id = row[1]
    calls = tuple(
        instruction for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "sum_dim_int64"
    )
    assert len(calls) == 1
    assert calls[0].res.id == result_id
    assert tuple(arg.id for arg in calls[0].args[2:4]) == fact[5:7]
    assert len(calls[0].args) == 5

    result_ref = book.latest_ref(SSA_VALUE, (scope, result_id))
    assert result_ref is not None
    result_fact = book.page(SSA_VALUE).cell(result_ref.row, result_ref.column)
    assert (
        result_fact.dtype, result_fact.physical_dtype,
        result_fact.storage, result_fact.rank,
    ) == ("int64", "int64", "span", 1)
    descriptor = module.tensor_tables[function.name].by_id(result_id)
    assert descriptor is not None and descriptor.metadata_state == "dynamic"
    assert (
        descriptor.shape_value_id, descriptor.rank_value_id,
        descriptor.element_count_value_id,
    ) == fact[8:11]
    owner_ref = book.latest_ref(
        TENSOR_OWNED_HEAP_BINDING, (scope, function.name, result_id),
    )
    domain_ref = book.latest_ref(TENSOR_REDUCTION_DOMAIN, row)
    assert owner_ref is not None
    assert domain_ref is not None
    assert book.page(TENSOR_OWNED_HEAP_BINDING).cell(
        owner_ref.row, owner_ref.column,
    ) == (
        descriptor.data_value_id, descriptor.heap_owner_value_id,
        fact[10], "int64",
    )
    selected = module.functions["sum_dim_int64"]
    assert selected.metadata.get("ssa_output_argument") == 1
    formal_refs = tuple(
        book.latest_ref(SSA_VALUE, (function_scope_of(selected), formal.id))
        for formal in selected.args[:2]
    )
    assert all(ref is not None for ref in formal_refs)
    causes = {ref for ref, _ in book.edges_into(result_ref)}
    assert {owner_ref, domain_ref, *formal_refs} <= causes

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "runtime_axis_int64_sum")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    for values in (
        np.empty((0, 3), dtype=np.int64),
        np.asarray([[2**53, 1, -(2**53)]], dtype=np.int64),
        np.asarray([[2**53], [1], [-(2**53)]], dtype=np.int64),
        np.asarray([[np.iinfo(np.int64).max], [1]], dtype=np.int64),
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
    ):
        expected = np.sum(values, axis=0, dtype=np.int64).sum(dtype=np.int64)
        actual = np.asarray(
            artifact.prepare_execution({parameter: values}).run().buffers[output]
        )
        assert actual.dtype == np.dtype("int64")
        assert actual.item() == int(expected)


def test_runtime_axis_int64_sum_rank_one_scalar_native(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(values: AbstractTensor) -> int:\n"
        "    return values.sum(dim=0).item()\n",
        "root", name="runtime_axis_int64_sum_scalar_native",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    domain_page = book.page(TENSOR_REDUCTION_DOMAIN)
    domain_row, = (
        row for row in domain_page.rows()
        if (fact := domain_page.latest(row))
        and fact[0] == "axis-runtime" and fact[2] == "sum"
    )
    fact = domain_page.latest(domain_row)
    function = module.functions[domain_row[0]]
    result = book.page(SSA_VALUE).latest((function_scope_of(function), domain_row[1]))
    assert (result.dtype, result.physical_dtype,
            result.storage, result.rank) == ("int64", "int64", "scalar", 0)
    assert fact[8:11] != fact[5:8]
    assert module.tensor_tables[function.name].by_id(
        domain_row[1]
    ).metadata_state == "static"
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "runtime_axis_int64_sum_scalar")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    for values in (
        np.empty((0,), dtype=np.int64),
        np.asarray([2**53, 1, -(2**53)], dtype=np.int64),
        np.asarray([np.iinfo(np.int64).max, 1], dtype=np.int64),
        np.asarray([3, 4, 5], dtype=np.int64),
    ):
        expected = int(np.sum(values, dtype=np.int64))
        execution = artifact.prepare_execution({parameter: values})
        for _ in range(2):
            actual = np.asarray(execution.run().buffers[output])
            assert actual.dtype == np.dtype("int64")
            assert actual.item() == expected


def test_runtime_axis_int64_sum_keepdim_native(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 2,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(values: AbstractTensor) -> int:\n"
        "    return values.sum(dim=0, keepdim=True).sum().item()\n",
        "root", name="runtime_axis_int64_sum_keepdim_native",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "runtime_axis_int64_sum_keepdim")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    for values in (
        np.empty((0, 3), dtype=np.int64),
        np.asarray([[2**53, 1, -(2**53)]], dtype=np.int64),
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
    ):
        expected = int(np.sum(
            np.sum(values, axis=0, keepdims=True, dtype=np.int64),
            dtype=np.int64,
        ))
        actual = np.asarray(
            artifact.prepare_execution({parameter: values}).run().buffers[output]
        )
        assert actual.dtype == np.dtype("int64")
        assert actual.item() == expected
