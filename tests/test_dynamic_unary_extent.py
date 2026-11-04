"""Dynamic int64 unary results keep their exact input extent and native lanes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_ELEMENTWISE_EXTENT_BINDING, TENSOR_OWNED_HEAP_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import function_scope_of


@pytest.mark.parametrize(
    ("operation", "body"),
    (
        ("neg", "    return (-data).sum().item()\n"),
        ("neg", "    priority = data * (count + 1) + index\n"
                "    return (-priority).sum().item()\n"),
        ("abs", "    priority = data * (count + 1) + index\n"
                "    return priority.abs().sum().item()\n"),
    ),
    ids=("simple-neg", "priority-neg", "priority-abs"),
)
def test_dynamic_int64_unary_inherits_extent_and_runs_natively(
    tmp_path, operation, body,
):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "data", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = (
        "def root(data: AbstractTensor) -> int:\n"
        + (
            "    count = data.shape[0]\n"
            "    index = AbstractTensor.arange(count, cls=type(data))\n"
            if "priority" in body else ""
        )
        + body
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name=f"dynamic_unary_{operation}_{'priority' in body}",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert not CorrelationTable.build(module).findings(module)

    book = identity_book(module)
    page = book.page(TENSOR_ELEMENTWISE_EXTENT_BINDING)
    unary = tuple(
        (row, page.latest(row)) for row in page.rows()
        if len(page.latest(row)) == 5 and page.latest(row)[0] == operation
    )
    assert len(unary) == 1
    row, fact = unary[0]
    descriptor = module.tensor_tables[row[1]].by_id(row[2])
    assert descriptor is not None and descriptor.metadata_state == "dynamic"
    assert (descriptor.shape_value_id, descriptor.rank_value_id,
            descriptor.element_count_value_id) == fact[2:]
    owner_ref = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, row)
    assert owner_ref is not None
    carrier, owner, count, dtype = book.page(TENSOR_OWNED_HEAP_BINDING).cell(
        owner_ref.row, owner_ref.column,
    )
    assert (descriptor.data_value_id, descriptor.heap_owner_value_id,
            descriptor.element_count_value_id, descriptor.dtype) == (
                carrier, owner, count, dtype,
            )

    function = module.functions[row[1]]
    unary_instruction, = (
        instruction for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None and instruction.res.id == row[2]
    )
    original_args = unary_instruction.args
    unary_instruction.args = [unary_instruction.res]
    try:
        assert CorrelationTable._tensor_elementwise_extent_binding_findings(module)
    finally:
        unary_instruction.args = original_args

    table = module.tensor_tables[row[1]]
    table.tensors[row[2]] = replace(
        descriptor, element_count_value_id=descriptor.shape_value_id,
    )
    try:
        assert CorrelationTable._tensor_elementwise_extent_binding_findings(module)
    finally:
        table.tensors[row[2]] = descriptor

    values = book.page(SSA_VALUE)
    source_ref = book.latest_ref(
        SSA_VALUE, (function_scope_of(function), fact[1]),
    )
    assert source_ref is not None
    source_fact = values.cell(source_ref.row, source_ref.column)
    assert source_fact.physical_dtype == "int64"
    values.cells[(source_ref.row, source_ref.column)] = replace(
        source_fact, physical_dtype=None,
    )
    try:
        assert CorrelationTable._tensor_elementwise_extent_binding_findings(module)
    finally:
        values.cells[(source_ref.row, source_ref.column)] = source_fact

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "dynamic_unary")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["data"]
    output, = [value.id for block in root.blocks.values()
               for instruction in block.instrs if instruction.op == "Ret"
               for value in instruction.args]
    for values in ((2**53 + 1, -7, 3), ()):
        data = np.asarray(values, dtype=np.int64)
        priority = data * (len(data) + 1) + np.arange(len(data), dtype=np.int64)
        selected = data if "priority" not in body else priority
        transformed = -selected if operation == "neg" else np.abs(selected)
        expected = int(transformed.sum())
        actual = np.asarray(artifact.prepare_execution({parameter: data}).run().buffers[output])
        assert actual.dtype == np.dtype("int64")
        assert actual.item() == expected


def test_dynamic_float64_unary_uses_exact_native_geometry(tmp_path):
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": "root", "parameter": "data", "storage": "span",
                    "dtype": "float64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(data: AbstractTensor) -> float:\n    return (-data).sum().item()\n",
        "root", name="dynamic_unary_float64", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert not CorrelationTable.build(module).findings(module)
    page = identity_book(module).page(TENSOR_ELEMENTWISE_EXTENT_BINDING)
    assert any(len(page.latest(row)) == 5 and page.latest(row)[0] == "neg" for row in page.rows())
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "float64_unary")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["data"]
    output, = [value.id for block in root.blocks.values() for instruction in block.instrs
               if instruction.op == "Ret" for value in instruction.args]
    for values in ((), (0.5, -1.25, 4.0)):
        data = np.asarray(values, dtype=np.float64)
        actual = artifact.prepare_execution({parameter: data}).run().buffers[output]
        np.testing.assert_array_equal(actual, (-data).sum())
