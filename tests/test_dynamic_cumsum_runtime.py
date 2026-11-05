"""A runtime-length cumsum keeps its source geometry and owns its output."""

from pathlib import Path

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    TENSOR_OWNED_HEAP_BINDING, TENSOR_RUNTIME_SCAN_GEOMETRY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_runtime_float64_cumsum_private_result_native(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "float64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = (
        "def root(values: AbstractTensor) -> float:\n"
        "    scanned = values.cumsum(dim=0)\n"
        "    return scanned.sum().item()\n"
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="dynamic_cumsum_native",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY)
    rows = tuple(row for row in page.rows()
                 if row[1].startswith("dynamic_cumsum_native"))
    assert len(rows) == 1
    row = rows[0]
    operation, source_id, axis_id, normalized_axis, shape_id, rank_id, count_id, carrier_id, callee = page.latest(row)
    assert (operation, normalized_axis, callee) == (
        "cumsum", 0, "cumsum_dim_double",
    )
    table = module.tensor_tables[row[1]]
    descriptor = table.by_id(row[2])
    assert descriptor is not None and descriptor.metadata_state == "dynamic"
    assert (descriptor.shape_value_id, descriptor.rank_value_id,
            descriptor.element_count_value_id, descriptor.data_value_id) == (
                shape_id, rank_id, count_id, carrier_id,
            )
    owner_ref = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, row)
    assert owner_ref is not None
    assert book.page(TENSOR_OWNED_HEAP_BINDING).cell(
        owner_ref.row, owner_ref.column,
    ) == (carrier_id, descriptor.heap_owner_value_id, count_id, "float64")
    function = module.functions[row[1]]
    selected, = (
        instruction for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call" and instruction.res is not None
        and instruction.res.id == row[2]
    )
    assert selected.attributes["callee"] == callee
    assert selected.attributes["ssa_output_argument"] == 1
    assert tuple(value.id for value in selected.args[:4]) == (
        source_id, carrier_id, shape_id, rank_id,
    )
    axis_constants = tuple(
        instruction for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Const" and instruction.res is not None
        and instruction.res.id == selected.args[4].id
    )
    assert len(axis_constants) == 1
    assert axis_constants[0].attributes.get("constant") == normalized_axis
    assert axis_id != selected.args[4].id

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "dynamic_cumsum")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    for data in (
        np.array([], dtype=np.float64),
        np.array([3.25], dtype=np.float64),
        np.array([1.5, -2.0, 3.25, 0.0, 4.5, -1.75, 2.0], dtype=np.float64),
    ):
        actual = np.asarray(
            artifact.prepare_execution({parameter: data}).run().buffers[output]
        )
        assert actual.dtype == np.dtype("float64")
        assert actual.item() == np.cumsum(data).sum()


def test_runtime_int64_cumsum_private_result_native(tmp_path):
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
    source = (
        "def root(values: AbstractTensor) -> int:\n"
        "    scanned = values.cumsum(dim=0)\n"
        "    return scanned.sum().item()\n"
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="dynamic_int64_cumsum_native",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_SCAN_GEOMETRY)
    rows = tuple(row for row in page.rows()
                 if row[1].startswith("dynamic_int64_cumsum_native"))
    assert len(rows) == 1
    row = rows[0]
    fact = page.latest(row)
    assert fact[0] == "cumsum" and fact[3] == 0
    assert fact[8] == "cumsum_dim_int64"
    descriptor = module.tensor_tables[row[1]].by_id(row[2])
    assert descriptor is not None
    assert descriptor.metadata_state == "dynamic"
    assert descriptor.element_count_value_id == fact[6]
    assert descriptor.data_value_id == fact[7]
    owner_ref = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, row)
    assert owner_ref is not None
    assert book.page(TENSOR_OWNED_HEAP_BINDING).cell(
        owner_ref.row, owner_ref.column,
    ) == (fact[7], descriptor.heap_owner_value_id, fact[6], "int64")

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "dynamic_int64_cumsum")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    for data in (
        np.array([], dtype=np.int64),
        np.array([2**53 + 1], dtype=np.int64),
        np.array([2**53 + 1, -1, 3, 0, 4, -2, 1], dtype=np.int64),
        np.array([np.iinfo(np.int64).max, 1], dtype=np.int64),
    ):
        expected = np.cumsum(data).sum(dtype=np.int64)
        actual = np.asarray(
            artifact.prepare_execution({parameter: data}).run().buffers[output]
        )
        assert actual.dtype == np.dtype("int64")
        assert actual.item() == int(expected)


def test_static_int64_cumsum_uses_typed_kernel_native(tmp_path):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "shape": [3], "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = (
        "def root(values: AbstractTensor) -> int:\n"
        "    scanned = values.cumsum(dim=0)\n"
        "    return scanned.sum().item()\n"
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="static_int64_cumsum_native",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert any(
        instruction.op == "Call"
        and instruction.attributes.get("callee") == "cumsum_dim_int64"
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "static_int64_cumsum")
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    data = np.array([2**53 + 1, -1, 2], dtype=np.int64)
    actual = np.asarray(
        artifact.prepare_execution({parameter: data}).run().buffers[output]
    )
    assert actual.dtype == np.dtype("int64")
    assert actual.item() == int(np.cumsum(data).sum(dtype=np.int64))
