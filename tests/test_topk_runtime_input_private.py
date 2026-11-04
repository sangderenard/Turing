"""Runtime topk selection retains the authored input extent and owned leaves."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.compression import huffman
from src.compiler.concordance_declarations import (
    AGGREGATE_PASSTHROUGH_REBINDING, CALL_RECORD,
    TENSOR_METADATA_RESULT_BINDING, TENSOR_RUNTIME_BROADCAST_VALIDITY,
    TENSOR_RUNTIME_TOPK_SELECTION,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.precompile_to_ssa import SSALoweringIncompleteError
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def _contract(*parameters, rank=1):
    return ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "root", "parameter": name, "storage": "span",
             "dtype": "int64", "rank": rank,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"}
            for name in parameters
        ],
    })


def _lower(source, name, contract, *, emit=True):
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name=name,
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    assert tuple(module.metadata["identity_book"].page(
        TENSOR_RUNTIME_TOPK_SELECTION,
    ).rows())
    artifact = emit_ssa_module_to_c(module, exports[0]) if emit else None
    if artifact is not None:
        assert artifact.complete, artifact.shortfalls
    return module, exports[0], artifact


def _port(module, root_name, parameter):
    names = dict(module.functions[root_name].metadata["parameter_names"])
    return int(names[parameter])


def _result(module, root_name):
    return next(
        int(value.id)
        for block in module.functions[root_name].blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
        for value in instruction.args
    )


def test_runtime_input_and_other_axis_dimension_use_checked_min(tmp_path):
    module, root, artifact = _lower(
        "def root(values: AbstractTensor) -> int:\n"
        "    k = values.shape[1]\n"
        "    picked = AbstractTensor.topk(values, k=k, dim=0).indices\n"
        "    return picked.sum().item()\n",
        "topk_runtime_input", _contract("values", rank=2),
    )
    artifact.compile(tmp_path / "topk_runtime_input")
    for width, expected in ((0, 0), (1, 0), (2, 4), (4, 12)):
        values = np.repeat(np.array([
            [2**53 + 5], [2**53 + 1], [2**53 + 3],
        ], dtype=np.int64), width, axis=1)
        result = artifact.prepare_execution({
            _port(module, root, "values"): values,
        }).run().buffers[_result(module, root)]
        assert int(np.asarray(result).item()) == expected


def test_runtime_topk_zero_other_axis_keeps_zero_count(tmp_path):
    module, root, artifact = _lower(
        "def root(values: AbstractTensor) -> int:\n"
        "    k = values.shape[0]\n"
        "    picked = AbstractTensor.topk(values, k=k, dim=2).indices\n"
        "    return picked.sum().item()\n",
        "topk_runtime_zero_other_axis", _contract("values", rank=3),
    )
    artifact.compile(tmp_path / "topk_runtime_zero_other_axis")
    cases = (
        (np.empty((2, 0, 3), dtype=np.int64), 0),
        (np.array([[[1, 3, 2]], [[4, 6, 5]]], dtype=np.int64), 6),
    )
    for values, expected in cases:
        result = artifact.prepare_execution({
            _port(module, root, "values"): values,
        }).run().buffers[_result(module, root)]
        assert int(np.asarray(result).item()) == expected


def test_authored_ascending_order_runtime_topk_native(tmp_path):
    source = inspect.getsource(huffman._ascending_order)
    source += (
        "\ndef root(weights: AbstractTensor) -> int:\n"
        "    selected = _ascending_order(weights)\n"
        "    return (selected * weights).sum().item()\n"
    )
    module, root, artifact = _lower(
        source, "topk_runtime_ascending", _contract("weights"),
    )
    book = module.metadata["identity_book"]
    validity_page = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY)
    mixed_rows = tuple(row for row in validity_page.rows()
                       if validity_page.latest(row)[0] == "mul")
    assert len(mixed_rows) == 1
    mixed_ref = book.latest_ref(TENSOR_RUNTIME_BROADCAST_VALIDITY,
                                mixed_rows[0])
    causes = {cause for cause, _stage in book.edges_into(mixed_ref)}
    assert mixed_ref not in causes
    result_causes = tuple(cause for cause in causes
                          if cause.page == TENSOR_METADATA_RESULT_BINDING
                          and book.latest_ref(cause.page, cause.row) == cause)
    record_causes = tuple(cause for cause in causes
                          if cause.page == CALL_RECORD
                          and book.latest_ref(cause.page, cause.row) == cause)
    rebinding_causes = tuple(cause for cause in causes
                             if cause.page == AGGREGATE_PASSTHROUGH_REBINDING)
    assert len(result_causes) == len(record_causes) == 1
    assert len(rebinding_causes) == 3
    artifact.compile(tmp_path / "topk_runtime_ascending")
    for items in ((), (5,), (5, 3, 3, 1), (9, 2, 8, 2, 1, 1)):
        values = np.asarray(items, dtype=np.int64)
        expected = int(np.sum(
            np.argsort(values, kind="stable") * values,
            dtype=np.int64,
        ))
        result = artifact.prepare_execution({
            _port(module, root, "weights"): values,
        }).run().buffers[_result(module, root)]
        assert int(np.asarray(result).item()) == expected
    result_ref = result_causes[0]
    result_page = book.page(TENSOR_METADATA_RESULT_BINDING)
    original = result_page.cell(result_ref.row, result_ref.column)
    result_page.cells[(result_ref.row, result_ref.column)] = (
        original[0], (original[1][0] + 1, *original[1][1:]), original[2],
    )
    try:
        assert CorrelationTable._tensor_metadata_result_binding_findings(
            module)
    finally:
        result_page.cells[(result_ref.row, result_ref.column)] = original
    result_page.cells[(result_ref.row, result_ref.column)] = (
        original[0], original[1],
        (original[2][0] + 1, *original[2][1:]),
    )
    try:
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(
            module)
    finally:
        result_page.cells[(result_ref.row, result_ref.column)] = original


def test_runtime_topk_selection_rejects_changed_min_or_count():
    module, root, _ = _lower(
        "def root(values: AbstractTensor) -> int:\n"
        "    k = values.shape[1]\n"
        "    picked = AbstractTensor.topk(values, k=k, dim=0).indices\n"
        "    return picked.sum().item()\n",
        "topk_runtime_selection_audit", _contract("values", rank=2),
        emit=False,
    )
    page = module.metadata["identity_book"].page(TENSOR_RUNTIME_TOPK_SELECTION)
    row = next(iter(page.rows()))
    fact = page.latest(row)
    region = module.functions[row[1]]
    selected = next(
        instruction for block in region.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
        and instruction.res.id == fact.selection_cell.row[-1]
    )
    selected.args.reverse()
    assert any(finding.kind == "tensor-runtime-topk-selection-disagreement"
               for finding in CorrelationTable.build(module).findings(module))
    selected.args.reverse()
    assert CorrelationTable.build(module).findings(module) == []
    descriptor = module.tensor_tables[region.name].tensors[row[2]]
    from dataclasses import replace
    module.tensor_tables[region.name].tensors[row[2]] = replace(
        descriptor, element_count_value_id=fact.input_count_cell.row[-1],
    )
    assert any(finding.kind == "tensor-runtime-topk-selection-disagreement"
               for finding in CorrelationTable.build(module).findings(module))


def test_runtime_topk_unproved_scalar_k_keeps_named_shortfall():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "root", "parameter": "values", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "k", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    with pytest.raises(
        SSALoweringIncompleteError,
        match="dynamic named topk lacks exact nonnegative guarded k domain|"
              "runtime named topk k lacks exact nonnegative source dimension",
    ):
        lower_ast_source_to_ssa(
            "def root(values: AbstractTensor, k: int) -> int:\n"
            "    return AbstractTensor.topk(values, k=k, dim=0).indices.sum().item()\n",
            "root", name="topk_unproved_scalar_k",
            python_bindings={"AbstractTensor": AbstractTensor},
            extraction_contract=contract,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
