"""Native output capacity and native-written logical extent stay distinct."""

import numpy as np
import pytest
from pathlib import Path
from copy import deepcopy
import hashlib

from src.compiler.runtime_abi import (
    apply_native_result_extents,
    runtime_abi_description,
)


def _description(**relations):
    return runtime_abi_description(
        entry="root", backend="c", extent_type="int64_t",
        buffer_order=(1, 2, 3, 4),
        buffer_dtypes=("int64", "int64", "int64", "int64"),
        buffer_shapes=((), (), (1,), (1,)),
        buffer_ranks=(1, 1, 1, 1),
        buffer_roles=("input", "output", "input", "input"),
        extent_order=((1, "dimension", 0),), **relations,
    )


@pytest.mark.parametrize("length", (0, 2, 4))
def test_native_length_presents_only_written_prefix(length):
    allocated = np.array([10, 20, 30, 40], dtype=np.int64)
    original = allocated.copy()
    buffers = {2: allocated, 3: np.array([length], dtype=np.int64)}
    apply_native_result_extents(
        buffers, {2: allocated}, length_sources=((2, 3),),
    )
    assert buffers[2].tolist() == original[:length].tolist()
    assert np.shares_memory(buffers[2], allocated) is (length > 0)
    np.testing.assert_array_equal(allocated, original)


def test_native_shape_vector_presents_contiguous_nonlast_axis_result():
    # A top-k result along axis 0 of a (3, 4) input has physical capacity
    # twelve cells, but the native pair reports a (2, 4) logical shape.
    allocated = np.arange(12, dtype=np.int64).reshape(3, 4)
    buffers = {2: allocated, 4: np.array([2, 4], dtype=np.int64)}
    apply_native_result_extents(
        buffers, {2: allocated}, shape_sources=((2, 4),),
    )
    assert buffers[2].shape == (2, 4)
    assert buffers[2].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert np.shares_memory(buffers[2], allocated)
    assert allocated.shape == (3, 4)


def test_repeated_native_lengths_use_one_proven_capacity():
    allocated = np.array([10, 20, 30, 40], dtype=np.int64)
    buffers = {2: allocated, 3: np.array([2], dtype=np.int64)}
    for count in (2, 4, 0, 3):
        buffers[2] = allocated
        buffers[3][0] = count
        apply_native_result_extents(
            buffers, {2: allocated}, length_sources=((2, 3),),
        )
        assert buffers[2].tolist() == allocated[:count].tolist()
    assert allocated.tolist() == [10, 20, 30, 40]


@pytest.mark.parametrize("bad", (
    np.array([-1], dtype=np.int64),
    np.array([5], dtype=np.int64),
    np.array([2.0], dtype=np.float64),
    np.array([2, 3], dtype=np.int64),
))
def test_invalid_native_length_refuses_without_changing_any_result(bad):
    first = np.array([7, 8, 9, 10], dtype=np.int64)
    second = np.array([11, 12, 13, 14], dtype=np.int64)
    buffers = {2: first, 5: second, 3: np.array([2], dtype=np.int64), 6: bad}
    with pytest.raises(ValueError, match="native result length"):
        apply_native_result_extents(
            buffers, {2: first, 5: second},
            length_sources=((2, 3), (5, 6)),
        )
    assert buffers[2] is first and buffers[5] is second


@pytest.mark.parametrize("bad", (
    np.array([-1, 2], dtype=np.int64),
    np.array([4, 4], dtype=np.int64),
    np.array([3.0, 2.0], dtype=np.float64),
    np.array([2], dtype=np.int64),
))
def test_invalid_native_shape_refuses_atomically(bad):
    first = np.arange(12, dtype=np.int64).reshape(3, 4)
    second = np.arange(6, dtype=np.int64).reshape(2, 3)
    buffers = {
        2: first, 5: second,
        3: np.array([3, 2], dtype=np.int64), 6: bad,
    }
    with pytest.raises(ValueError, match="native result shape"):
        apply_native_result_extents(
            buffers, {2: first, 5: second},
            shape_sources=((2, 3), (5, 6)),
        )
    assert buffers[2] is first and buffers[5] is second


def test_v1_extent_contract_is_unchanged_without_computed_output():
    abi = _description(output_extent_sources=((2, 1),))
    assert abi["schema"] == "turing-runtime-abi-v1"
    assert abi["buffers"][1]["extents_from"] == 0
    assert "capacity_from" not in abi["buffers"][1]
    assert "logical_length_from" not in abi["buffers"][1]


def test_computed_output_contract_names_capacity_and_native_length():
    abi = _description(
        output_capacity_sources=((2, 1),),
        output_length_sources=((2, 3),),
    )
    assert abi["schema"] == "turing-runtime-abi-v2"
    assert abi["buffers"][1]["capacity_from"] == 0
    assert abi["buffers"][1]["logical_length_from"] == 2
    assert "extents_from" not in abi["buffers"][1]


def test_computed_rank_two_output_names_native_shape_vector():
    abi = runtime_abi_description(
        entry="root", backend="c", extent_type="int64_t",
        buffer_order=(1, 2, 3),
        buffer_dtypes=("int64", "int64", "int64"),
        buffer_shapes=((), (), (2,)),
        buffer_ranks=(2, 2, 1),
        buffer_roles=("input", "output", "input"),
        extent_order=((1, "dimension", 0), (1, "dimension", 1)),
        output_capacity_sources=((2, 1),),
        output_shape_sources=((2, 3),),
    )
    assert abi["schema"] == "turing-runtime-abi-v2"
    assert abi["buffers"][1]["capacity_from"] == 0
    assert abi["buffers"][1]["logical_shape_from"] == 2


@pytest.mark.parametrize("relations", (
    {"output_capacity_sources": ((2, 1), (2, 4))},
    {"output_length_sources": ((2, 3), (2, 4))},
    {"output_shape_sources": ((2, 3), (2, 4))},
    {"output_length_sources": ((2, 3),), "output_shape_sources": ((2, 4),)},
    {"output_capacity_sources": ((2, 2),)},
    {"output_capacity_sources": ((2, 99),)},
))
def test_ambiguous_or_unbound_computed_output_relation_refuses(relations):
    with pytest.raises(ValueError):
        _description(**relations)


def test_runtime_abi_receipt_audit_rejects_forged_capacity_index():
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.concordance_declarations import (
        ArtifactFact, ArtifactPart, Backend, EMISSION_ARTIFACT,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    source = """def root(packet: bytes, n: int) -> bytes:
    return packet[:n]
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "packet", "storage": "span",
             "dtype": "int64", "rank": 1, "python_type": "builtins.bytes"},
            {"function": "root", "parameter": "n", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    artifact = emit_ssa_module_to_c(module, "root__root", entry_name="slices_native")
    assert artifact.complete, artifact.shortfalls
    assert CorrelationTable._runtime_result_abi_findings(module) == []

    book = module.metadata["identity_book"]
    row = ("slices_native", Backend.C_MODULE, ArtifactPart.RUNTIME_ABI)
    page = book.page(EMISSION_ARTIFACT)
    cell = book.latest_ref(EMISSION_ARTIFACT, row)
    original = page.cell(cell.row, cell.column)
    forged = deepcopy(original.location[0])
    output = next(buffer for buffer in forged["buffers"]
                  if "capacity_from" in buffer)
    output["capacity_from"] = next(
        buffer["index"] for buffer in forged["buffers"]
        if buffer["value_id"] != output["value_id"]
        and buffer["index"] != output["capacity_from"]
    )
    payload = repr(forged).encode("utf-8")
    page.cells[(cell.row, cell.column)] = ArtifactFact(
        hashlib.sha256(payload).hexdigest(), len(payload), (forged,),
    )
    try:
        assert CorrelationTable._runtime_result_abi_findings(module)
    finally:
        page.cells[(cell.row, cell.column)] = original
