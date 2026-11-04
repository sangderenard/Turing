"""A tensor index span selects every requested element or row in native C."""

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize(
    ("data", "indices", "expected", "data_dtype"),
    (
        ((1, 2, 3, 4), (1, 3), (2, 4), np.int64),
        ((1, 2, 3, 4), (-1, 1), (4, 2), np.int64),
        ((2**53, 2**53 + 1, 2**53 + 2, 2**53 + 3),
         (1, 3), (2**53 + 1, 2**53 + 3), np.int64),
        (((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)),
         (3, 1), ((9, 10, 11), (3, 4, 5)), np.int64),
        ((((-1, 0), (1, 2), (3, 4)),
          ((5, 6), (7, 8), (9, 10))),
         (1, 0),
         (((5, 6), (7, 8), (9, 10)),
          ((-1, 0), (1, 2), (3, 4))), np.int64),
        ((1.25, 2.5, 3.75, 5.0), (1, 3), (2.5, 5.0), np.float64),
    ),
    ids=("elements", "negative-index", "int64-above-float-exactness",
         "rows", "rank3-blocks", "float64-elements"),
)
def test_native_tensor_indexed_gather_writes_entire_result(
    tmp_path, data, indices, expected, data_dtype,
):
    data_array = np.asarray(data, dtype=data_dtype)
    indices_array = np.asarray(indices, dtype=np.int64)
    artifact, root = _compile_direct_gather(tmp_path, data_array, indices_array)
    names = dict(root.metadata["parameter_names"])
    feeds = {
        int(names["data"]): data_array,
        int(names["indices"]): indices_array,
    }
    buffers = artifact.prepare_execution(feeds).run().buffers
    result_ids = [argument.id for block in root.blocks.values()
                  for instruction in block.instrs if instruction.op == "Ret"
                  for argument in instruction.args]
    assert len(result_ids) == 1
    actual = np.asarray(buffers[result_ids[0]])
    assert actual.dtype == data_array.dtype
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=data_array.dtype))


def _compile_direct_gather(tmp_path, data_array, indices_array):
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {
                "function": "root", "parameter": name, "storage": "span",
                "dtype": str(array.dtype), "rank": array.ndim,
                "shape": list(array.shape),
                "python_type": "src.common.tensors.abstraction.AbstractTensor",
            }
            for name, array in (("data", data_array), ("indices", indices_array))
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        "def root(data: AbstractTensor, indices: AbstractTensor) -> AbstractTensor:\n"
        "    return data[indices]\n",
        "root", name="native_tensor_indexed_gather",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native_tensor_indexed_gather")
    root = module.functions[exports[0]]
    return artifact, root


def test_native_tensor_indexed_gather_rejects_out_of_range_indices(tmp_path):
    data = np.asarray((1, 2, 3, 4), dtype=np.int64)
    artifact, root = _compile_direct_gather(
        tmp_path, data, np.asarray((1, 3), dtype=np.int64),
    )
    names = dict(root.metadata["parameter_names"])
    for invalid in ((4, 1), (-5, 1)):
        feeds = {
            int(names["data"]): data,
            int(names["indices"]): np.asarray(invalid, dtype=np.int64),
        }
        with pytest.raises(IndexError):
            artifact.prepare_execution(feeds).run()


@pytest.mark.parametrize(
    ("data_shape", "declared_data_shape", "cases"),
    (
        ((4,), True, (((1, 3), 6), ((3,), 4), ((), 0))),
        ((4, 3), False, (((3, 1), 42), ((3,), 30), ((), 0))),
    ),
    ids=("rank1-fixed-data", "rank2-runtime-data"),
)
def test_private_tensor_indexed_gather_uses_runtime_index_count(
    tmp_path, data_shape, declared_data_shape, cases,
):
    data_array = np.arange(1, 5, dtype=np.int64) if declared_data_shape else (
        np.arange(12, dtype=np.int64).reshape(data_shape)
    )
    data_abi = {
        "function": "root", "parameter": "data", "storage": "span",
        "dtype": "int64", "rank": len(data_shape),
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }
    if declared_data_shape:
        data_abi["shape"] = list(data_shape)
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            data_abi,
            {
                "function": "root", "parameter": "indices", "storage": "span",
                "dtype": "int64", "rank": 1,
                "python_type": "src.common.tensors.abstraction.AbstractTensor",
            },
        ],
        "records": {}, "bindings": [],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        "def root(data: AbstractTensor, indices: AbstractTensor) -> int:\n"
        "    leaves = data[indices]\n"
        "    return leaves.sum().item()\n",
        "root", name="native_private_tensor_indexed_gather",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native_private_tensor_indexed_gather")
    root = module.functions[exports[0]]
    names = dict(root.metadata["parameter_names"])
    output_ids = [argument.id for block in root.blocks.values()
                  for instruction in block.instrs if instruction.op == "Ret"
                  for argument in instruction.args]
    assert len(output_ids) == 1
    for index_values, expected in cases:
        feeds = {
            int(names["data"]): data_array,
            int(names["indices"]): np.asarray(index_values, dtype=np.int64),
        }
        execution = artifact.prepare_execution(feeds).run()
        actual = np.asarray(execution.buffers[output_ids[0]])
        assert actual.dtype == np.dtype(np.int64)
        assert actual.item() == expected
    if declared_data_shape:
        int64_max = np.iinfo(np.int64).max
        exact_cases = (
            ((0, 2**53 + 1, 0, 0), (1,), 2**53 + 1),
            ((0, 2**53 + 1, 0, 0), (1, 1), 2**54 + 2),
            ((int64_max, 1, 0, 0), (0, 1), np.iinfo(np.int64).min),
        )
        for data_values, index_values, expected in exact_cases:
            feeds = {
                int(names["data"]): np.asarray(data_values, dtype=np.int64),
                int(names["indices"]): np.asarray(index_values, dtype=np.int64),
            }
            execution = artifact.prepare_execution(feeds).run()
            actual = np.asarray(execution.buffers[output_ids[0]])
            assert actual.dtype == np.dtype(np.int64)
            assert actual.item() == expected
