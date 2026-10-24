"""Linked SSA coverage for Token Orchard's padded COO target construction."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from tests.test_runtime_sized_spans import BACKENDS, _run


CONTRACT = "extraction_contracts/program_extraction.yaml"

SOURCE = """
def position(tokens):
    token_column = tokens.reshape((-1, 1))
    dimensions = AbstractTensor.arange(0, 3, 1, dtype="int64").reshape((1, -1))
    mixed = (token_column * 1103515245 + (dimensions + 1) * 12345 + 1013904223) % 2147483647
    mixed = (mixed * (dimensions * 2 + 3) + 69069) % 2147483647
    coordinates = mixed.to_dtype("float64") / 1073741823.5 - 1.0
    radius = (coordinates * coordinates).sum(dim=1, keepdim=True).sqrt()
    return coordinates / radius


def ordered(child_vectors, parent_index, ordinal, parent_length):
    dimensions = AbstractTensor.arange(0, 3, 1, dtype="int64").reshape((1, -1))
    ordinal_column = ordinal.reshape((-1, 1))
    length_column = parent_length.reshape((-1, 1))
    mixed = ((ordinal_column + 1) * (dimensions * 2 + 3) + (length_column + 1) * (dimensions + 5) + 8191) % 104729
    modulation = mixed.to_dtype("float64") / 104728.0 + 0.5
    contributions = child_vectors * modulation
    sums = AbstractTensor.scatter(
        AbstractTensor.zeros((2, 3), dtype="float64"),
        parent_index,
        contributions,
        dim=0,
        reduce="sum",
    )
    counts = AbstractTensor.scatter(
        AbstractTensor.zeros((2, 1), dtype="float64"),
        parent_index,
        AbstractTensor.full((4, 1), 1.0, dtype="float64"),
        dim=0,
        reduce="sum",
    )
    return sums / counts


def root(child_tokens, parent_index, ordinal, parent_length):
    return ordered(position(child_tokens), parent_index, ordinal, parent_length)
"""


def _lower_target_graph(source=SOURCE):
    shapes = {
        "child_tokens": ((4,), "int64"),
        "parent_index": ((4,), "int64"),
        "ordinal": ((4,), "int64"),
        "parent_length": ((4,), "int64"),
    }
    contract = ExtractionContract(CONTRACT).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": "root",
                "parameter": parameter,
                "storage": "span",
                "dtype": dtype,
                "rank": len(shape),
                "shape": list(shape),
                "python_type": "src.common.tensors.abstraction.AbstractTensor",
            }
            for parameter, (shape, dtype) in shapes.items()
        ],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        source,
        "root",
        name="token_orchard_target_link",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
        progress_bars=False,
    )
    entry = exports[0]
    function = module.functions[entry]
    parameters = dict(function.metadata["parameter_names"])
    returned = next(
        instruction.args[0]
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    return module, entry, parameters, returned


def test_scatter_reduction_policy_survives_capture_and_refuses_wrong_kernel():
    with pytest.raises(
        ValueError,
        match=(
            "scatter reduce='mean' has no matching repository SSA kernel; "
            "only reduce='sum' is currently implemented"
        ),
    ):
        _lower_target_graph(
            SOURCE.replace('reduce="sum"', 'reduce="mean"', 1)
        )


def _reference(child_tokens, parent_index, ordinal, parent_length):
    namespace = {"AbstractTensor": AbstractTensor}
    exec(SOURCE, namespace)
    with AbstractTensor.use_backend("numpy", "cpu"):
        result = namespace["root"](
            AbstractTensor.tensor(child_tokens, dtype="int64"),
            AbstractTensor.tensor(parent_index, dtype="int64"),
            AbstractTensor.tensor(ordinal, dtype="int64"),
            AbstractTensor.tensor(parent_length, dtype="int64"),
        )
        return np.asarray(result.data)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "child_tokens",
    (
        np.asarray([1114368, 1114369, 1114370, 1114371], dtype=np.int64),
        np.asarray([4294967292, 4294967293, 4294967294, 4294967295], dtype=np.int64),
    ),
    ids=("aggregate-base", "binary64-inexact-as-double"),
)
def test_padded_coo_target_graph_executes_through_linked_ssa(
    tmp_path, backend, child_tokens,
):
    module, entry, parameters, returned = _lower_target_graph()
    parent_index = np.asarray([0, 0, 0, 1], dtype=np.int64)
    ordinal = np.asarray([0, 1, 2, 0], dtype=np.int64)
    parent_length = np.asarray([3, 3, 3, 1], dtype=np.int64)
    expected = _reference(
        child_tokens, parent_index, ordinal, parent_length,
    )
    artifact, execution = _run(
        backend,
        module,
        entry,
        {
            parameters["child_tokens"]: child_tokens,
            parameters["parent_index"]: parent_index,
            parameters["ordinal"]: ordinal,
            parameters["parent_length"]: parent_length,
        },
        tmp_path,
    )
    assert getattr(artifact, "complete", True)
    actual = np.asarray(execution.buffers[int(returned.id)]).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
