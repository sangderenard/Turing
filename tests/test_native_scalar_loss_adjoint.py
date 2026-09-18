"""Scalar loss joins must retain the specialized unit adjoint in native code."""

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from src.compiler.llvm_training_runtime import compile_native_graph_reverse
from src.compiler.ssa_llvm_backend import prepare_artifact_execution


@pytest.mark.parametrize("unit_seed", [True, False])
def test_native_sum_of_scalar_losses_preserves_both_unit_adjoints(tmp_path, unit_seed):
    program = SSATensorProgram("scalar_loss_join")
    left = SSATensorOperations.input(program, (2, 3))
    right = SSATensorOperations.input(program, (2, 3))
    loss = (left * left).sum() + (right * right).sum()
    ids = [int(value.data.value.id) for value in (left, right)]
    reverse = compile_native_graph_reverse(
        loss, bindings={"left": left, "right": right}, wrt_value_ids=ids,
        name="scalar_loss_join_reverse", directory=tmp_path,
        unit_output_seed=unit_seed,
    )
    for offset in (0.0, 0.25):
        values = {
            ids[0]: np.arange(6, dtype=np.float64).reshape(2, 3) - 2 + offset,
            ids[1]: np.arange(6, dtype=np.float64).reshape(2, 3) * -0.3 + offset,
        }
        seed = 1.0 if unit_seed else 2.5
        feeds = {**values, **{
            seed_id: np.asarray(seed)
            for seed_id in reverse.seed_value_ids.values()
        }}
        execution = prepare_artifact_execution(reverse.artifact, feeds)
        # Poison outputs so a missing native write fails deterministically.
        for gradient_id in reverse.gradient_value_ids.values():
            execution.buffers[gradient_id].fill(np.nan)
        execution.run()
        for value_id, value in values.items():
            np.testing.assert_allclose(
                execution.buffers[reverse.gradient_value_ids[value_id]],
                2.0 * seed * value, rtol=1e-12, atol=1e-12,
            )
