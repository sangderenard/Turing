from __future__ import annotations

import numpy as np

from src.common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from src.compiler.llvm_training_runtime import (
    NativeParameterGroup,
    compile_native_training_schedule,
    run_parameter_group,
)


def test_native_parameter_groups_step_only_the_selected_state(tmp_path):
    program = SSATensorProgram("independent_parameter_groups")
    x, weight, bias, target = [
        SSATensorOperations.input(program, shape)
        for shape in ((2, 1), (1, 1), (1, 1), (2, 1))
    ]
    prediction = x @ weight + bias
    residual = prediction - target
    loss = (residual * residual).mean()
    schedule = compile_native_training_schedule(
        loss,
        bindings={
            "x": x, "weight": weight, "bias": bias, "target": target,
        },
        parameter_groups=(
            NativeParameterGroup("generator", (1,), 0.1),
            NativeParameterGroup("discriminator", (2,), 0.1),
        ),
        observed_outputs={"prediction": int(prediction.data.value.id)},
        name="independent_parameter_groups",
        directory=tmp_path,
    )
    buffers = {
        0: np.asarray([[1.0], [2.0]]),
        1: np.asarray([[0.25]]),
        2: np.asarray([[0.1]]),
        3: np.asarray([[0.0], [1.0]]),
    }

    before_weight = buffers[1].copy()
    before_bias = buffers[2].copy()
    run_parameter_group(schedule, "generator", buffers)
    assert not np.array_equal(buffers[1], before_weight)
    np.testing.assert_array_equal(buffers[2], before_bias)

    before_weight = buffers[1].copy()
    before_bias = buffers[2].copy()
    run_parameter_group(schedule, "discriminator", buffers)
    np.testing.assert_array_equal(buffers[1], before_weight)
    assert not np.array_equal(buffers[2], before_bias)
