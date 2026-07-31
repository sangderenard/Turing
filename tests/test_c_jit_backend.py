from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.artifact_cache import (
    RepositoryArtifactCache,
)
from src.common.tensors.accelerator_backends.c_jit_backend import (
    compile_torture_case_to_c,
)
from src.common.tensors.accelerator_backends.tensor_torture import (
    capture_torture_case,
    tensor_torture_cases,
)
from src.compiler.glsl_deployment_strategy import DeploymentProfiler


CASES = {case.name: case for case in tensor_torture_cases()}


@pytest.mark.parametrize(
    "name",
    (
        "add",
        "scalar_broadcast",
        "sin",
        "where",
        "flat_sum",
        "dim_sum",
        "cumsum",
        "matmul",
        "operator_grab_bag",
        "advanced_tensor_topology",
    ),
)
def test_c_jit_torture_executes_whole_tape_through_c_shell(name, tmp_path):
    case = CASES[name]
    program = compile_torture_case_to_c(
        capture_torture_case(case),
        cache=RepositoryArtifactCache(tmp_path),
    )
    profiler = DeploymentProfiler(enabled=True)
    execution = program.execute(
        case.inputs,
        profiler=profiler,
        profile_path=f"torture/c/{name}",
    )

    for output_name, expected in case.numpy_reference().items():
        np.testing.assert_allclose(
            execution.outputs[output_name],
            expected,
            rtol=case.rtol,
            atol=case.atol,
        )
    assert execution.profile.status == 1
    assert execution.profile.shell_ns > 0
    assert execution.profile.device_ns == 0
    assert any(
        row["section"] == "compiled-c-shell"
        for row in profiler.report()["rows"]
    )


def test_c_jit_reuses_repository_source_and_compiled_module(tmp_path):
    case = CASES["operator_grab_bag"]
    cache = RepositoryArtifactCache(tmp_path)
    first = compile_torture_case_to_c(
        capture_torture_case(case),
        cache=cache,
    )
    second = compile_torture_case_to_c(
        capture_torture_case(case),
        cache=cache,
    )

    assert not first.source_artifact.hit
    assert second.source_artifact.hit
    assert first.source_artifact.identity == second.source_artifact.identity
