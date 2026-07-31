from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.artifact_cache import (
    RepositoryArtifactCache,
)
from src.common.tensors.accelerator_backends.glsl_backend import (
    GLContextUnavailable,
    require_gl_context,
)
from src.common.tensors.accelerator_backends.glsl_jit_backend import (
    compile_torture_case_to_glsl,
)
from src.common.tensors.accelerator_backends.tensor_torture import (
    capture_torture_case,
    tensor_torture_cases,
)
from src.compiler.glsl_deployment_strategy import DeploymentProfiler


CASES = {case.name: case for case in tensor_torture_cases()}


@pytest.fixture(scope="module")
def gl():
    try:
        return require_gl_context()
    except GLContextUnavailable as error:
        pytest.skip(f"no OpenGL 4.3+ compute context: {error}")


@pytest.mark.parametrize(
    "name",
    ("add", "scalar_broadcast", "where", "operator_grab_bag"),
)
def test_glsl_jit_torture_uses_c_shell_and_gpu_timer(name, tmp_path, gl):
    case = CASES[name]
    program = compile_torture_case_to_glsl(
        capture_torture_case(case),
        cache=RepositoryArtifactCache(tmp_path),
    )
    profiler = DeploymentProfiler(enabled=True)
    execution = program.execute(
        case.inputs,
        profiler=profiler,
        profile_path=f"torture/glsl/{name}",
    )

    for output_name, expected in case.numpy_reference().items():
        np.testing.assert_allclose(
            execution.outputs[output_name],
            expected,
            rtol=max(case.rtol, 2.0e-5),
            atol=max(case.atol, 2.0e-5),
        )
    assert execution.profile.shell_ns > 0
    assert execution.profile.device_ns > 0
    row = next(
        row
        for row in profiler.report()["rows"]
        if row["section"] == "compiled-c-shell"
    )
    assert row["cpu_ms"] > 0.0
    assert row["gpu_ms"] > 0.0


def test_glsl_advanced_torture_runs_as_compiled_stages(tmp_path, gl):
    case = CASES["advanced_tensor_topology"]
    program = compile_torture_case_to_glsl(
        capture_torture_case(case),
        cache=RepositoryArtifactCache(tmp_path),
    )
    execution = program.execute(case.inputs)

    for output_name, expected in case.numpy_reference().items():
        np.testing.assert_allclose(
            execution.outputs[output_name],
            expected,
            rtol=5.0e-5,
            atol=5.0e-5,
        )
    assert len(program.captured.execution_programs) > 1
