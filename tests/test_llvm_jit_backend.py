from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.artifact_cache import (
    RepositoryArtifactCache,
)
from src.common.tensors.accelerator_backends.llvm_jit_backend import (
    compile_torture_case_to_llvm,
)
from src.common.tensors.accelerator_backends.tensor_torture import (
    capture_torture_case,
    tensor_torture_cases,
)
from src.compiler.glsl_deployment_strategy import DeploymentProfiler


CASES = {case.name: case for case in tensor_torture_cases()}
LLVM_BASELINE = (
    "add",
    "scalar_broadcast",
    "sin",
    "where",
    "flat_sum",
    "matmul",
    "operator_grab_bag",
    "dim_sum",
    "cumsum",
    "advanced_tensor_topology",
)


@pytest.mark.parametrize("name", LLVM_BASELINE)
def test_llvm_jit_torture_baseline_uses_profiled_c_shell(name, tmp_path):
    case = CASES[name]
    captured = capture_torture_case(case)
    compiler = compile_torture_case_to_llvm(
        captured,
        cache=RepositoryArtifactCache(tmp_path),
    )
    profiler = DeploymentProfiler(enabled=True)
    execution = compiler.execute(
        case.inputs,
        profiler=profiler,
        profile_path=f"torture/llvm/{name}",
    )

    expected = case.numpy_reference()
    for output_name, expected_value in expected.items():
        np.testing.assert_allclose(
            execution.outputs[output_name],
            expected_value,
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


def test_llvm_torture_artifact_is_reused_without_relowering(tmp_path):
    case = CASES["operator_grab_bag"]
    cache = RepositoryArtifactCache(tmp_path)
    first = compile_torture_case_to_llvm(
        capture_torture_case(case),
        cache=cache,
    )
    second = compile_torture_case_to_llvm(
        capture_torture_case(case),
        cache=cache,
    )

    assert not first.cache_artifact.hit
    assert second.cache_artifact.hit
    assert first.cache_artifact.identity == second.cache_artifact.identity
    assert first.llvm_ir == second.llvm_ir


def test_llvm_advanced_case_executes_new_layout_and_reduction_stages(tmp_path):
    case = CASES["advanced_tensor_topology"]
    program = compile_torture_case_to_llvm(
        capture_torture_case(case),
        cache=RepositoryArtifactCache(tmp_path),
    )
    execution = program.execute(case.inputs)

    for output_name, expected in case.numpy_reference().items():
        np.testing.assert_allclose(
            execution.outputs[output_name],
            expected,
            rtol=case.rtol,
            atol=case.atol,
        )
    assert "@transpose_double" in program.llvm_ir
    assert "@stack_double" in program.llvm_ir
    assert "@cat_double" in program.llvm_ir
    assert "@reduce_dim_double" in program.llvm_ir
