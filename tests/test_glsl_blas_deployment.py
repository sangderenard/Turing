from __future__ import annotations

import json

from src.common.tensors.blas import BLAS_ROLES, GEMM_SOURCE, blas_role
from src.compiler.deployment_lowering import ComputeDispatchLimits
from src.compiler.glsl_blas_deployment import build_gemm_deployment_pair
from src.compiler.work_contract import (
    PRESETS,
    ShaderOptimizationContract,
    WorkContract,
)


LIMITS = ComputeDispatchLimits(
    max_group_count=(65535, 65535, 65535),
    max_group_size=(1024, 1024, 64),
    max_invocations=1024,
)


def test_finite_blas_roles_are_derived_from_the_authored_kernel_set():
    assert tuple(BLAS_ROLES) == ("scal", "axpy", "dot", "gemv", "gemm", "rot")
    role = blas_role("gemm")
    assert role.source is GEMM_SOURCE
    assert role.identity == "blas.gemm"
    assert role.abstract_operator == "matmul"


def test_glsl_gemm_defaults_fast_and_source_lowering_is_contract_configurable():
    assert all(
        contract.shaders.blas_gemm == "glslblas_gemm"
        for contract in PRESETS.values()
    )
    source_contract = WorkContract(
        "source-proof",
        register_reuse=False,
        inexact_identities=False,
        contract_multiply_add=False,
        shaders=ShaderOptimizationContract(blas_gemm="source_algorithm"),
    )
    assert source_contract.shaders.blas_gemm == "source_algorithm"


def test_source_and_intrinsic_gemm_are_comparable_standalone_deployments(tmp_path):
    source, intrinsic = build_gemm_deployment_pair(32, 48, 16, limits=LIMITS)

    assert source.role == intrinsic.role == "blas.gemm"
    assert source.variant == "source_algorithm"
    assert intrinsic.variant == "glslblas_gemm"
    assert source.shader_source != intrinsic.shader_source
    assert "for (uint p = 0u" in source.shader_source
    assert "shared float left_tile" in intrinsic.shader_source
    assert source.manifest["role_source_sha256"] == intrinsic.manifest[
        "role_source_sha256"
    ]
    repeated_source, repeated_intrinsic = build_gemm_deployment_pair(
        32, 48, 16, limits=LIMITS
    )
    assert source.manifest["shader_plan_identity"] == repeated_source.manifest[
        "shader_plan_identity"
    ]
    assert intrinsic.manifest["shader_plan_identity"] == repeated_intrinsic.manifest[
        "shader_plan_identity"
    ]
    assert source.manifest["arena_abi"] == intrinsic.manifest["arena_abi"]
    assert source.manifest["standalone"]["owns_hidden_opengl_context"]
    assert "SDL_GL_CreateContext" in source.shell_source
    assert "glDispatchCompute" in source.shell_source
    assert "glGetBufferSubData" in source.shell_source
    assert "Py_Initialize" not in source.shell_source
    assert "#include <Python.h>" not in source.shell_source

    source_files = source.write(tmp_path / "source")
    intrinsic_files = intrinsic.write(tmp_path / "intrinsic")
    for files in (source_files, intrinsic_files):
        assert files.shader_path.is_file()
        assert files.shell_path.is_file()
        manifest = json.loads(files.manifest_path.read_text(encoding="utf-8"))
        assert manifest["recommended_dispatch"]["groups"]
        assert not manifest["standalone"]["python_runtime_dependency"]


def test_standalone_shell_feeds_launch_controls_as_storage_blocks(tmp_path):
    """The shell binds what the shader declares: no loose uniforms, the
    slot table at binding 1 and the count/extent table at binding 2. It is
    Windows C, so it is compiled for that target here when a cross compiler
    is available; running it needs Windows."""

    import importlib.util
    import subprocess
    import sys

    import pytest

    source, intrinsic = build_gemm_deployment_pair(32, 48, 16, limits=LIMITS)
    for deployment in (source, intrinsic):
        assert "uniform" not in deployment.shader_source
        assert "glUniform" not in deployment.shell_source
        assert "glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1u," in deployment.shell_source
        assert "glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2u," in deployment.shell_source
    if importlib.util.find_spec("ziglang") is None:
        pytest.skip("no cross compiler (ziglang) for the Windows shell")
    for deployment in (source, intrinsic):
        files = deployment.write(tmp_path / deployment.variant)
        compiled = subprocess.run(
            [sys.executable, "-m", "ziglang", "cc", "-target", "x86_64-windows-gnu",
             "-Wall", "-Werror", "-c", str(files.shell_path),
             "-o", str(tmp_path / f"{deployment.variant}.o")],
            capture_output=True, text=True,
        )
        assert compiled.returncode == 0, compiled.stderr
