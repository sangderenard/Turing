from __future__ import annotations

from src.compiler.backend_identities import apply_backend_identities
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue
from src.transmogrifier.graph.python_identity_programs import resolve_python_identity


def _module():
    left = SSAValue(0, "float64")
    right = SSAValue(1, "float64")
    result = SSAValue(2, "float64")
    function = Function("add", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Add", [left, right], result),
            Instr("Ret", [result], None),
        ]),
    })
    return IRModule({"add": function}), {"add": (result,)}


def test_glsl_and_webgpu_share_the_same_float_storage_identity():
    module, outputs = _module()
    glsl = apply_backend_identities(
        module, outputs, backend="glsl", licensed_inexact=True,
    )
    webgpu = apply_backend_identities(
        module, outputs, backend="webgpu", licensed_inexact=True,
    )

    assert glsl.decisions == webgpu.decisions
    assert glsl.decisions[0].applied
    assert glsl.decisions[0].before_sha256 != glsl.decisions[0].after_sha256
    assert glsl.outputs["add"][0].dtype == "float32"
    assert webgpu.outputs["add"][0].dtype == "float32"
    assert module.functions["add"].args[0].dtype == "float64"


def test_exact_contract_refuses_the_inexact_backend_swap():
    module, outputs = _module()
    result = apply_backend_identities(
        module, outputs, backend="webgpu", licensed_inexact=False,
    )

    assert not result.decisions[0].applied
    assert result.decisions[0].before_sha256 == result.decisions[0].after_sha256
    assert "forbids inexact" in " ".join(result.decisions[0].reasons)
    assert result.outputs["add"][0].dtype == "float64"


def test_abstract_tensor_blas_identity_exposes_a_backend_intrinsic_family():
    program = resolve_python_identity(
        "src.common.tensors.abstraction.AbstractTensor.matmul"
    )

    assert program is not None
    assert program.direct_operator == "matmul"
    assert program.direct_attributes["semantic_library"] == "abstract_tensor_blas"
    assert program.direct_attributes["backend_intrinsic_family"] == "blas.gemm"
