from __future__ import annotations

import ast

from src.compiler.backend_identities import apply_backend_identities
from src.compiler.tensor_ssa_lowering import lower_tensor_calls_to_repository_ssa
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue
from src.transmogrifier.graph.python_identity_programs import resolve_python_identity
from src.transmogrifier.graph.python_special_cases import (
    interpret_python_special_case,
)


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
    assert program.direct_attributes["semantic_library"] == "src.common.tensors.blas"
    assert program.direct_attributes["semantic_kernel"] == "gemm"
    assert program.direct_attributes["semantic_source_symbol"] == "GEMM_SOURCE"
    assert program.direct_attributes["semantic_parameters"] == {
        "alpha": 1.0,
        "beta": 0.0,
    }
    assert program.direct_attributes["backend_intrinsic_family"] == "blas.gemm"


def test_abstract_tensor_matmul_entry_points_use_the_blas_semantic_seam():
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.abstraction_methods import blas
    from src.common.tensors.blas import GEMM_SOURCE

    assert blas.GEMM_SOURCE is GEMM_SOURCE
    assert blas.MATMUL_BLAS_SEMANTICS == {
        "library": "src.common.tensors.blas",
        "kernel": "gemm",
        "source_symbol": "GEMM_SOURCE",
        "intrinsic_family": "blas.gemm",
        "alpha": 1.0,
        "beta": 0.0,
    }

    calls = []
    tensor = object.__new__(AbstractTensor)
    tensor._apply_operator = lambda op, left, right: calls.append(
        (op, left, right)
    ) or "result"
    other = object()

    assert tensor.matmul(other) == "result"
    assert tensor.__matmul__(other) == "result"
    assert tensor.__rmatmul__(other) == "result"
    assert tensor.__imatmul__(other) == "result"
    assert calls == [
        ("matmul", tensor, other),
        ("matmul", tensor, other),
        ("matmul", other, tensor),
        ("imatmul", tensor, other),
    ]


def _lower_flagged_matmul():
    call_node = ast.parse("left.matmul(right)", mode="eval").body
    call_node._extraction_contract = {
        "identity": "src.common.tensors.abstraction.AbstractTensor.matmul",
        "classification": "repository_python",
        "action": "intrinsic",
        "rule_id": "abstract-tensor-vocabulary-is-intrinsic",
        "parameters": {"lowering_namespace": "abstract_tensor"},
    }
    special = interpret_python_special_case(call_node)
    assert special is not None and special.type == "matmul"

    left = SSAValue(100, "float32", shape=(2, 3))
    right = SSAValue(101, "float32", shape=(3, 4))
    result = SSAValue(102, "float32", shape=(2, 4))
    caller = Function("flagged", [left, right], {
        "entry": BasicBlock("entry", [
            Instr(special.type, [left, right], result,
                  attributes=dict(special.attributes)),
            Instr("Ret", [result], None),
        ]),
    })
    module = IRModule({caller.name: caller})
    assert lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference()
    ) == ()
    lowered = next(
        instruction
        for instruction in module.functions["flagged"].blocks["entry"].instrs
        if instruction.attributes.get("callee") == "matmul_double"
    )
    return module, result, lowered


def test_process_graph_intrinsic_flag_survives_tensor_lowering_and_swaps_for_glsl():
    module, result, lowered = _lower_flagged_matmul()

    assert lowered.attributes["backend_intrinsic_candidate"] == {
        "semantic_identity": (
            "src.common.tensors.abstraction.AbstractTensor.matmul"
        ),
        "lowering_namespace": "abstract_tensor",
        "ingested_fallback": False,
    }
    swapped = apply_backend_identities(
        module, {"flagged": (result,)}, backend="glsl",
        licensed_inexact=False,
    )
    intrinsic = next(
        instruction
        for instruction in swapped.module.functions["flagged"].blocks[
            "entry"
        ].instrs
        if instruction.op == "BackendIntrinsic"
    )
    record = intrinsic.attributes["backend_intrinsic"]
    assert record["semantic_family"] == "blas.gemm"
    assert record["symbol"] == "glslblas_gemm"
    assert record["consumption"] == "deployment_bypass"
    assert record["operand_positions"] == [0, 1]
    assert intrinsic.attributes["backend_intrinsic_original"] == {
        "op": "Call",
        "callee": "matmul_double",
    }
    assert swapped.decisions[0].identity == "backend_intrinsic_location_swap"
    assert swapped.decisions[0].applied
    assert swapped.decisions[0].before_sha256 != swapped.decisions[0].after_sha256
    assert lowered.op == "Call"  # backend swap never mutates universal SSA


def test_glsl_intrinsic_location_accepts_an_explicit_gestalt_override():
    from src.common.tensors.accelerator_backends.glsl_backend import (
        execute_backend_intrinsic,
    )

    module, result, _lowered = _lower_flagged_matmul()
    location = "external.demo_blas:gemm"
    swapped = apply_backend_identities(
        module, {"flagged": (result,)}, backend="glsl",
        licensed_inexact=False,
        intrinsic_overrides={
            "blas.gemm": {
                "location": location,
                "symbol": "gemm",
                "consumption": "deployment_bypass",
                "lowering_namespaces": ["abstract_tensor"],
                "operand_positions": [0, 1],
            },
        },
    )
    intrinsic = next(
        instruction
        for instruction in swapped.module.functions["flagged"].blocks[
            "entry"
        ].instrs
        if instruction.op == "BackendIntrinsic"
    )
    observed = []
    left, right = object(), object()

    answer = execute_backend_intrinsic(
        intrinsic,
        {
            int(intrinsic.args[0].id): left,
            int(intrinsic.args[1].id): right,
        },
        gestalt_overrides={
            location: lambda first, second: observed.append(
                (first, second)
            ) or "external-result",
        },
    )

    assert answer == "external-result"
    assert observed == [(left, right)]
