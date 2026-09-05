"""Value-precision casts agree with the reference in every lane.

The reference semantics is the numpy backend's ``_cast_`` map:

    float -> np.float32 values    double -> np.float64 values
    int   -> truncated integers   bool   -> nonzero -> 1

``double`` and an explicit ``to_dtype("float64")`` previously shared the
single-precision narrowing kernel (``fptrunc``+``fpext``), silently truncating
every mantissa; ``to_dtype("bool")`` shared integer truncation (2.7 -> 2 where
the reference says 1).  These tests execute the LLVM lane natively against the
reference and hold the Fortran spelling to the same map, so the agreement is
measured, not assumed -- the same pattern as the trig-solver cross-check.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.ssa_fortran_backend import emit_module
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.compiler.tensor_ssa_lowering import (
    lower_tensor_calls_to_repository_ssa,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from src.transmogrifier.ssa_registry import Handler


#: One value per failure mode: a mantissa below single precision, a repeating
#: fraction, a truncatable positive/negative, an exact zero.
FEED = np.array([1.0 + 2.0 ** -40, 1.0 / 3.0, 2.7, -2.7], dtype=np.float64)

#: The numpy backend's ``_cast_`` map, applied to FEED and read back into the
#: double-typed working representation each backend stores values in.
REFERENCE = {
    "float": FEED.astype(np.float32).astype(np.float64),
    "double": FEED.astype(np.float64),
    "int": FEED.astype(np.int64).astype(np.float64),
    "bool": FEED.astype(np.bool_).astype(np.float64),
}

EXPECTED_KERNEL = {
    "float": "cast_double_to_float_values",
    "double": "cast_double_to_double_values",
    "int": "cast_double_to_int_values",
    "bool": "cast_double_to_bool_values",
}


def _cast_module(operation: str, attributes: dict | None = None) -> tuple:
    source = SSAValue(1000, dtype="float64", shape=(4,))
    result = SSAValue(1001, dtype="float64", shape=(4,))
    function = Function(
        "program",
        [source],
        {
            "entry": BasicBlock("entry", [
                Instr(
                    Handler.Call.value,
                    [source],
                    result,
                    attributes={
                        "tensor_operation": operation,
                        **(attributes or {}),
                    },
                ),
                Instr(Handler.Ret.value, [result], None),
            ])
        },
    )
    module = IRModule({"program": function})
    shortfalls = lower_tensor_calls_to_repository_ssa(
        module, c_backend_repository_ssa_reference(),
    )
    assert shortfalls == (), f"{operation}: lowering shortfalls {shortfalls!r}"
    return module, function, source, result


def _lowered_callee(function) -> str:
    call = next(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == Handler.Call.value
        and instruction.attributes.get("callee")
    )
    return str(call.attributes["callee"])


@pytest.mark.parametrize("operation", sorted(EXPECTED_KERNEL))
def test_cast_lowers_to_its_own_kernel(operation):
    _module, function, _source, _result = _cast_module(operation)
    assert _lowered_callee(function) == EXPECTED_KERNEL[operation]


@pytest.mark.parametrize(
    "target_dtype, kernel",
    [
        ("float64", "cast_double_to_double_values"),
        ("double", "cast_double_to_double_values"),
        ("float32", "cast_double_to_float_values"),
        ("float", "cast_double_to_float_values"),
        ("int64", "cast_double_to_int_values"),
        ("bool", "cast_double_to_bool_values"),
    ],
)
def test_to_dtype_dispatches_on_the_requested_dtype(target_dtype, kernel):
    _module, function, _source, _result = _cast_module(
        "to_dtype", {"target_dtype": target_dtype},
    )
    assert _lowered_callee(function) == kernel


@pytest.mark.parametrize("operation", sorted(EXPECTED_KERNEL))
def test_cast_executes_to_the_reference_values(operation, tmp_path):
    module, _function, source, result = _cast_module(operation)
    artifact = emit_ssa_function_to_llvm(module, "program")
    assert artifact.shortfalls == ()
    native = compile_artifact(
        artifact, directory=tmp_path / f"cast_{operation}",
    )
    execution = prepare_artifact_execution(
        native, {int(source.id): FEED.copy()},
    )
    execution.run()
    produced = np.asarray(execution.buffers[int(result.id)]).reshape(-1)
    expected = REFERENCE[operation]
    assert produced.tolist() == expected.tolist(), (
        f"{operation}: {produced!r} != {expected!r}"
    )


def test_double_preserves_the_full_mantissa(tmp_path):
    """The defect this file exists for: double must not narrow."""

    module, _function, source, result = _cast_module("double")
    artifact = emit_ssa_function_to_llvm(module, "program")
    native = compile_artifact(artifact, directory=tmp_path / "mantissa")
    execution = prepare_artifact_execution(
        native, {int(source.id): FEED.copy()},
    )
    execution.run()
    produced = np.asarray(execution.buffers[int(result.id)]).reshape(-1)
    assert produced[0] == 1.0 + 2.0 ** -40
    assert produced[0] != np.float64(np.float32(1.0 + 2.0 ** -40))


@pytest.mark.parametrize("operation", sorted(EXPECTED_KERNEL))
def test_fortran_spells_each_cast_from_the_same_map(operation):
    module, _function, _source, _result = _cast_module(operation)
    emitted = emit_module(module)
    assert not emitted.shortfalls, (
        f"{operation}: fortran shortfalls "
        f"{[ (s.operation, s.reason) for s in emitted.shortfalls ]!r}"
    )
    spelling = {
        "float": "c_float",
        "double": None,               # identity: no conversion text at all
        "int": "int(",
        "bool": "merge(",
    }[operation]
    if spelling is not None:
        assert spelling in emitted.source


def test_c_runtime_bool_cast_matches_the_reference():
    """The C runtime lane accepts bool now instead of raising."""

    c_backend = pytest.importorskip(
        "src.common.tensors.accelerator_backends.c_backend"
    )
    operations = c_backend.CTensorOperations()
    tensor = operations.tensor_from_list_(
        FEED.tolist(), dtype="float64", device=None,
    )
    produced = operations.to_dtype_(tensor, "bool")
    values = [
        float(value) for value in operations.tolist_(produced)
    ]
    assert values == REFERENCE["bool"].tolist()
