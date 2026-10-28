"""Python ``int`` powers lower natively and exactly.

``a ** b``, ``pow(a, b)`` and ``pow(a, b, m)`` on ints are the shared exact
power reduction (``ir_identities.reduce_integer_pow``): checked
square-and-multiply in repository SSA, so C and LLVM compute Python's exact
int or raise a named fault -- never a float ``pow`` of an int, never a
wrapped product.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

INT64_MAX = 2 ** 63 - 1
INT64_MIN = -(2 ** 63)

PROGRAMS = {
    "variable": "a ** b",
    "builtin": "pow(a, b)",
    "cube": "a ** 3",
    "zeroth": "a ** 0",
    "first": "a ** 1",
    "sixty_third": "a ** 63",
    "large_literal": "a ** 1000003",
    "modular": "pow(a, b, m)",
    "modular_literal": "pow(a, b, 1000000007)",
    "negative_literal": "a ** -2",
}

def _source(expression):
    return f"""
def step(a: int, b: int, m: int) -> int:
    return {expression}


def root(a: int, b: int, m: int):
    return step(a, b, m) + 0
"""


def _lower(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="int_pow",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


_LOWERED: dict = {}


def _lowered(source):
    # Lowering is the slow step; each lane emits from its own fresh copy.
    import copy

    if source not in _LOWERED:
        _LOWERED[source] = _lower(source)
    return copy.deepcopy(_LOWERED[source])


def _inputs(parameters, a, b, m):
    return {
        parameters[name]: np.array([value], dtype=np.int64)
        for name, value in (("a", a), ("b", b), ("m", m))
        if name in parameters
    }


def _c_lane(tmp_path, source, a, b, m):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lowered(source)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    execution = artifact.prepare_execution(_inputs(parameters, a, b, m)).run()
    return execution.buffers[outputs[0]].reshape(-1)[0].item()


def _llvm_lane(tmp_path, source, a, b, m):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered(source)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    execution = prepare_artifact_execution(
        artifact, _inputs(parameters, a, b, m)
    ).run()
    return execution.buffers[outputs[0]].reshape(-1)[0].item()


LANES = pytest.mark.parametrize("lane", [_c_lane, _llvm_lane], ids=["c", "llvm"])


def _python(source, a, b, m):
    namespace = {}
    exec(source, namespace)
    return namespace["root"](a, b, m)


EXACT = [
    ("variable", 3, 39, 0), ("variable", -3, 39, 0), ("variable", 0, 0, 0),
    ("variable", 7, 0, 0), ("variable", 0, 5, 0), ("variable", 1, INT64_MAX, 0),
    ("variable", -1, INT64_MAX, 0), ("variable", -1, INT64_MAX - 1, 0),
    ("variable", 0, INT64_MAX, 0), ("variable", 2, 62, 0),
    ("variable", -2, 63, 0), ("variable", 3037000499, 2, 0),
    ("variable", -9, 19, 0), ("variable", INT64_MIN, 1, 0),
    ("builtin", 5, 27, 0), ("builtin", -7, 22, 0),
    ("cube", 2097151, 0, 0), ("cube", -2097152, 0, 0),
    ("zeroth", 0, 0, 0), ("zeroth", INT64_MIN, 0, 0),
    ("first", INT64_MIN, 0, 0), ("sixty_third", -2, 0, 0),
    ("sixty_third", 1, 0, 0), ("large_literal", -1, 0, 0),
    ("modular", 3, 200, 1000), ("modular", -3, 201, 1000),
    ("modular", 3, 201, -1000), ("modular", 5, 0, 1), ("modular", 5, 0, -7),
    ("modular", 12345, INT64_MAX, 3037000499), ("modular", INT64_MIN, 3, -1),
    ("modular", INT64_MIN, 5, 97), ("modular", 0, 0, 13),
    ("modular_literal", 2, 10 ** 18, 0),
]


@LANES
@pytest.mark.parametrize("program,a,b,m", EXACT)
def test_int_power_is_exact(tmp_path, lane, program, a, b, m):
    source = _source(PROGRAMS[program])
    expected = _python(source, a, b, m)
    result = lane(tmp_path, source, a, b, m)
    assert type(result) is int and result == expected


FAULTS = [
    ("variable", 2, 63, 0, "NativeIntOverflow"),
    ("variable", -2, 64, 0, "NativeIntOverflow"),
    ("variable", 3, 40, 0, "NativeIntOverflow"),
    ("variable", 2, INT64_MAX, 0, "NativeIntOverflow"),
    ("variable", 3037000500, 2, 0, "NativeIntOverflow"),
    ("builtin", 10, 19, 0, "NativeIntOverflow"),
    ("cube", 2097153, 0, 0, "NativeIntOverflow"),
    ("sixty_third", 2, 0, 0, "NativeIntOverflow"),
    ("large_literal", 2, 0, 0, "NativeIntOverflow"),
    ("variable", 0, -1, 0, "NativeZeroDivisionError"),
    ("variable", 2, -1, 0, "NativeIntPowerNegativeExponent"),
    # The frontend types every int ** int as int; Python's float result
    # is reported, never returned as an int.
    ("negative_literal", 3, 0, 0, "NativeIntPowerNegativeExponent"),
    ("negative_literal", 0, 0, 0, "NativeZeroDivisionError"),
    ("builtin", -3, -5, 0, "NativeIntPowerNegativeExponent"),
    ("modular", 3, 5, 0, "NativeValueError"),
    ("modular", 3, -1, 7, "NativeIntPowerNegativeExponent"),
    # Python is exact here; the product of two residues left int64.
    ("modular", 3037000499, 3, 2 ** 62, "NativeIntOverflow"),
]


@LANES
@pytest.mark.parametrize("program,a,b,m,fault", FAULTS)
def test_int_power_fault_is_named(tmp_path, lane, program, a, b, m, fault):
    from src.compiler import native_faults

    source = _source(PROGRAMS[program])
    python_exception = {
        "NativeValueError": ValueError,
        "NativeZeroDivisionError": ZeroDivisionError,
    }.get(fault)
    if python_exception is not None:
        with pytest.raises(python_exception):
            _python(source, a, b, m)
    with pytest.raises(getattr(native_faults, fault)):
        lane(tmp_path, source, a, b, m)


def test_no_float_pow_reaches_an_int_power():
    module, _export, _parameters, _outputs = _lowered(_source(PROGRAMS["variable"]))
    operations = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    assert not any(instruction.op == "Pow" for instruction in operations)
    assert any(
        instruction.op == "Mul" and instruction.attributes.get("integer_power")
        for instruction in operations
    )
