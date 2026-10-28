"""Python ``int`` values crossing a native boundary keep their value.

Each program's ``root`` runs on the C and LLVM lanes; the native result
either equals Python's or raises a named fault:

* a store or contract conversion into an integer slot that cannot hold the
  value (``integer_narrowing: trap``) reports ``INT_OVERFLOW``;
* ``int / int`` divides once when both ints are exactly float64, reports a
  zero divisor as ``ZeroDivisionError`` and an inexact operand as
  ``INT_TRUE_DIVISION_INEXACT``;
* a uint64 result returns as the exact Python int.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"


def _lower(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="int_boundaries",
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


def _feeds(parameters, arguments):
    return {
        parameters[name]: np.array([value], dtype=np.int64)
        for name, value in arguments.items()
    }


def _c_lane(tmp_path, source, **arguments):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lower(source)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    execution = artifact.prepare_execution(_feeds(parameters, arguments)).run()
    return execution.buffers[outputs[0]].reshape(-1)[0].item()


def _llvm_lane(tmp_path, source, **arguments):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lower(source)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    execution = prepare_artifact_execution(
        artifact, _feeds(parameters, arguments),
    ).run()
    return execution.buffers[outputs[0]].reshape(-1)[0].item()


LANES = pytest.mark.parametrize("lane", [_c_lane, _llvm_lane], ids=["c", "llvm"])


def _python(source, **arguments):
    namespace = {}
    exec(source, namespace)
    return namespace["root"](**arguments)


# --- narrowing stores ----------------------------------------------------
#
# ``Box.v`` is assigned an int64 (``0``, ``a``) and, from ``put_big``, a
# uint64 above 2**63 (``big(3)``).  The field settles as an int64 slot; the
# uint64 store wrote its bits and the field read back ``big(3) - 2**64``.

FIELD_SOURCE = '''
class Box:
    def __init__(self):
        self.v = 0

    def big(self, n: int) -> int:
        h = (n * 2654435761 + 12345) % 4294967296
        return h * (1 << 32) + h

    def put_big(self, n: int):
        self.v = self.big(n)

    def put(self, a: int):
        self.v = a


def root(a: int, b: int):
    box = Box()
    box.put_big(b)
    x = box.v % 1000003
    box.put(a)
    return x + box.v
'''


@LANES
@pytest.mark.parametrize("a,b", [(-(2 ** 62), 7), (5, 0)])
def test_field_store_that_fits_matches_python(tmp_path, lane, a, b):
    assert (1 << 63) > _python(FIELD_SOURCE.replace("return x + box.v", "return box.big(b)"), a=a, b=b)
    assert lane(tmp_path, FIELD_SOURCE, a=a, b=b) == _python(FIELD_SOURCE, a=a, b=b)


@LANES
@pytest.mark.parametrize("a", [5, -1])
def test_uint64_into_int64_field_is_reported_not_wrapped(tmp_path, lane, a):
    from src.compiler.native_faults import NativeIntOverflow

    assert _python(FIELD_SOURCE.replace("return x + box.v", "return box.big(b)"), a=a, b=3) >= 1 << 63
    with pytest.raises(NativeIntOverflow):
        lane(tmp_path, FIELD_SOURCE, a=a, b=3)


def test_narrowing_bounds_follow_the_layouts_and_the_contract(monkeypatch):
    from src.compiler import ssa_call_input_adapters
    from src.compiler.ssa_llvm_backend import integer_narrowing_bounds

    assert integer_narrowing_bounds("uint64", "int64") == (None, (1 << 63) - 1)
    assert integer_narrowing_bounds("int64", "uint64") == (0, None)
    assert integer_narrowing_bounds("int64", "int32") == (-(1 << 31), (1 << 31) - 1)
    assert integer_narrowing_bounds("int128", "uint64") == (0, (1 << 64) - 1)
    assert integer_narrowing_bounds("int64", "int128") is None
    assert integer_narrowing_bounds("uint32", "int64") is None
    assert integer_narrowing_bounds("int64", "float64") is None
    assert integer_narrowing_bounds("bool", "int8") is None
    modes = {
        mode: {**contract, "integer_narrowing": "wrap"}
        for mode, contract in ssa_call_input_adapters._NUMERIC_CONVERSION_MODES.items()
    }
    monkeypatch.setattr(ssa_call_input_adapters, "_NUMERIC_CONVERSION_MODES", modes)
    assert integer_narrowing_bounds("uint64", "int64") is None


def test_narrowing_c_check_spells_each_side():
    from src.compiler.ssa_llvm_backend import (
        integer_narrowing_c_check, integer_narrowing_lines,
    )

    check = integer_narrowing_c_check("int64", "int32", "int64_t", "x")
    assert "< ((int64_t)INT64_C(-2147483648))" in check
    assert "> ((int64_t)INT64_C(2147483647))" in check
    assert "turing_validation_error(86u)" in check
    unsigned = integer_narrowing_c_check("uint64", "int64", "uint64_t", "x")
    assert "<" not in unsigned.split("turing")[0]
    lines = integer_narrowing_lines("uint64", "int64", "i64", "%x", "%r")
    assert lines[0] == "%r.above = icmp ugt i64 %x, 9223372036854775807"
    assert lines[-1] == "call void @turing_int_overflow(i1 %r.above)"


# --- int / int -----------------------------------------------------------

DIVISION_SOURCE = '''
def root(a: int, b: int):
    return a / b
'''

EXACT_DIVISIONS = [
    (7, 2), (-7, 2), (1, 3), (2 ** 53, 3), (-(2 ** 63), 2 ** 62),
    (2 ** 62, -(2 ** 61) - 2 ** 9), (0, -5),
]


@LANES
@pytest.mark.parametrize("a,b", EXACT_DIVISIONS)
def test_true_division_of_exact_floats_matches_python(tmp_path, lane, a, b):
    assert lane(tmp_path, DIVISION_SOURCE, a=a, b=b) == a / b


@LANES
def test_true_division_by_zero_raises_zero_division(tmp_path, lane):
    from src.compiler.native_faults import NativeZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        _python(DIVISION_SOURCE, a=1, b=0)
    with pytest.raises(NativeZeroDivisionError):
        lane(tmp_path, DIVISION_SOURCE, a=1, b=0)


@LANES
@pytest.mark.parametrize("a,b", [(2 ** 53 + 1, 3), (3, 2 ** 63 - 1)])
def test_true_division_of_inexact_operand_is_reported(tmp_path, lane, a, b):
    from src.compiler.native_faults import NativeIntOverflow

    # float(a) / float(b) is one ulp from Python's quotient for the first.
    assert a / b != float(a) / float(b) or a == 3
    with pytest.raises(NativeIntOverflow, match="not exact in float64"):
        lane(tmp_path, DIVISION_SOURCE, a=a, b=b)


# --- int -> float, and back to Python ------------------------------------

FLOAT_SOURCE = '''
def root(a: int, b: int):
    return float(a) + b
'''


@LANES
@pytest.mark.parametrize("a", [2 ** 53 + 1, 2 ** 53 + 3, 2 ** 63 - 1, -(2 ** 53) - 1])
def test_int_to_float_rounds_as_python(tmp_path, lane, a):
    assert lane(tmp_path, FLOAT_SOURCE, a=a, b=0) == float(a)


UINT64_SOURCE = '''
def root(b: int):
    h = (b * 2654435761 + 12345) % 4294967296
    return h * (1 << 32) + h
'''


@LANES
def test_uint64_result_returns_the_exact_python_int(tmp_path, lane):
    expected = _python(UINT64_SOURCE, b=3)
    assert expected >= 1 << 63
    value = lane(tmp_path, UINT64_SOURCE, b=3)
    assert type(value) is int and value == expected


# --- int dict keys: exact identity ---------------------------------------

KEY_SOURCE = '''
def root(a: int, b: int):
    d = {}
    d[a] = 1
    d[b] = 2
    return d[a] * 10 + len(d)
'''


@LANES
@pytest.mark.parametrize("a,b", [
    (5, 5), (2 ** 61 - 1 + 5, 5), (-1, -2), (2 ** 63 - 1, -(2 ** 63)),
])
def test_int_keys_with_equal_python_hash_stay_distinct(tmp_path, lane, a, b):
    assert lane(tmp_path, KEY_SOURCE, a=a, b=b) == _python(KEY_SOURCE, a=a, b=b)
