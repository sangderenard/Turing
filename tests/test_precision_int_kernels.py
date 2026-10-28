"""Exact Python ``int`` arithmetic past int64: what Precision carries, what limbs add.

Two things are held to Python ``int`` here.

1. ``Precision[n]`` (``src/common/tensors/extended_precision.py``) carrying an
   exact integer as integer-valued double limbs.  ``two_sum`` and
   ``two_product`` are error-free, so ``+ - *`` are exact whenever the result
   fits ``n`` limbs and stays below the binary64 exponent range; both are
   checked here in CPython and on the compiled C and LLVM lanes.
2. ``src.compiler.precision_int_kernels``, the 30-bit integer limb kernels
   for what Precision has no form of (floor division, modulo, shifts,
   two's complement bitwise, int64/int128 conversion), against ``int`` on
   random and edge values.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from src.compiler import precision_int_kernels as K

_RANDOM = random.Random(20261028)
EDGES = [
    0, 1, -1, 2, -2, 2**63, -(2**63), 2**63 - 1, -(2**63) + 1, 2**64, -(2**64),
    2**64 - 1, 2**30, 2**30 - 1, -(2**30), 2**60, 2**90 - 1, 3**100, -(3**100),
    7**55, 2**120 - 1, -(2**150) + 1, 2**300, -(2**300 - 1),
]
VALUES = EDGES + [
    _RANDOM.randint(-(2 ** _RANDOM.randint(1, 300)), 2 ** _RANDOM.randint(1, 300))
    for _ in range(120)
]
PAIRS = [(a, b) for a in VALUES[:40] for b in VALUES[:40]] + [
    (_RANDOM.choice(VALUES), _RANDOM.choice(VALUES)) for _ in range(2000)
]


def _run(kernel, *arguments):
    out = [5, 5, 5]
    kernel(*arguments, out)
    return out


def _check(out, want):
    assert out == K.from_python(want), (out, want)
    assert K.to_python(out) == want


# --- the limb kernels, in CPython -----------------------------------------


@pytest.mark.parametrize("name, kernel, python", [
    ("add", K.add, lambda a, b: a + b),
    ("sub", K.sub, lambda a, b: a - b),
    ("mul", K.mul, lambda a, b: a * b),
    ("and", K.and_, lambda a, b: a & b),
    ("or", K.or_, lambda a, b: a | b),
    ("xor", K.xor, lambda a, b: a ^ b),
])
def test_binary_kernel_matches_python_int(name, kernel, python):
    for a, b in PAIRS:
        _check(_run(kernel, K.from_python(a), K.from_python(b)), python(a, b))


def test_floordiv_and_mod_floor_with_python_sign_rules():
    for a, b in PAIRS:
        if b == 0:
            continue
        A, B = K.from_python(a), K.from_python(b)
        _check(_run(K.floordiv, A, B), a // b)
        _check(_run(K.mod, A, B), a % b)
    # Knuth D's add-back step: a quotient digit estimate one too large.
    for a, b in [
        (2**180 - 1, 2**120 + 1), (2**240 - 2**200, 2**120 - 1),
        (0x7FFFFFFF800000010000000000000000, 0x800000008000000200000005),
        (2**90 * (2**29 + 3), 2**60 + 2**30 * 7 + 5),
    ]:
        for sa in (1, -1):
            for sb in (1, -1):
                A, B = K.from_python(sa * a), K.from_python(sb * b)
                _check(_run(K.floordiv, A, B), (sa * a) // (sb * b))
                _check(_run(K.mod, A, B), (sa * a) % (sb * b))


def test_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        _run(K.floordiv, K.from_python(5), [0])
    with pytest.raises(ZeroDivisionError):
        _run(K.mod, K.from_python(-5), [0])


def test_compare_neg_and_shifts_match_python_int():
    for a, b in PAIRS[:3000]:
        A, B = K.from_python(a), K.from_python(b)
        assert K.compare(A, B) == (a > b) - (a < b)
        assert K.less(A, B) == (a < b)
        assert K.equal(A, B) == (a == b)
    for a in VALUES:
        A = K.from_python(a)
        _check(_run(K.neg, A), -a)
        for count in (0, 1, 29, 30, 31, 59, 60, 63, 64, 100, 400):
            _check(_run(K.shift_left, A, count), a << count)
            _check(_run(K.shift_right, A, count), a >> count)
    with pytest.raises(ValueError):
        _run(K.shift_right, K.from_python(1), -1)


def test_normalise_strips_top_zero_limbs_and_zero_has_sign_zero():
    _check(_run(K.normalise, [1, 5, 0, 0]), 5)
    _check(_run(K.normalise, [-1, 0, 0]), 0)
    assert _run(K.normalise, [1]) == [0]


def test_int64_and_int128_conversions():
    for x in [0, 1, -1, 2**63 - 1, -(2**63), 2**31, -(2**31) - 7] + [
        _RANDOM.randint(-(2**63), 2**63 - 1) for _ in range(500)
    ]:
        _check(_run(K.from_int64, x), x)
        assert K.to_int64_checked(K.from_python(x)) == x
    for value in [0, -1, 2**64, -(2**64), 2**127 - 1, -(2**127)] + [
        _RANDOM.randint(-(2**127), 2**127 - 1) for _ in range(500)
    ]:
        hi, lo = value >> 64, value & (2**64 - 1)
        lo_word = lo - 2**64 if lo >= 2**63 else lo
        _check(_run(K.from_int128, hi, lo_word), value)
    for value in [2**63, -(2**63) - 1, 2**64, -(2**90), 2**200]:
        with pytest.raises(OverflowError):
            K.to_int64_checked(K.from_python(value))


# --- Precision[n] carries exact integers natively ------------------------

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"


def _double_limbs(value: int, width: int) -> list[float]:
    """``value`` as ``width`` integer-valued doubles summing to it exactly."""

    limbs = []
    for _ in range(width):
        head = float(value)
        limbs.append(head)
        value -= int(head)
    assert value == 0
    return limbs


PRECISION_OPERATIONS = {
    "square": ("z * z", lambda v: v * v),
    "square_plus": ("z * z + z", lambda v: v * v + v),
    "square_minus": ("z * z - z * 3.0", lambda v: v * v - 3 * v),
}


def _precision_values(width: int) -> list[int]:
    bits = 53 * width // 2 - 2
    generator = random.Random(width)
    edges = [0, 1, -1, 2**50 + 1, 2**63, -(2**63), 2**64 + 1, -(2**64) + 1, 3**40]
    return [v for v in edges if abs(v).bit_length() <= bits] + [
        generator.randint(-(2**bits), 2**bits) for _ in range(40)
    ]


@pytest.mark.parametrize("width", (2, 4))
@pytest.mark.parametrize("operation", sorted(PRECISION_OPERATIONS))
def test_precision_expansion_is_exact_on_integers_in_cpython(width, operation):
    from src.common.tensors.extended_precision import Precision
    from src.common.tensors.numpy_backend import NumPyTensorOperations

    _source, python = PRECISION_OPERATIONS[operation]
    values = _precision_values(width)
    limbs = []
    for index in range(width):
        tensor = NumPyTensorOperations()
        tensor.data = np.array(
            [_double_limbs(v, width)[index] for v in values], dtype=np.float64,
        )
        limbs.append(tensor)
    z = Precision(limbs, width)
    result = {"square": z * z, "square_plus": z * z + z,
              "square_minus": z * z - z * 3.0}[operation]
    rows = [result.term(index).tolist() for index in range(width)]
    for position, value in enumerate(values):
        assert sum(int(row[position]) for row in rows) == python(value)


@pytest.mark.parametrize("lane", ("c", "llvm"))
@pytest.mark.parametrize("operation", sorted(PRECISION_OPERATIONS))
def test_precision_expansion_is_exact_on_integers_natively(tmp_path, lane, operation):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    from tools.benchmark_precision_cores import _feeds, _roles

    width = 4
    body, python = PRECISION_OPERATIONS[operation]
    source = (
        f"def f(x: Precision[{width}], y: Precision[{width}], n):\n"
        "    for i in range(n):\n"
        "        z = x[i]\n"
        f"        y[i] = {body}\n"
        "    return y\n"
    )
    contract = ExtractionContract(CONTRACT).with_program_abi(
        {"records": {}, "bindings": [], "values": []}
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "f", extraction_contract=contract, progress=lambda _: None,
    )
    wrapper = next(name for name in module.functions if name.endswith("__f"))
    roles = _roles(module.functions, wrapper)
    values = _precision_values(width)
    x = np.array([limb for v in values for limb in _double_limbs(v, width)])
    y = np.zeros(len(values) * width)
    feeds = _feeds(roles, width, [], x, y, len(values))
    if lane == "c":
        artifact = emit_ssa_module_to_c(module, wrapper)
        assert not artifact.shortfalls, artifact.shortfalls
        execution = artifact.compile(tmp_path).prepare_execution(feeds).run()
    else:
        artifact = emit_ssa_function_to_llvm(module, wrapper)
        assert not artifact.shortfalls, artifact.shortfalls
        execution = prepare_artifact_execution(
            compile_artifact(artifact, directory=tmp_path), feeds,
        ).run()
    rows = np.asarray(execution.buffers[roles["output"].id]).reshape(len(values), width)
    for value, row in zip(values, rows):
        assert sum(int(limb) for limb in row) == python(value), value
