"""Exact Python ``int`` arithmetic on Precision expansions, natively.

``WideExpansionEmitter`` carries an integer past the native width as an
expansion of integer-valued binary64 limbs.  Here its integer kernels --
``integer_from_int64``, ``integer_to_int64``, ``integer_divmod`` (Python's
floor ``//`` and ``%``) and ``renormalise_exact`` -- are built into SSA,
compiled on the C and LLVM lanes and held to Python's ``int``: every value
exact, ``ZeroDivisionError`` and int64 overflow reported as predicates, and a
result that does not fit its width reported, never rounded.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.compiler.ir_identities import WideExpansionEmitter
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue

WIDTH = 4
MIX_MODULUS = 4611686018427387847
_RANDOM = random.Random(20261029)


def _limbs(value: int, width: int) -> list[float]:
    """``value`` as ``width`` integer-valued doubles summing to it exactly."""

    limbs = []
    for _ in range(width):
        head = float(value)
        limbs.append(head)
        value -= int(head)
    assert value == 0
    return limbs


def _build(name, inputs, body):
    """One function: the emitter's SSA, then a second block that returns.

    Two blocks take the repository module emitter on LLVM, the path every
    lowered program takes.
    """

    emitted: list = []
    function = Function(name, list(inputs), {"entry": BasicBlock("entry", emitted)})
    outputs = [
        value for value in body(WideExpansionEmitter(function, emitted))
        if value is not None
    ]
    emitted.append(Instr("Br", [], None, attributes={"target": "exit"}))
    function.blocks["entry"].successors = ["exit"]
    function.blocks["exit"] = BasicBlock("exit", [Instr("Ret", outputs, None)])
    return IRModule({name: function}), name, outputs


def _compile(module, name, lane, directory):
    if lane == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(directory / "native")
        return lambda feeds: artifact.prepare_execution(feeds).run().buffers
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    emitted = emit_ssa_function_to_llvm(module, name)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=directory)
    return lambda feeds: prepare_artifact_execution(artifact, feeds).run().buffers


def _read(buffers, value):
    return np.asarray(buffers[value.id]).reshape(-1)[0]


def _integer(buffers, values) -> int:
    return sum(int(_read(buffers, value)) for value in values)


def _int64(value: int):
    return np.array([value], dtype=np.int64)


LANES = pytest.mark.parametrize("lane", ["c", "llvm"])

INT64_EDGES = [
    0, 1, -1, 2**63 - 1, -(2**63), 2**32, 2**32 - 1, -(2**32), -(2**32) - 1,
    2**53 + 1, -(2**53) - 1, 2**62, MIX_MODULUS,
]


@LANES
def test_int64_enters_an_expansion_exactly(tmp_path, lane):
    source = SSAValue(0, "int64", shape=())
    module, name, outputs = _build(
        "integer_from_int64", [source],
        lambda emitter: emitter.integer_from_int64(source, WIDTH),
    )
    run = _compile(module, name, lane, tmp_path)
    values = INT64_EDGES + [
        _RANDOM.randint(-(2**63), 2**63 - 1) for _ in range(80)
    ]
    for value in values:
        assert _integer(run({source.id: _int64(value)}), outputs) == value


@LANES
def test_expansion_leaves_for_int64_checked(tmp_path, lane):
    limbs = [SSAValue(index, "float64", shape=()) for index in range(WIDTH)]
    like = SSAValue(WIDTH, "int64", shape=())
    module, name, (value, outside) = _build(
        "integer_to_int64", [*limbs, like],
        lambda emitter: emitter.integer_to_int64(limbs, like),
    )
    run = _compile(module, name, lane, tmp_path)
    values = INT64_EDGES + [
        2**63, -(2**63) - 1, 2**64, -(2**64), 2**90 + 5, -(2**100), 3**120,
    ] + [_RANDOM.randint(-(2**66), 2**66) for _ in range(80)]
    for number in values:
        feeds = {
            limb.id: np.array([part])
            for limb, part in zip(limbs, _limbs(number, WIDTH))
        }
        feeds[like.id] = _int64(0)
        buffers = run(feeds)
        if -(2**63) <= number < 2**63:
            assert not bool(_read(buffers, outside)), number
            assert int(_read(buffers, value)) == number
        else:
            assert bool(_read(buffers, outside)), number


@LANES
def test_divmod_is_pythons_floor_division_and_modulus(tmp_path, lane):
    left = [SSAValue(index, "float64", shape=()) for index in range(WIDTH)]
    right = [SSAValue(WIDTH + index, "float64", shape=()) for index in range(2)]
    state = {}

    def body(emitter):
        quotient, remainder, zero, inexact = emitter.integer_divmod(
            left, right, WIDTH,
        )
        state.update(quotient=quotient, remainder=remainder)
        return [*quotient, *remainder, zero, inexact]

    module, name, outputs = _build("integer_divmod", [*left, *right], body)
    zero, inexact = outputs[2 * WIDTH], outputs[2 * WIDTH + 1]
    run = _compile(module, name, lane, tmp_path)
    cases = [
        (17000110000177 * 1000003 + 17000110000177, MIX_MODULUS),
        (-5, 3), (5, -3), (-5, -3), (0, 7), (6, 3), (-6, 3), (6, -3),
        (2**180 + 12345, 1000000007), (-(2**170) - 1, 2**62 + 3),
        (2**63, -1), (-(2**63), -1), (3**100, -(2**61 + 1)),
        ((-(2**63)) ** 3 + 2**63 - 1, 2**63 - 1),
        (2**200, 2**63 - 1), (-(2**200), -(2**63) + 1),
    ] + [
        (
            _RANDOM.randint(-(2**200), 2**200),
            _RANDOM.choice([1, -1]) * _RANDOM.randint(1, 2**63),
        )
        for _ in range(60)
    ]
    for number, divisor in [*cases, (7, 0), (-(2**150), 0)]:
        feeds = {
            limb.id: np.array([part])
            for limb, part in zip(left, _limbs(number, WIDTH))
        }
        feeds.update({
            limb.id: np.array([part])
            for limb, part in zip(right, _limbs(divisor, 2))
        })
        buffers = run(feeds)
        if divisor == 0:
            assert bool(_read(buffers, zero))
            continue
        assert not bool(_read(buffers, zero))
        assert not bool(_read(buffers, inexact)), (number, divisor)
        assert (
            _integer(buffers, state["quotient"]),
            _integer(buffers, state["remainder"]),
        ) == divmod(number, divisor), (number, divisor)


def _int64_chain(name, body):
    """int64 arguments in, an int64 result and its fault predicates out."""

    arguments = [SSAValue(index, "int64", shape=()) for index in range(3)]

    def emit(emitter):
        value, faults = body(emitter, arguments)
        return [value, *faults]

    module, name, outputs = _build(name, arguments, emit)
    return module, name, arguments, outputs


def _product(emitter, left, right):
    terms = [
        term for high in left for low in right
        for term in emitter.two_product(high, low)
    ]
    return emitter.renormalise_exact(terms, WIDTH)


@LANES
def test_cube_plus_modulus_matches_python(tmp_path, lane):
    """``(a * a * a + b) % m``: about 2**189 in the middle, int64 at the end."""

    def body(emitter, arguments):
        a, b, m = (emitter.integer_from_int64(v, WIDTH) for v in arguments)
        square, lost_square = _product(emitter, a, a)
        cube, lost_cube = _product(emitter, square, a)
        total, lost_total = emitter.renormalise_exact([*cube, *b], WIDTH)
        _quotient, remainder, zero, lost_divmod = emitter.integer_divmod(
            total, m[:2], WIDTH,
        )
        value, outside = emitter.integer_to_int64(remainder, arguments[0])
        inexact = emitter._join_inexact(
            lost_square, lost_cube, lost_total, lost_divmod,
        )
        return value, (outside, zero, inexact)

    module, name, arguments, outputs = _int64_chain("cube_mod", body)
    run = _compile(module, name, lane, tmp_path)
    cases = [
        (4611686018427387000, 1, 1000000007), (5, 3, 1000000007),
        (-(2**63), 2**63 - 1, 2**63 - 1), (2**63 - 1, -5, -(2**62) - 7),
        (12345, 0, 1), (-7, 0, 2),
    ] + [
        (
            _RANDOM.randint(-(2**63), 2**63 - 1),
            _RANDOM.randint(-(2**63), 2**63 - 1),
            _RANDOM.choice([1, -1]) * _RANDOM.randint(1, 2**63 - 1),
        )
        for _ in range(40)
    ]
    for a, b, m in cases:
        buffers = run({
            argument.id: _int64(number)
            for argument, number in zip(arguments, (a, b, m))
        })
        value, *faults = (_read(buffers, output) for output in outputs)
        assert not any(bool(fault) for fault in faults), (a, b, m)
        assert int(value) == (a * a * a + b) % m


@LANES
def test_mix_hash_digest_through_expansions_matches_python(tmp_path, lane):
    """The string-core digest ``(d * 1000003 + token + 7) % M`` per round."""

    def body(emitter, arguments):
        d, token, seven = arguments
        digest = emitter.integer_from_int64(d, WIDTH)
        factor = emitter.integer_from_int64(
            emitter._typed_constant(1000003, "int64", d), WIDTH,
        )
        product, lost_product = _product(emitter, digest, factor)
        total, lost_total = emitter.renormalise_exact(
            [*product, *emitter.integer_from_int64(token, WIDTH),
             *emitter.integer_from_int64(seven, WIDTH)],
            WIDTH,
        )
        modulus = emitter.integer_from_int64(
            emitter._typed_constant(MIX_MODULUS, "int64", d), 2,
        )
        _quotient, remainder, zero, lost = emitter.integer_divmod(
            total, modulus, WIDTH,
        )
        value, outside = emitter.integer_to_int64(remainder, d)
        return value, (
            outside, zero,
            emitter._join_inexact(lost_product, lost_total, lost),
        )

    module, name, arguments, outputs = _int64_chain("mix_round", body)
    run = _compile(module, name, lane, tmp_path)
    for seed, rounds in [(7, 3), (42, 20), (0, 1), (2**62, 5)]:
        digest = expected = 0
        for index in range(rounds):
            token = seed + index
            expected = (expected * 1000003 + token + 7) % MIX_MODULUS
            buffers = run({
                arguments[0].id: _int64(digest),
                arguments[1].id: _int64(token),
                arguments[2].id: _int64(7),
            })
            value, *faults = (_read(buffers, output) for output in outputs)
            assert not any(bool(fault) for fault in faults)
            digest = int(value)
            assert digest == expected, (seed, index)


@LANES
def test_a_result_past_the_width_is_reported_not_rounded(tmp_path, lane):
    """Two limbs hold about 106 bits; ``(2**63 - 1)**2`` needs 126."""

    source = SSAValue(0, "int64", shape=())
    state = {}

    def body(emitter):
        value = emitter.integer_from_int64(source, 2)
        terms = [
            term for high in value for low in value
            for term in emitter.two_product(high, low)
        ]
        square, inexact = emitter.renormalise_exact(terms, 2)
        state["square"] = square
        return [*square, inexact]

    module, name, outputs = _build("square_width_two", [source], body)
    run = _compile(module, name, lane, tmp_path)
    exact = run({source.id: _int64(3037000499)})
    assert not bool(_read(exact, outputs[-1]))
    assert _integer(exact, state["square"]) == 3037000499**2
    assert bool(_read(run({source.id: _int64(2**63 - 1)}), outputs[-1]))
