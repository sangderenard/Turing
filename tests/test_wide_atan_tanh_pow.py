"""Correctly rounded atan, atan2, tanh and pow from ``WideExpansionEmitter``.

Each test builds a tiny SSA function with the emitter's full-range
composition (``signal_symbolic`` proof cores + exact lane selection), emits
it through the C backend, runs it and compares bit for bit -- sign of zero
included -- with mpmath's value rounded once to a double (subnormals rounded
here: mpmath's own ``float()`` rounds a subnormal twice).  Special values are
compared with Python's own result.  A fault the emitter returns is guarded
the way ``reduce_integer_pow`` guards its own: a ``turing_validation_error``
call in a fault block, raised by the run as the named Python exception.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from src.compiler.ir_identities import WideExpansionEmitter
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue

INF, NAN = float("inf"), float("nan")
SPECIAL = (0.0, -0.0, INF, -INF, NAN, 5e-324, -5e-324, 2.0 ** -1022,
           1.7976931348623157e308, -1.7976931348623157e308, 1.0, -1.0, 0.5, -0.5)

_BUILT: dict = {}


def _built(kind, tmp_path_factory, **options):
    key = (kind, tuple(sorted(options.items())))
    if key in _BUILT:
        return _BUILT[key]
    y, x = SSAValue(0, "float64"), SSAValue(1, "float64")
    emitted: list = []
    function = Function("root", [y, x], {"entry": BasicBlock("entry", [])})
    emitter = WideExpansionEmitter(function, emitted)
    faults = ()
    if kind == "atan":
        limbs = emitter.atan_expansion(y)
    elif kind == "tanh":
        limbs = emitter.tanh_expansion(y)
    elif kind == "atan2":
        limbs = emitter.atan2_expansion(y, x)
    else:
        limbs, faults = emitter.pow_expansion(y, x, **options)
    total = limbs[0]
    for limb in limbs[1:]:
        total = emitter.put("Add", (total, limb), total)
    entry = function.blocks["entry"]
    entry.instrs = list(emitted)
    current = entry
    for index, (test, code, failure) in enumerate(faults):
        fail, passed = BasicBlock(f"fault{index}", []), BasicBlock(f"pass{index}", [])
        current.instrs.append(Instr("CondBr", [test], None, attributes={
            "true_target": fail.name, "false_target": passed.name,
        }))
        current.successors = [fail.name, passed.name]
        fail.instrs = [
            Instr("Call", [], None, attributes={
                "callee": "turing_validation_error", "error_code": int(code),
                "failure": failure,
            }),
            Instr("Br", [], None, attributes={"target": passed.name}),
        ]
        fail.successors = [passed.name]
        function.blocks[fail.name] = fail
        function.blocks[passed.name] = passed
        current = passed
    current.instrs.append(Instr("Ret", [total], None))
    artifact = emit_ssa_to_c(IRModule({"root": function}), "root")
    assert artifact.complete, artifact.shortfalls
    # The composition spells no host transcendental: only exact IEEE ops.
    spelled = {str(instruction.op) for instruction in emitted}
    assert not spelled & {"Exp", "Log", "Pow", "sin", "atan", "tanh", "Sin"}
    artifact.compile(tmp_path_factory.mktemp(kind) / "native")
    _BUILT[key] = (artifact, int(total.id))
    return _BUILT[key]


def _run(built, y, x=0.0):
    artifact, out = built
    buffers = artifact.prepare_execution({
        0: np.array([y], dtype=np.float64), 1: np.array([x], dtype=np.float64),
    }).run().buffers
    return float(np.asarray(buffers[out]).reshape(-1)[0])


# 2400 bits: a subnormal result's distance from its rounding midpoint can be
# as small as x**3 relative to x (atan2(3*2**-1000, 2**75)), ~2**-2150.
mpmath.mp.prec = 2400
_TINY = mpmath.mpf(2) ** -1022


def _rounded(value):
    if abs(value) < _TINY:
        result = float(mpmath.nint(value * mpmath.mpf(2) ** 1074)) * 2.0 ** -1074
        return result if result != 0 else (-0.0 if value < 0 else 0.0)
    return float(value)


def _same(got, want):
    if math.isnan(got) or math.isnan(want):
        return math.isnan(got) and math.isnan(want)
    return got == want and math.copysign(1.0, got) == math.copysign(1.0, want)


def _check(built, cases, truth, python):
    wrong = []
    for args in cases:
        got = _run(built, *args)
        exact = all(math.isfinite(a) and a != 0 for a in args)
        want = _rounded(truth(*(mpmath.mpf(a) for a in args))) if exact else python(*args)
        if not _same(got, want):
            wrong.append((args, got, want))
    assert not wrong, wrong[:5]


_RNG = np.random.default_rng(1110)


def _spread(count, low=-1074.0, high=1023.9):
    magnitudes = np.exp2(_RNG.uniform(low, high, count))
    return [float(v) for v in magnitudes * _RNG.choice([-1.0, 1.0], count)]


def test_atan_is_correctly_rounded_over_the_full_range(tmp_path_factory):
    built = _built("atan", tmp_path_factory)
    cases = [(v,) for v in (*SPECIAL, 0.41421356237309503, 0.4142135623730951,
                            1.0000000000000002, 0.9999999999999999, 1e16, -3.7)]
    cases += [(v,) for v in _spread(150)]
    cases += [(float(v),) for v in _RNG.uniform(-5.0, 5.0, 150)]
    _check(built, cases, mpmath.atan, math.atan)


def test_tanh_is_correctly_rounded_near_zero_and_saturating(tmp_path_factory):
    built = _built("tanh", tmp_path_factory)
    cases = [(v,) for v in (*SPECIAL, 0.5, 0.5000000000000001, 19.0, 19.07,
                            22.0, 22.000000000000004, 1e-8, -1e-300)]
    cases += [(v,) for v in _spread(100, -1074.0, -1.0)]
    cases += [(float(v),) for v in _RNG.uniform(-23.0, 23.0, 200)]
    _check(built, cases, mpmath.tanh, math.tanh)


def test_atan2_quadrants_specials_and_subnormal_quotients(tmp_path_factory):
    built = _built("atan2", tmp_path_factory)
    cases = [(a, b) for a in SPECIAL for b in SPECIAL]
    cases += list(zip(_spread(150), _spread(150)))
    cases += list(zip(_spread(100, -20, 20), _spread(100, -20, 20)))
    # A subnormal quotient exactly on a rounding midpoint: atan lies just
    # below it, so the correct result rounds down (CPython's rounds to even).
    cases += [(3 * 2.0 ** -1000, 2.0 ** 75), (-3 * 2.0 ** -1000, 2.0 ** 75)]
    _check(built, cases, mpmath.atan2, math.atan2)
    assert _run(built, 3 * 2.0 ** -1000, 2.0 ** 75) == 5e-324


def _python_pow(x, y):
    try:
        result = x ** y
    except (ZeroDivisionError, OverflowError) as error:
        return type(error)
    return "complex" if isinstance(result, complex) else result


def _native_pow(built, x, y):
    from src.compiler.native_faults import NativeFloatPowComplexResult

    try:
        return _run(built, x, y)
    except NativeFloatPowComplexResult:
        return "complex"
    except (ZeroDivisionError, OverflowError, ValueError) as error:
        return next(kind for kind in (ZeroDivisionError, OverflowError, ValueError)
                    if isinstance(error, kind))


def test_pow_is_correctly_rounded_with_exact_midpoints(tmp_path_factory):
    built = _built("pow", tmp_path_factory, form="operator")
    cases = [(a, b) for a in SPECIAL for b in SPECIAL]
    cases += [
        (3.0, 34.0),                  # 54-bit odd integer: a tie, to even
        (68718952449.0, 1.5),         # (2**18-1)**3: a tie through sqrt
        (2.0, -1075.0), (2.0 ** -25, 43.0),  # 2**-1075: tie to +0
        (3 * 2.0 ** -215, 5.0),       # 243 * 2**-1075: subnormal tie
        (0.5, 1074.3), (1.5, -1800.0), (10.0, 308.0), (10.0, 309.0),
        (0.9999999999, 1e12), (1.0 + 2.0 ** -52, 2.0 ** 60), (1e-310, 0.5),
        (-2.0, 3.0), (-2.0, 0.5), (-1e-300, 3.0), (0.0, -1.0), (-0.0, -3.0),
        (-0.0, 3.0), (0.0, -INF), (-1.0, INF), (NAN, 0.0), (1.0, NAN),
        (1.7976931348623157e308, 1.7976931348623157e308),
    ]
    magnitudes = np.exp2(_RNG.uniform(-1074.0, 1023.9, 150))
    targets = _RNG.uniform(-1110.0, 1060.0, 150)
    cases += [(float(a), float(t / np.log2(a))) for a, t in zip(magnitudes, targets)]
    cases += [(float(a), float(b)) for a, b in zip(_RNG.uniform(-20, 20, 100),
                                                   _RNG.integers(-300, 300, 100))]
    wrong = []
    for x, y in cases:
        got, want = _native_pow(built, x, y), _python_pow(x, y)
        if isinstance(want, float) and all(math.isfinite(a) and a != 0 for a in (x, y)) \
                and math.isfinite(want) and want != 0:
            want = _rounded(mpmath.power(mpmath.mpf(x), mpmath.mpf(y)))
        if isinstance(got, float) and isinstance(want, float):
            matches = _same(got, want)
        else:
            matches = got == want
        if not matches:
            wrong.append(((x, y), got, want))
    assert not wrong, wrong[:5]


def test_math_pow_form_reports_math_faults(tmp_path_factory):
    from src.compiler.native_faults import NativeFloatOverflow, NativeValueError

    built = _built("pow", tmp_path_factory, form="math")
    for x, y in ((0.0, -1.0), (-0.0, -3.0), (-8.0, 1.0 / 3.0)):
        with pytest.raises(NativeValueError, match="math domain error"):
            _run(built, x, y)
    with pytest.raises(NativeFloatOverflow, match="math range error"):
        _run(built, 10.0, 400.0)
    assert _run(built, 2.0, 10.0) == math.pow(2.0, 10.0)
    assert _run(built, 0.0, -INF) == math.pow(0.0, -INF)
