"""sin/cos/tan of one float64 as wide SSA from the signal_symbolic proofs.

``WideExpansionEmitter.circular_double`` reduces over the whole double range
(Cody-Waite below 2**10, the windowed 2/pi reduction above), evaluates the
``sin``/``cos`` proof cores in two limbs, selects the quadrant and collapses
once. The test builds the tiny SSA function with the emitter, runs it through
the C backend and requires every result to be the correctly rounded double
(mpmath), bit for bit, signed zeros included. (trig cores, package S2b)
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from src.compiler.ir_identities import (
    CIRCULAR_CODY_WAITE_LIMIT, WideExpansionEmitter,
    circular_payne_hanek_windows,
)
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue

#: Only IEEE-exact or correctly rounded primitives: no host transcendental.
_EXACT_OPS = {
    "Const", "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Fma", "Floor",
    "Round", "Select", "Eq", "Lt", "Le", "Ge", "Gt", "Ret",
}


def _arguments() -> list[float]:
    rng = np.random.default_rng(1029)

    def logu(lo, hi, n):
        values = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        return list(values * rng.choice([-1.0, 1.0], n))

    subnormals = [
        struct.unpack("<d", struct.pack("<Q", int(bits)))[0]
        for bits in rng.integers(1, 1 << 52, 20)
    ]
    mp.mp.prec = 2400
    near_quadrants = [
        float(mp.mpf(int(k)) * mp.pi / 2)
        for k in [*rng.integers(1, 1 << 20, 20), *rng.integers(1, 1 << 62, 20)]
    ]
    return [
        *rng.uniform(-math.pi / 4, math.pi / 4, 60),
        *rng.uniform(-10.0, 10.0, 60),
        *logu(10.0, CIRCULAR_CODY_WAITE_LIMIT, 60),
        *logu(CIRCULAR_CODY_WAITE_LIMIT, 1e15, 60),
        *logu(1e15, 1.79e308, 80),
        *logu(1e-300, 1e-1, 40),
        *subnormals, *(-value for value in subnormals[:10]),
        *near_quadrants, *(-value for value in near_quadrants[:10]),
        0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308,
        1.7976931348623157e308, -1.7976931348623157e308,
        # the double nearest a multiple of pi/2 (|r| ~ 2**-60.9)
        6381956970095103 * 2.0 ** 797, -6381956970095103 * 2.0 ** 797,
        CIRCULAR_CODY_WAITE_LIMIT, math.nextafter(CIRCULAR_CODY_WAITE_LIMIT, 0),
        2.0 ** 24, 2.0 ** 1008, math.pi / 4, math.pi / 2, math.pi, 1e22,
    ]


def _build(name: str, tmp_path):
    source = SSAValue(0, "float64", shape=())
    emitted: list = []
    function = Function(
        f"circular_{name}", [source], {"entry": BasicBlock("entry", emitted)},
        metadata={"output_names": ("value", "domain_error")},
    )
    value, domain = WideExpansionEmitter(function, emitted).circular_double(
        name, source,
    )
    emitted.append(Instr("Ret", [value, domain], None))
    assert {instruction.op for instruction in emitted} <= _EXACT_OPS
    artifact = emit_ssa_to_c(IRModule({function.name: function}), function.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / name)

    def run(argument: float) -> tuple[float, bool]:
        execution = artifact.prepare_execution({
            source.id: np.array([argument], dtype=np.float64),
        }).run()
        return (
            float(np.asarray(execution.buffers[value.id]).reshape(-1)[0]),
            bool(np.asarray(execution.buffers[domain.id]).reshape(-1)[0]),
        )

    return run


def test_windows_are_exact_cuts_of_two_over_pi():
    from fractions import Fraction

    from src.common.tensors.signal_symbolic import constant_rational

    windows = circular_payne_hanek_windows()
    assert len(windows) == 1023 // 24 + 1
    two_over_pi = Fraction(2) / constant_rational("pi", 500)
    for index in (0, 1, 17, len(windows) - 1):
        exact = (two_over_pi * (1 << (24 * index))) % (1 << 54)
        assert abs(sum(map(Fraction, windows[index])) - exact) < Fraction(1, 1 << 260)


@pytest.mark.parametrize("name", ["sin", "cos", "tan"])
def test_circular_double_is_correctly_rounded_over_the_double_range(
    name, tmp_path,
):
    run = _build(name, tmp_path)
    exact_of = {"sin": mp.sin, "cos": mp.cos, "tan": mp.tan}[name]
    mp.mp.prec = 2400
    wrong = []
    for argument in _arguments():
        got, domain = run(argument)
        if argument == 0.0:
            want = 1.0 if name == "cos" else argument
        else:
            want = float(exact_of(mp.mpf(argument)))
        if domain or got != want or (
            math.copysign(1.0, got) != math.copysign(1.0, want)
        ):
            wrong.append((argument.hex(), got, want, domain))
    assert not wrong, wrong[:5]


@pytest.mark.parametrize("name", ["sin", "cos", "tan"])
def test_circular_double_special_values_follow_python(name, tmp_path):
    run = _build(name, tmp_path)
    # nan in, nan out, and no domain error -- as math.sin(nan).
    got, domain = run(math.nan)
    assert math.isnan(got) and not domain
    # +-inf: Python raises ValueError; the emitter hands the lowering the
    # predicate to guard, with NaN in the value lane.
    for infinite in (math.inf, -math.inf):
        with pytest.raises(ValueError):
            getattr(math, name)(infinite)
        got, domain = run(infinite)
        assert domain and math.isnan(got)
