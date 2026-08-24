"""What the precision surface computes, pinned against exact arithmetic.

These tests exist to make a STORAGE change safe. The limbs are moving from
interleaved channels to a planar stack, and the whole risk of that move is
that a layout error is silent: a value read with the wrong stride is still
a perfectly ordinary tensor of plausible numbers. So every assertion here
is against an exact ``Fraction`` reference computed from the limbs
themselves, never against a remembered float, and the seams that cross the
boundary -- the eager cores and the compiled kernels' feeds -- are checked
by value rather than by shape.
"""

from fractions import Fraction

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.extended_precision import Precision


VALUES = [0.1, 0.25, 1.5, -0.75]


def wide(values=VALUES, width=2):
    return Precision.of(
        AbstractTensor.get_tensor(np.asarray(values, dtype=np.float64)), width
    )


def exact(value: Precision):
    """The number a Precision denotes: the exact sum of its limbs."""

    rows = value.to_float_lists()
    count = len(rows[0])
    return [
        sum((Fraction(row[index]) for row in rows), Fraction())
        for index in range(count)
    ]


@pytest.mark.parametrize("width", (2, 3, 4))
def test_arithmetic_is_exact_at_every_width(width):
    left, right = wide(width=width), wide([2.0, 3.0, 0.5, -4.0], width)
    reference = [Fraction(v) for v in VALUES]
    other = [Fraction(v) for v in [2.0, 3.0, 0.5, -4.0]]

    for produced, expected in (
        (left + right, [a + b for a, b in zip(reference, other)]),
        (left - right, [a - b for a, b in zip(reference, other)]),
        (left * right, [a * b for a, b in zip(reference, other)]),
        (-left, [-a for a in reference]),
    ):
        for got, want in zip(exact(produced), expected):
            assert abs(got - want) < Fraction(1, 10 ** (14 * width))


def test_width_buys_precision_that_a_double_cannot_hold():
    """A third of one, at two widths: the wider answer is nearer.

    The point of the type in one assertion. If a storage change quietly
    collapsed the limbs, both widths would agree exactly -- and that
    agreement is the failure, not the pass.
    """

    third_2 = exact(wide([1.0], 2) / 3.0)[0]
    third_4 = exact(wide([1.0], 4) / 3.0)[0]
    truth = Fraction(1, 3)
    assert abs(third_4 - truth) < abs(third_2 - truth)
    assert abs(third_2 - truth) < Fraction(1, 10 ** 28)
    assert abs(third_4 - truth) < Fraction(1, 10 ** 60)


def test_reductions_keep_the_limbs_they_were_given():
    """Summing must not round each element to a double on the way."""

    total = exact(wide().sum())[0]
    assert abs(total - sum(Fraction(v) for v in VALUES)) < Fraction(1, 10 ** 28)

    average = exact(wide().mean())[0]
    expected = sum(Fraction(v) for v in VALUES) / len(VALUES)
    assert abs(average - expected) < Fraction(1, 10 ** 28)


def test_elementwise_surface_matches_exact_arithmetic():
    # abs is exact: it only changes signs, which no limb has to round.
    assert exact(abs(wide())) == [abs(Fraction(v)) for v in VALUES]
    # A cube needs three times the input's bits, which is more than two
    # limbs hold, so this is checked to the width's own resolution rather
    # than to equality -- demanding exactness would be demanding precision
    # the representation never claimed.
    for got, want in zip(exact(wide() ** 3), [Fraction(v) ** 3 for v in VALUES]):
        assert abs(got - want) < Fraction(1, 10 ** 28)
    assert [float(v) for v in wide().floor().tolist()] == [
        float(np.floor(v)) for v in VALUES
    ]
    assert [float(v) for v in wide().sign().tolist()] == [
        float(np.sign(v)) for v in VALUES
    ]


def test_sqrt_converges_to_the_exact_root():
    root = exact(wide([2.0, 9.0], 3).sqrt())
    assert abs(root[1] - 3) < Fraction(1, 10 ** 40)
    assert abs(root[0] * root[0] - 2) < Fraction(1, 10 ** 40)


def test_transcendentals_are_evaluated_at_the_value_s_own_width():
    """Routed to the cores, which hold the width rather than collapsing it."""

    from src.common.tensors import signal_symbolic as proof

    arguments = [0.1, -0.3, 0.5]
    errors = {}
    for width in (2, 3):
        produced = Precision.of(
            AbstractTensor.get_tensor(np.asarray(arguments)), width
        ).sin()
        assert isinstance(produced, Precision)
        assert produced.limbs == width
        truth = proof.exact_evaluator("sin", proof.CORE_RADII["sin"], 70)
        errors[width] = max(
            abs(got - truth(value))
            for got, value in zip(exact(produced), arguments)
        )
    # Each limb has to buy precision, or the routing collapsed somewhere.
    assert errors[3] < errors[2]
    assert errors[2] < Fraction(1, 10 ** 30)


def test_a_core_refuses_outside_the_interval_it_was_proven_on():
    """Extrapolating a core returns a plausible number, so it must not."""

    beyond = Precision.of(AbstractTensor.get_tensor(np.asarray([2.0])), 2)
    with pytest.raises(ValueError, match="outside the core"):
        beyond.sin()


def test_operations_with_no_wide_meaning_still_refuse():
    for name in ("erf", "gamma", "arcsinh"):
        with pytest.raises((AttributeError, TypeError, NotImplementedError)):
            getattr(wide(), name)()


def test_eager_core_evaluates_wide_and_beats_the_double_path():
    """The seam to signal_symbolic: a wide argument stays wide through it."""

    from src.common.tensors import signal_symbolic as proof

    argument = 0.3
    narrow = float(proof.evaluate_proof("sin", argument, 1))
    widened = proof.evaluate_proof(
        "sin", AbstractTensor.get_tensor(np.asarray([argument])), 3,
    )
    assert isinstance(widened, Precision)
    assert widened.limbs == 3

    truth = proof.exact_evaluator("sin", proof.CORE_RADII["sin"], 60)(argument)
    wide_error = abs(exact(widened)[0] - truth)
    narrow_error = abs(Fraction(narrow) - truth)
    assert wide_error < narrow_error
    assert wide_error < Fraction(1, 10 ** 30)


@pytest.mark.parametrize("name", ("tau", "pi", "e", "ln2", "ln10"))
@pytest.mark.parametrize("width", (1, 2, 4, 8))
def test_constants_are_derived_correctly_at_any_width(name, width):
    """Every constant, derived not borrowed, to whatever width is asked.

    A constant taken from libm is the one thing this stack exists to
    replace, and a constant that stops converging is worse than one that
    refuses -- it silently caps the precision of everything downstream.
    ``ln10`` did exactly that: its derivation added a term belonging to a
    different identity and returned ln(10) + 2*ln(2), which nothing
    noticed while nothing consumed it.
    """

    import mpmath

    from src.common.tensors.signal_symbolic import constant_limbs

    parts = constant_limbs(name, width)
    assert len(parts) == width
    total = sum((Fraction(part) for part in parts), Fraction())
    with mpmath.workdps(40 + 20 * width):
        truth = {
            "tau": 2 * mpmath.pi, "pi": mpmath.pi, "e": mpmath.e,
            "ln2": mpmath.log(2), "ln10": mpmath.log(10),
        }[name]
        error = abs(mpmath.mpf(total.numerator) / total.denominator - truth)
        # Each limb has to buy roughly fifteen digits, or the derivation
        # has stopped converging and the width is decorative.
        assert error < mpmath.mpf(10) ** (-14 * width)
