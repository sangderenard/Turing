from fractions import Fraction

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.extended_precision import (
    ComplexPrecision,
    ComplexRational,
    ComplexRationalPrecision,
    Precision,
    Rational,
    RationalPrecision,
    _numeric_features,
)


def _tensor(value, dtype="float64"):
    with AbstractTensor.use_backend("numpy"):
        return AbstractTensor.get_tensor(value, dtype=dtype)


def _represented_integer_ratio(value: Rational) -> list[Fraction]:
    numerators = value.numerator.tolist()
    denominators = value.denominator.tolist()
    return [
        Fraction(int(numerator), int(denominator))
        for numerator, denominator in zip(numerators, denominators)
    ]


def test_rational_basic_arithmetic_keeps_the_quotient_structural():
    left = Rational.ratio(_tensor([1.0]), _tensor([3.0]))
    right = Rational.ratio(_tensor([2.0]), _tensor([5.0]))

    expected = {
        "add": ([11.0], [15.0], [11.0 / 15.0]),
        "sub": ([-1.0], [15.0], [-1.0 / 15.0]),
        "mul": ([2.0], [15.0], [2.0 / 15.0]),
        "div": ([5.0], [6.0], [5.0 / 6.0]),
    }
    produced = {
        "add": left + right,
        "sub": left - right,
        "mul": left * right,
        "div": left / right,
    }
    for name, value in produced.items():
        numerator, denominator, collapsed = expected[name]
        assert isinstance(value, Rational)
        assert value.numerator.tolist() == numerator
        assert value.denominator.tolist() == denominator
        assert value.collapse().tolist() == pytest.approx(collapsed)


def test_repeated_integer_division_matches_fraction_without_intermediate_division():
    value = Rational.of(_tensor([1], "int64"))
    oracle = Fraction(1)
    divisors = [(3, 2), (7, 5), (11, 13), (17, 19)]

    for numerator, denominator in divisors:
        divisor = Rational.ratio(
            _tensor([numerator], "int64"),
            _tensor([denominator], "int64"),
        )
        value = value / divisor
        oracle /= Fraction(numerator, denominator)

    assert _represented_integer_ratio(value) == [oracle]
    assert "int64" in str(value.numerator.get_dtype())
    assert "int64" in str(value.denominator.get_dtype())


def test_rational_arithmetic_is_direct_component_algebra():
    left = Rational.ratio(_tensor([3.0]), _tensor([4.0]))
    right = Rational.ratio(_tensor([2.0]), _tensor([5.0]))

    product = left * right

    assert product.numerator.tolist() == [6.0]
    assert product.denominator.tolist() == [20.0]


def test_rational_division_keeps_products_structural_until_collapse():
    value = Rational.ratio(_tensor([3.0]), _tensor([3.0]))

    quotient = value / value
    assert quotient.numerator.tolist() == [9.0]
    assert quotient.denominator.tolist() == [9.0]
    assert quotient.collapse().tolist() == [1.0]


def test_precision_and_complex_precision_promotion_is_operand_order_independent():
    tensor = _tensor([2.0])
    precision = Precision.of(tensor, 2)
    complex_precision = ComplexPrecision(_tensor([1.0]), _tensor([3.0]), 2)

    for operation in (
        lambda left, right: left + right,
        lambda left, right: left - right,
        lambda left, right: left * right,
        lambda left, right: left / right,
    ):
        assert isinstance(operation(precision, complex_precision), ComplexPrecision)
        assert isinstance(operation(complex_precision, precision), ComplexPrecision)


def _feature_values():
    real = _tensor([2.0])
    native_complex = _tensor([2.0 + 1.0j], "complex128")
    precision = Precision.of(real, 2)
    rational = Rational.ratio(real, _tensor([3.0]))
    complex_precision = ComplexPrecision.of(native_complex, 2)
    rational_precision = RationalPrecision.ratio(real, _tensor([3.0]), 2)
    complex_rational = ComplexRational.of(native_complex)
    complex_rational_precision = ComplexRationalPrecision.of(native_complex, 2)
    return (
        real,
        native_complex,
        precision,
        rational,
        complex_precision,
        rational_precision,
        complex_rational,
        complex_rational_precision,
    )


@pytest.mark.parametrize("operation", (
    lambda left, right: left + right,
    lambda left, right: left - right,
    lambda left, right: left * right,
    lambda left, right: left / right,
))
def test_every_ordered_feature_pair_promotes_to_the_union(operation):
    values = _feature_values()
    for left in values:
        for right in values:
            expected = _numeric_features(left) | _numeric_features(right)
            produced = operation(left, right)
            assert _numeric_features(produced) == expected, (
                type(left).__name__, type(right).__name__, type(produced).__name__
            )


def test_rational_precision_delays_the_only_wide_division_until_collapse():
    numerator = Precision.of(_tensor([1.0]), 3)
    denominator = Precision.of(_tensor([3.0]), 3)
    value = RationalPrecision.ratio(numerator, denominator, 3)

    assert value.components() == (numerator, denominator)
    reciprocal = value.reciprocal()
    assert reciprocal.numerator is denominator
    assert reciprocal.denominator is numerator
    quotient = value.quotient()
    assert isinstance(quotient, Precision)
    assert quotient.limbs == 3
