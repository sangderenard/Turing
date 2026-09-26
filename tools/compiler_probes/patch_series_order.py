"""Size each constant's series by ITS OWN argument, not by a shared guess."""
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "src/common/tensors/signal_symbolic.py"

OLD = '''    order = int(digits * 1.6) + 24
    if name == "pi":
        return (16 * _atan_rational(Fraction(1, 5), order)
                - 4 * _atan_rational(Fraction(1, 239), order))'''

NEW = '''    # The order a series needs is set by its ARGUMENT, not by the digits
    # alone. Each term of atan or atanh multiplies by the argument
    # squared, so the series gains -2*log10(|x|) digits per term: 1.4 at
    # a fifth, 4.8 at a two-hundred-and-thirty-ninth, 0.95 at a third,
    # 0.74 at three sevenths. One shared order therefore over-serves the
    # fast arguments and silently UNDER-serves the slow ones -- which is
    # not a refusal but a cap, and a capped constant quietly limits the
    # precision of everything downstream of it. Sizing per argument is
    # what makes "ask for more digits and the same code produces them"
    # true at every width rather than only at narrow ones.
    if name == "pi":
        return (16 * _atan_rational(Fraction(1, 5),
                                    _series_order(Fraction(1, 5), digits))
                - 4 * _atan_rational(Fraction(1, 239),
                                     _series_order(Fraction(1, 239), digits)))'''

OLD_E = '''    order = int(digits * 1.6) + 24'''

OLD_LN2 = '''    if name == "ln2":
        return 2 * _atanh_rational(Fraction(1, 3), order)'''
NEW_LN2 = '''    if name == "ln2":
        return 2 * _atanh_rational(
            Fraction(1, 3), _series_order(Fraction(1, 3), digits)
        )'''

OLD_LN10 = '''        return (2 * _atanh_rational(Fraction(3, 7), order)
                + 2 * constant_rational("ln2", digits))'''
NEW_LN10 = '''        return (2 * _atanh_rational(
                    Fraction(3, 7), _series_order(Fraction(3, 7), digits)
                ) + 2 * constant_rational("ln2", digits))'''

HELPER = '''

def _series_order(argument: Fraction, digits: int) -> int:
    """The polynomial order a power series in ``argument`` needs.

    ``atan`` and ``atanh`` advance by the square of their argument, so
    each term is worth ``-2*log10(|argument|)`` decimal digits and the
    term count is the digits wanted divided by that. The order is twice
    the term count because both series are odd, plus a margin that costs
    nothing in exact arithmetic and covers the last term's own size.
    """

    import math

    magnitude = abs(float(argument))
    if not 0.0 < magnitude < 1.0:
        raise ValueError(
            f"a power series in {argument!r} does not converge; the "
            "argument must lie strictly inside the unit interval"
        )
    per_term = -2.0 * math.log10(magnitude)
    return int(2.0 * (float(digits) / per_term)) + 24

'''


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "_series_order" in text:
        raise SystemExit("already sized per argument; refusing to re-apply")
    for anchor in (OLD, OLD_LN2, OLD_LN10):
        if text.count(anchor) != 1:
            raise SystemExit(f"anchor not unique: {anchor[:40]!r}")
    text = text.replace(OLD, NEW, 1)
    text = text.replace(OLD_LN2, NEW_LN2, 1)
    text = text.replace(OLD_LN10, NEW_LN10, 1)
    # ``e`` still uses a shared order; its series is factorial and converges
    # far faster than any of these, so it keeps the old sizing.
    text = text.replace(
        '    if name == "e":',
        '''    order = int(digits * 1.6) + 24  # the factorial series, which
    # outruns every power series here and needs no argument sizing.
    if name == "e":''',
        1,
    )
    marker = "\ndef constant_rational("
    text = text.replace(marker, HELPER + marker, 1)
    text = text.replace(
        '"ln10": "ln2 * 10/3 corrected, via 2*atanh(9/11) + 2*ln2"',
        '"ln10": "2*atanh(3/7) + 2*ln2   (ln(5/2) + ln(4))"',
        1,
    )
    PATH.write_text(text, encoding="utf-8")
    print("series sized per argument")


if __name__ == "__main__":
    main()
