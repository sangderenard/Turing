"""Derive tau, and stop taking the module constants from libm."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYMBOLIC = ROOT / "src/common/tensors/signal_symbolic.py"
MATH = ROOT / "src/common/tensors/signal_math.py"

OLD_TAU_ANCHOR = '''    if name == "ln2":'''
NEW_TAU = '''    if name == "tau":
        # A whole turn, derived rather than doubled from a rounded pi.
        # Multiplying by two is exact in binary, so at ONE limb the two
        # spellings agree -- but a limb decomposition of 2*pi is not the
        # doubling of pi's decomposition, because each limb is rounded
        # separately, and the reduction that consumes tau needs every one
        # of them right.
        return 2 * constant_rational("pi", digits)
    if name == "ln2":'''

OLD_CONSTANTS = '''TAU = 2.0 * math.pi'''

MATH_ANCHOR_LN = '''LN2 = math.log(2.0)
LN10 = math.log(10.0)'''

NEW_CONSTANTS = '''#: The turn, the natural logarithm of two, and of ten -- DERIVED, never
#: borrowed. ``math.log`` is the platform's libm, which is the one thing
#: this stack exists to replace; taking a core constant from it means the
#: replacement rests on the thing replaced. Each of these is now the
#: correctly rounded double of an exactly derived rational, and the same
#: derivation serves a wide caller at any width through
#: ``signal_symbolic.constant_limbs``.
def _derived(name: str) -> float:
    from .signal_symbolic import constant_rational

    return float(constant_rational(name, 40))


TAU = _derived("tau")'''

NEW_LN = '''LN2 = _derived("ln2")
LN10 = _derived("ln10")'''


def main() -> None:
    text = SYMBOLIC.read_text(encoding="utf-8")
    if '"tau"' in text:
        raise SystemExit("tau already derivable; refusing to re-apply")
    if text.count(OLD_TAU_ANCHOR) != 1:
        raise SystemExit("ln2 anchor not unique")
    SYMBOLIC.write_text(text.replace(OLD_TAU_ANCHOR, NEW_TAU, 1), encoding="utf-8")

    text = MATH.read_text(encoding="utf-8")
    if "_derived(" in text:
        raise SystemExit("constants already derived")
    for anchor in (OLD_CONSTANTS, MATH_ANCHOR_LN):
        if text.count(anchor) != 1:
            raise SystemExit(f"anchor not unique: {anchor[:30]!r}")
    text = text.replace(OLD_CONSTANTS, NEW_CONSTANTS, 1)
    text = text.replace(MATH_ANCHOR_LN, NEW_LN, 1)
    MATH.write_text(text, encoding="utf-8")
    print("tau derivable; module constants no longer from libm")


if __name__ == "__main__":
    main()
