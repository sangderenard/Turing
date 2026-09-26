"""Route a wide value's transcendentals to the cores that can hold it."""
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "src/common/tensors/extended_precision.py"

ANCHOR = '''    def sum(self) -> "Precision":'''

ADDITION = '''    # -- transcendentals: the cores, at this width -----------------------
    #
    # These are not arithmetic and cannot be built from it, so they are
    # not implemented here: they are ROUTED to the materialised proof
    # cores, which are the same programs the compiler lowers and which
    # measure their own error against an exact oracle. The width travels
    # with the argument, so a wide value gets a wide Horner chain rather
    # than a double's worth of answer widened afterwards.
    #
    # A core is valid on ITS OWN INTERVAL and carries no range reduction,
    # so an argument outside that interval is refused by name rather than
    # pushed through a polynomial that approximates nothing there. That is
    # the same rule the compiled kernels follow -- their router falls back
    # to the eager surface beyond the radius -- and it is deliberate: a
    # silently extrapolated core returns a plausible number, which is the
    # one outcome this type exists to prevent. Reduction belongs to the
    # caller, or to the signal surface that owns turns and binades.

    def _core(self, name: str) -> "Precision":
        from .signal_symbolic import CORE_RADII, evaluate_proof

        radius = CORE_RADII.get(name)
        if radius is None:
            raise AttributeError(
                f"no proof core is registered for {name!r}"
            )
        magnitude = abs(self).collapse()
        try:
            worst = max(
                abs(float(each))
                for each in _flatten(magnitude.tolist())
            )
        except (TypeError, ValueError):
            worst = float("inf")
        if worst > float(radius):
            raise ValueError(
                f"{name}: argument reaches {worst!r}, outside the core's "
                f"proven interval +-{radius}. Reduce the range first -- a "
                "core evaluated beyond its interval approximates nothing "
                "and would return a plausible wrong answer at every limb"
            )
        return evaluate_proof(name, self, self.limbs)

    def sin(self) -> "Precision":
        return self._core("sin")

    def cos(self) -> "Precision":
        return self._core("cos")

    def tan(self) -> "Precision":
        return self._core("tan")

    def exp(self) -> "Precision":
        return self._core("exp")

    def expm1(self) -> "Precision":
        return self._core("expm1")

    def log1p(self) -> "Precision":
        return self._core("log1p")

    def sinh(self) -> "Precision":
        return self._core("sinh")

    def cosh(self) -> "Precision":
        return self._core("cosh")

    def tanh(self) -> "Precision":
        return self._core("tanh")

    def atan(self) -> "Precision":
        return self._core("atan")

'''

HELPER = '''

def _flatten(nested):
    """Every leaf of a nested list, in order."""

    if isinstance(nested, list):
        for item in nested:
            yield from _flatten(item)
    else:
        yield nested

'''


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "transcendentals: the cores" in text:
        raise SystemExit("already routed; refusing to re-apply")
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"anchor appears {text.count(ANCHOR)} times")
    text = text.replace(ANCHOR, ADDITION + ANCHOR, 1)
    marker = "\ndef limbs_of("
    if text.count(marker) != 1:
        raise SystemExit("helper anchor not unique")
    text = text.replace(marker, HELPER + marker, 1)
    PATH.write_text(text, encoding="utf-8")
    print("transcendentals routed to the cores")


if __name__ == "__main__":
    main()
