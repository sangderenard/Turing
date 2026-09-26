"""Give limb layout a name, a conversion, and a per-lane declaration."""
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "src/common/tensors/extended_precision.py"

ANCHOR = '''    @property
    def _value(self):'''

ADDITION = '''    # -- layout: a boundary fact, declared per destination ---------------
    #
    # There is no universally best arrangement of limbs in a buffer, and
    # the measurements say so plainly. A compiled per-element kernel wants
    # ELEMENT-MAJOR ("interleaved"): element i's limbs land on one cache
    # line, and on the CPU lanes that measured 1.2x to 1.7x faster than
    # the alternative. A GPU wants LIMB-MAJOR ("blocked"): adjacent
    # invocations then read adjacent addresses instead of addresses a
    # stride apart, which is what coalescing rewards. The eager surface
    # wants neither, because it holds its limbs planar and never packs
    # them at all.
    #
    # So the layout is not a property of the value. It is a property of
    # the DESTINATION, chosen where the value crosses into one, and the
    # only thing this type owes is an exact conversion in both directions.

    def interleaved(self):
        """Element-major: element i, limb k at flat index ``i * limbs + k``.

        What every compiled kernel in this tree addresses today, and what
        the artifacts and feeds already on disk contain.
        """

        return interleave(list(self._terms))

    def blocked(self):
        """Limb-major: element i, limb k at flat index ``k * count + i``.

        Each limb contiguous. Offered because a dispatch-parallel
        destination reads it better, not because anything here prefers it.
        """

        from .abstraction import AbstractTensor

        return AbstractTensor.concat(
            [term.reshape(-1) for term in self._terms], dim=0,
        )

    @classmethod
    def from_interleaved(cls, value, limbs: int) -> "Precision":
        """Adopt an element-major buffer without copying its meaning."""

        return cls(value, int(limbs))

    @classmethod
    def from_blocked(cls, value, limbs: int) -> "Precision":
        """Adopt a limb-major buffer, splitting it back into terms."""

        from .abstraction import AbstractTensor

        flat = AbstractTensor.get_tensor(value).reshape(-1)
        width = max(int(limbs), 1)
        count = int(flat.shape[0]) // width
        return cls(
            [flat[position * count:(position + 1) * count]
             for position in range(width)],
            width,
        )

'''


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "layout: a boundary fact" in text:
        raise SystemExit("layouts already declared; refusing to re-apply")
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"anchor appears {text.count(ANCHOR)} times")
    text = text.replace(ANCHOR, ADDITION + ANCHOR, 1)
    PATH.write_text(text, encoding="utf-8")
    print("layout vocabulary added")


if __name__ == "__main__":
    main()
