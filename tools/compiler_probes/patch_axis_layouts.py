"""Express both buffer layouts as axis orders of the one stack."""
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "src/common/tensors/extended_precision.py"

OLD_INTERLEAVED = '''        return interleave(self.terms())

    def blocked(self):'''

NEW_INTERLEAVED = '''        # The limb axis moved to the END, then flattened. Both layouts
        # are the SAME STACK in two axis orders, which is the whole
        # argument for holding the limbs as a dimension rather than as a
        # tuple of arrays: a destination is handed the memory it wants by
        # permuting an axis and flattening, never by packing element by
        # element.
        stack = self._stack
        order = tuple(range(1, len(tuple(stack.shape)))) + (0,)
        return stack.transpose(*order).reshape(-1)

    def blocked(self):'''

OLD_BLOCKED = '''        from .abstraction import AbstractTensor

        return AbstractTensor.concat(
            [term.reshape(-1) for term in self.terms()], dim=0,
        )'''

NEW_BLOCKED = '''        # Already the stack's own axis order: limb zero's elements, then
        # limb one's, and so on. Flattening is the whole conversion.
        return self._stack.reshape(-1)'''


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "SAME STACK in two axis orders" in text:
        raise SystemExit("already axis-ordered; refusing to re-apply")
    for old in (OLD_INTERLEAVED, OLD_BLOCKED):
        if text.count(old) != 1:
            raise SystemExit(f"anchor appears {text.count(old)} times")
    text = text.replace(OLD_INTERLEAVED, NEW_INTERLEAVED, 1)
    text = text.replace(OLD_BLOCKED, NEW_BLOCKED, 1)
    PATH.write_text(text, encoding="utf-8")
    print("layouts are axis orders now")


if __name__ == "__main__":
    main()
