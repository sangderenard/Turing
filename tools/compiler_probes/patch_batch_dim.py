"""Hold the limbs as a leading BATCH DIMENSION rather than a tuple.

A tuple of limbs is planar but it is not a tensor, so a precision value has
no shape of its own and nothing can reason about it without unpacking. A
leading axis makes the width an ordinary dimension: shape[0] is the limb
count, shape[1:] is what the caller declared, and every structural
operation on the trailing axes means what it says.
"""
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "src/common/tensors/extended_precision.py"

OLD_INIT_TAIL = '''        self.limbs = int(limbs)
        if isinstance(value, (list, tuple)):
            self._terms = tuple(value)
        else:
            self._terms = tuple(
                limb(value, index, self.limbs) for index in range(self.limbs)
            )'''

NEW_INIT_TAIL = '''        from .abstraction import AbstractTensor

        self.limbs = int(limbs)
        if isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            parts = [
                limb(value, index, self.limbs) for index in range(self.limbs)
            ]
        # The limbs are a LEADING AXIS, not a tuple and not channels.
        #
        # A tuple is planar but it is not a tensor, so the value has no
        # shape of its own and nothing can reason about it without
        # unpacking first. Channels were worse: they made shape[-1] a lie,
        # so every operation that did not know about limbs computed
        # confidently on limbs*n elements. A leading axis is the only
        # arrangement where the width is an ordinary dimension -- shape[0]
        # is the limb count, shape[1:] is exactly what the caller declared
        # -- so widening a program does not disturb a single shape
        # assumption below it, and the trailing axes keep meaning what
        # they always meant.
        self._stack = AbstractTensor.stack(
            [AbstractTensor.get_tensor(part) for part in parts], dim=0,
        )'''

OLD_TERMS_ATTR = [
    ('''        total = self._terms[0]
        for term in self._terms[1:]:
            total = plain(total, "add", term)
        return total''',
     '''        total = self._stack[0]
        for index in range(1, self.limbs):
            total = plain(total, "add", self._stack[index])
        return total'''),
    ('''        return [term.tolist() for term in self._terms]''',
     '''        return [self._stack[index].tolist() for index in range(self.limbs)]'''),
    ('''        return self._terms[int(index)]''',
     '''        return self._stack[int(index)]'''),
    ('''            return list(self._terms)
        if width < self.limbs:
            return list(self._terms[:width])
        zero = plain(self._terms[0], "mul", 0.0)
        return list(self._terms) + [zero] * (width - self.limbs)''',
     '''            return [self._stack[index] for index in range(self.limbs)]
        if width < self.limbs:
            return [self._stack[index] for index in range(width)]
        held = [self._stack[index] for index in range(self.limbs)]
        zero = plain(held[0], "mul", 0.0)
        return held + [zero] * (width - self.limbs)'''),

    ('''        return AbstractTensor.concat(
            [term.reshape(-1) for term in self._terms], dim=0,
        )''',
     '''        return AbstractTensor.concat(
            [term.reshape(-1) for term in self.terms()], dim=0,
        )'''),
    ('''        return Precision(
            interleave([function(term) for term in self.terms()]), self.limbs
        )''',
     '''        return Precision(
            [function(term) for term in self.terms()], self.limbs
        )'''),
    ('''        return Precision(
            interleave([
                term.reshape(-1)[start:stop] for term in self.terms()
            ]),
            self.limbs,
        )''',
     '''        return Precision(
            [term.reshape(-1)[start:stop] for term in self.terms()],
            self.limbs,
        )'''),
]

SHAPE_PROPERTY = '''    @property
    def shape(self):
        """The shape the CALLER declared: the stack without its limb axis.

        This is the whole point of holding the limbs as a leading axis. A
        two-limb field of a hundred elements reports a hundred, not two
        hundred and not a tuple of two arrays, so every shape assumption
        written against the narrow program still holds when it is widened.
        """

        return tuple(self._stack.shape)[1:]

    @property
    def stacked(self):
        """The payload: limbs on axis zero, the declared shape after it."""

        return self._stack

'''


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "_stack" in text:
        raise SystemExit("already a batch dimension; refusing to re-apply")
    if text.count(OLD_INIT_TAIL) != 1:
        raise SystemExit("init anchor not unique")
    text = text.replace(OLD_INIT_TAIL, NEW_INIT_TAIL, 1)
    for old, new in OLD_TERMS_ATTR:
        if text.count(old) != 1:
            raise SystemExit(f"anchor appears {text.count(old)} times: {old[:48]!r}")
        text = text.replace(old, new, 1)
    # Both the named conversion and the compatibility view build the same
    # interleaved buffer, so both move off the tuple together.
    text = text.replace(
        "        return interleave(list(self._terms))",
        "        return interleave(self.terms())",
    )
    text = text.replace('__slots__ = ("_terms", "limbs")',
                        '__slots__ = ("_stack", "limbs")', 1)
    anchor = '''    @property
    def _value(self):'''
    if text.count(anchor) != 1:
        raise SystemExit("_value anchor not unique")
    text = text.replace(anchor, SHAPE_PROPERTY + anchor, 1)
    PATH.write_text(text, encoding="utf-8")
    print("limbs are a leading batch dimension")


if __name__ == "__main__":
    main()
