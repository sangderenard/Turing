from .bitbitindex import BitBitIndex
from .bitbititem import BitBitItem

class BitBitSlice(BitBitItem):
    """
    Immutable view on an *aligned* bit‑range.
    """
    __slots__ = ("reversed",)

    def __init__(self, buffer, start_bit, length, reversed=False, plane='mask'):
        self._plane = plane
        stride = buffer.bitsforbits
        padded = buffer.intceil(length, stride)
        padding = padded - length
        super().__init__(buffer, start_bit, padded, cast=bytearray, padding=padding, reversed=reversed)
        self.reversed = reversed

    @property
    def plane(self):
        """Which plane are we targeting?"""
        return self._plane

    def __repr__(self):
        spec = BitBitIndex(self, slice(None), mode='repr')
        return self.buffer.indexer.access(spec)
    def __iter__(self):
        """Yield bits honoring this slice's reversed flag."""
        if getattr(self, "reversed", False):
            for i in range(self.useful_length - 1, -1, -1):
                yield int(BitBitItem.__getitem__(self, i))
        else:
            for i in range(self.useful_length):
                yield int(BitBitItem.__getitem__(self, i))

    def __reversed__(self):
        """Explicit reversed() support, independent of the .reversed flag."""
        for i in range(self.useful_length - 1, -1, -1):
            yield int(BitBitItem.__getitem__(self, i))
