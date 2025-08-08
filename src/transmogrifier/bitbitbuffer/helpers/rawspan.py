class _RawSpan(bytearray):
    """A zero‑copy view (slice) tied to a BitBitBuffer plane."""
    __slots__ = ("_bitbit_cap", "_origin", "_offset")
    def __init__(self, backing, start_bit=0, length_bits=None, readonly=False):
        pass
    def __new__(cls, backing, offset, length):
        # backing: original bytearray; offset, length in bits
        view = super().__new__(cls,
            backing[offset // 8 : (offset + length + 7) // 8]
        )
        view._origin = backing
        view._offset = offset
        view._bitbit_cap = True

        return view

    @property
    def plane(self):
        """Which plane are we targeting?"""
        # default to mask; override in caller wrappers if needed
        return self._plane
