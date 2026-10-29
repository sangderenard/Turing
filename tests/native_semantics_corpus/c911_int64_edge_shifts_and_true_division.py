def root(a: int, b: int) -> tuple[int, int, int, float]:
    # ``<<`` is exact up to the sign bit and 0 << n is 0 for any n; ``>>``
    # floors, so a count past 63 leaves 0 or -1.  ``a / b`` of ints that
    # are exactly float64 is the one correctly rounded division.
    return a << b, a >> b, (-1 - a) >> (b + 100), a / (b + 1)

# INPUTS [{'a': 1, 'b': 62}, {'a': -1, 'b': 63}, {'a': 0, 'b': 200}, {'a': -8, 'b': 1}, {'a': -9223372036854775808, 'b': 0}, {'a': 9007199254740992, 'b': 2}, {'a': -4611686018427387904, 'b': 1}, {'a': 3, 'b': 61}]
