def root(a: int, b: int) -> tuple[int, int, int, int, int]:
    # Overflow-checked int arithmetic whose exact results land on the int64
    # edges: every value here is Python's, none is wrapped or faulted.
    return a + b, a - b, a * b, -a, abs(a)

# INPUTS [{'a': 9223372036854775807, 'b': 0}, {'a': -9223372036854775807, 'b': -1}, {'a': 3037000499, 'b': 3037000499}, {'a': -4611686018427387904, 'b': 2}, {'a': 9223372036854775806, 'b': 1}, {'a': -9223372036854775807, 'b': 1}]
