# int.bit_length counts the bits of the magnitude: 0 for 0, and a negative
# formal answers like its absolute value.
def root(a: int, b: int) -> int:
    return a.bit_length() * 100 + b.bit_length()
# INPUTS [{'a': 0, 'b': 1}, {'a': 255, 'b': -256}, {'a': -1, 'b': 1099511627776}]
