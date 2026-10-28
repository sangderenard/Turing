# bit_length of a loop-carried int feeds an accumulator.
def root(n: int) -> int:
    total = 0
    v = n
    while v > 0:
        total = total + v.bit_length()
        v = v // 3
    return total
# INPUTS [{'n': 0}, {'n': 1}, {'n': 1000}, {'n': 123456789}]
