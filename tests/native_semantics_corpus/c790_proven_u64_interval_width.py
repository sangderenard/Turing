# Token Orchard's fingerprint combine: high and low are each proven below
# 2**32 by a floor modulus, so high * (1 << 32) + low is a full u64. Python
# never wraps it. Before the interval record every int operator result was
# int64 by class alone, so C and LLVM wrapped the combine negative whenever
# high >= 2**31 with no finding (n=1 and n=3 here), and C's floored modulus
# then narrowed its int128 operand through long long.
def root(n: int):
    high = (n * 2654435761 + 12345) % 4294967296
    low = (n * 40503 + 777) % 4294967296
    first = high * (1 << 32) + low
    if first < 0 or first > 18446744073709551615:
        return -1
    return first % 1000000007

# INPUTS [{'n': 0}, {'n': 1}, {'n': 3}, {'n': 7}]
