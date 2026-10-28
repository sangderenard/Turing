# Token Orchard's fingerprint lanes, reduced: each limb is proven below
# 2**32 by a floor modulus inside a tuple-returning callee, then carried
# through two loops and a second callee before the combine
# high * (1 << 32) + low, a full u64 in Python. The interval has to cross
# the call returns, the formals and the loop-carried values, or the combine
# stays int64 and wraps negative natively whenever high >= 2**31 (n=0, n=5).
# The callees' tuple returns are unannotated: each unpacked member takes its
# class from the callee's returned value.
U32 = 4294967296


def limb_step(high: int, low: int, byte: int):
    mixed = low ^ byte
    product = mixed * 435
    next_low = product % U32
    carry = product // U32
    next_high = (high * 435 + carry + (mixed % 16777216) * 256) % U32
    return next_high, next_low


def atom_step(high: int, low: int, atom: int):
    remaining = atom
    index = 0
    while index < 4:
        high, low = limb_step(high, low, remaining % 256)
        remaining = remaining // 256
        index += 1
    return high, low


def root(n: int) -> int:
    high, low = limb_step(3421674724, 2216829733, 1)
    i = 0
    while i < n:
        high, low = atom_step(high, low, 65 + i)
        i += 1
    first = high * (1 << 32) + low
    if first < 0:
        return -1
    return first % 1000000007

# INPUTS [{'n': 0}, {'n': 1}, {'n': 2}, {'n': 3}, {'n': 5}]
