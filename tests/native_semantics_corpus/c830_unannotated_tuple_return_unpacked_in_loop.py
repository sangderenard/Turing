# An unannotated callee returns a tuple of ints (each reduced by a modulus);
# its caller unpacks the tuple in a loop and returns the carried members as a
# tuple again. Each unpacked member is the callee's returned int, never a
# float default.
M = 4294967296


def step(high: int, low: int, byte: int):
    mixed = low ^ byte
    product = mixed * 435
    return (high * 435 + product // M) % M, product % M


def run(high: int, low: int, atom: int):
    remaining = atom
    index = 0
    while index < 4:
        high, low = step(high, low, remaining % 256)
        remaining = remaining // 256
        index += 1
    return high, low


def root(n: int) -> int:
    x, y = step(3421674724, 2216829733, 1)
    i = 0
    while i < n:
        x, y = run(x, y, 65 + i)
        i += 1
    return (x * 1000003 + y) % 1000000007

# INPUTS [{'n': 0}, {'n': 1}, {'n': 2}, {'n': 5}]
