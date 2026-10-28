def step(high, low, byte):
    return (high + byte) % 1000, (low ^ byte) % 1000

def inner(high, low, atom):
    index = 0
    while index < 4:
        high, low = step(high, low, index + atom)
        index += 1
    return high, low

def root(domain, n):
    first_high, first_low = step(3, 5, domain)
    index = 0
    while index < n:
        first_high, first_low = inner(first_high, first_low, index + 1000)
        index += 1
    return first_high, first_low
# INPUTS [{'domain': 1, 'n': 3}]
