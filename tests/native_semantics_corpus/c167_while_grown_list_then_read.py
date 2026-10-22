def root(n: int) -> int:
    xs0 = []
    k0 = n % 5 + 2
    while k0 > 0:
        xs0.append(k0)
        k0 -= 2
    return len(xs0) * 10 + xs0[0]

# INPUTS [{'n': 0}, {'n': 6}]
