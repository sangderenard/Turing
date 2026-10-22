def root(n: int) -> int:
    xs0 = [6, n, 2]
    for j0 in range(len(xs0)):
        xs0[j0] = xs0[j0] * 2
    return xs0[0] + xs0[1] + xs0[2]

# INPUTS [{'n': 3}, {'n': 7}]
