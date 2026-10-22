def root(n: int) -> int:
    xs1 = [1, 9, n]
    xs1.reverse()
    xs0 = [6, n, 2]
    for j0 in range(len(xs0)):
        xs0[j0] = xs0[j0] * 2
    return (xs0[0] + xs0[1] + xs0[2]) + (xs1[0] - xs1[2])

# INPUTS [{'n': 3}, {'n': 7}]
