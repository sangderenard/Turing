def root(n: int) -> int:
    xs1 = [4, n]
    xs1[0] += n
    xs2 = [5, n, 0]
    xs2[1] = 7
    xs2[0] += n
    xs3 = [1, 0] + [n, 8]
    return xs1[0] + (xs2[0] + xs2[1]) * 100 + (len(xs3) * 10 + xs3[2]) * 10000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 5}]
