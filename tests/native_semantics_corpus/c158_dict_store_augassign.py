def root(n: int) -> int:
    d = {1: 10, 2: 20}
    d[3] = n
    d[1] += 5
    return d[1] * 1000 + d[2] * 10 + d[3] + len(d) * 100000

# INPUTS [{'n': 4}, {'n': 7}]
