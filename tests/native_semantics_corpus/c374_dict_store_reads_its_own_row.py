def root(n: int) -> int:
    d = {0: 1, 1: 2**55 + 1}
    d[0] += n
    d[0] = d[1] + d[0]
    e = {1: 10, 2: 20}
    for i in range(5):
        e[i % 2 + 1] += i * n
    return (d[0] - 2**55) * 1000 + e[1] * 10 + e[2]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
