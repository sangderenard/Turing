def counted(n: int) -> int:
    d = {}
    for i in range(n):
        d[i % 2] = d.get(i % 2, 0) + 5
    return d[0] + len(d)


def seeded(n: int) -> int:
    d = {0: 1}
    for i in range(n):
        d[i % 3] = d.get(i % 3, 2) * 2
    return d[0] * 100 + d.get(2, 9) * 10 + len(d)


def stepped(n: int) -> int:
    d = {}
    d[0] = n
    d[1] = d.get(0, 0) + 5
    return d[1] * 10 + len(d)


def root(n: int) -> int:
    return counted(n) + seeded(n) * 1000 + stepped(n) * 10000000

# INPUTS [{'n': 3}, {'n': 1}, {'n': 4}, {'n': 7}]
