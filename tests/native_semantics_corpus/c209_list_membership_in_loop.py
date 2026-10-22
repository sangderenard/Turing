def root(n: int) -> int:
    ts = [2, 3]
    ts.append(9)
    c = 0
    for i in range(n % 6 + 2):
        if i in ts:
            c += 10
    return c + len(ts)

# INPUTS [{'n': 1}, {'n': 5}, {'n': 3}]
