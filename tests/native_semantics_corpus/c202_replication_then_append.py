def root(n: int) -> int:
    ys = [7] * (n % 5)
    ys.append(n)
    return len(ys) * 10 + ys[0]

# INPUTS [{'n': 1}, {'n': 5}, {'n': 2}, {'n': 0}]
