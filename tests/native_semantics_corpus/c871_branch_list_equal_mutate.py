def root(n: int) -> int:
    if n > 2:
        x = [1, 2, 3]
    else:
        x = [4, 5, 6]
    x.append(n)
    x[1] = x[1] + 10
    return len(x) * 1000 + x[1] * 10 + x[3]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
