def root(n: int) -> int:
    xs = [n, 3, 8]
    ys = [x * 2 for x in xs if x > 2]
    return len(ys) * 100 + sum(ys)

# INPUTS [{'n': 1}, {'n': 5}]
