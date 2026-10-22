def root(n: int) -> int:
    xs = [1, 4, 9]
    a = any([x > n for x in xs])
    ys = [x > n for x in xs]
    return int(a) * 100 + int(all(ys)) * 10 + int(any(ys))

# INPUTS [{'n': 0}, {'n': 5}, {'n': 8}, {'n': 9}, {'n': 20}]
