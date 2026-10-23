def root(n: int) -> int:
    ys = [1, n]
    ys.extend([1, 4])
    zs = [3, -2, 4]
    ws = [z - n for z in zs]
    return ys[3] * 10 + sum(ws) + len(ys)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 5}]
