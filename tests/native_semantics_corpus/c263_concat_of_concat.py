def root(n: int) -> int:
    xs = [n, 1]
    xs.append(n + 1)
    ys = [7, 8] + xs
    zs = ys + [n]
    ws = [n] + [1, 2] + [n * 2]
    return len(zs) * 1000 + zs[4] * 100 + len(ws) * 10 + ws[3]
# INPUTS [{'n':0},{'n':3},{'n':-2}]
