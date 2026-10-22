def root(n: int) -> int:
    xs = [4, 5, 6]
    xs.append(7)
    y = xs[n - 1]
    return len(xs) * 1000 + y
# INPUTS [{'n':1},{'n':3},{'n':4}]
