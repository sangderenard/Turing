def root(n: int) -> int:
    xs = [4, 5, 6]
    xs.append(7)
    return len(xs) * 1000 + (xs[n - 1] if n > 0 else 0)
# INPUTS [{'n':0},{'n':1},{'n':3},{'n':4}]
