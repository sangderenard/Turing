def root(n: int) -> int:
    xs = []
    i = 0
    while i < n:
        xs.append(i * 3)
        i += 1
    last = 0
    if n > 0:
        last = xs[n - 1]
    return len(xs) * 1000 + last
# INPUTS [{'n':0},{'n':1},{'n':7},{'n':100}]
