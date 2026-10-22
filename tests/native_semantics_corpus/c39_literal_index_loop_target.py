def root(n: int) -> int:
    xs = [5, 7, 11]
    total = 0
    for i in range(n):
        total += xs[i]
    return total
# INPUTS [{'n':0},{'n':1},{'n':3}]
