# Lane E (Fortran list growth): sorting a list grown at run time uses
# scratch arenas with their own owners and capacity cells.
def root(n: int) -> int:
    xs = []
    k = n % 4 + 3
    while k > 0:
        xs.append((k * 7 + n) % 10)
        k -= 1
    ys = sorted(xs)
    return ys[0] * 1000 + ys[1] * 100 + ys[-1] * 10 + len(ys)

# INPUTS [{'n': 0}, {'n': 5}, {'n': 2}]
