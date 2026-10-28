# Rows of floats keep float64 through the pool.
def root(n):
    g = [[1.5, n * 0.5], [2.25, 4.0]]
    i = n % 2
    return g[i][1] + g[1][i] * 2.0
# INPUTS [{'n': 3}, {'n': 4}]
