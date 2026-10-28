# A list of lists: each row is its own list; g[i][j] reads row i, column j.
def root(n):
    g = [[1, 2], [3, n]]
    i = n % 2
    return g[i][1] * 10 + g[i][0] + g[1][i]
# INPUTS [{'n': 3}, {'n': 4}]
