# Rows of different lengths share one stride; each reads its own members.
def root(n):
    g = [[1, 2, 3], [4], [n, 5]]
    i = n % 3
    return g[i][0] * 10 + g[2][1] + g[0][i]
# INPUTS [{'n': 3}, {'n': 5}]
