# A list of constant lists is read by runtime row and column.
def root(n: int) -> int:
    g = [[1, 2], [3, 4]]
    i = n % 2
    j = (n + 1) % 2
    return g[i][j] * 10 + g[j][i]
# INPUTS [{'n': 3}, {'n': 4}]
