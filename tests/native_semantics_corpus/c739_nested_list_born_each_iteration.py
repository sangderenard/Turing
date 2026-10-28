# The list of lists is born again on every pass of the loop.
def root(n: int) -> int:
    total = 0
    for k in range(n):
        g = [[k, 1], [2, k + 1]]
        total += g[k % 2][1] * 3 + g[1][k % 2]
    return total
# INPUTS [{'n': 3}, {'n': 5}]
