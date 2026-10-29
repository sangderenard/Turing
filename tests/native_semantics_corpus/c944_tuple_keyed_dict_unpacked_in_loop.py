# A dict keyed by tuples, filled in a loop, then iterated with a tuple
# target (bootstrap: ``for row, column in state["cells"]``).  The tuple
# target's names had no rule over an unannotated mapping, so ``row * 10``
# reached its region with no dtype.
def root(n: int) -> int:
    cells = {}
    for i in range(n):
        cells[(i % 2, i)] = i
    total = 0
    for row, column in cells:
        total += row * 10 + column
    return total
# INPUTS [{'n': 0}, {'n': 1}, {'n': 4}]
