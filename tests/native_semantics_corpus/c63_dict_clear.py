def root(n: int) -> int:
    cells = {1: 10, 2: 20}
    if n > 0:
        cells.clear()
    return len(cells)

# INPUTS [{'n': 0}, {'n': 1}]
