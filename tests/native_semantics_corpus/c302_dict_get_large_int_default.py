def root(n: int) -> int:
    cells = {1: 10, 2: 20}
    if n > 1:
        cells[n] = n * 3
    return cells.get(5, 36028797018963969) + cells.get(2, 7) + len(cells)

# INPUTS [{'n': 0}, {'n': 2}, {'n': 5}]
