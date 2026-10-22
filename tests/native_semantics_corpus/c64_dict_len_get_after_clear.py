def root(n: int) -> int:
    cells = {1: 10, 2: 20, 3: 30}
    before = len(cells) * 1000 + cells.get(2, 9) * 10
    if n > 1:
        cells.clear()
    return before + len(cells) * 100 + cells.get(2, 7)

# INPUTS [{'n': 0}, {'n': 1}, {'n': 2}]
