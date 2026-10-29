# An annotated empty dict filled in a loop, then iterated
# (bootstrap: identity_spill._reindex_in_place's ``row_columns``).  Its data
# arena was refined to the row dtype in the formals while the store calls
# read the earlier value of the same id, so the unused-formal pruning
# dropped the arena; and the loop's versions of the dict each minted a
# table of their own beside it.
def root(n: int) -> int:
    cells: dict[int, int] = {}
    for i in range(n):
        cells[i * 3] = i
    total = 0
    for key in cells:
        total += key * 10 + cells[key]
    return total
# INPUTS [{'n': 0}, {'n': 1}, {'n': 3}]
