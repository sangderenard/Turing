def root(n: int) -> int:
    i = 0
    found = -1
    broke = False
    while i < 8:
        if i * 3 == n:
            found = i
            broke = True
            break
        i += 1
    if not broke:
        found = 100
    return found

# INPUTS [{'n': 0}, {'n': 9}, {'n': 10}]
