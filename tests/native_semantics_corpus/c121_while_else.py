def root(n: int) -> int:
    i = 0
    found = -1
    while i < 8:
        if i * 3 == n:
            found = i
            break
        i += 1
    else:
        found = 100
    return found

# INPUTS [{'n': 0}, {'n': 9}, {'n': 10}]
