def root(n: int) -> int:
    i = 0
    while i < 8:
        if i * 3 >= n:
            break
        i += 1
    return i

# INPUTS [{'n': 0}, {'n': 9}, {'n': 10}, {'n': 100}]
