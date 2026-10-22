def root(n: int) -> int:
    t = 0
    for i in range(6):
        if i not in (1, 4) and n > i:
            t += i
    return t + (5 if n * 2 in {6, 8} else 0)

# INPUTS [{'n': 3}, {'n': 4}, {'n': 0}]
