def root(n: int) -> int:
    i = -5
    for i in range(10):
        if i * i > n:
            break
    return i

# INPUTS [{'n': 5}, {'n': 50}, {'n': 100}]
