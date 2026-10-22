def root(n: int) -> int:
    j = -5
    for i in range(10):
        j = i
        if i * i > n:
            break
    return j

# INPUTS [{'n': 5}, {'n': 50}, {'n': 100}]
