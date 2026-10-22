def root(n: int) -> int:
    total = 0
    for x in range(6):
        if x * 2 > n:
            total += x
        else:
            total -= 1
    return total

# INPUTS [{'n': 0}, {'n': 4}, {'n': 9}]
