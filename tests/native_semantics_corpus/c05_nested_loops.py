def root(n: int) -> int:
    total = 0
    for i in range(5):
        for j in range(4):
            if (i + j) % 2 == 0:
                total += i * j + n
            else:
                total -= j
    return total
# INPUTS [{'n':0},{'n':3}]
