def root(n: int) -> int:
    total = 0
    i = 0
    while True:
        i += 1
        if i > n:
            break
        if i % 3 == 0:
            continue
        total += i
    return total
# INPUTS [{'n':10},{'n':0},{'n':3}]
