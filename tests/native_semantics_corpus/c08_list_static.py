def root(n: int) -> int:
    xs = []
    for i in range(8):
        xs.append((i * n) % 5)
    best = -1
    for x in xs:
        if x > best:
            best = x
    count = 0
    for x in xs:
        if x == best:
            count += 1
    return best * 10 + count
# INPUTS [{'n':1},{'n':3},{'n':0}]
