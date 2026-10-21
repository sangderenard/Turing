def root(n: int) -> int:
    xs = []
    for i in range(6):
        xs.append(i * 2 + n)
    total = 0
    for x in xs:
        total += x
    if 4 in xs:
        total += 100
    return total + len(xs)
# INPUTS [{'n':0},{'n':1},{'n':3}]
