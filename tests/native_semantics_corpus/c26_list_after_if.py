def root(n: int) -> int:
    xs = []
    for i in range(5):
        xs.append(i * 2)
    total = 0
    for x in xs:
        total += x
    if 4 in xs:
        total += n
    return total
# INPUTS [{'n':1},{'n':7}]
