def root(n: int) -> int:
    xs = []
    for i in range(n):
        xs.append(i * i)
    total = 0
    for x in xs:
        total += x
    return total + len(xs)
# INPUTS [{'n':0},{'n':1},{'n':5},{'n':40}]
