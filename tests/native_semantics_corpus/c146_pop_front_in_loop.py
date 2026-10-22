def root(n: int) -> int:
    xs = []
    for i in range(n):
        xs.append(i)
        if i % 2 == 1:
            xs.pop(0)
    total = 0
    for x in xs:
        total = total * 10 + x
    return total

# INPUTS [{'n': 4}, {'n': 7}]
