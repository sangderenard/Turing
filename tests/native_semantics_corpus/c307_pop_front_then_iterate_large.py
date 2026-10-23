def root(n: int) -> int:
    xs = []
    for i in range(n):
        xs.append(i * 3)
        if i % 3 == 2:
            xs.pop(0)
    total = 0
    for x in xs:
        total = total * 1009 + x
    return total

# INPUTS [{'n': 5}, {'n': 9}]
