def root(n: int) -> float:
    acc = n
    for i in range(200):
        acc = acc * 0.5 + i
    return acc
# INPUTS [{'n':3}]
