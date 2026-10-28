# A repeated display member extended inside a loop.
def root(n: int, v: int) -> int:
    xs = [v]
    for i in range(n):
        xs.extend([i, v, v])
    total = 0
    for x in xs:
        total += x
    return len(xs) * 1000 + total
# INPUTS [{'n': 0, 'v': 2}, {'n': 3, 'v': 5}]
