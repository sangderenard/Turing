def root(x: float, n: int) -> int:
    xs = [3, 1, 4]
    for i in range(n):
        xs.append(i)
    r = 0
    if 2 < len(xs) <= 5:
        r += 1
    if 0.5 < abs(x) < 7.0:
        r += 10
    if n > 0 and 1 <= abs(int(x / n)) < 3:
        r += 100
    return r

# INPUTS [{'x': 5.5, 'n': 2}, {'x': -6.9, 'n': 0}, {'x': 0.25, 'n': 3}, {'x': -12.0, 'n': 5}, {'x': 7.0, 'n': 1}]
