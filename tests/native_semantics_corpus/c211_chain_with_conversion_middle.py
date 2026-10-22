def root(x: float, n: int) -> int:
    r = 0
    if 0 <= int(x) <= 9:
        r += 1
    if n != 0 and 0 < int(100 / n) < 10:
        r += 10
    total = 0.0
    for i in range(n):
        if 1.5 < float(i) < x:
            total += i
    return r + int(total) * 100

# INPUTS [{'x': 5.5, 'n': 20}, {'x': 9.99, 'n': 0}, {'x': -0.5, 'n': 11}, {'x': 10.0, 'n': -4}, {'x': 3.0, 'n': 3}]
