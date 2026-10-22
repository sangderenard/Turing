def root(n: int) -> int:
    xs = [i * 3 for i in range(n)]
    a = any(x > 10 for x in xs)
    b = all(x % 3 == 0 for x in xs)
    return int(a) * 10 + int(b)

# INPUTS [{'n': 2}, {'n': 6}]
