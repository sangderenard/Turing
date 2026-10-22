def root(n: int) -> int:
    xs = [i * i for i in range(n) if i % 2 == 1]
    return len(xs) * 1000 + sum(xs)

# INPUTS [{'n': 1}, {'n': 6}]
