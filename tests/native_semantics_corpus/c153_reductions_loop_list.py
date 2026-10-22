def root(n: int) -> int:
    xs = []
    for i in range(n):
        xs.append((i * 7) % 11)
    return max(xs) * 10000 + min(xs) * 100 + sum(xs)

# INPUTS [{'n': 1}, {'n': 5}, {'n': 9}]
