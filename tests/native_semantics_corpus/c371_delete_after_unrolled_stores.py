def root(n: int) -> int:
    e = {}
    for i in range(4):
        e[i] = i + n
    del e[8 % 4]
    f = {}
    for j in range(3):
        f[j] = j * n
    del f[n % 3]
    return len(e) * 10 + len(f)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
