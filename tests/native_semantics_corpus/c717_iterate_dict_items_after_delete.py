def root(n: int) -> int:
    # items() over a table with a deleted key visits only the live rows.
    d = {1: n, 2: 20, 3: 30, 4: 40}
    del d[1]
    del d[3]
    total = 0
    for k, v in d.items():
        total = total * 100 + k * 10 + v % 10
    return total + len(d)

# INPUTS [{'n': 5}, {'n': 8}]
