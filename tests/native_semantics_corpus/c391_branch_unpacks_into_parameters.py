def root(n: int, t: int, u: int) -> int:
    # Tuple unpacking and an elif rebind parameters no arm reads.
    if n > 5:
        t, u = n, n * 2
    elif n < -5:
        t = 0
    return t * 10 + u

# INPUTS [{'n': -9, 't': 7, 'u': 2}, {'n': -1, 't': 7, 'u': 2}, {'n': 8, 't': 7, 'u': 2}]
