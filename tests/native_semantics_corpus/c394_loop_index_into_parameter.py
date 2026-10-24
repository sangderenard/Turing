def root(n: int, t: int) -> int:
    # Both families together: an unread parameter rebound to the loop index
    # inside a conditional arm.
    for i in range(n):
        if i % 3 == 1:
            t = i
    return t * 10 + n

# INPUTS [{'n': 0, 't': 7}, {'n': 2, 't': 7}, {'n': 6, 't': 7}]
