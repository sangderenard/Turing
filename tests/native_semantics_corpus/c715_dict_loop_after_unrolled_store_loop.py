# A loop over a dict that an earlier unrolled loop stored into iterates the
# stored dict, not the removed node the first loop left behind.
def root(n: int) -> int:
    d = {1: n, 2: 5, 3: 7}
    for i in range(3):
        d[1] = i + n
    first = 0
    for k in d:
        first = k
        break
    return first
# INPUTS [{'n': 0}]
