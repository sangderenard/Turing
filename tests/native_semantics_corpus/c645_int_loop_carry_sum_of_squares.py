# An int loop-carried scalar read after the loop keeps int64 across the
# carry cast; a negative result must not become a float64 bit pattern.
def root(n: int) -> int:
    s = 0
    k = n
    for i in range(5):
        s = s * 3 + k - i
        k = k - 7
    return s - 99999
# INPUTS [{'n': 0}, {'n': -11}, {'n': 6}]
