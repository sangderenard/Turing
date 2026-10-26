# Lane E parity probe: Python's ``>>`` on a negative int is an arithmetic
# shift (it rounds toward minus infinity).  A logical shift in one backend
# returns a huge positive value instead.
def root(n: int) -> int:
    a = (n - 100) >> 2
    b = -(n * 3 + 1) >> (n % 3 + 1)
    return a * 1000 + b

# INPUTS [{'n': 0}, {'n': 5}, {'n': 50}]
