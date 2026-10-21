def root(a: int) -> int:
    x = a
    x *= 3
    x -= 4
    x //= 2
    x %= 7
    x += a
    return x
# INPUTS [{'a':10},{'a':-5},{'a':0}]
