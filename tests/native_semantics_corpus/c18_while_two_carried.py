def root(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a
# INPUTS [{'a':48,'b':18},{'a':17,'b':5},{'a':0,'b':7}]
