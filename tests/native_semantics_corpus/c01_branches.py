def root(a: int, b: int) -> int:
    if a > b:
        r = a - b
    elif a == b:
        r = 0
    else:
        r = (b - a) * 2
    return r + 1
# INPUTS [{'a':5,'b':3},{'a':3,'b':3},{'a':1,'b':4}]
