def root(n: bool) -> int:
    acc = n
    for i in range(40):
        acc = acc * 2 + i
    return acc
# INPUTS [{'n':True}, {'n':False}]
