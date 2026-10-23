def root(n: int, flag: int) -> int:
    acc = 0
    for i in range(n):
        s = None if (i + flag) % 2 == 0 else i
        t = None if s is None else -s
        if t is not None:
            acc += t
    return acc
# INPUTS [{'n':5,'flag':1},{'n':6,'flag':0}]
