def root(a: int, b: int, flag: bool) -> int:
    c = (a > 0 and b > 0) or flag
    d = not (a == b)
    out = 0
    if c and d:
        out = 1
    elif c:
        out = 2
    elif d:
        out = 3
    return out
# INPUTS [{'a':1,'b':2,'flag':False},{'a':1,'b':1,'flag':False},{'a':-1,'b':2,'flag':False},{'a':-1,'b':-1,'flag':False},{'a':-1,'b':-1,'flag':True}]
