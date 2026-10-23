def root(b: int, flag: int) -> int:
    s = None if flag > 0 else b
    t = (s - 3) if s is not None else None
    if t is None:
        return 100
    return t * 2
# INPUTS [{'b':5,'flag':1},{'b':5,'flag':0},{'b':-4,'flag':0}]
