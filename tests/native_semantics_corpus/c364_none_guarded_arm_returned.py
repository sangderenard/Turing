def bump(b: int, flag: int):
    s = None if flag > 0 else b
    return None if s is None else s * 2 + 1

def root(b: int, flag: int) -> int:
    v = bump(b, flag)
    if v is None:
        return -1
    return v
# INPUTS [{'b':5,'flag':1},{'b':5,'flag':0},{'b':-4,'flag':0}]
