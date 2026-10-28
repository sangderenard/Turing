def span_of(lo: int, hi: int) -> tuple[int, int, int] | None:
    if hi < lo:
        return None
    return (lo, hi, hi - lo)


def root(lo: int, hi: int, use: int) -> int:
    s = span_of(lo, hi) if use > 0 else None
    if s is None:
        return -1
    first, last, width = s
    return first * 100 + last * 10 + width
# INPUTS [{'lo':1,'hi':4,'use':1},{'lo':5,'hi':2,'use':1},{'lo':1,'hi':4,'use':0},{'lo':-2,'hi':3,'use':1}]
