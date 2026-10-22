def fill(dst: list[int], v: int) -> None:
    if v == 0:
        dst.append(7)
        return
    dst.append(8)
    dst.append(9)


def root(v: int) -> int:
    xs = []
    fill(xs, v)
    return len(xs) * 100 + xs[0]
# INPUTS [{'v':0},{'v':5}]
