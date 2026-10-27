def root(n: int) -> int:
    values = [0, 0, 0, 0]
    index = 0
    while index < 4:
        values[index] = index + n
        index += 1
    sliced = values[1:3]
    return sliced[0] * 10 + sliced[1]

# INPUTS [{'n': 4}, {'n': -1}, {'n': 0}]
