def root(n: int) -> int:
    everything = [0, 2, 3]
    count = 3
    position = 0
    index = 0
    total = 0
    for k in range(n):
        while index < count and everything[index] == position:
            position += 1
            index += 1
        total += position * 10 + k
        position += 1
    return total
# INPUTS [{'n':0},{'n':1},{'n':5}]
