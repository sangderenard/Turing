def root(n: int) -> int:
    everything = [5, 7, 11]
    index = 0
    total = 0
    while index < n:
        total += everything[index]
        index += 1
    return total
# INPUTS [{'n':0},{'n':1},{'n':3}]
