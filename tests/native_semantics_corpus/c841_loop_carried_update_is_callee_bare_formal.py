def step(high, low, byte):
    return low ^ byte, high

def root(n):
    h = 1
    l = 2
    for i in range(n):
        h, l = step(h, l, i)
    return h + l * 10
# INPUTS [{'n': 5}]
