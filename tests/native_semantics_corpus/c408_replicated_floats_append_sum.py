# Lane E (E22a): the float case of c407; C refused to cast the arena
# pointer to double.
def root(n: int) -> float:
    ys = [0.5] * (n % 4)
    ys.append(n * 1.25)
    ys.append(2.0)
    return len(ys) * 100.0 + sum(ys) + ys[0]

# INPUTS [{'n': 0}, {'n': 1}, {'n': 3}, {'n': 6}]
