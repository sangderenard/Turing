# The one-caller form of c900: ``trim([1, 2, 3, 4])`` alone, where the
# catalogue fold (not a callsite copy) made the formal a literal.
def trim(values: list[int]) -> int:
    values.pop()
    total = 0
    for v in values:
        total += v
    return total


def root(rounds: int):
    return trim([1, 2, 3, 4]) + rounds

# INPUTS [{'rounds': 2}, {'rounds': 0}]
