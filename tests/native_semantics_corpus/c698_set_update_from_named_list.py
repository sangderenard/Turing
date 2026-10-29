# set.update from a list held in a name, after the list grew by append.
def root(a: int, b: int) -> int:
    seen = {a}
    more = [b]
    more.append(a + b)
    seen.update(more)
    return len(seen) * 10 + (b in seen)
# INPUTS [{'a': 1, 'b': 2}, {'a': 3, 'b': 3}, {'a': 0, 'b': 0}, {'a': 2, 'b': -2}]
