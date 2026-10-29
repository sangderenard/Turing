# set.update inserts each element of one iterable under the set's unique
# policy; members already present do not grow it.
def root(a: int, b: int) -> int:
    seen = {a, b}
    seen.update([b, a + b, a])
    return len(seen) * 10 + (a + b in seen)
# INPUTS [{'a': 1, 'b': 2}, {'a': 3, 'b': 3}, {'a': 0, 'b': 0}, {'a': 2, 'b': -2}]
