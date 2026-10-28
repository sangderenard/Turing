# A display that names one value twice extends by both occurrences.
def root(a: int, b: int) -> int:
    seen = [a]
    seen.extend([a, b, b])
    return len(seen) * 10 + seen[3]
# INPUTS [{'a': 1, 'b': 2}, {'a': 3, 'b': 3}, {'a': -4, 'b': 0}]
