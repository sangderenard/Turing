def root(n: int) -> int:
    # A runtime member may equal a literal one: dedupe, truth and length
    # over it are runtime facts.
    s = {1, 2, n}
    xs = [0, n]
    return len(s) * 100 + int(any(xs)) * 10 + int(all(xs))

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}, {'n': 5}]
