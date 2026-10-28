def root(n: int) -> int:
    # Each pass stores a key and deletes it: live keys never exceed the
    # initial one, so the table never fills however long the loop runs.
    d = {2: 5}
    for i in range(n):
        d[1] = i
        del d[1]
    return d[2] + len(d)

# INPUTS [{'n': 0}, {'n': 1}, {'n': 4}]
