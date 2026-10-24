# Lane E (E22a): a runtime-count replication grows on the heap, so ``sum``
# reads its pointer cell.  The cell's (1,) shape was read as one row and C
# added the arena address to the sum.
def root(n: int) -> int:
    ys = [5] * (n % 4)
    ys.append(n)
    ys.append(36028797018963969)
    return sum(ys) + ys[-1] % 1000 + len(ys)

# INPUTS [{'n': 0}, {'n': 3}, {'n': -6}]
