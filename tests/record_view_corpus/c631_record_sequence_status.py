from array import array


class Pool:
    def __init__(self):
        self.atoms = array("I")

    def append_span(self, values: list[int], begin: int, end: int):
        i = begin
        while i < end:
            self.atoms.append(values[i])
            i += 1
        return len(self.atoms), 1

    def batch(self, values: list[int], offsets: list[int]):
        results = [0] * (len(offsets) - 1)
        i = 0
        while i < len(offsets) - 1:
            count, marker = self.append_span(values, offsets[i], offsets[i + 1])
            results[i] = count - marker
            i += 1
        return results


def root(values: list[int], offsets: list[int]):
    pool = Pool()
    return pool.batch(values, offsets)
