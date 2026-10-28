from array import array


class Vocabulary:
    def __init__(self):
        self.atoms = array("I")

    def inner(self, values: list[int], begin: int, end: int):
        i = begin
        while i < end:
            self.atoms.append(values[i])
            i += 1
        return len(self.atoms), 0

    def batch(self, values: list[int], offsets: list[int]):
        results = [0] * (len(offsets) - 1)
        i = 0
        while i < len(offsets) - 1:
            count, status = self.inner(values, offsets[i], offsets[i + 1])
            results[i] = count + status
            i += 1
        return results


def root(values: list[int], offsets: list[int]):
    vocabulary = Vocabulary()
    return vocabulary.batch(values, offsets)
