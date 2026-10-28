from array import array


class Vocabulary:
    def __init__(self):
        self.exact_atoms = array("I")

    def put(self, values: list[int], begin: int, end: int):
        for i in range(begin, end):
            self.exact_atoms.append(values[i])
        return len(self.exact_atoms)


def root(values: list[int], begin: int, end: int):
    vocabulary = Vocabulary()
    return vocabulary.put(values, begin, end)
