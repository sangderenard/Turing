from array import array


class Vocabulary:
    def __init__(self):
        self.exact_begin = array("Q")

    def inspect(self):
        return len(self.exact_begin)

    def relay(self):
        return self.inspect()


def root():
    vocabulary = Vocabulary()
    return vocabulary.relay()
