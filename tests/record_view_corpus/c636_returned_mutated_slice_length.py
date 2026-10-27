from array import array

class Pool:

    def __init__(self):
        self.counts = array('I')

    def cover(self, values: list[int], begin: int, end: int):
        n = end - begin
        result = [0] * n
        i = 0
        while i < n:
            result[i] = values[begin + i]
            i += 1
        return result[:i]

    def put(self, values: list[int], begin: int, end: int):
        cover = self.cover(values, begin, end)
        self.counts.append(len(cover))
        return len(self.counts)

def root(values: list[int], begin: int, end: int):
    p = Pool()
    return p.put(values, begin, end)
