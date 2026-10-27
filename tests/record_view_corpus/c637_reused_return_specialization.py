DOMAIN_BYTES = 2
BYTE_ATOM_BASE = 1112067
TEXT_ATOM_BASE = 3
SURROGATE_END = 57344
SURROGATE_COUNT = 2048

class Pool:

    def __init__(self):
        self.epoch = 0

    def _atom_token(self, domain: int, atom: int) -> int:
        if domain == DOMAIN_BYTES:
            return BYTE_ATOM_BASE + atom
        rank = atom
        if atom >= SURROGATE_END:
            rank = atom - SURROGATE_COUNT
        return TEXT_ATOM_BASE + rank

    def first(self, domain: int, atom: int):
        return self._atom_token(domain, atom)

    def second(self, domain: int, atom: int):
        return self._atom_token(domain, atom)

def root(domain: int, atom: int):
    p = Pool()
    left = p.first(domain, atom)
    right = p.second(domain, atom)
    return left + right
