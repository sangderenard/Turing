def lcm(self, cells):
    from math import gcd
    from functools import reduce
    def lcm(a, b):
        return a * b // gcd(a, b)
    return reduce(lcm, (cell.stride for cell in cells if hasattr(cell, 'stride')), 1)
