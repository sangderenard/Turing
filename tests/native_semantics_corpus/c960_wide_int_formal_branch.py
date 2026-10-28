# Lane X2 (2026-10-24): the independent Wasm core typed Python int formals as
# i32, so a = 2**31 or 2**32 + 1 wrapped on entry, compared below b, and
# returned the other branch's value with no refusal. Python int is int64:
# every lane must carry these values whole, mix them into float arithmetic
# by the C conversions, and return an int result as int.
def root(a: int, b: int, x: float) -> float:
    if a > b:
        return x + float(a - b)
    return x - a / b
# INPUTS [{'a': 5, 'b': 3, 'x': 1.5}, {'a': 2147483648, 'b': 3, 'x': 1.5}, {'a': 4294967297, 'b': 3, 'x': 1.5}, {'a': -4294967297, 'b': 2147483647, 'x': 0.25}, {'a': 3, 'b': 3, 'x': 2.0}]
