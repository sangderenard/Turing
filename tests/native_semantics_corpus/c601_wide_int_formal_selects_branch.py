# Lane X2 (2026-10-24): silently wrong on the Wasm core before the fix. The
# int formals entered as i32, so a = 2**31 compared as -2**31 and a = 2**32 + 1
# as 1, and each row returned the other branch with no refusal.
def root(a: int, b: int, x: float) -> float:
    if a > b:
        return x
    return 0.0 - x
# INPUTS [{'a': 5, 'b': 3, 'x': 1.5}, {'a': 4294967297, 'b': 3, 'x': 1.5}, {'a': 2147483648, 'b': 3, 'x': 1.5}, {'a': -2147483649, 'b': 0, 'x': 1.5}]
