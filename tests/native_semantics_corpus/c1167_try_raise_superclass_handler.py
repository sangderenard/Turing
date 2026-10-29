# The authored ValueError reaches a broader Exception handler through the
# concorded builtin class-subtype edge, while the protected update remains
# live at handler entry and on the next loop iteration.
def root(n: int) -> int:
    x = 0
    for i in range(n):
        try:
            x = x + i
            raise ValueError("bad")
        except Exception:
            x = x + 2
    return x


# INPUTS [{'n': 0}, {'n': 1}, {'n': 3}]
