# ``c = b`` then a loop carries ``c`` and reads ``b``.  ``b``'s read of the
# preheader value was rebound to ``c``'s header Phi (the body added c + c);
# in an unrolled loop each copy threaded ``b``'s read as ``c``.
def root(k: int):
    b = k + 2
    c = b
    while c < 40:
        c = c + b
    total = b
    for i in range(k):
        total = total + b + i
    u = b
    for j in range(3):
        u = u * 2 + b
    return c, total, u
# INPUTS [{'k': 0}, {'k': 3}, {'k': 4}]
