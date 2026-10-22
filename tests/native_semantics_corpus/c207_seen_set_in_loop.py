def root(n: int) -> int:
    seen = {n % 3}
    total = 0
    for i in range(6):
        if i not in seen:
            seen.add(i)
            total += i
    return total * 10 + len(seen)

# INPUTS [{'n': 1}, {'n': 5}, {'n': 3}]
