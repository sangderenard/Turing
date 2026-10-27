def root(n: int, end: int) -> int:
    values = []
    for index in range(n):
        values.append(index + 3)
    selected = values[:end]
    return selected[0] if len(selected) else 0

# INPUTS [{'n': 5, 'end': 3}, {'n': 0, 'end': 4}, {'n': 4, 'end': 0}]
