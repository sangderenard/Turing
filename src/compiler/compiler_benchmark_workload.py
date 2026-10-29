"""Scalar-heavy authored workload for the ordinary project-product harness."""


def recurrence(seed: int, rounds: int) -> int:
    state = seed % 65521
    checksum = 0
    index = 0
    while index < rounds:
        state = (state * 25173 + 13849) % 65521
        checksum = (checksum + state) % 65521
        index += 1
    return checksum * 65521 + state


def adjacent(value: int) -> list[int]:
    return [value, value + 1]
