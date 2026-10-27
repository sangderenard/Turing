"""An authored consumer of the separately verified recurrence library."""
from src.compiler.compiler_benchmark_workload import recurrence


def leaf(seed: int, rounds: int) -> int:
    return recurrence(seed, rounds)


def linked_checksum(seed: int, rounds: int) -> int:
    return leaf(seed, rounds) + 1
