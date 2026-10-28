"""Scalar entry importing a private Counter through normal source pursuit."""
from src.compiler.compiler_counter_benchmark_workload import Counter


def counter_checksum(rounds: int) -> int:
    counter = Counter()
    for index in range(rounds):
        counter.add(index)
    return counter.read()
