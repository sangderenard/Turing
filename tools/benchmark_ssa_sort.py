"""Compile and execute a small source-level resident-list sort benchmark."""

from __future__ import annotations

import argparse
import pathlib
import tempfile
import time
import warnings

import numpy as np

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


CONTRACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)

SOURCE = """
def sort_values(values: list[int]):
    values.sort()
    return values
"""

KEYED_SOURCE = """
def sort_values(values: list[int], positions: dict[int, int]):
    values.sort(key=positions.__getitem__)
    return values
"""

PROJECTION_SOURCE = """
def sort_values(values: list[tuple[int, int]]):
    values.sort(key=lambda row: row[1])
    return values
"""

SORTED_SOURCE = """
def sort_values(values: list[int]):
    return sorted(values)
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=20)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--keyed",
        action="store_true",
        help="sort through a resident dict.__getitem__ key mapping",
    )
    modes.add_argument(
        "--projection",
        action="store_true",
        help="sort fixed-width rows through lambda row: row[1]",
    )
    modes.add_argument(
        "--builtin-sorted",
        action="store_true",
        help="copy and sort through Python's value-producing sorted()",
    )
    arguments = parser.parse_args()
    if arguments.size < 0 or arguments.repetitions < 1:
        parser.error("size must be nonnegative and repetitions must be positive")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            (
                KEYED_SOURCE if arguments.keyed
                else PROJECTION_SOURCE if arguments.projection
                else SORTED_SOURCE if arguments.builtin_sorted
                else SOURCE
            ),
            "sort_values",
            name="ssa_sort_benchmark",
            extraction_contract=CONTRACT,
            progress=lambda _message: None,
            progress_bars=False,
        )
    entrypoint = "ssa_sort_benchmark__sort_values"
    function = module.functions[entrypoint]
    parameters = dict(function.metadata["parameter_names"])
    sequence_table = module.sequence_tables[entrypoint]
    artifact = emit_ssa_function_to_llvm(module, entrypoint)
    if artifact.shortfalls:
        raise RuntimeError(f"sort benchmark emission shortfalls: {artifact.shortfalls!r}")
    native = compile_artifact(
        artifact,
        directory=pathlib.Path(tempfile.mkdtemp()) / "ssa_sort_benchmark",
    )

    rng = np.random.default_rng(20261014)
    if arguments.projection:
        first = rng.permutation(arguments.size).astype(np.int64)
        second = rng.integers(
            -(1 << 30), 1 << 30, size=arguments.size, dtype=np.int64,
        )
        order = np.argsort(second, kind="stable")
        original = first
        expected = (first[order], second[order])
        mapping_keys = mapping_values = None
    elif arguments.keyed:
        original = rng.permutation(arguments.size).astype(np.int64)
        mapping_keys = original.copy()
        mapping_values = rng.permutation(arguments.size).astype(np.int64)
        expected = original[np.argsort(mapping_values, kind="stable")]
    else:
        original = rng.integers(
            -(1 << 30), 1 << 30, size=arguments.size, dtype=np.int64,
        )
        mapping_keys = mapping_values = None
        expected = np.sort(original, kind="stable")
    elapsed = 0.0
    produced = None
    for _ in range(arguments.repetitions):
        source_buffer = original.copy()
        feed = {parameters["values"]: source_buffer}
        for descriptor in sequence_table.sequences.values():
            is_values = int(descriptor.sequence_id) == int(parameters["values"])
            is_positions = (
                arguments.keyed
                and int(descriptor.sequence_id) == int(parameters["positions"])
            )
            for column_index, column in enumerate(descriptor.column_value_ids):
                authored_column = (
                    (
                        original if column_index == 0 else second
                    ) if is_values and arguments.projection
                    else original if is_values
                    else mapping_keys if is_positions and column_index == 0
                    else mapping_values if is_positions and column_index == 1
                    else None
                )
                feed.setdefault(
                    int(column),
                    authored_column.copy() if authored_column is not None else np.zeros(
                        arguments.size, dtype=np.int64,
                    ),
                )
            feed[int(descriptor.length_address_id)] = np.array([
                arguments.size if is_values or is_positions else 0
            ], dtype=np.int64)
            feed[int(descriptor.capacity_value_id)] = np.array([
                arguments.size
            ], dtype=np.int64)
            if descriptor.status_address_id is not None:
                feed[int(descriptor.status_address_id)] = np.array(
                    [0], dtype=np.int32,
                )
        for formal in function.args:
            identifier = int(formal.id)
            if identifier not in feed:
                feed[identifier] = (
                    np.array([0], dtype=np.int64)
                    if formal.shape == ()
                    else np.zeros(max(1, arguments.size), dtype=np.int64)
                )
        execution = prepare_artifact_execution(native, feed)
        started = time.perf_counter()
        execution.run()
        elapsed += time.perf_counter() - started
        if arguments.projection:
            values_descriptor = sequence_table.by_id(parameters["values"])
            produced = tuple(
                np.asarray(execution.buffers[int(column)]).copy()
                for column in values_descriptor.column_value_ids
            )
        else:
            output_id = int(_outputs[entrypoint][0].id)
            produced = np.asarray(execution.buffers[output_id]).copy()

        if arguments.builtin_sorted and not np.array_equal(
            np.asarray(execution.buffers[parameters["values"]]), original,
        ):
            raise AssertionError("compiled sorted() mutated its input sequence")

    agrees = (
        all(np.array_equal(actual, wanted) for actual, wanted in zip(produced, expected))
        if arguments.projection else np.array_equal(produced, expected)
    )
    if not agrees:
        def buffer(value_id: int):
            value = execution.buffers.get(int(value_id))
            return None if value is None else np.asarray(value).copy()

        states = tuple(
            (
                int(descriptor.sequence_id),
                buffer(descriptor.length_address_id),
                buffer(descriptor.capacity_value_id),
                None if descriptor.status_address_id is None else buffer(
                    descriptor.status_address_id
                ),
                tuple(
                    buffer(column)
                    for column in descriptor.column_value_ids
                ),
            )
            for descriptor in sequence_table.sequences.values()
        )
        raise AssertionError(
            f"compiled sort disagrees with Python/NumPy stable order: "
            f"produced={produced!r}, expected={expected!r}, states={states!r}"
        )
    print(
        f"sorted {arguments.size} int64 values correctly"
        f"{' through resident mapping keys' if arguments.keyed else ''}; "
        f"{'two-column rows by projection; ' if arguments.projection else ''}"
        f"mean native time={elapsed / arguments.repetitions:.6f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
