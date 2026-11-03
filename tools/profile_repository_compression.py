"""Print the repository tile codec's compression/latency frontier.

The default rows have the repeated scope, stage and fact shapes carried by
concordance pages.  Pickle baselines use the same tile boundaries so the
comparison does not give a monolithic stream an artificial ratio advantage.
The zlib/LZMA rows are measurements only; repository storage continues to use
the backend-independent AbstractTensor codec.
"""

from __future__ import annotations

import argparse
import lzma
import pickle
from pathlib import Path
import statistics
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.compressed_tiles import CompressedTable


def _rows(count: int) -> tuple:
    return tuple(
        (
            ("source-evidence", index),
            (
                "stable repeated concordance fact " * 12,
                index,
                ("source", index % 17),
                ("stage", "type-evidence"),
            ),
        )
        for index in range(count)
    )


def _chunks(rows: tuple, tile_rows: int) -> tuple[tuple, ...]:
    return tuple(
        rows[start:start + tile_rows]
        for start in range(0, len(rows), tile_rows)
    )


def _median_run(repeats: int, operation):
    elapsed = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        elapsed.append(time.perf_counter() - started)
    return result, statistics.median(elapsed) * 1000.0


def _baseline(
    label: str, chunks: tuple[tuple, ...], repeats: int, encode, decode,
) -> tuple[str, int, int, float, float]:
    blobs, encode_ms = _median_run(
        repeats,
        lambda: tuple(encode(pickle.dumps(chunk, protocol=5)) for chunk in chunks),
    )
    restored, decode_ms = _median_run(
        repeats,
        lambda: tuple(pickle.loads(decode(blob)) for blob in blobs),
    )
    if restored != chunks:
        raise AssertionError(f"{label} did not round-trip its rows")
    return label, sum(map(len, blobs)), 0, encode_ms, decode_ms


def profile(
    *, row_count: int, tile_rows: int, levels: tuple[int, ...], repeats: int,
) -> tuple[tuple[str, int, int, float, float], ...]:
    rows = _rows(row_count)
    chunks = _chunks(rows, tile_rows)
    records = [
        _baseline("pickle", chunks, repeats, bytes, bytes),
        _baseline("pickle+zlib-1", chunks, repeats,
                  lambda data: zlib.compress(data, 1), zlib.decompress),
        _baseline("pickle+zlib-9", chunks, repeats,
                  lambda data: zlib.compress(data, 9), zlib.decompress),
        _baseline("pickle+lzma-6", chunks, repeats,
                  lambda data: lzma.compress(data, preset=6), lzma.decompress),
    ]

    # Pay one-time tensor/backend imports before measuring a codec level.
    warm = CompressedTable(rows[:min(8, len(rows))], tile_rows=8, level=1)
    tuple(warm.items())
    for level in levels:
        table, encode_ms = _median_run(
            repeats,
            lambda level=level: CompressedTable(
                rows, tile_rows=tile_rows, level=level,
            ),
        )
        restored, decode_ms = _median_run(repeats, lambda: tuple(table.items()))
        if restored != rows:
            raise AssertionError(f"repository level {level} did not round-trip")
        stats = table.stats()
        records.append((
            f"repository-{level}", stats.payload_bytes, stats.resident_bytes,
            encode_ms, decode_ms,
        ))
    return tuple(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--levels", default="0,1,3,6,9")
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args(argv)
    if arguments.rows <= 0 or arguments.tile_rows <= 0 or arguments.repeats <= 0:
        parser.error("rows, tile-rows and repeats must be positive")
    try:
        levels = tuple(int(item) for item in arguments.levels.split(","))
    except ValueError:
        parser.error("levels must be comma-separated integers in [0, 9]")
    if not levels or any(level < 0 or level > 9 for level in levels):
        parser.error("levels must be comma-separated integers in [0, 9]")

    records = profile(
        row_count=arguments.rows,
        tile_rows=arguments.tile_rows,
        levels=levels,
        repeats=arguments.repeats,
    )
    pickle_bytes = records[0][1]
    print(
        f"repository compression profile: rows={arguments.rows} "
        f"tile_rows={arguments.tile_rows} repeats={arguments.repeats}"
    )
    print(
        f"{'codec':<20} {'payload':>10} {'resident':>10} "
        f"{'pickle/x':>9} {'encode ms':>11} {'scan ms':>10}"
    )
    for label, payload, resident, encode_ms, decode_ms in records:
        print(
            f"{label:<20} {payload:>10,d} "
            f"{(f'{resident:,}' if resident else '-'):>10} "
            f"{pickle_bytes / max(1, payload):>9.3f} "
            f"{encode_ms:>11.3f} {decode_ms:>10.3f}"
        )
    print(
        "payload excludes the repository random-access index; resident includes "
        "that index, tile headers, codebook and payloads"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
