"""Build the math tables once and cache them in the repository.

    python -m src.compiler.build_math_cache

Sampling fifteen functions finely enough to hit 1e-6 takes long enough that
doing it on every build is a waste, and the result is deterministic: the
same epsilon always produces the same bytes. So they are built once, written
next to the compiler, and read back afterwards.

Each table is a flat little-endian f64 array -- the exact bytes a
WebAssembly data segment or a fetch() into linear memory wants -- with one
manifest describing every one of them: domain, interval count, whether the
argument wraps or clamps, the error bound it was sized to, and the error it
actually delivered when measured against the function it came from.

That manifest is what lets the HTML shell load these instead of carrying
them: a page that needs sin and cos fetches two files totalling 64 KB rather
than embedding them in its own source.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from .wasm_math_tables import (
    DEFAULT_EPSILON,
    TABULATED,
    build_series,
    build_table,
    measure_error,
    SERIES_CAPABLE,
)

CACHE_DIRECTORY = Path(__file__).resolve().parent / "math_cache"
MANIFEST = "manifest.json"


def build(directory: Path = CACHE_DIRECTORY,
          epsilon: float = DEFAULT_EPSILON,
          verify: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    total = 0
    for name in sorted(TABULATED):
        table = build_table(name, epsilon)
        payload = b"".join(struct.pack("<d", value) for value in table.values)
        (directory / f"{name}.f64").write_bytes(payload)
        entry = table.to_mapping()
        entry["file"] = f"{name}.f64"
        if verify:
            # The bound is a prediction. Recording what the table actually
            # delivers means a caller never has to take the prediction on
            # trust, and a regression in the sizing shows up here rather
            # than in someone's output.
            measured = measure_error(table, 20001)
            entry["measured"] = measured
            # Curvature is sampled, and a function whose second derivative
            # is singular at an endpoint (acosh at 1, atanh at +/-1) can
            # peak between samples -- so the prediction can be optimistic
            # there. The number a caller should rely on is whichever is
            # worse, and it is recorded rather than the flattering one.
            entry["achieved"] = max(table.bound, measured)
            entry["bound_met"] = measured <= table.bound
            if measured > table.bound:
                print(f"    ! {name}: measured {measured:.2e} exceeds the "
                      f"predicted {table.bound:.2e}; 'achieved' is authoritative")
        entries[name] = entry
        total += len(payload)
        print(f"  {name:7} {table.intervals:>8} intervals "
              f"{len(payload)/1024:>8.0f} KB  bound {table.bound:.2e}"
              + (f"  measured {entry['measured']:.2e}" if verify else ""))

    series = {}
    for name in sorted(SERIES_CAPABLE):
        expansion = build_series(name, epsilon)
        leading, coefficients = expansion.horner()
        series[name] = {
            "terms": len(coefficients),
            "leading_power": leading,
            "stride": 2 if name in ("sin", "cos", "atan") else 1,
            "coefficients": list(coefficients),
            "lower": expansion.lower,
            "upper": expansion.upper,
            "bound": expansion.bound,
        }

    manifest = {
        "schema": "turing-wasm-math-cache-v1",
        # Read "achieved" per table, not "bound": the bound is what the
        # sizing predicted, achieved is what the table was measured to do.
        "accuracy_field": "achieved",
        "epsilon": epsilon,
        "element": "f64-le",
        "tables": entries,
        "series": series,
        "total_bytes": total,
    }
    path = directory / MANIFEST
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(entries)} tables, {total/1024/1024:.1f} MB, "
          f"{len(series)} series -> {path}")
    return path


def load_manifest(directory: Path = CACHE_DIRECTORY) -> dict:
    path = directory / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"no math cache at {path}; run "
            "`python -m src.compiler.build_math_cache` to build it"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(name: str, directory: Path = CACHE_DIRECTORY) -> tuple[float, ...]:
    manifest = load_manifest(directory)
    entry = manifest["tables"].get(name)
    if entry is None:
        raise KeyError(f"{name} is not in the cache; {sorted(manifest['tables'])}")
    payload = (directory / entry["file"]).read_bytes()
    return struct.unpack(f"<{len(payload)//8}d", payload)


if __name__ == "__main__":
    build()
