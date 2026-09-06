"""Rebuild every standard-object deployment matrix with live status output."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from src.compiler.standard_object_blas import blas_standard_object
from src.compiler.standard_object_linalg import linalg_standard_object
from src.compiler.standard_object_product import cook_standard_object
from src.compiler.standard_object_trigonometry import trigonometry_standard_object


SCHEMA = "turing.standard-object-rebuild.v1"


def _sizes(text: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(
        int(item.strip()) for item in str(text).split(",") if item.strip()
    ))
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated integers")
    return values


def _prepare_root(path: Path) -> Path:
    root = path.resolve()
    marker = root / ".turing-standard-object-rebuild"
    if root.exists() and any(root.iterdir()):
        if not marker.is_file() or marker.read_text(encoding="ascii") != SCHEMA:
            raise RuntimeError(
                f"refusing to replace unowned rebuild directory: {root}"
            )
        for child in root.iterdir():
            if child == marker:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(SCHEMA, encoding="ascii")
    return root


def _status(started: float):
    def emit(event: Mapping[str, Any]) -> None:
        elapsed = time.perf_counter() - started
        stage = str(event["stage"])
        object_name = str(event.get("object", "-"))
        method = str(event.get("method", "-"))
        parameters = event.get("parameters")
        suffix = "" if parameters is None else " " + json.dumps(
            parameters, sort_keys=True, separators=(",", ":"),
        )
        print(
            f"[{elapsed:9.2f}s] {object_name:16s} {method:12s} {stage}{suffix}",
            flush=True,
        )
    return emit


def _linalg_domains(sizes: Sequence[int]) -> dict[str, dict[str, Sequence[int]]]:
    values = tuple(map(int, sizes))
    return {
        "eye": {"n": values},
        "dot": {"length": values},
        "norm": {"length": values},
        "trace": {"n": values},
        "det": {"n": values},
        "solve": {"n": values},
        "inv": {"n": values},
        "eigh": {"n": values},
        "cholesky": {"n": values},
    }


def _linalg_baselines(sizes: Sequence[int]) -> dict[str, dict[str, int]]:
    first = int(tuple(sizes)[0])
    return {
        "eye": {"n": first},
        "dot": {"length": first},
        "norm": {"length": first},
        "trace": {"n": first},
        "det": {"n": first},
        "solve": {"n": first},
        "inv": {"n": first},
        "eigh": {"n": first},
        "cholesky": {"n": first},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("build/standard-object-matrices"),
    )
    parser.add_argument("--contract", default="fast")
    parser.add_argument("--sizes", type=_sizes, default=(2, 3, 4))
    parser.add_argument(
        "--wasm-reverses", action="store_true",
        help="also compile every baseline reverse to WebAssembly",
    )
    args = parser.parse_args(argv)

    root = _prepare_root(args.output)
    started = time.perf_counter()
    status = _status(started)
    reverse_backends = ("native", "wasm") if args.wasm_reverses else ("native",)
    jobs = (
        ("blas", blas_standard_object()),
        ("trigonometry", trigonometry_standard_object()),
        (
            "linalg",
            linalg_standard_object(
                parameter_domains=_linalg_domains(args.sizes),
                baseline_parameters=_linalg_baselines(args.sizes),
            ),
        ),
    )
    records = {}
    print(
        f"Rebuild started {datetime.now().isoformat(timespec='seconds')} "
        f"at {root}",
        flush=True,
    )
    for name, spec in jobs:
        product = cook_standard_object(
            spec,
            directory=root / name,
            contract=args.contract,
            reverse_backends=reverse_backends,
            progress=status,
        )
        records[name] = {
            "path": name,
            "product_id": product.manifest["product_id"],
            "deployment_rows": len(product.manifest["deployment_matrix"]),
        }
    manifest = {
        "schema": SCHEMA,
        "contract": args.contract,
        "sizes": list(args.sizes),
        "reverse_backends": list(reverse_backends),
        "products": records,
    }
    temporary = root / "rebuild-manifest.json.tmp"
    destination = root / "rebuild-manifest.json"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8", newline="\n",
    )
    temporary.replace(destination)
    print(
        f"[{time.perf_counter() - started:9.2f}s] ALL COMPLETE {destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
