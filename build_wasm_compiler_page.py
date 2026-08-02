"""Publish Turing's direct WASM compiler into the workspace gallery."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.compiler.site_bundle import (
    DEFAULT_PUBLISH_ROOT,
    build_source_inspection_bundle,
)

TURING_ROOT = Path(__file__).resolve().parent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_PUBLISH_ROOT,
        help="published-site root (default: parent of the Turing repository)",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    bundle = build_source_inspection_bundle(
        TURING_ROOT / "src/compiler/wasm_binary.py",
        arguments.destination,
        title="Turing WebAssembly Compiler",
        slug="turing-webassembly-compiler",
    )
    print(bundle.page_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
