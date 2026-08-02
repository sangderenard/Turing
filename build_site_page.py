"""Compile one Python file into a standard versioned inspection-page bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.compiler.site_bundle import (  # noqa: E402
    DEFAULT_PUBLISH_ROOT,
    build_program_bundle,
    build_source_inspection_bundle,
    discover_source_contract,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_PUBLISH_ROOT,
        help="published-site root (default: parent of the Turing repository)",
    )
    parser.add_argument("--entrypoint")
    parser.add_argument("--title")
    parser.add_argument("--slug")
    parser.add_argument("--probes-json", default="{}")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--no-backends", action="store_true")
    parser.add_argument("--no-mathematics", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    probes = json.loads(arguments.probes_json)
    if not isinstance(probes, dict):
        raise SystemExit("--probes-json must decode to an object")
    source = arguments.source.read_text(encoding="utf-8")
    try:
        discover_source_contract(
            source,
            entrypoint=arguments.entrypoint or None,
            title=arguments.title or None,
            slug=arguments.slug or None,
            probes=probes,
        )
    except ValueError as error:
        if "no public top-level function" not in str(error):
            raise
        bundle = build_source_inspection_bundle(
            arguments.source,
            arguments.destination,
            title=arguments.title or None,
            slug=arguments.slug or None,
        )
    else:
        bundle = build_program_bundle(
            source,
            arguments.destination,
            source_filename=arguments.source.name,
            entrypoint=arguments.entrypoint or None,
            title=arguments.title or None,
            slug=arguments.slug or None,
            probes=probes,
            include_backends=not arguments.no_backends,
            include_mathematics=not arguments.no_mathematics,
        )
    result = {
        "ok": True,
        "bundle": str(bundle.manifest_path),
        "page": str(bundle.page_path),
        "url": bundle.url,
        "manifest": bundle.manifest,
    }
    encoded = json.dumps(result)
    if arguments.result_json:
        arguments.result_json.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
