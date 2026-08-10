"""Compile one Python file into a standard versioned inspection-page bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument("--no-final-fused-reduction", action="store_true")
    return parser.parse_args()


def _print_progress(record) -> None:
    """Write straight to the real terminal, not whatever ``sys.stdout``
    currently is. The compiler wraps its own noisy phases in
    ``contextlib.redirect_stdout`` so third-party library prints don't spam
    the terminal; a ``print()`` call made from inside a telemetry
    subscriber that fires during one of those phases would obey that same
    redirect and silently vanish into the capture buffer instead of
    reaching anyone watching. ``sys.__stdout__`` is the one handle that
    redirect never touches."""

    detail = record.to_mapping() if hasattr(record, "to_mapping") else dict(record)
    path = f" [{detail['path']}]" if detail.get("path") else ""
    nanoseconds = detail.get("detail", {}).get("nanoseconds")
    duration = f" ({nanoseconds / 1e6:.1f}ms)" if nanoseconds else ""
    stream = sys.__stdout__ or sys.stdout
    stream.write(f"{detail['kind']:>8}{path} {detail['message']}{duration}\n")
    stream.flush()


def _announce(message: str) -> None:
    """Raw, unconditional visibility for moments before any TelemetryChannel
    exists yet (argument parsing, reading the source file, the CLI's own
    pre-check parse). Same real-stdout rule as ``_print_progress``."""

    stream = sys.__stdout__ or sys.stdout
    stream.write(f"    open {message}\n")
    stream.flush()


def _infer_python_package(source: Path) -> str | None:
    package_parts = []
    directory = source.resolve().parent
    while (directory / "__init__.py").is_file():
        package_parts.append(directory.name)
        directory = directory.parent
    if not package_parts:
        return None
    return ".".join(reversed(package_parts))


def main() -> int:
    _announce("build_site_page.py starting")
    arguments = _arguments()
    _announce(f"arguments parsed: --source {arguments.source}")
    python_package = _infer_python_package(arguments.source)
    if python_package is not None:
        _announce(f"inferred Python package: {python_package}")
    probes = json.loads(arguments.probes_json)
    if not isinstance(probes, dict):
        raise SystemExit("--probes-json must decode to an object")
    _announce(f"reading source file {arguments.source}")
    source = arguments.source.read_text(encoding="utf-8")
    _announce(f"source read: {len(source)} bytes; running CLI-side pre-check discover_source_contract")
    try:
        discover_source_contract(
            source,
            entrypoint=arguments.entrypoint or None,
            title=arguments.title or None,
            slug=arguments.slug or None,
            probes=probes,
            progress=_announce,
        )
    except ValueError as error:
        if "no public top-level function" not in str(error):
            raise
        _announce("no public top-level function; calling build_source_inspection_bundle")
        bundle = build_source_inspection_bundle(
            arguments.source,
            arguments.destination,
            title=arguments.title or None,
            slug=arguments.slug or None,
        )
    else:
        _announce("pre-check passed; calling build_program_bundle")
        bundle = build_program_bundle(
            source,
            arguments.destination,
            source_filename=arguments.source.name,
            python_package=python_package,
            entrypoint=arguments.entrypoint or None,
            title=arguments.title or None,
            slug=arguments.slug or None,
            probes=probes,
            include_backends=not arguments.no_backends,
            include_mathematics=not arguments.no_mathematics,
            final_fused_reduction=(
                False if arguments.no_final_fused_reduction else None
            ),
            progress_sink=_print_progress,
        )
    _announce(f"build_program_bundle returned: {bundle.page_path}")
    result = {
        "ok": True,
        "bundle": str(bundle.manifest_path),
        "page": str(bundle.page_path),
        "url": bundle.url,
        "manifest": bundle.manifest,
    }
    encoded = json.dumps(result)
    if arguments.result_json:
        _announce(f"writing result json to {arguments.result_json}")
        arguments.result_json.write_text(encoded, encoding="utf-8")
    _announce("done")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
