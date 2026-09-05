"""Compile one or many Python sources into published-site bundles, in one pass.

Source selection (``--source``) works one of three ways:

  --source path/to/file.py
      Compile that one file. With no ``--targets``, its single entrypoint is
      auto-detected the same way ``build_site_page.py`` already does.

  --source path/to/directory
      Compile every ``*.py`` file directly inside that directory (not its
      subdirectories).

  --source path/to/directory --recursive
      Compile every ``*.py`` file anywhere under that directory tree.

``--targets`` (optional) restricts which top-level functions are compiled as
entrypoints, per file -- e.g. ``--targets kernel,presentation`` builds one
bundle per name, for every file that defines it. A file missing a named
target is skipped for that name, not an error. Only top-level functions are
ever candidates; class methods are never picked up as entrypoints -- that
exclusion isn't something this script adds, it's how the AST scan already
works (a method lives inside a ``ClassDef`` body, never in the module's own
top-level statement list). Omit ``--targets`` to auto-detect the one
implicit entrypoint per file, same as today.

``--backends`` (optional) restricts published source tabs to specific
languages (e.g. ``--backends glsl,webgpu``) instead of every backend
``collect_backend_sources`` can serve. See ``build_program_bundle``'s
``backend_targets`` parameter.

``--probes-json`` takes a JSON object of feed values. If its top-level keys
match discovered entrypoint names, it's treated as per-entrypoint
(``{"kernel": {"x": [1, 2, 3]}}``); otherwise the whole object is applied as
one flat probes dict to every entrypoint compiled this run.

Every bundle is published under ``--destination`` (default: the parent
workspace root, i.e. the actual site the local server and GitHub Pages
serve -- never Turing's own repository; see PUBLISHING_BUNDLES_TO_ROOT.md at
the workspace root for the two-repository layout this defaults against).

``--full`` regenerates instead of building fresh: it ignores
``--source``/``--targets``/``--backends``/``--probes-json``, walks every
program's ``site/programs/<slug>/origin.json`` under ``--destination``
(one per program, shared across however many versions it already has, not
one per version), and recompiles each from its embedded source text plus
the exact entrypoint/probes/backend_targets it was built with
(``build_program_bundle`` writes/refreshes this record on every build,
specifically so this doesn't depend on the original external source file
still existing at its original path). Programs published before this
record existed have no ``origin.json`` and are skipped with a note, not
silently rebuilt with guessed defaults. Each regenerated program is
published as one additional version (never overwriting an existing one) --
use this to pick up a compiler change across everything already live
without re-typing every original build command.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.compiler.site_bundle import (  # noqa: E402
    DEFAULT_PUBLISH_ROOT,
    TURING_REPOSITORY_ROOT,
    _TextInspectionSubject,
    build_program_bundle,
    build_source_inspection_bundle,
    discover_source_contract,
    resolve_publish_root,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path,
        help="a .py file, or a directory of them; required unless --full",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="when --source is a directory, scan it recursively",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="ignore --source/--targets/--backends/--probes-json; instead "
        "find every bundle already published under --destination and "
        "recompile each one from its own stored origin record (the "
        "published source copy plus the entrypoint/probes/backend_targets "
        "it was built with) -- the way to pick up a compiler change across "
        "everything already live without re-supplying each build's "
        "original arguments by hand",
    )
    parser.add_argument(
        "--targets",
        help="comma-separated top-level function names to compile as "
        "entrypoints; omit to auto-detect one implicit entrypoint per file",
    )
    parser.add_argument(
        "--backends",
        help="comma-separated backend languages to publish "
        "(e.g. glsl,webgpu); omit to publish every backend",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_PUBLISH_ROOT,
        help="published-site root (default: parent of the Turing repository)",
    )
    parser.add_argument("--probes-json", default="{}")
    parser.add_argument("--no-mathematics", action="store_true")
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def _announce(message: str) -> None:
    """Raw, unconditional visibility -- see build_site_page.py for why this
    writes to ``sys.__stdout__`` rather than a plain ``print()``."""

    stream = sys.__stdout__ or sys.stdout
    stream.write(f"    open {message}\n")
    stream.flush()


def _print_progress(record) -> None:
    detail = record.to_mapping() if hasattr(record, "to_mapping") else dict(record)
    path = f" [{detail['path']}]" if detail.get("path") else ""
    nanoseconds = detail.get("detail", {}).get("nanoseconds")
    duration = f" ({nanoseconds / 1e6:.1f}ms)" if nanoseconds else ""
    stream = sys.__stdout__ or sys.stdout
    stream.write(f"{detail['kind']:>8}{path} {detail['message']}{duration}\n")
    stream.flush()


def _discover_files(source: Path, *, recursive: bool) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise SystemExit(f"--source {source} is neither a file nor a directory")
    pattern = "**/*.py" if recursive else "*.py"
    return sorted(path for path in source.glob(pattern) if path.is_file())


def _discover_program_origins(destination: Path) -> list[Path]:
    """One ``origin.json`` per program (``site/programs/<slug>/origin.json``),
    not one per version -- a program's origin is shared across every version
    it has on disk, so --full walks these, not each version's bundle.json."""

    programs_root = destination / "site" / "programs"
    if not programs_root.is_dir():
        return []
    return sorted(programs_root.glob("*/origin.json"))


def _public_top_level_functions(source_text: str) -> list[str]:
    """Names ``discover_source_contract`` would accept as an entrypoint.

    Deliberately mirrors that function's own filter (module.body-level
    ``FunctionDef`` nodes, non-underscore names) rather than re-deriving a
    different notion of "public function" -- a name this misses or adds
    would silently disagree with what the compiler actually does. Class
    methods never appear here because they live inside a ``ClassDef``'s
    body, never in ``module.body`` itself.
    """

    module = ast.parse(source_text)
    return [
        node.name for node in module.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def _probes_for(entrypoint: str | None, probes: dict) -> dict:
    if entrypoint is not None and entrypoint in probes and isinstance(probes[entrypoint], dict):
        return probes[entrypoint]
    return probes


def _build_one(
    source_text: str, source_filename: str, entrypoint: str | None, *,
    original_source: str, destination: Path,
    backend_targets: tuple[str, ...] | None, probes: dict,
    include_backends: bool, include_mathematics: bool,
    title: str | None = None, slug: str | None = None,
    force_new_version: bool = False,
) -> dict:
    try:
        discover_source_contract(
            source_text, entrypoint=entrypoint, title=title, slug=slug,
            probes=probes, progress=_announce,
        )
    except ValueError as error:
        if "no public top-level function" not in str(error):
            raise
        _announce(f"{source_filename}: no public top-level function; inspection bundle only")
        bundle = build_source_inspection_bundle(
            _TextInspectionSubject(
                source_text, source_filename,
                Path(source_filename).stem, "file",
            ),
            destination, title=title, slug=slug,
        )
    else:
        _announce(
            f"{source_filename}: compiling entrypoint "
            f"{entrypoint or '(auto-detected)'}"
        )
        bundle = build_program_bundle(
            source_text,
            destination,
            source_filename=source_filename,
            entrypoint=entrypoint,
            title=title,
            slug=slug,
            probes=probes,
            backend_targets=backend_targets,
            include_backends=include_backends,
            include_mathematics=include_mathematics,
            progress_sink=_print_progress,
            force_new_version=force_new_version,
        )
    return {
        "ok": True,
        "source": original_source,
        "entrypoint": entrypoint,
        "bundle": str(bundle.manifest_path),
        "page": str(bundle.page_path),
        "url": bundle.url,
        "directory": str(bundle.directory),
    }


def _full_regeneration(destination: Path) -> tuple[list[dict], list[dict]]:
    """Rebuild every program that has a program-level origin record.

    Each program (one ``site/programs/<slug>/origin.json``, shared across
    however many versions it already has) is recompiled from its embedded
    source text -- self-contained, so this never depends on an external
    file still existing -- and published as one additional version
    (``force_new_version=True``), never overwriting or reusing an existing
    version directory.
    """

    origin_files = _discover_program_origins(destination)
    _announce(f"{len(origin_files)} program origin record(s) found under {destination}")

    results: list[dict] = []
    errors: list[dict] = []
    for origin_json in origin_files:
        program_dir = origin_json.parent
        try:
            origin = json.loads(origin_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append({"source": str(program_dir), "error": f"{type(error).__name__}: {error}"})
            continue

        source_text = origin.get("source")
        if not source_text:
            errors.append({
                "source": str(program_dir),
                "error": "origin.json has no embedded source text",
            })
            continue

        source_filename = origin.get("source_filename") or "program.py"
        backend_targets = (
            tuple(origin["backend_targets"]) if origin.get("backend_targets") else None
        )
        try:
            results.append(_build_one(
                source_text, source_filename, origin.get("entrypoint"),
                original_source=str(program_dir),
                destination=destination,
                backend_targets=backend_targets,
                probes=origin.get("probes") or {},
                include_backends=bool(origin.get("include_backends", True)),
                include_mathematics=bool(origin.get("include_mathematics", True)),
                title=origin.get("title"), slug=origin.get("slug"),
                force_new_version=True,
            ))
        except Exception as error:  # one failure does not sink the batch
            errors.append({
                "source": str(program_dir), "entrypoint": origin.get("entrypoint"),
                "error": f"{type(error).__name__}: {error}",
            })
            _announce(f"{program_dir} regeneration failed: {error}")
    return results, errors


def _fresh_source_build(
    source: Path, *, recursive: bool, requested_targets: list[str] | None,
    backend_targets: tuple[str, ...] | None, probes: dict,
    include_mathematics: bool, destination: Path,
) -> tuple[list[dict], list[dict]]:
    files = _discover_files(source, recursive=recursive)
    _announce(f"{len(files)} source file(s) discovered under {source}")

    results: list[dict] = []
    errors: list[dict] = []
    for file in files:
        source_text = file.read_text(encoding="utf-8")
        try:
            available = _public_top_level_functions(source_text)
        except SyntaxError as error:
            errors.append({"source": str(file), "error": f"SyntaxError: {error}"})
            continue

        if requested_targets is None:
            entrypoints: list[str | None] = [None] if available else [None]
        else:
            entrypoints = [name for name in requested_targets if name in available]
            if not entrypoints:
                _announce(
                    f"{file.name}: none of {requested_targets} are top-level "
                    "functions here; skipped"
                )
                continue

        for entrypoint in entrypoints:
            try:
                results.append(_build_one(
                    source_text, file.name, entrypoint,
                    original_source=str(file),
                    destination=destination,
                    backend_targets=backend_targets,
                    probes=_probes_for(entrypoint, probes),
                    include_backends=True,
                    include_mathematics=include_mathematics,
                ))
            except Exception as error:  # one failure does not sink the batch
                errors.append({
                    "source": str(file), "entrypoint": entrypoint,
                    "error": f"{type(error).__name__}: {error}",
                })
                _announce(f"{file.name} [{entrypoint}] failed: {error}")
    return results, errors


def main() -> int:
    _announce("publish_bundles.py starting")
    arguments = _arguments()

    destination = resolve_publish_root(arguments.destination)
    _announce(f"destination resolved: {destination}")

    if arguments.full:
        if arguments.source or arguments.targets or arguments.backends or arguments.probes_json != "{}":
            _announce(
                "--full given; --source/--targets/--backends/--probes-json "
                "are ignored, each bundle's own origin record is used instead"
            )
        results, errors = _full_regeneration(destination)
    else:
        if arguments.source is None:
            raise SystemExit("--source is required unless --full is given")
        probes = json.loads(arguments.probes_json)
        if not isinstance(probes, dict):
            raise SystemExit("--probes-json must decode to an object")
        backend_targets = (
            tuple(item.strip() for item in arguments.backends.split(",") if item.strip())
            if arguments.backends else None
        )
        requested_targets = (
            [item.strip() for item in arguments.targets.split(",") if item.strip()]
            if arguments.targets else None
        )
        results, errors = _fresh_source_build(
            arguments.source, recursive=arguments.recursive,
            requested_targets=requested_targets, backend_targets=backend_targets,
            probes=probes, include_mathematics=not arguments.no_mathematics,
            destination=destination,
        )

    summary = {
        "ok": not errors,
        "destination": str(destination),
        "in_root_repo": destination != TURING_REPOSITORY_ROOT,
        "built": len(results),
        "failed": len(errors),
        "bundles": results,
        "errors": errors,
    }
    encoded = json.dumps(summary)
    if arguments.result_json:
        arguments.result_json.write_text(encoded, encoding="utf-8")
    print(encoded)

    if results:
        _announce(
            f"done: published {len(results)} bundle(s) to the root repo at "
            f"{destination}"
        )
    if errors:
        _announce(f"{len(errors)} bundle(s) failed; see errors in the result JSON")
    return 1 if errors and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
