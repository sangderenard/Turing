"""Scan a clean git checkout into neutral repository world-map records.

This is the producer for the repository world map described in
``docs/REPO_WORLD_MAP_DESIGN.md``.  It knows nothing about HTML, geometry,
or the Living Data Map page.  Given the path of a clone it emits a plain
record set:

    regions   one per tracked directory (the repository root included);
              a directory's parent region is its parent directory
    rooms     one per tracked file
    portals   one PAIR per resolved cross-file reference: an ``out`` end on
              the referencing file's room and an ``in`` end on the referenced
              file's room, with the source line that made the reference

Only two reference extractors exist: Python imports read with :mod:`ast`,
and JavaScript ES-module static imports (``import … from``, bare
``import "…"``, ``export … from``) read line by line.  A reference whose
target is not a tracked file is kept as an *unresolved reference* so nothing
disappears silently; it is not a portal because a portal needs two rooms.

Every identity is keyed by the commit hash, so two scans of the same commit
produce identical records and a scan of a different commit never collides.
The scanner refuses a tree with uncommitted or untracked changes: the record
set describes a commit, never a working state.

The sanctioned input is a GitHub repository, not the owner's local tree: the
local scratch checkout is messy and holds things that must not be exposed.
:func:`scan_github_repository` makes a fresh shallow clone of
``https://github.com/<owner>/<repo>.git`` in a temporary directory (Git LFS
smudge disabled, so no large binaries are fetched), scans it, and deletes the
clone.  Only what GitHub serves can reach the record set.

    python -m src.compiler.repo_world_scan owner/repo [--ref main] [--json]
    python -m src.compiler.repo_world_scan --path <clone-path> [--json]   # fixtures
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

REPO_WORLD_SCAN_VERSION = "repo-world-scan-v0"
GITHUB_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

PYTHON_SUFFIXES = (".py",)
JAVASCRIPT_SUFFIXES = (".js", ".mjs")
_JS_RESOLUTION_SUFFIXES = ("", ".js", ".mjs", "/index.js", "/index.mjs")

# ES-module static import forms.  The specifier is always the last string
# literal on the statement line, so one pattern per form is enough.
_JS_IMPORT_PATTERNS = (
    # import x from "spec" / import {a} from "spec" / import * as ns from "spec"
    re.compile(r"""^\s*import\s+(?:[^'";]*?)\s+from\s+(['"])([^'"]+)\1"""),
    # import "spec"
    re.compile(r"""^\s*import\s+(['"])([^'"]+)\1"""),
    # export {a} from "spec" / export * from "spec"
    re.compile(r"""^\s*export\s+(?:[^'";]*?)\s+from\s+(['"])([^'"]+)\1"""),
)


class RepoScanError(RuntimeError):
    """The path is not a scannable repository clone."""


class DirtyCloneError(RepoScanError):
    """The checkout has uncommitted or untracked changes."""


@dataclass(frozen=True, slots=True)
class RepoRegion:
    """A directory.  ``parent`` is None only for the repository root."""

    identity: str
    path: str
    name: str
    parent: str | None
    depth: int

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "region", "path": self.path,
            "name": self.name, "parent": self.parent, "depth": self.depth,
            "dependencies": ([] if self.parent is None else [
                {"relationship": "contained-by", "target": self.parent},
            ]),
        }


@dataclass(frozen=True, slots=True)
class RepoRoom:
    """A tracked file inside exactly one region."""

    identity: str
    path: str
    name: str
    region: str
    language: str

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "room", "path": self.path,
            "name": self.name, "region": self.region, "language": self.language,
            "dependencies": [{"relationship": "contained-by", "target": self.region}],
        }


@dataclass(frozen=True, slots=True)
class RepoPortalEnd:
    """One end of a portal pair, owned by one room's boundary.

    ``transmission`` is the declared, empty slot that a later consumer (the
    particle simulation, the studio's shaders or ray tracers) fills with
    optical or transport properties.  The scanner never writes it.
    """

    identity: str
    room: str
    role: str  # "out" on the referencing room, "in" on the referenced room
    pair: str
    transmission: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.role not in {"in", "out"}:
            raise ValueError(f"portal end role must be 'in' or 'out', not {self.role!r}")

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "portal-end", "room": self.room,
            "role": self.role, "pair": self.pair,
            "transmission": None if self.transmission is None else dict(self.transmission),
            "dependencies": [
                {"relationship": "on-boundary-of", "target": self.room},
                {"relationship": "end-of", "target": self.pair},
            ],
        }


@dataclass(frozen=True, slots=True)
class RepoPortalPair:
    """A directed cross-file reference, realized as two portal ends."""

    identity: str
    source_room: str
    target_room: str
    source_line: int
    reference_kind: str  # "python-import" | "es-module-import"
    specifier: str       # the text the source used to name the target
    out_end: RepoPortalEnd
    in_end: RepoPortalEnd

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "portal-pair",
            "source_room": self.source_room, "target_room": self.target_room,
            "direction": "source-to-target", "source_line": self.source_line,
            "reference_kind": self.reference_kind, "specifier": self.specifier,
            "ends": [self.out_end.to_data(), self.in_end.to_data()],
        }


@dataclass(frozen=True, slots=True)
class RepoUnresolvedReference:
    """A reference whose target is not a tracked file (stdlib, package, typo)."""

    source_room: str
    source_line: int
    reference_kind: str
    specifier: str
    reason: str

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": "unresolved-reference", "source_room": self.source_room,
            "source_line": self.source_line, "reference_kind": self.reference_kind,
            "specifier": self.specifier, "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepoWorldScan:
    commit: str
    origin: str | None
    regions: tuple[RepoRegion, ...]
    rooms: tuple[RepoRoom, ...]
    portals: tuple[RepoPortalPair, ...]
    unresolved: tuple[RepoUnresolvedReference, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def root_region(self) -> RepoRegion:
        return self.regions[0]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": REPO_WORLD_SCAN_VERSION,
            "identity": f"repo:{self.commit}",
            "commit": self.commit,
            "origin": self.origin,
            "regions": [item.to_data() for item in self.regions],
            "rooms": [item.to_data() for item in self.rooms],
            "portals": [item.to_data() for item in self.portals],
            "unresolved_references": [item.to_data() for item in self.unresolved],
            "notes": list(self.notes),
        }


# --- git -----------------------------------------------------------------------

def _git(clone: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(clone), *args],
            capture_output=True, text=True, check=True, encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise RepoScanError("git is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RepoScanError(
            f"git {' '.join(args)} failed in {clone}: {exc.stderr.strip()}"
        ) from exc
    return out.stdout


def _require_clean_clone(clone: Path) -> None:
    toplevel = _git(clone, "rev-parse", "--show-toplevel").strip()
    if Path(toplevel).resolve() != clone.resolve():
        raise RepoScanError(
            f"{clone} is not the top level of its repository ({toplevel})"
        )
    status = _git(clone, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        lines = status.rstrip("\n").splitlines()
        shown = "\n".join(lines[:10])
        more = "" if len(lines) <= 10 else f"\n... {len(lines) - 10} more"
        raise DirtyCloneError(
            f"{clone} has uncommitted or untracked changes; the scan only "
            f"describes a clean checkout:\n{shown}{more}"
        )


def _tracked_files(clone: Path) -> tuple[str, ...]:
    raw = _git(clone, "ls-files", "-z")
    return tuple(sorted(part for part in raw.split("\0") if part))


# --- identities ----------------------------------------------------------------

def region_identity(commit: str, directory: str) -> str:
    return f"repo:{commit}/dir:{directory}"


def room_identity(commit: str, path: str) -> str:
    return f"repo:{commit}/file:{path}"


def _pair_identity(commit: str, source: str, line: int, target: str) -> str:
    return f"repo:{commit}/portal:{source}:{line}->{target}"


# --- tree ----------------------------------------------------------------------

def _language(path: str) -> str:
    suffix = PurePosixPath(path).suffix
    if suffix in PYTHON_SUFFIXES:
        return "python"
    if suffix in JAVASCRIPT_SUFFIXES:
        return "javascript"
    return "other"


def _build_tree(commit: str, files: Iterable[str]) -> tuple[tuple[RepoRegion, ...], tuple[RepoRoom, ...]]:
    directories: dict[str, RepoRegion] = {
        ".": RepoRegion(region_identity(commit, "."), ".", ".", None, 0),
    }
    rooms: list[RepoRoom] = []
    for path in files:
        parts = PurePosixPath(path).parts
        parent = "."
        for depth, part in enumerate(parts[:-1], start=1):
            directory = part if parent == "." else f"{parent}/{part}"
            if directory not in directories:
                directories[directory] = RepoRegion(
                    region_identity(commit, directory), directory, part,
                    directories[parent].identity, depth,
                )
            parent = directory
        rooms.append(RepoRoom(
            room_identity(commit, path), path, parts[-1],
            directories[parent].identity, _language(path),
        ))
    regions = tuple(sorted(directories.values(), key=lambda r: (r.depth, r.path)))
    return regions, tuple(rooms)


# --- references ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Reference:
    source: str
    line: int
    kind: str
    specifier: str
    target: str | None
    reason: str = ""


def _python_module_candidates(module: str) -> tuple[str, ...]:
    base = module.replace(".", "/")
    return (f"{base}.py", f"{base}/__init__.py")


def _resolve_python(module: str, tracked: frozenset[str]) -> str | None:
    for candidate in _python_module_candidates(module):
        if candidate in tracked:
            return candidate
    return None


def _python_references(path: str, text: str, tracked: frozenset[str]) -> tuple[list[_Reference], str | None]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        return [], f"{path}: not parsed as Python ({exc.msg} at line {exc.lineno})"
    package_parts = PurePosixPath(path).parts[:-1]
    refs: list[_Reference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_python(alias.name, tracked)
                refs.append(_Reference(
                    path, node.lineno, "python-import", alias.name, target,
                    "" if target else "no tracked module at that absolute name",
                ))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                up = node.level - 1
                if up > len(package_parts):
                    refs.append(_Reference(
                        path, node.lineno, "python-import",
                        "." * node.level + (node.module or ""), None,
                        "relative import escapes the repository root",
                    ))
                    continue
                base_parts = package_parts[: len(package_parts) - up]
                base = ".".join(base_parts)
                module = ".".join(part for part in (base, node.module or "") if part)
                spelled = "." * node.level + (node.module or "")
            else:
                module = node.module or ""
                spelled = module
            # ``from pkg import name`` names a submodule when pkg/name.py is
            # tracked; otherwise the reference is to the module ``pkg`` itself.
            module_target = _resolve_python(module, tracked) if module else None
            for alias in node.names:
                if alias.name == "*":
                    target, specifier = module_target, spelled
                else:
                    submodule = f"{module}.{alias.name}" if module else alias.name
                    target = _resolve_python(submodule, tracked) or module_target
                    specifier = f"{spelled}:{alias.name}"
                refs.append(_Reference(
                    path, node.lineno, "python-import", specifier, target,
                    "" if target else "no tracked module at that name",
                ))
    return refs, None


def _resolve_javascript(source: str, specifier: str, tracked: frozenset[str]) -> tuple[str | None, str]:
    if not specifier.startswith(("./", "../", "/")):
        return None, "bare specifier (package or URL), not a repository path"
    if specifier.startswith("/"):
        return None, "absolute URL path has no repository anchor"
    base = PurePosixPath(source).parent
    joined = PurePosixPath(*(base / specifier).parts)
    parts: list[str] = []
    for part in joined.parts:
        if part == "..":
            if not parts:
                return None, "relative specifier escapes the repository root"
            parts.pop()
        elif part not in {".", ""}:
            parts.append(part)
    stem = "/".join(parts)
    for suffix in _JS_RESOLUTION_SUFFIXES:
        candidate = f"{stem}{suffix}"
        if candidate in tracked:
            return candidate, ""
    return None, "no tracked file at that relative path"


def _javascript_references(path: str, text: str, tracked: frozenset[str]) -> list[_Reference]:
    refs: list[_Reference] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern in _JS_IMPORT_PATTERNS:
            match = pattern.match(line)
            if match:
                specifier = match.group(2)
                target, reason = _resolve_javascript(path, specifier, tracked)
                refs.append(_Reference(path, number, "es-module-import", specifier, target, reason))
                break
    return refs


def _read(clone: Path, path: str) -> str:
    return (clone / path).read_text(encoding="utf-8", errors="replace")


# --- entry ---------------------------------------------------------------------

def scan_repository(clone_path: str | Path) -> RepoWorldScan:
    """Scan one clean clone into regions, rooms and portal pairs.

    Raises :class:`DirtyCloneError` when the tree has uncommitted or untracked
    changes and :class:`RepoScanError` when the path is not a repository root.
    """

    clone = Path(clone_path)
    if not clone.is_dir():
        raise RepoScanError(f"{clone} is not a directory")
    _require_clean_clone(clone)
    commit = _git(clone, "rev-parse", "HEAD").strip()
    origin_raw = subprocess.run(
        ["git", "-C", str(clone), "config", "--get", "remote.origin.url"],
        capture_output=True, text=True, encoding="utf-8",
    )
    origin = origin_raw.stdout.strip() or None

    files = _tracked_files(clone)
    tracked = frozenset(files)
    regions, rooms = _build_tree(commit, files)

    references: list[_Reference] = []
    notes: list[str] = []
    for path in files:
        language = _language(path)
        if language == "python":
            refs, note = _python_references(path, _read(clone, path), tracked)
            references.extend(refs)
            if note:
                notes.append(note)
        elif language == "javascript":
            references.extend(_javascript_references(path, _read(clone, path), tracked))

    portals: list[RepoPortalPair] = []
    unresolved: list[RepoUnresolvedReference] = []
    seen_pairs: set[str] = set()
    for ref in sorted(references, key=lambda r: (r.source, r.line, r.specifier)):
        source_room = room_identity(commit, ref.source)
        if ref.target is None:
            unresolved.append(RepoUnresolvedReference(
                source_room, ref.line, ref.kind, ref.specifier, ref.reason,
            ))
            continue
        if ref.target == ref.source:
            unresolved.append(RepoUnresolvedReference(
                source_room, ref.line, ref.kind, ref.specifier,
                "self-reference; a portal needs two rooms",
            ))
            continue
        pair_id = _pair_identity(commit, ref.source, ref.line, ref.target)
        if pair_id in seen_pairs:
            # Two names imported on one line from one module are one portal.
            continue
        seen_pairs.add(pair_id)
        target_room = room_identity(commit, ref.target)
        portals.append(RepoPortalPair(
            pair_id, source_room, target_room, ref.line, ref.kind, ref.specifier,
            RepoPortalEnd(f"{pair_id}/end:out", source_room, "out", pair_id),
            RepoPortalEnd(f"{pair_id}/end:in", target_room, "in", pair_id),
        ))

    return RepoWorldScan(
        commit, origin, regions, rooms, tuple(portals), tuple(unresolved), tuple(notes),
    )


def github_clone_url(repository: str) -> str:
    """``owner/repo`` -> the public HTTPS clone URL.  Nothing else is accepted."""

    if not GITHUB_REPOSITORY.match(repository):
        raise RepoScanError(
            f"{repository!r} is not a GitHub repository spelled owner/repo"
        )
    return f"https://github.com/{repository}.git"


def scan_cloned_repository(clone_url: str, *, ref: str | None = None) -> RepoWorldScan:
    """Clone ``clone_url`` fresh into a temporary directory, scan it, delete it.

    The clone is shallow (one commit) and Git LFS smudge is disabled, so the
    scan reads only the tracked tree at that commit and no large binaries.
    The clone never persists; the record set names the commit it came from.
    """

    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    with tempfile.TemporaryDirectory(prefix="repo-world-scan-") as tmp:
        clone = Path(tmp) / "clone"
        command = ["git", "clone", "--quiet", "--depth", "1", "--no-tags"]
        if ref:
            command += ["--branch", ref]
        command += [clone_url, str(clone)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True,
                           encoding="utf-8", env=env)
        except FileNotFoundError as exc:
            raise RepoScanError("git is not on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RepoScanError(
                f"git clone of {clone_url}"
                f"{' at ' + ref if ref else ''} failed: {exc.stderr.strip()}"
            ) from exc
        return scan_repository(clone)


def scan_github_repository(repository: str, *, ref: str | None = None) -> RepoWorldScan:
    """Scan ``owner/repo`` as GitHub serves it, never a local working tree."""

    return scan_cloned_repository(github_clone_url(repository), ref=ref)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [item for item in args if item != "--json"]
    ref = None
    if "--ref" in args:
        index = args.index("--ref")
        if index + 1 >= len(args):
            print("--ref needs a branch or tag", file=sys.stderr)
            return 2
        ref = args[index + 1]
        del args[index:index + 2]
    local_path = None
    if "--path" in args:
        index = args.index("--path")
        if index + 1 >= len(args):
            print("--path needs a clone directory", file=sys.stderr)
            return 2
        local_path = args[index + 1]
        del args[index:index + 2]
    usage = "\n".join(__doc__.strip().splitlines()[-2:])
    try:
        if local_path is not None:
            if args or ref:
                print(usage, file=sys.stderr)
                return 2
            scan = scan_repository(local_path)
        elif len(args) == 1:
            scan = scan_github_repository(args[0], ref=ref)
        else:
            print(usage, file=sys.stderr)
            return 2
    except RepoScanError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if as_json:
        json.dump(scan.to_data(), sys.stdout, indent=2, sort_keys=False)
        print()
        return 0
    print(f"commit {scan.commit}")
    print(f"regions {len(scan.regions)}  rooms {len(scan.rooms)}  portal pairs {len(scan.portals)}  unresolved {len(scan.unresolved)}")
    for portal in scan.portals:
        print(f"  {portal.source_room.split('/file:', 1)[1]}:{portal.source_line} -> "
              f"{portal.target_room.split('/file:', 1)[1]}  [{portal.reference_kind} {portal.specifier}]")
    for note in scan.notes:
        print(f"  note: {note}")
    return 0


__all__ = [
    "REPO_WORLD_SCAN_VERSION", "DirtyCloneError", "RepoPortalEnd", "RepoPortalPair",
    "RepoRegion", "RepoRoom", "RepoScanError", "RepoUnresolvedReference",
    "RepoWorldScan", "github_clone_url", "region_identity", "room_identity",
    "scan_cloned_repository", "scan_github_repository", "scan_repository",
]


if __name__ == "__main__":
    sys.exit(main())
